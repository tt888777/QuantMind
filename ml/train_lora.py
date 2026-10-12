"""
LoRA 监督微调 —— 中文金融情感三分类（看跌/看涨/中性）
=====================================================
流程：底座模型（原始 BERT 或 DAPT 输出）→ 挂 LoRA adapter + 分类头 → 微调 → 评估 → 导出。

设计要点（针对本机 RTX 3060 6GB）：
  * 仅 LoRA 参数可训练，显存占用约 2.4GB
  * 分类头通过 modules_to_save 一起训练，否则只训 adapter 会欠拟合
  * dataloader_num_workers=0（Windows 下多进程 DataLoader 易卡死）
  * --smoke N 只跑 N 步，用于快速验证链路

用法：
  python train_lora.py --smoke 30                      # 最小验证
  python train_lora.py --epochs 5 --merge              # 正式训练并导出合并模型
  python train_lora.py --base models/dapt_bert --out models/lora_dapt  # 基于 DAPT 底座
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import csv
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

LABEL_NAMES = {0: "negative/看跌", 1: "positive/看涨", 2: "neutral/中性"}
ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
class TitleDataset(Dataset):
    def __init__(self, titles, labels, tokenizer, max_len):
        self.enc = tokenizer(
            titles, truncation=True, max_length=max_len, padding=False
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def load_rows(path: Path):
    """读取 title/label/split 三列的 CSV。"""
    train, test = [], []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            title = (r.get("title") or "").strip()
            if not title:
                continue
            try:
                label = int(float(r.get("label", "")))
            except ValueError:
                continue
            if label not in LABEL_NAMES:
                continue
            rec = (title, label)
            (test if (r.get("split") or "").strip() == "test" else train).append(rec)
    return train, test


# --------------------------------------------------------------------------
# 兼容性 shim（transformers 版本间参数名有变动）
# --------------------------------------------------------------------------
def split_titles(titles, labels):
    return {"text": titles}, np.array(labels)


def build_training_args(out: Path, args) -> TrainingArguments:
    """按当前 transformers 版本的实际签名构造参数（v4 / v5 API 有差异）。"""
    params = set(inspect.signature(TrainingArguments.__init__).parameters)
    kw = dict(
        output_dir=str(out / "_ckpt"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        seed=42,
        fp16=False,
    )
    # v5 删除了 warmup_ratio，改用 warmup_steps
    if "warmup_ratio" in params:
        kw["warmup_ratio"] = 0.1
    else:
        kw["warmup_steps"] = 50
    # v5 把 evaluation_strategy 改名为 eval_strategy
    kw["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "epoch"
    if args.smoke:
        kw["max_steps"] = args.smoke
        kw["num_train_epochs"] = 1
        kw["logging_steps"] = 5
        kw["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "no"
    kw = {k: v for k, v in kw.items() if k in params}
    return TrainingArguments(**kw)


def build_trainer(model, tokenizer, train_ds, eval_ds, targs, collator):
    params = set(inspect.signature(Trainer.__init__).parameters)
    tok_kw = "processing_class" if "processing_class" in params else "tokenizer"

    def compute_metrics(pred):
        logits, labels = pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": round(float(accuracy_score(labels, preds)), 4),
            "macro_f1": round(float(f1_score(labels, preds, average="macro", zero_division=0)), 4),
        }

    return Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        **{tok_kw: tokenizer},
    )


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="bert-base-chinese", help="底座模型路径或 HF id")
    p.add_argument("--data", default=str(ROOT / "data" / "guba_title_all.csv"))
    p.add_argument("--out", default=str(ROOT / "models" / "lora_guba"))
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4, help="LoRA 常用 1e-4 ~ 5e-4")
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--smoke", type=int, default=0, help=">0 时只跑这么多步做验证")
    p.add_argument("--merge", action="store_true", help="训练后导出合并权重")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("LoRA 监督微调 · 中文金融情感三分类")
    print("=" * 64)
    print(f"  底座模型 : {args.base}")
    print(f"  数据     : {args.data}")
    print(f"  设备     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"  显存     : {free/2**30:.1f}GB 可用 / {total/2**30:.1f}GB 总量")
    print(f"  模式     : {'SMOKE(' + str(args.smoke) + ' 步)' if args.smoke else '正式训练'}")

    print("\n[1/5] 加载数据")
    train_rows, test_rows = load_rows(Path(args.data))
    print(f"  train {len(train_rows):,} 条 / test {len(test_rows):,} 条")
    if args.smoke:
        train_rows, test_rows = train_rows[:512], test_rows[:256]
        print(f"  [SMOKE] 截断为 train {len(train_rows)} / test {len(test_rows)}")

    print("\n[2/5] 加载 tokenizer 与底座模型")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base,
        num_labels=len(LABEL_NAMES),
        id2label={i: v for i, v in LABEL_NAMES.items()},
        label2id={v: i for i, v in LABEL_NAMES.items()},
    )

    print("\n[3/5] 挂 LoRA adapter")
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query", "value"],
        modules_to_save=["classifier"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数 : {trainable:,} / {total_p:,}  ({trainable / total_p * 100:.2f}%)")

    print("\n[4/5] 构建数据集并训练")
    tr_titles = [r[0] for r in train_rows]
    tr_labels = [r[1] for r in train_rows]
    te_titles = [r[0] for r in test_rows]
    te_labels = [r[1] for r in test_rows]

    train_ds = TitleDataset(tr_titles, tr_labels, tokenizer, args.max_len)
    eval_ds = TitleDataset(te_titles, te_labels, tokenizer, args.max_len) if te_titles else None
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    targs = build_training_args(out, args)
    trainer = build_trainer(model, tokenizer, train_ds, eval_ds, targs, collator)

    t0 = time.time()
    trainer.train()
    print(f"  训练耗时 : {time.time() - t0:.1f}s")

    print("\n[5/5] 评估与导出")
    metrics = {}
    if eval_ds is not None:
        pred = trainer.predict(eval_ds)
        preds = np.argmax(pred.predictions, axis=-1)
        metrics = {
            "accuracy": round(float(accuracy_score(te_labels, preds)), 4),
            "macro_f1": round(float(f1_score(te_labels, preds, average="macro", zero_division=0)), 4),
            "report": classification_report(
                te_labels, preds,
                labels=sorted(LABEL_NAMES), target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
                zero_division=0, output_dict=True,
            ),
        }
        print(f"  Accuracy : {metrics['accuracy']}")
        print(f"  Macro-F1 : {metrics['macro_f1']}")
        print(classification_report(
            te_labels, preds, labels=sorted(LABEL_NAMES),
            target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)], zero_division=0, digits=3,
        ))

    model.save_pretrained(str(out / "adapter"))
    tokenizer.save_pretrained(str(out / "adapter"))
    merged = None
    if args.merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(str(out / "merged"), safe_serialization=True)
        tokenizer.save_pretrained(str(out / "merged"))
        print(f"  合并模型 -> {(out / 'merged')}")

    # 导出「底座 checkpoint 里没有、推理时又必须一致」的参数（典型是 bert.pooler）。
    # 不导出的话，任何「重新从 adapter 现场合并」的路径都会抽到另一份随机 pooler，
    # 分类头随之失效（详见 ml/lora_extra.py 的说明）。
    try:
        from lora_extra import export_extra_state

        n_extra = export_extra_state(merged if merged is not None else model,
                                     Path(args.base), out / "adapter")
        if n_extra:
            print(f"  额外权重 -> {out / 'adapter' / 'extra_state.safetensors'}（{n_extra} 个张量）")
    except Exception as e:
        print(f"  [警告] 额外权重导出失败（不影响本次训练，但 adapter 可能无法独立复现）：{e}")

    meta = {
        "base_model": args.base,
        "data": args.data,
        "smoke": args.smoke,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_len": args.max_len,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "trainable_params": trainable,
        "total_params": total_p,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "metrics": metrics,
    }
    (out / "metrics.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  adapter  -> {out / 'adapter'}")
    print(f"  指标     -> {out / 'metrics.json'}")
    print("\n完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
