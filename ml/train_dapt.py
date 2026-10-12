"""
DAPT 域自适应预训练（MLM 继续预训练）
=====================================
在中文金融语料上对 BERT 做二次预训练，让底座吸收领域词与表达，再交给 LoRA 微调。

语料来源（三选一）：
  * 本地 CSV ：--corpus data/guba_title_all.csv --text_col title
  * 本地 TXT ：--corpus corpus.txt          （每行一条文本）
  * HF 流式  ：--corpus hf:mulfsa/CN-Market-Corpus --hf_column text
               （流式取前 N 条，不会下载 13.8GB 全量）

显存策略（RTX 3060 6GB）：
  默认 batch=16 / seq_len=128 / fp16，峰值约 4GB。
  若仍 OOM，脚本会自动减半 batch 并从 checkpoint 重试。

用法：
  python train_dapt.py --corpus data/guba_title_all.csv --text_col title --smoke 20
  python train_dapt.py --corpus hf:mulfsa/CN-Market-Corpus --max_samples 200000 --epochs 1
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

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# 语料读取
# --------------------------------------------------------------------------
def read_local(path: Path, text_col: str):
    texts = []
    if path.suffix.lower() in (".txt", ".md"):
        texts = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    else:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"{path.name} 无表头")
            col = text_col if text_col in reader.fieldnames else reader.fieldnames[0]
            for r in reader:
                v = (r.get(col) or "").strip()
                if v:
                    texts.append(v)
    return [t for t in texts if len(t) >= 2]


def read_hf(spec: str, column: str, max_samples: int, split: str):
    from datasets import load_dataset

    dataset_id = spec.split(":", 1)[1]
    print(f"  [流式] {dataset_id} split={split} 取前 {max_samples:,} 条")
    ds = load_dataset(dataset_id, split=split, streaming=True)
    cols = list(getattr(ds, "features", {}).keys()) if hasattr(ds, "features") else []
    col = column if (not cols or column in cols) else (cols[0] if cols else column)
    texts = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        v = str(row.get(col, "") or "").strip()
        if len(v) >= 2:
            texts.append(v)
        if (i + 1) % 20000 == 0:
            print(f"    已读取 {i + 1:,} 条 ...")
    return texts


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.enc = tokenizer(
            texts, truncation=True, max_length=max_len, padding=False,
            return_special_tokens_mask=True,
        )

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}


def build_training_args(out: Path, args, batch_size, params_names):
    """按当前 transformers 版本的实际签名构造参数（v4 / v5 API 有差异）。"""
    kw = dict(
        output_dir=str(out / "_ckpt"),
        learning_rate=args.lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        seed=42,
        max_grad_norm=1.0,
    )
    # 混合精度：sm_80+ 优先 bf16（数值更稳，fp16 在 MLM 上易出现 inf 梯度）
    if params_names and "bf16" in params_names:
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        kw["bf16"] = use_bf16
        kw["fp16"] = (not use_bf16) and torch.cuda.is_available() and args.fp16
    else:
        kw["fp16"] = args.fp16 and torch.cuda.is_available()
    if "warmup_ratio" in params_names:
        kw["warmup_ratio"] = 0.05
    else:
        kw["warmup_steps"] = 100
    kw["eval_strategy" if "eval_strategy" in params_names else "evaluation_strategy"] = "no"
    if args.smoke:
        kw["max_steps"] = args.smoke
        kw["num_train_epochs"] = 1
        kw["logging_steps"] = 1
    kw = {k: v for k, v in kw.items() if k in params_names}
    return TrainingArguments(**kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="bert-base-chinese")
    p.add_argument("--corpus", default=str(ROOT / "data" / "guba_title_all.csv"))
    p.add_argument("--text_col", default="title", help="本地 CSV 中的文本列名")
    p.add_argument("--hf_column", default="text", help="HF 数据集中的文本列名")
    p.add_argument("--hf_split", default="train")
    p.add_argument("--max_samples", type=int, default=200000)
    p.add_argument("--out", default=str(ROOT / "models" / "dapt_bert"))
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no_fp16", dest="fp16", action="store_false")
    p.add_argument("--smoke", type=int, default=0, help=">0 时只跑这么多步")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("DAPT 域自适应预训练 (MLM)")
    print("=" * 64)
    print(f"  底座模型 : {args.model}")
    print(f"  语料     : {args.corpus}")
    print(f"  设备     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"  显存     : {free/2**30:.1f}GB 可用 / {total/2**30:.1f}GB 总量")
    print(f"  fp16     : {args.fp16 and torch.cuda.is_available()}")

    print("\n[1/5] 读取语料")
    if args.corpus.startswith("hf:"):
        texts = read_hf(args.corpus, args.hf_column, args.max_samples, args.hf_split)
    else:
        texts = read_local(Path(args.corpus), args.text_col)[: args.max_samples]
    if args.smoke:
        texts = texts[:512]
        print(f"  [SMOKE] 截断为 {len(texts)} 条")
    print(f"  语料条数 : {len(texts):,}")
    if not texts:
        print("  [错误] 语料为空")
        return 1
    avg_len = sum(len(t) for t in texts) / len(texts)
    print(f"  平均字数 : {avg_len:.1f}")

    print("\n[2/5] 加载 tokenizer 与模型")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)

    print("\n[3/5] 构建数据集")
    dataset = TextDataset(texts, tokenizer, args.seq_len)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    print(f"  样本数   : {len(dataset):,}  序列长度上限 {args.seq_len}")

    print("\n[4/5] 训练（OOM 自动降批）")
    params_names = set(inspect.signature(TrainingArguments.__init__).parameters)
    batch = args.batch_size
    trainer, log_history = None, []
    t0 = time.time()
    for attempt in range(4):
        try:
            targs = build_training_args(out, args, batch, params_names)
            trainer = Trainer(model=model, args=targs, train_dataset=dataset, data_collator=collator)
            trainer.train()
            log_history = trainer.state.log_history
            break
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] batch={batch} 失败，清理显存并减半重试")
            torch.cuda.empty_cache()
            batch = max(batch // 2, 1)
            if attempt == 3:
                print("  [失败] 多次 OOM，请调小 --seq_len 或 --max_samples")
                return 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [OOM] batch={batch} 失败，减半重试")
                torch.cuda.empty_cache()
                batch = max(batch // 2, 1)
                continue
            raise
    print(f"  训练耗时 : {time.time() - t0:.1f}s  (最终 batch={batch})")

    print("\n[5/5] 保存")
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    losses = [{"step": h.get("step"), "loss": h.get("loss")} for h in log_history if "loss" in h]
    (out / "loss_curve.json").write_text(
        json.dumps({"config": vars(args), "final_batch": batch, "history": losses},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  模型     -> {out}")
    print(f"  loss 曲线-> {out / 'loss_curve.json'}  ({len(losses)} 个记录点)")
    if losses:
        print(f"  首/末 loss: {losses[0]['loss']} -> {losses[-1]['loss']}")
    print("\n完成。下一步：python train_lora.py --base " + str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
