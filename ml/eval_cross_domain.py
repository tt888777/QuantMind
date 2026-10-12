"""
跨域（交叉）评测：拿 N 个模型 × M 个测试集，跑一张完整对照矩阵
=============================================================
动机：本项目有两个不同域的监督数据集
  - 股吧标题  ``data/guba_title_all.csv``     （散户口语，平均 20.6 字，上证50ETF）
  - 财经新闻  ``data/flame_zh_all.csv``       （媒体书面，平均 29.1 字，全市场）
在各自域内训练的模型，换到对方域上会掉多少？这张矩阵就是这个问题的答案。

用法：
  # 单个模型
  python eval_cross_domain.py --model models/lora_guba/merged --data data/guba_title_all.csv

  # 完整矩阵（模型 与 数据集 都可用「名字=路径」重复给）
  python eval_cross_domain.py \
    --model  股吧模型=models/lora_guba/merged \
    --model  新闻模型=models/lora_flame/merged \
    --data   股吧测试集=data/guba_title_all.csv \
    --data   新闻测试集=data/flame_zh_all.csv \
    --split test --out out/cross_domain_matrix.json
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
LABEL_NAMES = {0: "negative/看跌", 1: "positive/看涨", 2: "neutral/中性"}


def parse_kv(items, default_name):
    """把 ['名字=路径', ...] 或 ['路径', ...] 解析成 [(名字, 路径)]。"""
    out = []
    for i, it in enumerate(items or []):
        if "=" in it and not Path(it).exists():
            name, path = it.split("=", 1)
        else:
            name, path = (default_name + str(i + 1) if len(items) > 1 else default_name), it
        out.append((name.strip(), path.strip()))
    return out


def load_split(path: Path, split: str):
    """读 ``title,label,split`` 格式 CSV，返回 (titles, labels)。"""
    titles, labels = [], []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if split != "all" and (r.get("split") or "").strip() != split:
                continue
            t = (r.get("title") or "").strip()
            if not t:
                continue
            try:
                y = int(float(r.get("label", "")))
            except ValueError:
                continue
            if y not in LABEL_NAMES:
                continue
            titles.append(t)
            labels.append(y)
    return titles, labels


@torch.no_grad()
def predict(model, tok, titles, max_len, batch_size, device):
    preds = []
    for i in range(0, len(titles), batch_size):
        chunk = titles[i:i + batch_size]
        enc = tok(chunk, truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(device)
        logits = model(**enc).logits
        preds.extend(logits.argmax(-1).cpu().tolist())
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True, help="名字=模型目录（可重复）")
    p.add_argument("--data", action="append", required=True, help="名字=CSV 路径（可重复）")
    p.add_argument("--split", default="test", choices=["train", "test", "all"])
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out", default=str(ROOT / "out" / "cross_domain_matrix.json"))
    args = p.parse_args()

    models = parse_kv(args.model, "model")
    datasets = parse_kv(args.data, "data")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 78)
    print(f"跨域评测   split={args.split}   device={device}   max_len={args.max_len}")
    print("=" * 78)

    # 先加载所有测试集
    loaded = {}
    for name, path in datasets:
        titles, labels = load_split(Path(path), args.split)
        loaded[name] = (titles, labels)
        dist = Counter(LABEL_NAMES[y] for y in labels)
        print(f"  [数据] {name:<12} {len(titles):>6} 条   {dict(dist)}")

    print()
    results, reports = {}, {}
    for mname, mpath in models:
        print(f"  [模型] 加载 {mname} <- {mpath}")
        tok = AutoTokenizer.from_pretrained(mpath)
        mdl = AutoModelForSequenceClassification.from_pretrained(mpath).to(device).eval()
        results[mname] = {}
        for dname, (titles, labels) in loaded.items():
            preds = predict(mdl, tok, titles, args.max_len, args.batch_size, device)
            acc = float(accuracy_score(labels, preds))
            f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
            results[mname][dname] = {"n": len(titles), "accuracy": round(acc, 4),
                                     "macro_f1": round(f1, 4)}
            reports[f"{mname} on {dname}"] = classification_report(
                labels, preds, labels=sorted(LABEL_NAMES),
                target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
                zero_division=0, output_dict=True)
            print(f"        {mname} × {dname:<12} Acc {acc:.4f}  Macro-F1 {f1:.4f}")
        del mdl
        torch.cuda.empty_cache()

    # ---- 打印矩阵 ----
    dnames = [n for n, _ in datasets]
    print("\n" + "=" * 78)
    print("对照矩阵（对角线=同域，非对角线=跨域）")
    print("=" * 78)
    w = max(14, max(len(n) for n, _ in models) + 2)
    print(f"{'模型 \\ 测试集':<{w}}" + "".join(f"{d:>16}" for d in dnames))
    for mname, _ in models:
        row = f"{mname:<{w}}"
        for d in dnames:
            r = results[mname][d]
            row += f"{r['accuracy']:>16.4f}"
        print(row)
    print(f"\n{'（下表为 Macro-F1）':<{w - 4}}")
    print(f"{'':<{w}}" + "".join(f"{d:>16}" for d in dnames))
    for mname, _ in models:
        row = f"{mname:<{w}}"
        for d in dnames:
            r = results[mname][d]
            row += f"{r['macro_f1']:>16.4f}"
        print(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split": args.split, "results": results,
                               "reports": reports}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n已写出 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
