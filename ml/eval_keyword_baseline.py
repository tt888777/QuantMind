"""
关键词规则基线评测 —— 与 LoRA 模型在同一测试集上对照
=====================================================
直接复用 quantmind/modules/sentiment.py 里的 analyze_sentiment()（真实实现，不改动），
把它的中文标签映射到 3 类（0=看跌/1=看涨/2=中性），在 guba_title_all.csv 的 test 划分上评估。

这样得到的数字才和 models/lora_guba/metrics.json 可比。

用法：
  python eval_keyword_baseline.py
"""
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parent
SENTIMENT_PY = ROOT.parent / "quantmind" / "modules" / "sentiment.py"

LABEL_NAMES = {0: "negative/看跌", 1: "positive/看涨", 2: "neutral/中性"}

# 关键词引擎输出 -> 3 类标签
KW_TO_CLASS = {
    "强烈看涨": 1,
    "看涨": 1,
    "看跌": 0,
    "中性偏空": 0,
    "中性": 2,
    "中性偏多": 1,
}


def load_keyword_engine():
    """从源码文件加载 analyze_sentiment，避免触发包级依赖。"""
    spec = importlib.util.spec_from_file_location("_sentiment_mod", SENTIMENT_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.analyze_sentiment


def load_test_rows(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("split") or "").strip() != "test":
                continue
            title = (r.get("title") or "").strip()
            if not title:
                continue
            try:
                label = int(float(r.get("label", "")))
            except ValueError:
                continue
            if label not in LABEL_NAMES:
                continue
            rows.append((title, label))
    return rows


def main():
    analyze = load_keyword_engine()
    rows = load_test_rows(ROOT / "data" / "guba_title_all.csv")
    print(f"测试集样本数: {len(rows):,}")

    y_true, y_pred, raw_dist = [], [], Counter()
    for title, label in rows:
        raw_label, _conf = analyze(title)
        raw_dist[raw_label] += 1
        y_true.append(label)
        y_pred.append(KW_TO_CLASS.get(raw_label, 2))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    report = classification_report(
        y_true, y_pred, labels=sorted(LABEL_NAMES),
        target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
        zero_division=0, digits=3,
    )

    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  Macro-F1 : {macro_f1:.4f}")
    print()
    print(report)

    print("关键词引擎原始输出分布（映射前）:")
    for k, v in raw_dist.most_common():
        print(f"  {k:8} {v:6,}  ({v / len(rows) * 100:.1f}%)")

    print("\n模型预测分布（映射后 3 类）:")
    for k, v in sorted(Counter(y_pred.tolist()).items()):
        print(f"  {LABEL_NAMES[k]:14} {v:6,}  ({v / len(rows) * 100:.1f}%)")
    print("真实标签分布:")
    for k, v in sorted(Counter(y_true.tolist()).items()):
        print(f"  {LABEL_NAMES[k]:14} {v:6,}  ({v / len(rows) * 100:.1f}%)")

    out = {
        "method": "keyword_rule (quantmind/modules/sentiment.py:analyze_sentiment)",
        "test_rows": len(rows),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "report": report,
        "raw_label_dist": dict(raw_dist),
        "pred_class_dist": {LABEL_NAMES[k]: v for k, v in sorted(Counter(y_pred.tolist()).items())},
        "true_class_dist": {LABEL_NAMES[k]: v for k, v in sorted(Counter(y_true.tolist()).items())},
    }
    outp = ROOT / "out" / "keyword_baseline_metrics.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n指标 -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
