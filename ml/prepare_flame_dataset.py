"""
构建 FLAME 中文财经新闻三分类数据集
=====================================
数据源：``Kenpache/multilingual-financial-sentiment-v2`` 的 ``data/zh_labeled.csv``
（FLAME — Perspective-Aware Multilingual Financial Sentiment，14,855 条中文财经新闻标题）

本脚本做四件事：
  1. **繁转简** —— 数据集含港台媒体（aastocks / 鉅亨網 / Yahoo HK / RTHK），有相当比例繁体，
     统一转成简体，贴近 A 股语境（依赖 ``zhconv``，纯 Python，无编译）。
  2. **标签映射** —— FLAME 用 ``negative/neutral/positive``，映射到本项目约定
     ``0=看跌 / 1=看涨 / 2=中性``（与 ``train_lora.py`` 的 ``LABEL_NAMES`` 一致）。
  3. **噪声过滤** —— 丢掉中文字符占比过低（混入英文/日文）与过短的条目。
  4. **分层切分** —— 按标签分层切 train/test（默认 8:2，seed=42），
     输出列 ``title,label,split``，**可直接喂给 ``train_lora.py --data``**。

用法：
  python prepare_flame_dataset.py                      # 默认 8:2 切分
  python prepare_flame_dataset.py --test_ratio 0.2      # 等价
  python prepare_flame_dataset.py --check               # 只体检，不写文件
"""
import argparse
import csv
import json
import random
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# FLAME 原标签 -> 本项目标签（与 train_lora.py 的 LABEL_NAMES 保持一致）
LABEL_MAP = {"negative": 0, "positive": 1, "neutral": 2}
LABEL_CN = {0: "看跌", 1: "看涨", 2: "中性"}

# 发布方的 UI 后缀噪声（新浪财经标题常带" _新浪新闻"）
TRAILING_NOISE = ["_新浪新闻", " - 新浪财经", " - Investing.com 香港", " - Investing.com",
                  "_东方财富网", " - 东方财富网"]


def cjk_ratio(s: str) -> float:
    if not s:
        return 0.0
    n = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    return n / len(s)


def clean_title(t: str, converter) -> str:
    t = (t or "").strip()
    for suf in TRAILING_NOISE:
        if t.endswith(suf):
            t = t[: -len(suf)].strip()
            break
    t = " ".join(t.split())          # 折叠空白
    return converter(t)               # 繁 -> 简


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(ROOT / "data" / "flame_zh_news.csv"))
    p.add_argument("--out", default=str(ROOT / "data" / "flame_zh_all.csv"))
    p.add_argument("--report", default=str(ROOT / "out" / "flame_dataset_report.json"))
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--min_cjk_ratio", type=float, default=0.3, help="中文字符占比下限")
    p.add_argument("--min_len", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--check", action="store_true", help="只统计，不写文件")
    args = p.parse_args()

    try:
        from zhconv import convert as _convert
    except ImportError:
        raise SystemExit("缺少 zhconv，请先执行：pip install zhconv")

    def convert(s: str) -> str:          # zhconv.convert 需要显式 locale
        return _convert(s, "zh-cn")

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"找不到源文件 {src}（应从 HF 镜像下载 zh_labeled.csv 并另存为此路径）")

    raw = list(csv.DictReader(open(src, encoding="utf-8-sig", newline="")))
    print("=" * 66)
    print("FLAME 中文财经新闻数据集构建")
    print("=" * 66)
    print(f"  源文件     : {src}")
    print(f"  原始条数   : {len(raw):,}")

    stats = {"src": str(src), "raw": len(raw)}
    drop_lang = drop_short = drop_label = drop_dup = 0
    trad_hits = 0
    seen, kept = set(), []

    for r in raw:
        label_txt = (r.get("label") or "").strip().lower()
        if label_txt not in LABEL_MAP:
            drop_label += 1
            continue

        orig = (r.get("sentence") or "").strip()
        title = clean_title(orig, convert)
        if title != orig:
            trad_hits += 1                      # 繁转简实际改动了内容

        if not title or len(title) < args.min_len:
            drop_short += 1
            continue
        if cjk_ratio(title) < args.min_cjk_ratio:
            drop_lang += 1                      # 混入英文/日文的条目
            continue
        if title in seen:
            drop_dup += 1
            continue
        seen.add(title)
        kept.append({"title": title, "label": LABEL_MAP[label_txt],
                     "source": r.get("source", "")})

    print(f"  丢弃-标签异常 : {drop_label}")
    print(f"  丢弃-过短     : {drop_short}")
    print(f"  丢弃-非中文   : {drop_lang}")
    print(f"  丢弃-重复     : {drop_dup}")
    print(f"  繁转简改动    : {trad_hits:,} 条 ({trad_hits/max(len(raw),1)*100:.1f}%)")
    print(f"  保留          : {len(kept):,}")

    # ---- 分层切分 ----
    rng = random.Random(args.seed)
    by_label = {}
    for it in kept:
        by_label.setdefault(it["label"], []).append(it)
    train, test = [], []
    for lab, items in sorted(by_label.items()):
        rng.shuffle(items)
        n_test = max(1, int(round(len(items) * args.test_ratio)))
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    for it in train:
        it["split"] = "train"
    for it in test:
        it["split"] = "test"

    print("\n  切分结果：")
    print(f"    {'标签':<8}{'train':>8}{'test':>8}{'合计':>8}")
    for lab in sorted(by_label):
        a = sum(1 for x in train if x["label"] == lab)
        b = sum(1 for x in test if x["label"] == lab)
        print(f"    {LABEL_CN[lab]:<8}{a:>8}{b:>8}{a+b:>8}")
    print(f"    {'合计':<8}{len(train):>8}{len(test):>8}{len(train)+len(test):>8}")

    lens = [len(x["title"]) for x in kept]
    print(f"\n  标题长度：平均 {statistics.mean(lens):.1f} 字 / 中位 {statistics.median(lens):.0f} / "
          f"最短 {min(lens)} / 最长 {max(lens)}")
    print("  样例（繁转简后）：")
    for x in test[:5]:
        print(f"    [{LABEL_CN[x['label']]}] {x['title'][:74]}")

    stats.update({
        "kept": len(kept), "train": len(train), "test": len(test),
        "dropped": {"label": drop_label, "short": drop_short,
                    "nonchinese": drop_lang, "dup": drop_dup},
        "trad_converted": trad_hits,
        "label_counts": {LABEL_CN[k]: sum(1 for x in kept if x["label"] == k) for k in sorted(by_label)},
        "title_len": {"mean": round(statistics.mean(lens), 1),
                      "median": statistics.median(lens),
                      "min": min(lens), "max": max(lens)},
        "test_ratio": args.test_ratio, "seed": args.seed,
    })

    if args.check:
        print("\n[--check] 未写文件。")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["title", "label", "split", "source"])
        w.writeheader()
        for it in train + test:
            w.writerow(it)
    print(f"\n  已写出 -> {out}")

    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  统计   -> {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
