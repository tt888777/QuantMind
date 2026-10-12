"""
eastmoney_guba_title 数据集最小验证脚本
=====================================
用途：从 hf-mirror 拉取数据集，检查是否可用于 LoRA 监督微调。
依赖：仅 Python 标准库（无需 torch / datasets / pandas）。

产出：
  ml/data/train.csv, test.csv        原始文件
  ml/data/guba_title_all.csv         合并后的语料（供后续训练使用）
  ml/out/dataset_report.json         机器可读的统计报告
  ml/out/sample_review.csv           每类随机抽样，供人工抽检质量

运行：python check_guba_dataset.py
"""
import csv
import json
import random
import sys
import urllib.request
from collections import Counter
from pathlib import Path

MIRROR = "https://hf-mirror.com"
REPO = "HikasaHana/eastmoney_guba_title"
FILES = ["train.csv", "test.csv"]
LABEL_NAMES = {0: "negative(看跌)", 1: "positive(看涨)", 2: "neutral(中性)"}
SAMPLE_PER_LABEL = 30
RANDOM_SEED = 42

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"


def download(fname: str) -> Path:
    """从镜像下载单个文件，带跟随 302 重定向。"""
    dest = DATA_DIR / fname
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [跳过] {fname} 已存在 ({dest.stat().st_size:,} bytes)")
        return dest
    url = f"{MIRROR}/datasets/{REPO}/resolve/main/{fname}"
    print(f"  [下载] {fname} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    dest.write_bytes(data)
    print(f"  [完成] {fname}  {len(data):,} bytes")
    return dest


def load_csv(path: Path) -> list:
    """读取 CSV，做基本结构校验。"""
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path.name} 为空或无表头")
        missing = {"title", "label"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path.name} 缺少列: {missing}；实际列: {reader.fieldnames}")
        for i, r in enumerate(reader, start=2):
            title = (r.get("title") or "").strip()
            raw_label = (r.get("label") or "").strip()
            try:
                label = int(float(raw_label))
            except ValueError:
                rows.append({"title": title, "label": None, "bad_label": raw_label, "line": i})
                continue
            rows.append({"title": title, "label": label, "line": i})
    return rows


def describe(rows: list, name: str) -> dict:
    """单份数据的统计描述。"""
    labels = Counter(r["label"] for r in rows)
    bad = [r for r in rows if r["label"] is None]
    illegal = [r for r in rows if r["label"] is not None and r["label"] not in LABEL_NAMES]
    empty = [r for r in rows if not r["title"]]
    lengths = sorted(len(r["title"]) for r in rows if r["title"])

    def pct(p):
        if not lengths:
            return 0
        return lengths[min(int(len(lengths) * p), len(lengths) - 1)]

    total = len(rows)
    stats = {
        "rows": total,
        "label_dist": {LABEL_NAMES.get(k, str(k)): {"count": v, "pct": round(v / total * 100, 1)}
                       for k, v in sorted(labels.items(), key=lambda x: (x[0] is None, x[0]))},
        "bad_label_rows": len(bad),
        "illegal_label_values": sorted({r["label"] for r in illegal}),
        "empty_title_rows": len(empty),
        "length": {"min": lengths[0] if lengths else 0, "p25": pct(0.25),
                   "median": pct(0.5), "p75": pct(0.75), "p95": pct(0.95),
                   "max": lengths[-1] if lengths else 0},
    }
    print(f"\n--- {name} ---")
    print(f"  总行数        : {total:,}")
    for k, v in stats["label_dist"].items():
        print(f"  {k:<18}: {v['count']:>6,}  ({v['pct']:>5}%)")
    print(f"  标签异常行    : {len(bad)}  (无法解析为整数)")
    print(f"  非法标签值    : {stats['illegal_label_values'] or '无'}")
    print(f"  空标题        : {len(empty)}")
    print(f"  标题长度      : min={stats['length']['min']}  中位={stats['length']['median']}  "
          f"p95={stats['length']['p95']}  max={stats['length']['max']}")
    return stats


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("eastmoney_guba_title 数据集验证")
    print("=" * 62)

    print("\n[1/5] 下载数据")
    paths = {}
    for f in FILES:
        try:
            paths[f] = download(f)
        except Exception as e:
            print(f"  [失败] {f}: {e}")
            return 1

    print("\n[2/5] 读取与校验")
    train = load_csv(paths["train.csv"])
    test = load_csv(paths["test.csv"])

    print("\n[3/5] 统计描述")
    report = {"repo": REPO, "source": MIRROR}
    report["train"] = describe(train, "train.csv")
    report["test"] = describe(test, "test.csv")

    print("\n[4/5] 数据质量检查")
    tr_titles = [r["title"] for r in train if r["title"]]
    te_titles = [r["title"] for r in test if r["title"]]
    tr_set, te_set = set(tr_titles), set(te_titles)

    dup_train = len(tr_titles) - len(tr_set)
    dup_test = len(te_titles) - len(te_set)
    overlap = tr_set & te_set
    label_conflict = 0
    if overlap:
        tr_map = {}
        for r in train:
            tr_map.setdefault(r["title"], set()).add(r["label"])
        te_map = {}
        for r in test:
            te_map.setdefault(r["title"], set()).add(r["label"])
        for t in overlap:
            if tr_map.get(t) != te_map.get(t):
                label_conflict += 1

    checks = {
        "train_duplicate_titles": dup_train,
        "test_duplicate_titles": dup_test,
        "train_test_overlap_titles": len(overlap),
        "overlap_label_conflicts": label_conflict,
    }
    report["quality"] = checks
    print(f"  train 内重复标题        : {dup_train}")
    print(f"  test  内重复标题        : {dup_test}")
    print(f"  train/test 重叠标题     : {len(overlap)}")
    print(f"  重叠且标签冲突          : {label_conflict}")
    if overlap:
        print("  重叠样例:")
        for t in list(overlap)[:3]:
            print(f"    - {t[:40]}")

    print("\n[5/5] 导出产物")
    merged = DATA_DIR / "guba_title_all.csv"
    with open(merged, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["title", "label", "label_name", "split"])
        for split, rows in (("train", train), ("test", test)):
            for r in rows:
                if r["title"] and r["label"] in LABEL_NAMES:
                    w.writerow([r["title"], r["label"], LABEL_NAMES[r["label"]], split])
    print(f"  合并语料 -> {merged.relative_to(ROOT)}")

    rng = random.Random(RANDOM_SEED)
    review = OUT_DIR / "sample_review.csv"
    with open(review, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "label_name", "title", "人工判断(留空填 0/1/2)", "是否同意(y/n)"])
        for lab in sorted(LABEL_NAMES):
            pool = [r["title"] for r in train if r["label"] == lab and r["title"]]
            for t in rng.sample(pool, min(SAMPLE_PER_LABEL, len(pool))):
                w.writerow([lab, LABEL_NAMES[lab], t, "", ""])
    print(f"  抽检样本 -> {review.relative_to(ROOT)}  (每类 {SAMPLE_PER_LABEL} 条)")

    rep_path = OUT_DIR / "dataset_report.json"
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  统计报告 -> {rep_path.relative_to(ROOT)}")

    print("\n" + "=" * 62)
    print("结论")
    print("=" * 62)
    t = report["train"]
    ok = (t["bad_label_rows"] == 0 and not t["illegal_label_values"]
          and t["rows"] > 1000 and len(overlap) == 0)
    print(f"  可用性            : {'通过 - 可直接用于 LoRA 监督微调' if ok else '需注意 - 见上方异常项'}")
    print(f"  监督样本量        : {t['rows'] + report['test']['rows']:,} 条")
    print(f"  标签体系          : 3 类 (0/1/2)，需映射到本项目标签")
    print(f"  标题中位长度      : {t['length']['median']} 字，与 bert 截断长度 64 匹配良好")
    return 0


if __name__ == "__main__":
    sys.exit(main())
