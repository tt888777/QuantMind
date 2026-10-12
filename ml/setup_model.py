"""
一键恢复情感分析模型（新使用者只需跑这一个脚本）
================================================
仓库里只提交了体积很小的 LoRA adapter（约 1.2MB），底座与合并模型因为过大
（411MB / 391MB）被 .gitignore 排除。本脚本负责把这两样补齐：

  1. 从 HF 镜像下载 ``bert-base-chinese`` 底座 → ``ml/models/bert-base-chinese``
  2. 用仓库内自带的 LoRA adapter 与底座合并，导出 → ``ml/models/lora_guba/merged``

用法：
  python setup_model.py                 # 下载底座 + 导出合并模型（推荐）
  python setup_model.py --base-only     # 只下底座（推理后端会自动挂 adapter）
  python setup_model.py --check         # 只看当前状态，不下载任何东西
  python setup_model.py --force         # 已存在也重新生成

说明：
  * 不装 torch 也能用 ``--base-only``（纯下载）；导出合并模型需要 torch + peft。
  * 国内网络默认走 https://hf-mirror.com，可用 HF_ENDPOINT 覆盖。
  * 全部步骤幂等，重复执行不会重复下载。
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # ml/
MODELS = HERE / "models"
BASE_DIR = MODELS / "bert-base-chinese"
ADAPTER_DIR = MODELS / "lora_guba" / "adapter"
MERGED_DIR = MODELS / "lora_guba" / "merged"

DEFAULT_ENDPOINT = "https://hf-mirror.com"
BASE_REPO = "bert-base-chinese"

# 镜像站要求带 User-Agent，空 UA 会被直接拒绝（HTTP 403）
UA = "Mozilla/5.0 (compatible; QuantMind-setup/1.0)"

# 关键：新版 huggingface_hub 默认走 Xet 存储协议，而 hf-mirror 不支持，
# 会报 404 xet-read-token。禁用后回落到普通 HTTP 下载即可正常。
# 必须在 import huggingface_hub 之前设置。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# 底座必须有的文件（缺任一即视为不完整）
REQUIRED_BASE_FILES = ["config.json", "vocab.txt"]
WEIGHT_FILES = ["model.safetensors", "pytorch_model.bin"]


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024


def dir_size(p: Path):
    if not p.is_dir():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def is_base_ok(p: Path) -> bool:
    if not p.is_dir():
        return False
    if not all((p / f).exists() for f in REQUIRED_BASE_FILES):
        return False
    return any((p / f).exists() for f in WEIGHT_FILES)


def is_merged_ok(p: Path) -> bool:
    if not p.is_dir():
        return False
    has_cfg = (p / "config.json").exists()
    has_w = any((p / f).exists() for f in WEIGHT_FILES)
    has_vocab = (p / "vocab.txt").exists() or (p / "tokenizer.json").exists()
    return has_cfg and has_w and has_vocab


# ----------------------------------------------------------------- 下载
def download_base(endpoint: str, force: bool) -> bool:
    if is_base_ok(BASE_DIR) and not force:
        print(f"[跳过] 底座已存在：{BASE_DIR}（{human(dir_size(BASE_DIR))}）")
        return True

    if force and BASE_DIR.exists():
        shutil.rmtree(BASE_DIR)
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[下载] {BASE_REPO}  ←  {endpoint}")
    print(f"       目标目录：{BASE_DIR}")

    # 首选 huggingface_hub（自动处理跳转、断点、并发）
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=BASE_REPO,
            local_dir=str(BASE_DIR),
            endpoint=endpoint,
            # 排除用不上的格式；尤其 *.bin —— bert-base-chinese 同时提供
            # model.safetensors 与 pytorch_model.bin，不排除会重复下载一整套权重
            # （实测底座目录会从 ~400MB 涨到 ~785MB）。
            ignore_patterns=[
                "*.h5", "*.msgpack", "*.onnx", "*.ot", "*.bin", "*.pt", "*.ckpt",
                "flax_model*", "tf_model*",
            ],
            max_workers=4,
        )
    except Exception as e:
        print(f"       huggingface_hub 下载失败（{type(e).__name__}: {e}）")
        print("       回退到 urllib 直连下载 ...")
        if not _download_base_urllib(endpoint):
            return False

    if not is_base_ok(BASE_DIR):
        print(f"[失败] 底座文件不完整，请重试或手动下载到 {BASE_DIR}")
        return False

    # 清理可能被错误写入的占位文件（镜像对不存在的文件会返回错误文本）
    bogus = BASE_DIR / "special_tokens_map.json"
    if bogus.exists() and bogus.stat().st_size < 100:
        try:
            bogus.unlink()
            print("       [清理] 删除无效的 special_tokens_map.json")
        except OSError:
            pass

    print(f"[完成] 底座就绪：{human(dir_size(BASE_DIR))}")
    return True


def _download_base_urllib(endpoint: str) -> bool:
    """无 huggingface_hub 时的回退下载器（仅用标准库）。"""
    import json
    import urllib.request

    def _open(url, timeout=60):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        return urllib.request.urlopen(req, timeout=timeout)

    api = f"{endpoint}/api/models/{BASE_REPO}"
    try:
        with _open(api, 30) as r:
            files = [s["rfilename"] for s in json.loads(r.read()).get("siblings", [])]
    except Exception as e:
        print(f"       无法获取文件列表：{e}")
        return False

    skip = (".h5", ".msgpack", ".onnx", ".ot")
    files = [f for f in files if not f.lower().endswith(skip) and not f.startswith(("flax_", "tf_"))]

    for f in files:
        dst = BASE_DIR / f
        if dst.exists() and dst.stat().st_size > 0:
            continue
        url = f"{endpoint}/{BASE_REPO}/resolve/main/{f}"
        try:
            with _open(url, 180) as r, open(dst, "wb") as out:
                shutil.copyfileobj(r, out)
            print(f"       {f}  ({human(dst.stat().st_size)})")
        except Exception as e:
            print(f"       [警告] {f} 下载失败：{e}")
    return is_base_ok(BASE_DIR)


# ----------------------------------------------------------------- 合并
def export_merged(endpoint: str, force: bool) -> bool:
    if not ADAPTER_DIR.is_dir():
        print(f"[跳过] 未找到 LoRA adapter：{ADAPTER_DIR}")
        return False
    if is_merged_ok(MERGED_DIR) and not force:
        print(f"[跳过] 合并模型已存在：{MERGED_DIR}（{human(dir_size(MERGED_DIR))}）")
        return True
    if not is_base_ok(BASE_DIR):
        print("[跳过] 底座缺失，无法合并。请先执行：python setup_model.py --base-only")
        return False

    print("[合并] LoRA adapter + 底座 → merged")
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as e:
        print(f"[跳过] 缺少推理依赖（{e}）。")
        print("       这不影响使用：推理后端会自己在加载时挂载 adapter。")
        print("       想导出合并模型请先装 requirements-ml.txt。")
        return False

    os.environ.setdefault("HF_ENDPOINT", endpoint)
    LABELS = ("看跌", "看涨", "中性")

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))
        base = AutoModelForSequenceClassification.from_pretrained(
            str(BASE_DIR),
            num_labels=len(LABELS),
            id2label={i: v for i, v in enumerate(LABELS)},
            label2id={v: i for i, v in enumerate(LABELS)},
        )
        model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
        model = model.merge_and_unload()
        MERGED_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(MERGED_DIR), safe_serialization=True)
        tokenizer.save_pretrained(str(MERGED_DIR))
    except Exception as e:
        print(f"[失败] 合并出错：{type(e).__name__}: {e}")
        return False

    if not is_merged_ok(MERGED_DIR):
        print("[失败] 合并目录不完整")
        return False
    print(f"[完成] 合并模型就绪：{human(dir_size(MERGED_DIR))}")
    return True


# ----------------------------------------------------------------- 检查
def check():
    print("当前模型状态")
    print("=" * 62)
    rows = [
        ("底座 bert-base-chinese", BASE_DIR, is_base_ok(BASE_DIR)),
        ("LoRA adapter（仓库自带）", ADAPTER_DIR, ADAPTER_DIR.is_dir()),
        ("合并模型 merged", MERGED_DIR, is_merged_ok(MERGED_DIR)),
    ]
    for name, path, ok in rows:
        size = human(dir_size(path)) if path.exists() else "-"
        print(f"  [{'OK' if ok else '缺失'}] {name:<28} {size:>9}  {path}")

    print()
    loadable = is_base_ok(BASE_DIR) or is_merged_ok(MERGED_DIR)
    if is_merged_ok(MERGED_DIR):
        print("结论：可用 merged 模型直接推理（加载最快）")
    elif is_base_ok(BASE_DIR) and ADAPTER_DIR.is_dir():
        print("结论：可用 adapter + 底座推理（后端会在加载时自动合并）")
    elif is_base_ok(BASE_DIR):
        print("结论：只有底座，没有 adapter —— 模型会输出未微调的结果")
    else:
        print("结论：模型缺失，engine=model 会自动回退关键词规则")
        print("      执行 python setup_model.py 即可补齐")
    return loadable


def main():
    ap = argparse.ArgumentParser(
        description="一键恢复情感分析模型（下载底座 + 导出合并模型）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT),
                    help=f"HF 镜像地址，默认 {DEFAULT_ENDPOINT}")
    ap.add_argument("--base-only", action="store_true", help="只下载底座，不做合并")
    ap.add_argument("--check", action="store_true", help="只检查状态，不下载")
    ap.add_argument("--force", action="store_true", help="已存在也重新生成")
    args = ap.parse_args()

    # 让 huggingface_hub / transformers 都走镜像
    os.environ["HF_ENDPOINT"] = args.endpoint

    print("QuantMind 情感模型恢复工具")
    print("=" * 62)
    print(f"镜像地址：{args.endpoint}")
    print()

    if args.check:
        ok = check()
        sys.exit(0 if ok else 2)

    if not download_base(args.endpoint, args.force):
        sys.exit(1)

    if not args.base_only:
        print()
        export_merged(args.endpoint, args.force)

    print()
    ok = check()
    print()
    if ok:
        print("下一步：启动服务后，在「舆情决策」页把引擎切到「AI 模型」即可。")
    else:
        print("仍有缺失，请检查上面的报错。")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
