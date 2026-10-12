"""
模型准备工具（全新电脑部署的核心步骤）
======================================

本项目线上只使用**一个**情感模型：

    lora_dapt_flame = dapt_bert 领域自适应底座 + FLAME 中文财经新闻三分类 LoRA

它由三部分组成，其中前两部分体积过大（410MB / 391MB），不入 Git 仓库：

    ml/models/dapt_bert/                底座（DAPT 产物，gitignore）
    ml/models/lora_dapt_flame/merged/   合并后的完整模型（gitignore）
    ml/models/lora_dapt_flame/adapter/  LoRA 权重（~4MB，**已提交到仓库**）

因此全新电脑上必须先把模型补齐。两条路：

  ① 从 Hugging Face 下载合并好的完整模型（推荐，最快）
       python setup_model.py --from-hf <用户名>/<仓库名>

  ② 用仓库自带的 adapter + 底座现场合并（需要底座已存在）
       python setup_model.py --rebuild

用法：
  python setup_model.py --check                 # 只看当前状态，不下载
  python setup_model.py --from-hf aaa/bbb       # 从 HF 下载合并模型
  python setup_model.py --rebuild               # 本地 adapter + 底座 → merged
  python setup_model.py --force                 # 已存在也重新生成
  python setup_model.py --hf-repo aaa/bbb       # 只登记 HF 仓库地址，供后续默认使用

说明：
  * 国内网络默认走 https://hf-mirror.com，可用 --endpoint / HF_ENDPOINT 覆盖。
  * 所有步骤幂等，重复执行不会重复下载。
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # ml/
MODELS = HERE / "models"

# 线上唯一模型
MODULE_NAME = "lora_dapt_flame"
ADAPTER_DIR = MODELS / MODULE_NAME / "adapter"
MERGED_DIR = MODELS / MODULE_NAME / "merged"
BASE_DIR = MODELS / "dapt_bert"                 # adapter 必须挂在这个底座上

CONFIG_FILE = HERE / "model_source.json"        # 记录 HF 仓库地址，供下次默认使用

DEFAULT_ENDPOINT = "https://hf-mirror.com"
UA = "Mozilla/5.0 (compatible; QuantMind-setup/1.0)"

# 镜像站要求带 User-Agent（空 UA 会被 403 拒绝）
# 新版 huggingface_hub 默认走 Xet 协议，hf-mirror 不支持，必须禁用（否则 404 xet-read-token）
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

WEIGHT_FILES = ["model.safetensors", "pytorch_model.bin"]

# 仓库里这些子目录不属于「合并模型」，下载与兜底提升时都要跳过
SKIP_SUBDIRS = {"adapter", "dapt_base", "merged", "_ckpt"}


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024


def dir_size(p: Path):
    if not p.is_dir():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def is_merged_ok(p: Path) -> bool:
    """一个可以直接 ``from_pretrained`` 的完整模型目录。"""
    if not p.is_dir():
        return False
    has_cfg = (p / "config.json").exists()
    has_w = any((p / f).exists() for f in WEIGHT_FILES)
    has_tok = (p / "vocab.txt").exists() or (p / "tokenizer.json").exists()
    return has_cfg and has_w and has_tok


def is_base_ok(p: Path) -> bool:
    return p.is_dir() and (p / "config.json").exists() and any((p / f).exists() for f in WEIGHT_FILES)


# ------------------------------------------------------------------ HF 仓库地址
def load_hf_repo() -> str:
    env = os.environ.get("QUANTMIND_MODEL_HF_REPO")
    if env:
        return env.strip()
    if CONFIG_FILE.is_file():
        try:
            return (json.loads(CONFIG_FILE.read_text(encoding="utf-8")).get("hf_repo") or "").strip()
        except Exception:
            return ""
    return ""


def save_hf_repo(repo: str):
    CONFIG_FILE.write_text(json.dumps({"hf_repo": repo}, ensure_ascii=False, indent=2),
                           encoding="utf-8")


# ------------------------------------------------------------------ 下载
def download_from_hf(repo: str, endpoint: str, force: bool) -> bool:
    if not repo:
        print("[跳过] 未提供 HF 仓库地址。用 --from-hf <用户名>/<仓库名> 指定。")
        return False
    if is_merged_ok(MERGED_DIR) and not force:
        print(f"[跳过] 合并模型已存在：{MERGED_DIR}（{human(dir_size(MERGED_DIR))}）")
        return True

    if force and MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[下载] {repo}  ←  {endpoint}")
    print(f"       目标目录：{MERGED_DIR}")
    print("       （完整模型约 390MB，首次下载需要几分钟）")

    try:
        from huggingface_hub import snapshot_download

        # 只取仓库根目录的模型文件。
        # 本仓库根目录 = 合并模型，另有两个子目录 adapter/ 与 dapt_base/（合计再 410MB）
        # 是给「继续微调」和「复现」用的，下载合并模型时不需要，必须排除。
        snapshot_download(
            repo_id=repo,
            local_dir=str(MERGED_DIR),
            endpoint=endpoint,
            ignore_patterns=[
                "*.bin", "*.h5", "*.msgpack", "*.onnx", "*.ot", "*.pt",
                "adapter/*", "dapt_base/*", "merged/*", "_ckpt/*",
            ],
            max_workers=4,
        )
    except Exception as e:
        print(f"       huggingface_hub 下载失败（{type(e).__name__}: {e}）")
        if not _download_repo_urllib(repo, endpoint):
            print("[失败] 无法从 HF 获取模型。")
            print("       可选：① 换用 --endpoint 指定其它镜像；"
                  "② 用 --rebuild 在本机用 adapter + 底座现场合并。")
            return False

    # 有些仓库把模型放在子目录里，做一次兜底查找。
    # ⚠️ 必须排除 adapter/ 与 dapt_base/：dapt_base 是 MLM 底座，
    #    它同样有 config.json + 权重 + 分词器，会被 is_merged_ok() 误判为「完整模型」，
    #    一旦被提升上来当作分类模型用，会静默算出错误结果。
    if not is_merged_ok(MERGED_DIR):
        for cfg in sorted(MERGED_DIR.rglob("config.json")):
            cand = cfg.parent
            if cand == MERGED_DIR or cand.name in SKIP_SUBDIRS:
                continue
            if not is_merged_ok(cand):
                continue
            print(f"       在子目录中找到模型：{cand.relative_to(MERGED_DIR)}，已提升到根目录")
            for f in cand.iterdir():
                shutil.move(str(f), str(MERGED_DIR / f.name))
            break

    if not is_merged_ok(MERGED_DIR):
        print("[失败] 下载完成但模型目录不完整（缺 config.json / 权重 / 分词器）")
        return False

    save_hf_repo(repo)
    print(f"[完成] 合并模型就绪：{human(dir_size(MERGED_DIR))}")
    return True


def _download_repo_urllib(repo: str, endpoint: str) -> bool:
    """无 huggingface_hub 时的回退下载器（只用标准库）。"""
    import urllib.request

    def _open(url, timeout=60):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        return urllib.request.urlopen(req, timeout=timeout)

    try:
        with _open(f"{endpoint}/api/models/{repo}", 30) as r:
            files = [s["rfilename"] for s in json.loads(r.read()).get("siblings", [])]
    except Exception as e:
        print(f"       无法获取文件列表：{e}")
        return False

    if not files:
        print("       仓库为空")
        return False

    skip = (".h5", ".msgpack", ".onnx", ".ot", ".bin")
    files = [
        f for f in files
        if not f.lower().endswith(skip)
        and not f.startswith(("flax_", "tf_"))
        and f.split("/")[0] not in SKIP_SUBDIRS   # 排除 adapter/、dapt_base/ 等子目录
    ]

    for f in files:
        dst = MERGED_DIR / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size > 0:
            continue
        try:
            with _open(f"{endpoint}/{repo}/resolve/main/{f}", 300) as r, open(dst, "wb") as out:
                shutil.copyfileobj(r, out)
            print(f"       {f}  ({human(dst.stat().st_size)})")
        except Exception as e:
            print(f"       [警告] {f} 下载失败：{e}")
    return is_merged_ok(MERGED_DIR)


# ------------------------------------------------------------------ 本地合并
def rebuild_local(force: bool) -> bool:
    if not ADAPTER_DIR.is_dir():
        print(f"[跳过] 未找到 LoRA adapter：{ADAPTER_DIR}")
        print("       该目录应该随仓库一起提交，请确认代码完整。")
        return False
    if is_merged_ok(MERGED_DIR) and not force:
        print(f"[跳过] 合并模型已存在：{MERGED_DIR}（{human(dir_size(MERGED_DIR))}）")
        return True
    if not is_base_ok(BASE_DIR):
        print(f"[跳过] 底座缺失：{BASE_DIR}")
        print("       该 adapter 必须挂在 DAPT 底座上（不能换 bert-base-chinese）。")
        print("       获得底座的两种方式：")
        print("         a) 从 HF 下载已合并的完整模型：python setup_model.py --from-hf <仓库>")
        print("         b) 自己重新跑 DAPT：cd ml && python train_dapt.py --out models/dapt_bert")
        return False
    if not is_merged_ok(BASE_DIR):
        pass  # 底座只要求 config + 权重，tokenizer 会在下面从 adapter 取

    print("[合并] LoRA adapter + 底座 → merged")
    try:
        import torch  # noqa: F401
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from lora_extra import export_extra_state, read_extra_state
    except Exception as e:
        print(f"[跳过] 缺少依赖（{e}）。")
        print("       请先安装：pip install -r ../quantmind/requirements-ml.txt")
        return False

    os.environ.setdefault("HF_ENDPOINT", os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT))
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

        # 关键：回填训练时冻结的「底座里没有」的参数（bert.pooler）。
        # 不回填的话，这份随机初始化和训练时那份不同，分类头会失效。
        if read_extra_state(ADAPTER_DIR) is None:
            cfg = ADAPTER_DIR / "adapter_config.json"
            if cfg.is_file():
                print("       [提示] 未找到 extra_state.safetensors，正在从 adapter_config 推断并补写 ...")
                try:
                    n = export_extra_state(model, BASE_DIR, ADAPTER_DIR)
                    print(f"       [提示] 已补写 {n} 个非 LoRA 张量")
                except Exception as e:
                    print(f"       [警告] 补写失败：{e}")
        extra = read_extra_state(ADAPTER_DIR)
        if extra:
            model.load_state_dict(extra, strict=False)
            print(f"       回填额外权重 {len(extra)} 个")

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


# ------------------------------------------------------------------ 检查
def check():
    print("当前模型状态（线上唯一模型：DAPT 底座 + 财经新闻 LoRA）")
    print("=" * 66)
    rows = [
        ("合并模型 merged（推理直接用这个）", MERGED_DIR, is_merged_ok(MERGED_DIR)),
        ("LoRA adapter（随仓库提交）", ADAPTER_DIR, ADAPTER_DIR.is_dir()),
        ("底座 dapt_bert（本地合并时才需要）", BASE_DIR, is_base_ok(BASE_DIR)),
    ]
    for name, path, ok in rows:
        size = human(dir_size(path)) if path.exists() else "-"
        print(f"  [{'OK' if ok else '缺失'}] {name:<32} {size:>9}")

    repo = load_hf_repo()
    print(f"  HF 仓库地址：{repo or '（未设置）'}")

    print()
    if is_merged_ok(MERGED_DIR):
        print("结论：模型就绪，engine=model 可直接使用（加载最快）")
        return True
    if is_base_ok(BASE_DIR) and ADAPTER_DIR.is_dir():
        print("结论：可用 adapter + 底座现场合并 —— 执行 python setup_model.py --rebuild")
        return True
    if repo:
        print(f"结论：请从 HF 下载 —— python setup_model.py --from-hf {repo}")
        return False
    print("结论：模型缺失，engine=model 会自动回退关键词规则（系统仍可运行）")
    print("      补齐方式：python setup_model.py --from-hf <用户名>/<仓库名>")
    return False


def main():
    ap = argparse.ArgumentParser(
        description="准备情感分析模型（HF 下载 或 本地合并）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--from-hf", dest="from_hf", default=None, metavar="REPO",
                    help="从 Hugging Face 仓库下载合并模型，如 用户名/仓库名")
    ap.add_argument("--hf-repo", default=None, metavar="REPO",
                    help="仅登记 HF 仓库地址（写入 model_source.json），不下载")
    ap.add_argument("--rebuild", action="store_true", help="用本地 adapter + 底座合并出 merged")
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT),
                    help=f"HF 镜像地址，默认 {DEFAULT_ENDPOINT}")
    ap.add_argument("--check", action="store_true", help="只检查状态")
    ap.add_argument("--force", action="store_true", help="已存在也重新生成")
    args = ap.parse_args()

    os.environ["HF_ENDPOINT"] = args.endpoint

    print("QuantMind 模型准备工具")
    print("=" * 66)
    print(f"镜像地址：{args.endpoint}")
    print()

    if args.hf_repo:
        save_hf_repo(args.hf_repo.strip())
        print(f"[记录] HF 仓库地址已写入 {CONFIG_FILE}")
        print()

    if args.check:
        ok = check()
        sys.exit(0 if ok else 2)

    repo = args.from_hf or load_hf_repo()
    ok = False

    if args.rebuild:
        ok = rebuild_local(args.force)
    elif args.from_hf:
        ok = download_from_hf(repo, args.endpoint, args.force)
    else:
        # 自动：已有 merged 就跳过；否则优先 HF，其次本地合并
        if is_merged_ok(MERGED_DIR):
            print(f"[跳过] 合并模型已存在：{MERGED_DIR}（{human(dir_size(MERGED_DIR))}）")
            ok = True
        elif repo:
            ok = download_from_hf(repo, args.endpoint, args.force)
        elif is_base_ok(BASE_DIR) and ADAPTER_DIR.is_dir():
            print("[提示] 未设置 HF 仓库，改为用本地 adapter + 底座合并。")
            print()
            ok = rebuild_local(args.force)
        else:
            print("[提示] 既没有 HF 仓库地址，也没有本地底座。")
            print("       请二选一：")
            print("         python setup_model.py --from-hf <用户名>/<仓库名>")
            print("         python setup_model.py --rebuild   （需先有 ml/models/dapt_bert）")

    print()
    ok_final = check()
    print()
    if is_merged_ok(MERGED_DIR):
        print("下一步：启动服务，在「舆情决策」页把引擎切到「AI 模型」即可。")
    elif ok or ok_final:
        print("模型可用（adapter 模式，推理时现场合并，首次加载会慢几秒）。")
    else:
        print("模型仍未就绪；服务可以正常启动，engine=model 会自动回退关键词规则。")
    sys.exit(0 if (is_merged_ok(MERGED_DIR) or ok_final) else 2)


if __name__ == "__main__":
    main()
