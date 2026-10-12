"""
把训练好的情感模型发布到 Hugging Face Hub
=========================================

**这个脚本必须在一台能访问 huggingface.co 的机器上运行。**
（hf-mirror.com 只提供下载，不支持上传；国内部分网络 huggingface.co 会被 DNS 污染而不可达。）

上传内容：

    <repo>/                      合并后的完整模型（config.json / model.safetensors / 分词器）
    <repo>/adapter/              LoRA adapter（可继续微调）
    <repo>/README.md             Model Card（自动生成，含指标与训练配置）
    <repo>/dapt_base/            （可选 --include-dapt-base）DAPT 底座，便于他人复现

准备：
    1. 在 https://huggingface.co/settings/tokens 建一个 **write** 权限的 token
    2. 设置环境变量：`set HF_TOKEN=hf_xxx`（Windows）或 `export HF_TOKEN=hf_xxx`

用法：
    python upload_to_hf.py --repo Roeyyyyyy/quantmind-sentiment-dapt-flame --check
    python upload_to_hf.py --repo Roeyyyyyy/quantmind-sentiment-dapt-flame
    python upload_to_hf.py --repo Roeyyyyyy/quantmind-sentiment-dapt-flame --include-dapt-base
    python upload_to_hf.py --repo Roeyyyyyy/quantmind-sentiment-dapt-flame --private
    python upload_to_hf.py --repo Roeyyyyyy/quantmind-sentiment-dapt-flame --gh-owner 别的GitHub账号

上传完成后，其他人在全新电脑上执行：

    python setup_model.py --from-hf Roeyyyyyy/quantmind-sentiment-dapt-flame
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"

MODULE_NAME = "lora_dapt_flame"
MERGED_DIR = MODELS / MODULE_NAME / "merged"
ADAPTER_DIR = MODELS / MODULE_NAME / "adapter"
DAPT_BASE_DIR = MODELS / "dapt_bert"
METRICS_FILE = MODELS / MODULE_NAME / "metrics.json"

# 上传必须走官方站点；镜像站不支持写入
OFFICIAL_ENDPOINT = "https://huggingface.co"
MIRROR_HOSTS = ("hf-mirror.com", "hfmirror")

# Model Card 里 GitHub 链接的默认指向。
# ⚠️ 本项目的 HF 账号（Roeyyyyyy）与 GitHub 账号（Jwwws）不同名，
#    默认不能取 HF 用户名，否则会生成指向不存在仓库的链接。
DEFAULT_GH_OWNER = "tt888777"

WEIGHT_FILES = ["model.safetensors", "pytorch_model.bin"]

MODEL_CARD = """---
library_name: transformers
pipeline_tag: text-classification
language:
  - zh
tags:
  - finance
  - sentiment-analysis
  - chinese
  - bert
  - lora
  - peft
  - dapt
license: cc-by-nc-4.0
base_model: bert-base-chinese
datasets:
  - Kenpache/multilingual-financial-sentiment-v2
metrics:
  - accuracy
  - f1
widget:
  - text: "公司发布业绩预告，净利润同比增长80%"
    example_title: "业绩预增（看涨）"
  - text: "大股东拟减持不超过3%股份，股价承压"
    example_title: "减持利空（看跌）"
  - text: "公司召开股东大会审议年度报告"
    example_title: "中性公告"
---

# QuantMind 中文金融情感三分类（DAPT + 财经新闻 LoRA）

面向**中文财经新闻**的三分类情感模型：`看跌 / 看涨 / 中性`。
用于 [QuantMind](https://github.com/{gh_owner}/QuantMind) 量化分析系统的「AI 智能体舆情决策」模块。

## 模型说明

本模型是两阶段训练的产物：

```
bert-base-chinese
      │  ① DAPT：MLM 域自适应预训练（CN-Market-Corpus 20 万条，1 epoch）
      ▼
   dapt_bert
      │  ② LoRA 监督微调（FLAME 中文财经新闻 14,680 条，5 epoch）
      ▼
lora_dapt_flame   ← 本仓库
```

- 仅 **0.29%** 参数可训练（297,219 / 102,567,174），LoRA `r=8, alpha=16`，目标模块 `query/value`
- 分类头通过 `modules_to_save` 一起训练
- 推理长度 64 token，bf16 训练

## 实测指标

**新闻测试集（2,936 条）**

| 指标 | 值 |
| --- | --- |
| Accuracy | **{acc}** |
| Macro-F1 | **{f1}** |

逐类（precision / recall / F1）：

| 类别 | P | R | F1 | 支持数 |
| --- | --- | --- | --- | --- |
| 看跌 negative | {p0} | {r0} | {f0} | {s0} |
| 看涨 positive | {p1} | {r1} | {f1c} | {s1} |
| 中性 neutral | {p2} | {r2} | {f2} | {s2} |

**跨域对照**（同一套超参、只换底座或数据域）

| 模型 | 股吧测试集(2,130) | 新闻测试集(2,936) |
| --- | --- | --- |
| `bert-base-chinese` + 股吧 | 0.7460 | 0.5562 |
| `dapt_bert` + 股吧 | 0.7446 | 0.5725 |
| `bert-base-chinese` + 新闻 | 0.5549 | 0.8273 |
| **`dapt_bert` + 新闻（本模型）** | 0.5592 | **0.8290** |

> ⚠️ **适用域很重要**：本模型在**财经新闻**上表现最好，但**不擅长股吧口语**
> （股吧测试集仅 0.5592，接近"全猜多数类"）。若目标是股吧舆情，请改用股吧域模型。
> 另外 DAPT 环节在股吧与新闻两条支线上**都没有产生显著增益**（Δ 均在噪声内），
> 保留它主要是为了流程完整性与可复现性。

## 使用方法

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

repo = "{repo}"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()

texts = ["公司发布业绩预告，净利润同比增长80%", "大股东拟减持不超过3%股份，股价承压"]
enc = tok(texts, truncation=True, max_length=64, padding=True, return_tensors="pt")
with torch.no_grad():
    probs = torch.softmax(model(**enc).logits, dim=-1)

id2label = model.config.id2label          # {{0: '看跌', 1: '看涨', 2: '中性'}}
for t, p in zip(texts, probs):
    i = int(p.argmax())
    print(id2label[i], round(float(p[i]), 4), t)
```

## 训练数据

| 用途 | 数据集 | 规模 | 许可 |
| --- | --- | --- | --- |
| ② LoRA 监督 | [Kenpache/multilingual-financial-sentiment-v2](https://huggingface.co/datasets/Kenpache/multilingual-financial-sentiment-v2)（FLAME 中文子集） | 14,680 条（train 11,744 / test 2,936） | **CC BY-NC 4.0** |
| ① DAPT | [mulfsa/CN-Market-Corpus](https://huggingface.co/datasets/mulfsa/CN-Market-Corpus) | 200,000 条（取头部约 1GB） | 见数据集页 |

FLAME 的特点是**按本国投资者视角标注**，中文子集即"中国投资者怎么看这条新闻"。
原始数据混有港台媒体（38.7% 为繁体），已用 `zhconv` 统一转为简体。

> ⚠️ **许可提醒**：训练数据含 **CC BY-NC 4.0（仅限非商业用途）**，
> 因此本模型权重同样**仅可用于非商业/研究/教学**场景。

## 训练配置

| 项 | 值 |
| --- | --- |
| 底座 | `dapt_bert`（bert-base-chinese 经 1 epoch MLM 域自适应） |
| 数据 | `data/flame_zh_all.csv` {train_rows} 条 |
| epoch / batch / lr | {epochs} / {batch} / {lr} |
| max_len | {max_len} |
| LoRA | r={r}, alpha={alpha}, dropout={dropout} |
| 可训练参数 | {trainable:,} / {total:,}（{pct}%） |
| 训练环境 | Windows + RTX 3060 Laptop 6GB |

## 复现

```bash
# 1. 准备数据与底座
python prepare_dapt_corpus.py           # 构建 DAPT 语料（约 1GB）
python train_dapt.py --out models/dapt_bert --epochs 1
python prepare_flame_dataset.py         # 构建中文新闻集（繁转简 + 分层切分）

# 2. LoRA 微调 + 导出合并模型
python train_lora.py --base models/dapt_bert --data data/flame_zh_all.csv \\
  --out models/lora_dapt_flame --epochs 5 --batch_size 32 --lr 5e-4 \\
  --lora_r 8 --lora_alpha 16 --merge
```

## 引用

如果用到本模型，请注明来自 QuantMind 项目，并同时遵守上游数据集的许可条款。
"""


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024


def dir_size(p: Path):
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else 0


def list_files(p: Path, skip_weights_bin=True):
    out = []
    for f in sorted(p.rglob("*")):
        if not f.is_file():
            continue
        if skip_weights_bin and f.name == "pytorch_model.bin":
            continue
        out.append((f.relative_to(p).as_posix(), f.stat().st_size))
    return out


def build_model_card(repo: str, gh_owner: str) -> str:
    """把 metrics.json 里的真实数字填进模板。"""
    acc = f1 = "—"
    per = {}
    train_rows = total = trainable = "—"
    epochs = batch = lr = max_len = r = alpha = dropout = "—"

    if METRICS_FILE.is_file():
        m = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
        met = m.get("metrics", {})
        acc = f"{met.get('accuracy', 0):.4f}"
        f1 = f"{met.get('macro_f1', 0):.4f}"
        rep = met.get("report", {})
        for i, key in enumerate(("negative/看跌", "positive/看涨", "neutral/中性")):
            row = rep.get(key, {})
            per[i] = (f"{row.get('precision', 0):.3f}", f"{row.get('recall', 0):.3f}",
                      f"{row.get('f1-score', 0):.3f}", int(row.get("support", 0)))
        train_rows = f"{m.get('train_rows', '—')}"
        total = int(m.get("total_params", 0) or 0)
        trainable = int(m.get("trainable_params", 0) or 0)
        epochs = m.get("epochs", "—")
        batch = m.get("batch_size", "—")
        lr = m.get("lr", "—")
        max_len = m.get("max_len", "—")
        lo = m.get("lora", {})
        r, alpha, dropout = lo.get("r", "—"), lo.get("alpha", "—"), lo.get("dropout", "—")

    pct = f"{trainable / total * 100:.2f}" if isinstance(total, int) and total else "0.29"
    zero = ("—", "—", "—", 0)

    return MODEL_CARD.format(
        repo=repo, gh_owner=gh_owner,
        acc=acc, f1=f1,
        p0=per.get(0, zero)[0], r0=per.get(0, zero)[1], f0=per.get(0, zero)[2], s0=per.get(0, zero)[3],
        p1=per.get(1, zero)[0], r1=per.get(1, zero)[1], f1c=per.get(1, zero)[2], s1=per.get(1, zero)[3],
        p2=per.get(2, zero)[0], r2=per.get(2, zero)[1], f2=per.get(2, zero)[2], s2=per.get(2, zero)[3],
        train_rows=train_rows, total=total if isinstance(total, int) else 0,
        trainable=trainable if isinstance(trainable, int) else 0, pct=pct,
        epochs=epochs, batch=batch, lr=lr, max_len=max_len,
        r=r, alpha=alpha, dropout=dropout,
    )


def main():
    ap = argparse.ArgumentParser(
        description="把 lora_dapt_flame 模型发布到 Hugging Face Hub（需在能访问 huggingface.co 的机器上运行）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--repo", required=True, metavar="用户名/仓库名",
                    help="目标 HF 仓库，如 Roeyyyyyy/quantmind-sentiment-dapt-flame")
    ap.add_argument("--token", default=None,
                    help="HF write token；默认读环境变量 HF_TOKEN / HUGGINGFACE_HUB_TOKEN")
    ap.add_argument("--endpoint", default=OFFICIAL_ENDPOINT,
                    help=f"HF 站点地址，默认 {OFFICIAL_ENDPOINT}（上传必须用官方站）")
    ap.add_argument("--private", action="store_true", help="创建为私有仓库")
    ap.add_argument("--include-dapt-base", action="store_true",
                    help="连 DAPT 底座（约 390MB）一起上传到 dapt_base/，便于他人复现")
    ap.add_argument("--skip-adapter", action="store_true", help="不上传 adapter 子目录")
    ap.add_argument("--gh-owner", default=None,
                    help=f"Model Card 里指向的 GitHub 用户名，默认 {DEFAULT_GH_OWNER}"
                         "（本项目实际的协作仓库）")
    ap.add_argument("--check", action="store_true", help="只列出将要上传的文件，不真正上传")
    args = ap.parse_args()

    print("QuantMind 模型发布工具")
    print("=" * 68)
    print(f"目标仓库：{args.repo}")
    print(f"站点    ：{args.endpoint}")
    print()

    if any(h in args.endpoint for h in MIRROR_HOSTS):
        print("[错误] 上传不能走镜像站（hf-mirror.com 只支持下载）。")
        print(f"       请使用官方站点：--endpoint {OFFICIAL_ENDPOINT}")
        return 2

    # ---- 检查本地产物 ----
    if not MERGED_DIR.is_dir() or not any((MERGED_DIR / f).exists() for f in WEIGHT_FILES):
        print(f"[错误] 找不到合并模型：{MERGED_DIR}")
        print("       请先执行：python train_lora.py ... --merge")
        print("       或用 setup_model.py 从 HF 拉取/本地合并。")
        return 1

    merged_files = list_files(MERGED_DIR)
    adapter_files = list_files(ADAPTER_DIR) if ADAPTER_DIR.is_dir() else []
    base_files = list_files(DAPT_BASE_DIR) if (args.include_dapt_base and DAPT_BASE_DIR.is_dir()) else []

    print("将要上传：")
    print(f"  [根目录]    {len(merged_files):>3} 个文件  {human(dir_size(MERGED_DIR)):>9}  合并后的完整模型")
    if adapter_files and not args.skip_adapter:
        print(f"  [adapter/]  {len(adapter_files):>3} 个文件  {human(dir_size(ADAPTER_DIR)):>9}  LoRA adapter")
    if base_files:
        print(f"  [dapt_base/]{len(base_files):>3} 个文件  {human(dir_size(DAPT_BASE_DIR)):>9}  DAPT 底座")
    print("  [README.md] Model Card（自动生成）")
    total_bytes = dir_size(MERGED_DIR) + (0 if args.skip_adapter else dir_size(ADAPTER_DIR)) + dir_size(DAPT_BASE_DIR) * bool(base_files)
    print(f"\n  合计约 {human(total_bytes)}")

    if args.check:
        print("\n[检查模式] 未执行上传。去掉 --check 即可真正上传。")
        return 0

    # ---- 凭据 ----
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print("\n[错误] 未找到 HF token。")
        print("       1) 到 https://huggingface.co/settings/tokens 建一个 write token")
        print("       2) Windows:  set HF_TOKEN=hf_xxx")
        print("          Linux/mac: export HF_TOKEN=hf_xxx")
        print("       或者用 --token hf_xxx 直接传入。")
        return 1

    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("\n[错误] 缺少 huggingface_hub。请先：pip install huggingface_hub")
        return 1

    os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    repo_id = args.repo.strip()
    owner = repo_id.split("/")[0] if "/" in repo_id else ""
    gh_owner = args.gh_owner or DEFAULT_GH_OWNER
    if not args.gh_owner:
        print(f"\n[提示] 未指定 --gh-owner，Model Card 的 GitHub 链接按项目默认指向 "
              f"https://github.com/{gh_owner}/QuantMind")
        print(f"       要用别的账号请加：--gh-owner <GitHub用户名>")

    try:
        login(token=token, add_to_git_credential=False)
        api = HfApi(endpoint=args.endpoint)
        who = api.whoami(token=token)
        print(f"\n[认证] 已登录：{who.get('name')}")
    except Exception as e:
        print(f"\n[错误] 登录失败：{e}")
        print("       请检查 token 是否有效、是否具备 write 权限，以及本机能否访问 huggingface.co。")
        return 1

    # ---- 建仓库 ----
    try:
        api.create_repo(repo_id=repo_id, repo_type="model",
                        private=args.private, exist_ok=True)
        print(f"[仓库] 就绪：https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"[错误] 创建仓库失败：{e}")
        return 1

    # ---- Model Card ----
    card = build_model_card(repo_id, gh_owner)
    card_path = MERGED_DIR.parent / "README.md"
    card_path.write_text(card, encoding="utf-8")
    print(f"[卡片] 已生成 {card_path}")

    # ---- 上传 ----
    def _upload(local_dir: Path, path_in_repo: str, label: str):
        print(f"\n[上传] {label} → {path_in_repo or '<根目录>'}")
        api.upload_folder(
            repo_id=repo_id, folder_path=str(local_dir),
            path_in_repo=path_in_repo or None, repo_type="model",
        )
        print(f"       完成（{human(dir_size(local_dir))}）")

    try:
        _upload(MERGED_DIR, "", "合并模型")
        if adapter_files and not args.skip_adapter:
            _upload(ADAPTER_DIR, "adapter", "LoRA adapter")
        if base_files:
            _upload(DAPT_BASE_DIR, "dapt_base", "DAPT 底座")

        api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md",
                        repo_id=repo_id, repo_type="model")
        print("\n[上传] Model Card → README.md   完成")
    except Exception as e:
        print(f"\n[错误] 上传中断：{type(e).__name__}: {e}")
        print("       已上传的部分会保留，重新执行本脚本可续传（幂等）。")
        return 1

    print("\n" + "=" * 68)
    print("发布完成 ✅")
    print(f"  模型页面：https://huggingface.co/{repo_id}")
    print("\n其他人在全新电脑上这样用：")
    print(f"  cd ml && python setup_model.py --from-hf {repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
