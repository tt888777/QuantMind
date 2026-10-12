"""
LoRA adapter 的「非 LoRA 必需参数」导出与回填
=============================================

**为什么需要这个模块**

LoRA 只保存注意力层（query/value）的低秩增量，加上 ``modules_to_save`` 里的分类头。
但 ``BertForSequenceClassification`` 还有别的参数是「训练时存在、底座 checkpoint 里却没有」的：

  * ``bert.pooler.dense.weight / bias``

原因是 DAPT 底座（``dapt_bert``）由 ``BertForMaskedLM`` 保存，而新版 transformers 的
``BertForMaskedLM`` 用 ``BertModel(config, add_pooling_layer=False)``，**不含 pooler**。
于是 ``--base models/dapt_bert`` 训练时，pooler 被**随机初始化**，而它不参与 LoRA 微调
（可训练参数只有 LoRA 增量 + classifier），全程冻结。

后果很隐蔽：那份随机 pooler 就是分类头所依赖的**固定非线性特征映射**。
- 训练结束时 ``merge_and_unload()`` 存下来的 ``merged/`` 里带着**那一份** pooler → 结果正确；
- 之后任何人重新从 adapter 现场合并（``setup_model.py``、推理后端），都会抽到
  **另一份随机 pooler** → 分类头失效，输出退化成近似均匀分布（实测三类概率都在 0.33~0.4）。

**做法**：把「底座 checkpoint 里没有、又不属于 classifier」的参数单独存到
``adapter/extra_state.safetensors``，加载时回填。这样 adapter 目录才是自洽可复现的。

用到的接口：
  * :func:`export_extra_state` —— 训练/迁移时导出（train_lora.py 调用）
  * :func:`extra_state_path` / :func:`read_extra_state` —— 加载侧读取
  * 命令行：``python export_extra_state.py --all`` 为已有 adapter 补写（一次性迁移）
"""

import json
from pathlib import Path

EXTRA_STATE_NAME = "extra_state.safetensors"

# 分类头由 PEFT 的 modules_to_save 负责保存，不要重复导出
_SKIP_PREFIXES = ("classifier.", "score.")
# PeftModel 的 state_dict 前缀；merged 后的普通模型没有这个前缀
_PEFT_PREFIX = "base_model.model."


def _safe_open(path):
    from safetensors import safe_open
    return safe_open(str(path), framework="pt")


def safetensors_keys(model_dir):
    """列出模型目录里所有 .safetensors 文件的张量名（取并集）。"""
    keys = set()
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        try:
            with _safe_open(f) as fh:
                keys.update(fh.keys())
        except Exception:
            continue
    return keys


def _strip_peft_prefix(name):
    return name[len(_PEFT_PREFIX):] if name.startswith(_PEFT_PREFIX) else name


def export_extra_state(model, base_path, adapter_dir):
    """导出 ``base_path`` 里缺失、推理时又必须一致的参数。

    ``model`` 可以是 ``PeftModel``（未合并）也可以是合并后的普通模型。
    返回导出的张量个数（0 表示无需导出，会删掉可能存在的旧文件）。
    """
    from safetensors.torch import save_file

    def _tensor_value(t):
        """兼容 state_dict() 里的 DTensor / 分片包装。"""
        return getattr(t, "_local_tensor", t)

    base_keys = safetensors_keys(base_path)
    payload = {}
    for name, tensor in model.state_dict().items():
        origin = _strip_peft_prefix(name)
        if origin in base_keys:
            continue
        if origin.startswith(_SKIP_PREFIXES):
            continue
        # modules_to_save 包装过的名字（classifier.modules_to_save.default.*）已在上面被拦掉；
        # 这里再兜一层，防止不同 PEFT 版本换命名。
        if ".modules_to_save." in origin:
            continue
        payload[origin] = _tensor_value(tensor).detach().to("cpu").contiguous()

    out = Path(adapter_dir) / EXTRA_STATE_NAME
    if not payload:
        if out.exists():
            out.unlink()
        return 0
    save_file(payload, str(out))
    return len(payload)


def extra_state_path(adapter_dir):
    """返回 adapter 目录下额外权重文件路径（不存在则返回 None）。"""
    p = Path(adapter_dir) / EXTRA_STATE_NAME
    return p if p.is_file() else None


def read_extra_state(adapter_dir):
    """读取额外权重，返回 ``dict[name, tensor]``；不存在则返回 None。"""
    p = extra_state_path(adapter_dir)
    if p is None:
        return None
    from safetensors.torch import load_file
    return load_file(str(p), device="cpu")


# ------------------------------------------------------------------ 一次性迁移
def _backfill_one(module_dir: Path, force=False):
    """为 ``models/<name>/{adapter,merged}`` 补写 extra_state.safetensors。"""
    adapter = module_dir / "adapter"
    merged = module_dir / "merged"
    if not adapter.is_dir() or not merged.is_dir():
        return None, "缺少 adapter 或 merged，跳过"

    out = adapter / EXTRA_STATE_NAME
    if out.exists() and not force:
        return None, f"已存在（{out.stat().st_size / 1024:.0f} KB），跳过"

    cfg_file = adapter / "adapter_config.json"
    try:
        raw_base = json.loads(cfg_file.read_text(encoding="utf-8")).get("base_model_name_or_path")
    except Exception as e:
        return None, f"读 adapter_config.json 失败：{e}"

    # 训练时写的是相对路径（models/xxx），相对于仓库里的 ml/ 目录
    roots = [Path(raw_base), module_dir.parent / raw_base, module_dir.parent.parent / raw_base]
    base = next((r for r in roots if (r / "config.json").is_file()), None)
    if base is None:
        return None, f"找不到底座（adapter 里写的是 {raw_base}），跳过"

    from safetensors import safe_open  # noqa: F401  仅在需要时校验依赖
    from safetensors.torch import save_file

    base_keys = safetensors_keys(base)
    payload = {}
    with _safe_open(merged / "model.safetensors") as fh:
        for name in fh.keys():
            if name in base_keys or name.startswith(_SKIP_PREFIXES):
                continue
            payload[name] = fh.get_tensor(name)

    if not payload:
        return 0, "merged 相对底座没有额外参数（该 adapter 无需修复）"

    save_file(payload, str(out))
    total = sum(t.numel() for t in payload.values())
    return len(payload), f"补写 {len(payload)} 个张量（{total:,} 参数，{out.stat().st_size / 1024:.0f} KB）"


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="为已有的 LoRA adapter 补写 extra_state.safetensors（修复 pooler 缺失）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--models", default=str(Path(__file__).resolve().parent / "models"),
                    help="模型根目录，默认 ml/models")
    ap.add_argument("--only", default=None, help="只处理指定模块，如 lora_dapt_flame")
    ap.add_argument("--all", action="store_true", help="处理全部 lora_* 模块")
    ap.add_argument("--force", action="store_true", help="已存在也重写")
    args = ap.parse_args()

    root = Path(args.models)
    if args.only:
        targets = [root / args.only]
    elif args.all:
        targets = sorted(p for p in root.glob("lora_*") if p.is_dir())
    else:
        ap.error("请指定 --all 或 --only <模块名>")

    print("LoRA adapter 额外权重补写（extra_state.safetensors）")
    print("=" * 68)
    for d in targets:
        n, msg = _backfill_one(d, force=args.force)
        flag = "OK  " if (n is not None and n > 0) else "----"
        print(f"  [{flag}] {d.name:<20} {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
