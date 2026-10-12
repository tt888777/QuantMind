"""
一次性迁移：为已有的 LoRA adapter 补写 ``extra_state.safetensors``
=================================================================

只做一件事：把「底座 checkpoint 里没有、又不属于 classifier」的参数
（典型是 ``bert.pooler.dense.*``）从各自的 ``merged/`` 里取出来，存进
``adapter/extra_state.safetensors``。

**为什么需要它**：``dapt_bert`` 底座由 ``BertForMaskedLM`` 保存，不含 pooler；
LoRA 训练时 pooler 被随机初始化且冻结。少了这份权重，任何「重新从 adapter
现场合并」的路径都会得到不一致的模型。原理详见 ``ml/lora_extra.py``。

用法：
  python export_extra_state.py --all                 # 处理 ml/models/lora_*
  python export_extra_state.py --only lora_dapt_flame
  python export_extra_state.py --all --force         # 已存在也重写

新训练的模型不需要跑这个 —— ``train_lora.py`` 已在导出阶段自动生成。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lora_extra import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
