---
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
---

# QuantMind 中文金融情感三分类（DAPT + 财经新闻 LoRA）

面向**中文财经新闻**的三分类情感模型：`看跌 / 看涨 / 中性`。
用于 [QuantMind](https://github.com/tt888777/QuantMind) 量化分析系统的「AI 智能体舆情决策」模块。

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
| Accuracy | **0.8290** |
| Macro-F1 | **0.8075** |

逐类（precision / recall / F1）：

| 类别 | P | R | F1 | 支持数 |
| --- | --- | --- | --- | --- |
| 看跌 negative | 0.806 | 0.855 | 0.830 | 698 |
| 看涨 positive | 0.884 | 0.909 | 0.896 | 1447 |
| 中性 neutral | 0.737 | 0.660 | 0.696 | 791 |

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

repo = "Roeyyyyyy/quantmind-sentiment-dapt-flame"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()

texts = ["公司发布业绩预告，净利润同比增长80%", "大股东拟减持不超过3%股份，股价承压"]
enc = tok(texts, truncation=True, max_length=64, padding=True, return_tensors="pt")
with torch.no_grad():
    probs = torch.softmax(model(**enc).logits, dim=-1)

id2label = model.config.id2label          # {0: '看跌', 1: '看涨', 2: '中性'}
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
| 数据 | `data/flame_zh_all.csv` 11744 条 |
| epoch / batch / lr | 5.0 / 32 / 0.0005 |
| max_len | 64 |
| LoRA | r=8, alpha=16, dropout=0.1 |
| 可训练参数 | 297,219 / 102,567,174（0.29%） |
| 训练环境 | Windows + RTX 3060 Laptop 6GB |

## 复现

```bash
# 1. 准备数据与底座
python prepare_dapt_corpus.py           # 构建 DAPT 语料（约 1GB）
python train_dapt.py --out models/dapt_bert --epochs 1
python prepare_flame_dataset.py         # 构建中文新闻集（繁转简 + 分层切分）

# 2. LoRA 微调 + 导出合并模型
python train_lora.py --base models/dapt_bert --data data/flame_zh_all.csv \
  --out models/lora_dapt_flame --epochs 5 --batch_size 32 --lr 5e-4 \
  --lora_r 8 --lora_alpha 16 --merge
```

## 引用

如果用到本模型，请注明来自 QuantMind 项目，并同时遵守上游数据集的许可条款。
