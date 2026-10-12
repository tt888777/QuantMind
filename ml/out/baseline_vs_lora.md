# 情感分类对照实验结果（实训基线）

- 数据集：`ml/data/guba_title_all.csv`（东财股吧标题，人工标注 3 类）
- 测试集：官方 `test` 划分，**2,130 条**，与训练集零重叠
- 任务：3 分类（0=看跌 / 1=看涨 / 2=中性）
- 三条对照：关键词规则 → 原始 BERT + LoRA → **DAPT 底座 + LoRA**

## 一、总览

| # | 方法 | 底座 | Accuracy | Macro-F1 |
|---|---|---|---|---|
| A | 关键词规则（`sentiment.py: analyze_sentiment`） | — | 0.4545 | 0.3847 |
| B | BERT-base-chinese + LoRA（r=8, α=16, 5 epoch） | `bert-base-chinese` | **0.7460** | **0.7417** |
| C | DAPT → LoRA（超参与 B 完全一致） | `dapt_bert` | 0.7446 | 0.7408 |

**A → B 提升**：Accuracy **+29.2 pt**、Macro-F1 **+35.7 pt**
**B → C 变化**：Accuracy **−0.14 pt**、Macro-F1 **−0.09 pt**（≈ 无差异，且略负）

> 两个 LoRA 模型的可训练参数完全相同：**297,219 / 102,567,174 = 0.29%**。
> 唯一变量就是底座是否经过 DAPT —— 所以 B 与 C 的差异可干净地归因于 DAPT。

## 二、分类别指标

### A. 关键词规则基线

| 类别 | Precision | Recall | F1 | support |
|---|---|---|---|---|
| negative/看跌 | 0.476 | 0.259 | 0.335 | 641 |
| positive/看涨 | 0.562 | 0.162 | 0.251 | 618 |
| neutral/中性 | 0.438 | 0.806 | 0.568 | 871 |
| **macro avg** | 0.492 | 0.409 | **0.385** | 2130 |

### B. BERT + LoRA（`models/lora_guba`）

| 类别 | Precision | Recall | F1 | support |
|---|---|---|---|---|
| negative/看跌 | 0.687 | 0.668 | 0.677 | 641 |
| positive/看涨 | 0.779 | 0.757 | 0.768 | 618 |
| neutral/中性 | 0.765 | 0.796 | 0.780 | 871 |
| **macro avg** | 0.744 | 0.740 | **0.742** | 2130 |

### C. DAPT + LoRA（`models/lora_dapt`）

| 类别 | Precision | Recall | F1 | support |
|---|---|---|---|---|
| negative/看跌 | 0.673 | 0.680 | 0.676 | 641 |
| positive/看涨 | 0.785 | 0.749 | 0.767 | 618 |
| neutral/中性 | 0.770 | 0.789 | 0.779 | 871 |
| **macro avg** | 0.743 | 0.739 | **0.741** | 2130 |

**逐类对比 B vs C**：三类 F1 差异均在 **0.001 以内**（−0.001 / −0.001 / −0.001），
没有任何一类出现可解释的方向性变化。

## 三、关键观察

1. **关键词基线严重塌向"中性"**：2,130 条里预测了 1,603 条中性（75.3%），而真实中性只占 40.9%。
   它把 74% 的真实"看跌"和 84% 的真实"看涨"都吞成了中性 —— Recall 只有 0.26 / 0.16。
   原因很直接：股吧口语表达（"真牛" "绿盘" "怎么玩"）不在它那 3 张关键词表里。
2. **LoRA 模型三类均衡**：三个类的 F1 落在 0.68–0.78，没有塌缩到多数类。
3. **"看跌"是最难的一类**（F1 ≈ 0.677，最低）：
   - 关键词基线在这类上几乎失效（0.335）；
   - 与数据质检结论一致 —— 该数据集里讽刺句、纯疑问句被标为负面的噪声最多。
4. **DAPT 没有带来收益（本实验的负结果）**。原因分析：
   - **域不匹配**：DAPT 语料是 *CN-Market-Corpus* 的新闻/公告体（书面、长句），
     而下游监督数据是 *股吧标题*（口语化、平均 21 字的短句）。二者文体差异明显，
     预训练"学到的口音"和下游要考的口音不是同一种。
   - **DAPT 强度轻**：1 epoch × 20 万条 × seq128 ≈ **2,560 万 token**，
     对 BERT 这种已在大规模通用语料上预训练过的模型来说，属于"轻量微调"，
     主要让 loss 从 2.23 降到 1.59（见 `dapt_report.md`），属于**适应语料统计分布**，
     不足以改变下游表征能力。
   - **下游瓶颈不在底座**：LoRA 只训 0.29% 参数，且分类头重新初始化，
     8,516 条训练样本 + 5 epoch 已把任务信号学得较充分；
     底座多一层"金融味"并不能突破这一层的上限。
5. 训练动态：两个 LoRA 的 loss 轨迹几乎重合（起始 ≈1.11，末段 step loss ≈0.45–0.52），
   **轻微过拟合**，但 8,515 条训练量 + 5 epoch 属正常范围。

## 四、结论与建议

- **结论**：在本数据集上，"BERT + LoRA" 已接近饱和，**轻量 DAPT 无增益**
  （ΔAcc = −0.14 pt，属于随机波动量级）。
- **实训价值**：这是一个有价值的**负结果** —— 它说明"多加一个预训练阶段"不是万能钥匙，
  域匹配度与预训练强度才是关键，比单纯报一个漂亮数字更有说服力。
- **若想验证 DAPT 是否真有用**，可尝试：
  1. 换成**同域**语料做 DAPT（例如抓取更多股吧文本，但注意与测试集隔离，避免泄漏）；
  2. 加大 DAPT 强度（多 epoch / 更大 seq_len / 更多条数）；
  3. 在**小样本**场景下对比（如只用 500 条训练 LoRA），DAPT 的收益通常在数据稀缺时才显现。

## 五、复现命令

```bash
cd ml

# A. 关键词基线
.venv/Scripts/python.exe eval_keyword_baseline.py

# B. 原始 BERT + LoRA
.venv/Scripts/python.exe train_lora.py --base models/bert-base-chinese \
  --data data/guba_title_all.csv --out models/lora_guba \
  --epochs 5 --batch_size 32 --lr 5e-4 --lora_r 8 --lora_alpha 16 --merge

# C. DAPT 底座 + LoRA（超参与 B 完全一致，只换 --base/--out）
.venv/Scripts/python.exe train_lora.py --base models/dapt_bert \
  --data data/guba_title_all.csv --out models/lora_dapt \
  --epochs 5 --batch_size 32 --lr 5e-4 --lora_r 8 --lora_alpha 16 --merge
```

（DAPT 底座本身的训练见 `ml/out/dapt_report.md`。）
