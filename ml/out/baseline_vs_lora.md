# 情感分类对照实验结果（实训基线）

- 数据集：`ml/data/guba_title_all.csv`（东财股吧标题，人工标注 3 类）
- 测试集：官方 `test` 划分，**2,130 条**，与训练集零重叠
- 任务：3 分类（0=看跌 / 1=看涨 / 2=中性）

## 总览

| 方法 | Accuracy | Macro-F1 |
|---|---|---|
| A. 关键词规则基线（`sentiment.py: analyze_sentiment`） | 0.4545 | 0.3847 |
| B. BERT-base-chinese + LoRA（r=8, α=16, 5 epoch） | **0.7460** | **0.7417** |
| **提升** | **+29.2 pt** | **+35.7 pt** |

## 分类别指标

### A. 关键词规则基线

| 类别 | Precision | Recall | F1 | support |
|---|---|---|---|---|
| negative/看跌 | 0.476 | 0.259 | 0.335 | 641 |
| positive/看涨 | 0.562 | 0.162 | 0.251 | 618 |
| neutral/中性 | 0.438 | 0.806 | 0.568 | 871 |
| **macro avg** | 0.492 | 0.409 | **0.385** | 2130 |

### B. LoRA 模型

| 类别 | Precision | Recall | F1 | support |
|---|---|---|---|---|
| negative/看跌 | 0.687 | 0.668 | 0.677 | 641 |
| positive/看涨 | 0.779 | 0.757 | 0.768 | 618 |
| neutral/中性 | 0.765 | 0.796 | 0.780 | 871 |
| **macro avg** | 0.744 | 0.740 | **0.742** | 2130 |

## 关键观察

1. **关键词基线严重塌向"中性"**：2,130 条里预测了 1,603 条中性（75.3%），而真实中性只占 40.9%。
   它把 74% 的真实"看跌"和 84% 的真实"看涨"都吞成了中性 —— Recall 只有 0.26 / 0.16。
   原因很直接：股吧口语表达（"真牛" "绿盘" "怎么玩"）不在它那 3 张关键词表里。
2. **LoRA 模型三类均衡**：三个类的 F1 落在 0.68–0.78，没有塌缩到多数类。
3. **"看跌"是最难的一类**（F1 0.677，最低）：
   - 关键词基线在这类上几乎失效（0.335）；
   - 与数据质检结论一致 —— 该数据集里讽刺句、纯疑问句被标为负面的噪声最多。
4. 训练 train_loss 0.62 → 末段 step loss 约 0.44–0.52，eval_loss 0.6297，**轻微过拟合**，
   但 8,515 条训练量 + 5 epoch 属正常范围。

## 复现命令

```bash
# 模型
cd ml
.venv/Scripts/python.exe train_lora.py --base models/bert-base-chinese \
  --data data/guba_title_all.csv --out models/lora_guba \
  --epochs 5 --batch_size 32 --lr 5e-4 --lora_r 8 --lora_alpha 16 --merge

# 基线
.venv/Scripts/python.exe eval_keyword_baseline.py
```
