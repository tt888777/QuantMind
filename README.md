# QuantMind · 智能量化分析平台

一个本地运行的 A 股量化分析 Web 应用：数据采集 → 清洗 → 技术指标 → 策略回测 → 舆情情感分析 → 报告导出，
其中**舆情情感分析**支持两种可切换引擎：传统关键词规则，与微调后的 **BERT + LoRA** 神经网络模型。

> 本项目以「实习实训」为目标构建，重点在于**端到端的完整链路**与**可演示的工程实现**：
> 训练脚本、评测对照、模型接入、故障降级全部可用且有据可查。

---

## 目录

- [一、它长什么样](#一它长什么样)
- [二、环境要求](#二环境要求)
- [三、快速开始](#三快速开始)
  - [路径 A：只跑演示（3 分钟，不需要 AI）](#路径-a只跑演示3-分钟不需要-ai)
  - [路径 B：启用 AI 情感模型](#路径-b启用-ai-情感模型)
  - [路径 C：复现完整训练链路](#路径-c复现完整训练链路)
- [四、目录结构](#四目录结构)
- [五、情感引擎说明](#五情感引擎说明)
- [六、环境变量](#六环境变量)
- [七、接口清单](#七接口清单)
- [八、常见问题](#八常见问题)
- [九、数据来源](#九数据来源)
- [十、实测结果](#十实测结果)

---

## 一、它长什么样

启动后浏览器访问 <http://127.0.0.1:5000>，左侧导航共 5 个功能区：

| 页面 | 功能 |
|---|---|
| **概览** | 市场情绪指数、数据源状态、最近活动 |
| **数据导入** | 拖拽上传 CSV/Excel，或从接口抓取行情 |
| **数据处理** | 自动清洗报告（缺失填充 / 异常修正）+ MA / RSI / MACD 指标计算 |
| **量化研究** | 双均线策略回测（收益、年化、夏普、最大回撤） |
| **舆情决策** | 采集财经新闻/股吧 → 情感分析 → **情感引擎切换开关** → 情绪特征融合进策略 |
| **报告中心** | 导出图表与回测报告 |

舆情决策页顶部就是这个项目的核心演示点：

```
情感分析引擎   [ 关键词规则 | AI 模型（BERT+LoRA） ]   ● 模型就绪 · GPU · 看跌/看涨/中性
```

点一下即可**当场对比**两种引擎在同一批真实新闻上的判定差异。

---

## 二、环境要求

| 项 | 要求 | 说明 |
|---|---|---|
| 操作系统 | Windows（本项目在 Win10/11 实测） | 启动脚本为 `.bat`；代码本身跨平台 |
| Python | **3.10 ~ 3.12** | 推荐 3.12；ML 依赖的预编译包在 3.13 上支持不佳 |
| 数据库 | MySQL **或** SQLite | **没有 MySQL 也能跑**，走 SQLite 本地模式 |
| 显卡（可选） | 任意 NVIDIA GPU | 仅训练/加速推理需要；纯 CPU 也能推理，只是慢一些 |
| 磁盘 | 纯演示 < 100MB；含 AI 模型约 1.5GB | 底座 411MB + 合并模型 391MB |

---

## 三、快速开始

> **先 clone：**
> ```bash
> git clone https://github.com/tt888777/QuantMind.git
> cd QuantMind
> ```

### 路径 A：只跑演示（3 分钟，不需要 AI）

适合：只想看界面和功能，或者机器上没有显卡 / 不想装 torch。

```bash
# 1. 建虚拟环境（放在 quantmind/ 旁边即可）
python -m venv .venv
.venv\Scripts\activate

# 2. 装 Web 依赖（国内建议加清华镜像）
pip install -r quantmind\requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

然后**双击 `quantmind\start_server_local.bat`** 即可。

脚本会自动完成：检测数据库 → 不存在则用 `init_local_db.py` 建库并灌入 3 只示例股票 → 启动服务。
访问 <http://127.0.0.1:5000>。

> 此路径下情感分析走**关键词规则引擎**（功能完整，`engine=model` 会自动回退并给出提示）。
> 想在界面上看到 AI 模型，按路径 B 操作。

**如果你本机有 MySQL**，也可以改用原生的 `quantmind\start_server.bat`：
先执行一次 `quantmind_schema.sql` 建表，并确认 `quantmind/config.py` 里的连接串与你的 MySQL 一致。

---

### 路径 B：启用 AI 情感模型

适合：要在界面上看到 / 演示 BERT+LoRA 模型的效果。

> **为什么需要额外一步？** 模型权重文件过大（411MB / 391MB），超出 GitHub 单文件 100MB 限制，
> 因此**没有提交到仓库**——仓库里只包含 1.2MB 的 LoRA adapter（真正的训练产物）。
> 底座与合并模型用脚本一键恢复即可。

```bash
cd ml

# 1. 建 ML 环境（Web 依赖一起装，一套环境同时管界面和推理）
python -m venv .venv
.venv\Scripts\activate

# 2. 装 PyTorch —— 必须按本机 CUDA 版本单独装，别直接 pip install torch
pip install torch==2.9.1+cu126 --index-url https://download.pytorch.org/whl/cu126
#   没有 NVIDIA 显卡就装 CPU 版：pip install torch

# 3. 装其余依赖
pip install -r ..\quantmind\requirements-ml.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r ..\quantmind\requirements.txt    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```bash
# 4. 一键恢复模型（下载底座 + 用仓库自带 adapter 导出合并模型）
python setup_model.py

#    只想先下底座也行，推理后端会自动挂载 adapter：
#    python setup_model.py --base-only
```

```bash
# 5. 启动
cd ..\quantmind
start_server_local.bat
```

> **国内网络提示：** `huggingface.co` 在很多网络下不可达。`setup_model.py` 与启动脚本
> **已默认使用镜像 `https://hf-mirror.com`**，无需手动配置。若要用其它镜像，设 `HF_ENDPOINT` 即可。

启动后进「舆情决策」页，把引擎切到 **AI 模型**，徽标应显示 `模型就绪 · cuda`。

---

### 路径 C：复现完整训练链路

适合：想自己重训模型、或者做实训报告里的实验复现。

```bash
cd ml
.venv\Scripts\activate
```

```bash
# ① 下载并校验标注数据集（东方财富股吧标题，含人工标注）
python check_guba_dataset.py
#    → 产出 ml/data/guba_title_all.csv（10646 条）
#    → 产出 ml/out/dataset_report.json（标签分布 / 长度统计 / 重复检查）
#    → 产出 ml/out/sample_review.csv（每类 30 条人工抽检表）
```

```bash
# ② 领域自适应预训练（DAPT，可选但能提升效果）
#    纯股吧语料（快，约 3 分钟）
python train_dapt.py --model models/bert-base-chinese ^
                     --corpus data/guba_title_all.csv --text_col title ^
                     --epochs 1 --out models/dapt_bert

#    或混入大规模金融语料（流式采样，不会下载全部 13.8GB）
python train_dapt.py --model models/bert-base-chinese ^
                     --corpus hf:mulfsa/CN-Market-Corpus --max_samples 200000 ^
                     --out models/dapt_bert
```

```bash
# ③ LoRA 微调
python train_lora.py --base models/bert-base-chinese ^
                     --data data/guba_title_all.csv ^
                     --out models/lora_guba --epochs 5 --merge

#    基于 DAPT 底座再微调（对比 DAPT 带来的增益）
python train_lora.py --base models/dapt_bert --out models/lora_dapt --epochs 5 --merge
```

```bash
# ④ 关键词规则基线评测（在同一测试集上跑，用于对照）
python eval_keyword_baseline.py
```

全部脚本都支持 `--smoke N`（只跑 N 步）用于快速验证链路是否通。

---

## 四、目录结构

```
QuantMind/
├── prototype.html                  # 单页前端（Flask 直接托管，改它不用重启服务）
├── user-stories.md                 # 需求 / 用户故事
├── quantmind_schema.sql            # MySQL 建表脚本
│
├── quantmind/                      # Flask 后端
│   ├── app.py                      # 全部路由
│   ├── config.py                   # DB_URL（支持环境变量覆盖）
│   ├── db.py                       # SQLAlchemy 模型（MySQL / SQLite 双兼容）
│   ├── init_local_db.py            # 本地 SQLite 建库 + 示例数据
│   ├── start_server_local.bat      # ★ SQLite 模式启动（推荐）
│   ├── start_server.bat            # MySQL 模式启动
│   ├── requirements.txt            # Web 依赖
│   ├── requirements-ml.txt         # 模型推理依赖
│   └── modules/
│       ├── sentiment.py            # 舆情采集 + 可插拔情感分析
│       ├── sentiment_model.py      # ★ BERT+LoRA 推理后端（懒加载 / 降级）
│       ├── data_fetch.py           # 行情抓取
│       ├── data_process.py         # 清洗与指标
│       ├── backtest.py             # 双均线回测
│       └── visualization.py        # 绘图
│
└── ml/                             # 机器学习链路
    ├── setup_model.py              # ★ 一键恢复模型（下底座 + 导出 merged）
    ├── check_guba_dataset.py       # 数据集下载 / 校验 / 质量抽检
    ├── train_dapt.py               # MLM 域自适应预训练
    ├── train_lora.py               # LoRA 三分类微调
    ├── eval_keyword_baseline.py    # 关键词基线评测
    ├── data/                       # 标注语料（已提交）
    ├── out/                        # 报告、loss 曲线、抽检表
    └── models/
        ├── bert-base-chinese/      # 底座（需 setup_model.py 恢复）
        └── lora_guba/
            ├── adapter/            # ★ LoRA 权重（1.2MB，已提交）
            ├── merged/             # 合并模型（需 setup_model.py 恢复）
            └── metrics.json        # 训练指标
```

---

## 五、情感引擎说明

两种引擎通过 `engine` 参数切换，实现「**模型可插拔 + 规则保底**」：

| engine | 行为 |
|---|---|
| `keyword` | 只用关键词词表 + 正则（默认值，与原始行为完全一致） |
| `model` | 优先用 BERT+LoRA；**模型不可用时自动回退关键词**，绝不报错 |
| `auto` | 模型可用就用模型，否则关键词 |

**降级触发条件**（全部实测通过，均不会中断服务）：

- 未装 `torch` / `transformers`
- 模型目录不存在（如刚 clone 还没跑 `setup_model.py`）
- 权重文件损坏
- 推理过程抛异常

**标签对齐**：模型输出统一为 `看跌 / 看涨 / 中性`（置信度 ≥ 0.9 时输出 `强烈看跌 / 强烈看涨`），
与关键词引擎词表一致，因此后端的 `'看涨' in sentiment` 聚合逻辑和前端渲染**无需任何改动**。

**模型加载路径探测顺序**（`quantmind/modules/sentiment_model.py`）：

1. `ml/models/lora_dapt/merged`（若做过 DAPT 微调）
2. `ml/models/lora_guba/merged`（合并模型，加载最快）
3. `ml/models/lora_guba/adapter`（仓库自带，加载时自动与底座合并）

---

## 六、环境变量

全为可选，不设置则使用默认值。

| 变量 | 默认 | 作用 |
|---|---|---|
| `QUANTMIND_DB_URL` | `mysql+pymysql://root:...@localhost:3306/quantmind` | 数据库连接串；切 SQLite 就设成 `sqlite:///<绝对路径>/quantmind_local.db` |
| `QUANTMIND_SENTIMENT_MODEL` | 自动探测（见上） | 直接指定情感模型目录 |
| `QUANTMIND_SENTIMENT_BASE_MODEL` | `ml/models/bert-base-chinese` | adapter 对应的底座路径 |
| `QUANTMIND_SENTIMENT_DEVICE` | 自动（有 CUDA 用 cuda） | `cuda` / `cpu` |
| `QUANTMIND_SENTIMENT_MAX_LEN` | `64` | 最大 token 长度 |
| `QUANTMIND_SENTIMENT_BATCH` | `32` | 推理批大小 |
| `QUANTMIND_SENTIMENT_STRONG` | `0.9` | 「强烈」标签阈值，设 `0` 则禁用 |
| `QUANTMIND_SENTIMENT_WARMUP` | `0` | 设 `1` 则启动时预热模型（避免首个请求等待） |
| `HF_ENDPOINT` | 启动脚本已设为 `https://hf-mirror.com` | HuggingFace 镜像地址 |
| `HF_HUB_DISABLE_XET` | 启动脚本与 `setup_model.py` 已设为 `1` | 关闭 Xet 协议（镜像站不支持，见 Q1） |

---

## 七、接口清单

服务默认监听 `5000` 端口，全部为 JSON 接口。

**数据与行情**

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/stocks` | 已入库的股票列表 |
| POST | `/api/fetch` | 抓取行情入库 |
| POST | `/api/import_csv` `/api/import_file` | 导入 CSV / Excel |
| GET | `/api/quotation/<stock_code>` | 日线数据（`?days=30`） |
| POST | `/api/clean` | 数据清洗 |
| POST | `/api/indicators` | 计算技术指标 |
| POST | `/api/backtest` | 双均线回测 |
| POST | `/api/pipeline` | 一键跑完整流水线 |

**舆情与情感**

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/sentiment_collect` | 采集新闻/股吧（返回 `simulated` 标记是否为模拟数据） |
| POST | `/api/sentiment_analyze` | 情感聚合分析，**支持 `engine` 参数** |
| GET/POST | `/api/sentiment_engine_status` | 模型加载状态（`{"probe": true}` 会真正尝试加载） |
| POST | `/api/sentiment_predict` | 单条/批量预测，支持 `compare=true` 并列返回两引擎结果 |
| POST | `/api/sentiment_fuse` | 情绪特征融合进策略 |

**报告**

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/export_chart` `/api/export_report` | 导出图表 / 报告 |
| GET | `/api/reports` · `/api/reports/delete` | 报告列表 / 删除 |
| GET | `/exports/<filename>` | 下载导出文件 |

**示例：验证模型真的在工作**

```bash
curl -X POST http://127.0.0.1:5000/api/sentiment_predict \
  -H "Content-Type: application/json" \
  -d "{\"titles\":[\"公司发布业绩预告，净利润同比增长80%\"],\"engine\":\"model\"}"
```

返回体会带 `engine`（实际生效的后端）、`model_status`（设备/路径/错误原因）以及每条的概率向量——
**只要看到三个和为 1 的浮点数，就说明真在做神经网络推理**（关键词引擎只能给出 0.5 / 0.72 这类硬编码值）。

---

## 八、常见问题

**Q1：`huggingface.co` 连不上 / 下载模型卡住？**
国内网络常态。本项目所有脚本**已默认走 `https://hf-mirror.com`**，并且已经处理好镜像站的两个「专属坑」：

| 现象 | 原因 | 已处理方式 |
|---|---|---|
| 404 `xet-read-token` | 新版 `huggingface_hub` 默认使用 **Xet 存储协议**，镜像站不支持 | 脚本已设 `HF_HUB_DISABLE_XET=1` 回落到普通 HTTP 下载 |
| 403 Forbidden | 镜像站要求请求带 **User-Agent**，空 UA 会被直接拒绝 | 内置 UA 头 |

> 如果你自己写脚本下载模型，记得也带上这两项，否则会在这两个地方卡住。

如仍失败，可换镜像：`set HF_ENDPOINT=https://你的镜像`。

**Q2：我本机没有安装 MySQL，能跑吗？**
能。用 `quantmind\start_server_local.bat` 走 SQLite 模式，首次运行会自动建库。
代码通过 `BigInteger().with_variant(Integer, "sqlite")` 兼容两边，**原 MySQL 配置一行没改**。

**Q3：界面显示「模型不可用，已自动回退关键词规则」？**
三种原因，按顺序排查：
1. 还没跑 `ml\setup_model.py`（最常见）→ 跑一次，或先 `--check` 看缺什么；
2. `torch` / `transformers` 没装 → 见路径 B 第 2、3 步；
3. 启动用的是系统 Python 而非 `ml\.venv` → 启动脚本会打印实际使用的解释器，看一眼日志。

**Q4：模型在 CPU 上很慢？**
BERT-base 短文本在 CPU 上约 20~60ms/条。批量预测（`/api/sentiment_predict` 支持一次传多条）比逐条调用快得多。

**Q5：为什么仓库里没有模型文件？**
两个权重文件 411MB / 391MB，超出 GitHub 单文件 100MB 上限，推送会被直接拒绝。
它们在 `.gitignore` 中并附了恢复方式注释；`ml\setup_model.py` 一条命令即可补齐。
**1.2MB 的 LoRA adapter 是真正的训练产物，已提交。**

**Q6：为什么演示数据是模拟的？**
`collect_sentiment` 在**所有数据源都抓取失败**时会回退到模板生成的模拟新闻。
接口返回体里的 `simulated` 字段会明确标记。**做实验或答辩演示时建议确认该字段为 `False`**，避免拿模拟数据自证。

**Q7：情感分析真的在用 AI 吗？**
可以用三条硬证据自查：
① `/api/sentiment_engine_status?probe=true` 看 `loaded` 与 `device`；
② 预测结果里出现三分类 softmax 概率向量；
③ 用训练集里不存在的句子测试（如「董事长被采取强制措施，公司紧急停牌」），关键词引擎无法给出合理判断。

---

## 九、数据来源

| 用途 | 数据集 | 说明 |
|---|---|---|
| **LoRA 监督训练** | [HikasaHana/eastmoney_guba_title](https://huggingface.co/datasets/HikasaHana/eastmoney_guba_title) | 东方财富上证50ETF股吧标题，**人工标注** 三分类，约 1.07 万条 |
| DAPT 语料（可选） | [mulfsa/CN-Market-Corpus](https://huggingface.co/datasets/mulfsa/CN-Market-Corpus) | 2013–2023 中文金融市场文本，**无标签**，约 135 万条，Apache-2.0 |
| 底座模型 | [bert-base-chinese](https://huggingface.co/bert-base-chinese) | 中文 BERT-base，约 1.02 亿参数 |

**已知局限（会在论文/实训报告中说明）**：训练数据只有**上证50ETF 单一标的**，
因此模型对**个股**新闻（尤其是业绩公告类）泛化会打折。
这正是本项目保留 DAPT 环节、并做「关键词 vs 模型」对照实验的原因。

---

## 十、实测结果

**情感分类（测试集 2,130 条，3 分类）**

| 方法 | Accuracy | Macro-F1 |
|---|---|---|
| 关键词规则基线 | 0.4545 | 0.3847 |
| **BERT + LoRA** | **0.7460** | **0.7417** |
| 提升 | **+29.2 pt** | **+35.7 pt** |

- 可训练参数 **297,219 / 102,567,174 = 0.29%**（LoRA 的意义）
- 训练环境：RTX 3060 Laptop 6GB，bs=32 / seq=64 / 5 epoch 约 9 分钟
- DAPT 验证：全量 8,516 条跑 1 epoch，loss 从 **5.118 → 3.68**

完整对照报告见 `ml/out/baseline_vs_lora.md`，训练 curve 见 `ml/out/loss_curves/`。

**为什么基线会输得这么惨？** 关键词引擎把 2,130 条里的 1,603 条（75.3%）判成中性
（真实中性只占 40.9%），真实「看涨」的召回只有 0.16——股吧口语（「真牛」「绿盘」「怎么玩」）
根本不在词表里。这正是把规则引擎替换为模型的直接动机。

---

## 许可

本项目为学习/实训用途。引用第三方数据集与预训练模型时，请遵循其各自的许可协议。
