# QuantMind · 智能量化分析平台

一个本地运行的 A 股量化分析 Web 应用：数据采集 → 清洗 → 技术指标 → 策略回测 → 舆情情感分析 → 报告导出，

其中**舆情情感分析**支持两种可切换引擎：传统关键词规则，与微调后的 **BERT + LoRA** 神经网络模型。  
线上唯一的 AI 模型是 **`lora_dapt_flame`** = BERT 领域自适应预训练（DAPT）底座 + 中文财经新闻三分类 LoRA  
（选型理由见 [五、情感引擎说明](#五情感引擎说明)）。

> 本项目以「实习实训」为目标构建，重点在于**端到端的完整链路**与**可演示的工程实现**：
>
> 训练脚本、评测对照、模型接入、故障降级全部可用且有据可查。

---

## 目录

- [一、它长什么样](#一它长什么样)
- [二、从一台全新电脑部署](#二从一台全新电脑部署)
  - [步骤 1 · 获取代码](#步骤-1--获取代码)
  - [步骤 2 · 建 Python 环境](#步骤-2--建-python-环境)
  - [步骤 3 · 准备数据库](#步骤-3--准备数据库)
  - [步骤 4 · 准备 AI 模型](#步骤-4--准备-ai-模型)
  - [步骤 5 · 启动与自检](#步骤-5--启动与自检)
  - [只想快速跑演示（最短路径）](#只想快速跑演示最短路径)
- [三、环境要求](#三环境要求)
- [四、目录结构](#四目录结构)
- [五、情感引擎说明](#五情感引擎说明)
- [六、环境变量](#六环境变量)
- [七、接口清单](#七接口清单)
- [八、常见问题](#八常见问题)
- [九、数据来源](#九数据来源)
- [十、实测结果](#十实测结果)
- [十一、复现训练链路](#十一复现训练链路)
- [十二、发布模型到 Hugging Face](#十二发布模型到-hugging-face)

---

## 一、它长什么样

启动后浏览器访问 <http://127.0.0.1:5000>，左侧导航共 5 个功能区：

| 页面       | 功能                                          |
| -------- | ------------------------------------------- |
| **概览**   | 市场情绪指数、数据源状态、最近活动                           |
| **数据导入** | 拖拽上传 CSV/Excel，或从接口抓取行情                     |
| **数据处理** | 自动清洗报告（缺失填充 / 异常修正）+ MA / RSI / MACD 指标计算   |
| **量化研究** | 双均线策略回测（收益、年化、夏普、最大回撤）                      |
| **舆情决策** | 采集财经新闻/股吧 → 情感分析 → **情感引擎切换开关** → 情绪特征融合进策略 |
| **报告中心** | 导出图表与回测报告                                   |

舆情决策页顶部就是这个项目的核心演示点：

```
情感分析引擎   [ 关键词规则 | AI 模型（DAPT + 财经新闻 LoRA） ]   ● DAPT + 财经新闻 LoRA 就绪 · GPU · 看跌/看涨/中性
```

点一下即可**当场对比**两种引擎在同一批真实新闻上的判定差异。

---

## 二、从一台全新电脑部署

以下步骤假设你拿到的是**一台什么都没装的电脑**（Windows 为例；macOS/Linux 把 `\` 换成 `/` 即可）。  
按顺序做完 5 步就能跑起来，**约 15 分钟**（模型另需下载约 390MB）。

> 全程只需要两样外部资源：① GitHub 取代码；② 模型权重（从 HF 下载，或本机重建）。  
> **数据库不需要 MySQL**——默认走 SQLite，启动脚本会自动建库。

### 步骤 0 · 前置检查

```bash
python --version     # 需要 3.10 ~ 3.12（推荐 3.12）
git --version
```

没有 Python 去 <https://www.python.org/downloads/> 装（**务必勾选 Add Python to PATH**）；  
没有 git 去 <https://git-scm.com/downloads> 装。

### 步骤 1 · 获取代码

```bash
git clone https://github.com/tt888777/QuantMind.git
cd QuantMind
```

### 步骤 2 · 建 Python 环境

项目**只用一个虚拟环境**：`ml/.venv`，Web 依赖与模型推理依赖都装在里面。  
启动脚本会自动优先使用它，**所以路径和名字不要改**。

```bash
cd ml
python -m venv .venv
.venv\Scripts\activate

# PyTorch 要按你的显卡单独装，不要直接 pip install torch
# 有 NVIDIA 显卡（以 CUDA 12.6 为例）：
pip install torch --index-url https://download.pytorch.org/whl/cu126
# 没有显卡 / 不想装 CUDA：
# pip install torch

# 其余依赖（国内建议加清华镜像）
pip install -r ..\quantmind\requirements-ml.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r ..\quantmind\requirements.txt    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 只想跑界面的话可以**跳过 torch**：`quantmind/requirements.txt` 里没有 torch，  
> 界面完全不依赖它。没装 torch 时 `engine=model` 会自动回退关键词规则，其余功能照常。

### 步骤 3 · 准备数据库

**默认 SQLite，零配置** —— `start_server_local.bat` 第一次运行时会自动建库并灌入 3 只示例股票。

想用 MySQL：先执行 `quantmind_schema.sql` 建表，确认 `quantmind/config.py` 里的连接串，  
然后改用 `quantmind\start_server.bat`。也可以用环境变量 `QUANTMIND_DB_URL` 覆盖，不必改代码。

### 步骤 4 · 准备 AI 模型

这一步决定「AI 模型」引擎能不能用。**模型权重过大（底座 390MB + 合并模型 391MB），没有进仓库**，  
仓库里只有 3.8MB 的 LoRA adapter。二选一：

**方案 ①（推荐）从 Hugging Face 下载已合并的完整模型**

```bash
cd ml
python setup_model.py --from-hf <用户名>/<仓库名>
# 例：python setup_model.py --from-hf Roeyyyyyy/quantmind-sentiment-dapt-flame
```

约 390MB。下载完会把仓库地址写进 `ml/model_source.json`，之后直接 `python setup_model.py` 即可复用。

> 该 HF 仓库里还有 `adapter/` 与 `dapt_base/` 两个子目录（合计再约 410MB）。
> `--from-hf` **只取根目录的合并模型**、自动跳过这两个子目录，所以实际只下载 390MB。
> 需要它们时见 [十二](#十二发布模型到-hugging-face) 的说明。

**方案 ② 用仓库自带 adapter 现场合并**

适合拿不到 HF 的情况，但**前提是已有 DAPT 底座** `ml/models/dapt_bert/`  
（不在仓库里，需要用 `ml/train_dapt.py` 自己训，约 1 小时）：

```bash
cd ml
python setup_model.py --rebuild
```

**随时查看状态：**

```bash
python setup_model.py --check
```

> **国内网络提示**：`huggingface.co` 在部分网络下不可达。脚本与启动文件**已默认使用镜像  
> `https://hf-mirror.com`**，无需手动配置；换其它镜像设 `HF_ENDPOINT` 即可。  
> 注意**镜像只支持下载**——发布模型（见 [十二](#十二发布模型到-hugging-face)）必须用官方站。

### 步骤 5 · 启动与自检

```bash
cd ..\quantmind
start_server_local.bat
```

浏览器打开 <http://127.0.0.1:5000>，进&#x5165;**「舆情决策」**&#x9875;：

1. 顶部有引擎切换条，点 **AI 模型（DAPT + 财经新闻 LoRA）**
2. 徽标显示 `DAPT + 财经新闻 LoRA 就绪 · GPU · 看跌/看涨/中性` 即加载成功
3. 点「开始分析」，即可看到模型判定与置信度

命令行自检（不启服务也能验证模型）：

```bash
cd quantmind
..\ml\.venv\Scripts\python.exe modules\sentiment_model.py
```

会打印模型路径、设备与 4 条样例的判定。若显示 `"engine": "keyword"` 说明模型未就绪，  
按报错信息回到步骤 4。

### 只想快速跑演示（最短路径）

不装 torch、不下模型，只看界面：

```bash
cd ml && python -m venv .venv && .venv\Scripts\activate
pip install -r ..\quantmind\requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ..\quantmind && start_server_local.bat
```

情感分析走**关键词规则引擎**，其余功能完全一致。

---

## 三、环境要求

| 项      | 要求                          | 说明                             |
| ------ | --------------------------- | ------------------------------ |
| 操作系统   | Windows（本项目在 Win10/11 实测）   | 启动脚本为 `.bat`；代码本身跨平台           |
| Python | **3.10 ~ 3.12**             | 推荐 3.12；ML 依赖的预编译包在 3.13 上支持不佳 |
| 数据库    | MySQL **或** SQLite          | **没有 MySQL 也能跑**，默认走 SQLite    |
| 显卡（可选） | 任意 NVIDIA GPU               | 仅加速推理需要；纯 CPU 也能推理，只是慢一些       |
| 磁盘     | 纯演示 < 100MB；**启用 AI 约 1GB** | 合并模型 391MB（+ 可选 DAPT 底座 391MB） |
| 网络     | 首次装依赖、首次下模型需要               | 模型可从 HF 镜像下载（国内可用）             |

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
│   ├── start_server_local.bat      # SQLite 模式启动（推荐）
│   ├── start_server.bat            # MySQL 模式启动
│   ├── requirements.txt            # Web 依赖
│   ├── requirements-ml.txt         # 模型推理依赖
│   └── modules/
│       ├── sentiment.py            # 舆情采集 + 可插拔情感分析
│       ├── sentiment_model.py      # BERT+LoRA 推理后端（懒加载 / 降级）
│       ├── data_fetch.py           # 行情抓取
│       ├── data_process.py         # 清洗与指标
│       ├── backtest.py             # 双均线回测
│       └── visualization.py        # 绘图
│
└── ml/                             # 机器学习链路
    ├── setup_model.py              # ★ 准备模型（HF 下载 或 本地 adapter 合并）
    ├── upload_to_hf.py             # 发布模型到 Hugging Face（生成 Model Card）
    ├── check_guba_dataset.py       # 股吧数据集下载 / 校验 / 质量抽检
    ├── prepare_dapt_corpus.py      # 构建 DAPT 语料（Range 取 CN-Market-Corpus 头部）
    ├── prepare_flame_dataset.py    # 构建 FLAME 中文新闻集（繁转简 + 分层切分）
    ├── train_dapt.py               # MLM 域自适应预训练
    ├── train_lora.py               # LoRA 三分类微调
    ├── lora_extra.py               # LoRA adapter 的非 LoRA 参数导出/回填（修 pooler 问题）
    ├── export_extra_state.py       # 为旧 adapter 补写 extra_state.safetensors（一次性迁移）
    ├── eval_keyword_baseline.py    # 关键词基线评测
    ├── eval_cross_domain.py        # 跨域评测（N 模型 × M 测试集矩阵）
    ├── model_source.json           # 记录 HF 模型仓库地址（setup_model.py 自动生成）
    ├── data/                       # 标注语料（已提交）
    ├── out/                        # 报告、loss 曲线、对照矩阵
    └── models/
        ├── bert-base-chinese/      # 通用底座（已 gitignore）
        ├── dapt_bert/              # DAPT 底座（train_dapt.py 产出，已 gitignore）
        ├── lora_guba/              # A：原底座 + 股吧数据（adapter 已提交）
        ├── lora_dapt/              # B：DAPT 底座 + 股吧数据（adapter 已提交）
        ├── lora_flame/             # C：原底座 + 新闻数据（adapter 已提交）
        └── lora_dapt_flame/        # D：★ 线上模型（DAPT 底座 + 新闻数据）
            ├── adapter/            #   LoRA 权重 + extra_state（3.8MB，已提交）
            ├── merged/             #   合并后的完整模型（391MB，已 gitignore）
            └── metrics.json        #   训练指标
```

---

## 五、情感引擎说明

两种引擎通过 `engine` 参数切换，实现「**模型可插拔 + 规则保底**」：

| engine    | 行为                               |
| --------- | -------------------------------- |
| `keyword` | 只用关键词词表 + 正则（默认值，与原始行为完全一致）      |
| `model`   | 优先用 AI 模型；**模型不可用时自动回退关键词**，绝不报错 |
| `auto`    | 模型可用就用模型，否则关键词                   |

**降级触发条件**（全部实测通过，均不会中断服务）：

- 未装 `torch` / `transformers`
- 模型目录不存在（如刚 clone 还没跑 `setup_model.py`）
- 权重文件损坏
- 推理过程抛异常

**标签对齐**：模型输出统一为 `看跌 / 看涨 / 中性`（置信度 ≥ 0.9 时输出 `强烈看跌 / 强烈看涨`），  
与关键词引擎词表一致，因此后端的 `'看涨' in sentiment` 聚合逻辑和前端渲染**无需任何改动**。

### 模型加载路径探测顺序

`quantmind/modules/sentiment_model.py` 里**只配置了线上这一个模型**：

1. `ml/models/lora_dapt_flame/merged` —— 合并后的完整模型（加载最快，推荐）
2. `ml/models/lora_dapt_flame/adapter` —— 仓库自带的 LoRA 权重，加载时现场与底座合并

adapter 场景会去解析 `adapter_config.json` 里的底座路径，**必须是 `ml/models/dapt_bert`**。  
底座缺失时**直接报明确错误并回退关键词规则**，不会静默换用别的底座 —— 因为把 LoRA 权重挂到  
错误的底座上会产出一个「能跑但结果错」的模型，比直接失败更危险。

### 模型选择：线上只用哪一版、为什么

项目一共训了 4 个 LoRA 变体（2 底座 × 2 数据域），线上**只使用 `lora_dapt_flame`**：

| 变体                        | 底座                | 数据 | 新闻测试集 Acc  | 股吧测试集 Acc  |
| ------------------------- | ----------------- | -- | ---------- | ---------- |
| A `lora_guba`             | bert-base-chinese | 股吧 | 0.5562     | **0.7460** |
| B `lora_dapt`             | dapt_bert         | 股吧 | 0.5725     | 0.7446     |
| C `lora_flame`            | bert-base-chinese | 新闻 | 0.8273     | 0.5549     |
| **D `lora_dapt_flame`** ★ | dapt_bert         | 新闻 | **0.8290** | 0.5592     |

选 D 的理由：它是**新闻域表现最好**的版本（Acc 0.8290 / Macro-F1 0.8075），  
也保留了完整的「DAPT → LoRA」两阶段流程。

> ⚠️ **已知取舍**：D（以及所有新闻域模型）**不擅长股吧口语**，在股吧测试集上只有 0.5592，  
> 接近"全猜多数类"。如果你的业务是**股吧舆情**，把引擎指向股吧域模型即可：
>
> ```bash
> set QUANTMIND_SENTIMENT_MODEL=<仓库绝对路径>/ml/models/lora_guba/merged
> ```
>
> 仓库里 A/B/C 三个 adapter 都保留着，用 `train_lora.py --base ... --merge` 或  
> `setup_model.py` 的思路都能重建出对应的 `merged/`。

### 一个容易踩的坑：adapter 必须带 `extra_state.safetensors`

DAPT 底座由 `BertForMaskedLM` 保存，**不含 `bert.pooler`**；而  
`BertForSequenceClassification` 需要它。LoRA 训练时 pooler 被**随机初始化且全程冻结**  
（可训练参数只有 LoRA 增量 + 分类头），所以那份随机值就是分类头所依赖的特征映射。

后果：**任何"重新从 adapter 现场合并"的路径都会抽到另一份随机 pooler，导致分类头失效**  
（实测输出退化成近似均匀分布：三类概率都在 0.33~0.4）。

解决办法（已在仓库内实现）：

- `train_lora.py` 训练导出时，自动把「底座里没有、又不属于分类头」的参数存成  
  `adapter/extra_state.safetensors`
- 推理后端（`sentiment_model.py`）与 `setup_model.py` 在合并后**回填**这份权重
- 历史 adapter 用 `python export_extra_state.py --all` 一次性补写

原理与实现见 `ml/lora_extra.py`。**自检方法**：`merged/` 与 adapter 现场合并两条路径，  
对同一批文本应给出**逐位一致**的概率。

---

## 六、环境变量

全为可选，不设置则使用默认值。

| 变量                               | 默认                                                  | 作用                                                        |
| -------------------------------- | --------------------------------------------------- | --------------------------------------------------------- |
| `QUANTMIND_DB_URL`               | `mysql+pymysql://root:...@localhost:3306/quantmind` | 数据库连接串；切 SQLite 就设成 `sqlite:///<绝对路径>/quantmind_local.db` |
| `QUANTMIND_SENTIMENT_MODEL`      | `ml/models/lora_dapt_flame/merged`（不存在则 adapter）    | 直接指定情感模型目录；换回股吧域模型就用它                                     |
| `QUANTMIND_SENTIMENT_BASE_MODEL` | `ml/models/dapt_bert`                               | adapter 对应的底座路径（必须与训练时一致）                                 |
| `QUANTMIND_SENTIMENT_DEVICE`     | 自动（有 CUDA 用 cuda）                                   | `cuda` / `cpu`                                            |
| `QUANTMIND_SENTIMENT_MAX_LEN`    | `64`                                                | 最大 token 长度                                               |
| `QUANTMIND_SENTIMENT_BATCH`      | `32`                                                | 推理批大小                                                     |
| `QUANTMIND_SENTIMENT_STRONG`     | `0.9`                                               | 「强烈」标签阈值，设 `0` 则禁用                                        |
| `QUANTMIND_SENTIMENT_WARMUP`     | `0`                                                 | 设 `1` 则启动时预热模型（避免首个请求等待）                                  |
| `QUANTMIND_MODEL_HF_REPO`        | 读 `ml/model_source.json`                            | 模型所在的 HF 仓库（`用户名/仓库名`），供 `setup_model.py` 使用              |
| `HF_TOKEN`                       | 无                                                   | 上传模型到 HF 用的 write token（仅 `upload_to_hf.py` 需要）           |
| `HF_ENDPOINT`                    | 启动脚本已设为 `https://hf-mirror.com`                     | HuggingFace 镜像地址                                          |
| `HF_HUB_DISABLE_XET`             | 启动脚本与 `setup_model.py` 已设为 `1`                      | 关闭 Xet 协议（镜像站不支持，见 Q1）                                    |


## 八、常见问题

**Q1：`huggingface.co` 连不上 / 下载模型卡住？**

国内网络常态。本项目所有脚本**已默认走 `https://hf-mirror.com`**，并且已经处理好镜像站的两个「专属坑」：

| 现象                   | 原因                                            | 已处理方式                                     |
| -------------------- | --------------------------------------------- | ----------------------------------------- |
| 404 `xet-read-token` | 新版 `huggingface_hub` 默认使用 **Xet 存储协议**，镜像站不支持 | 脚本已设 `HF_HUB_DISABLE_XET=1` 回落到普通 HTTP 下载 |
| 403 Forbidden        | 镜像站要求请求带 **User-Agent**，空 UA 会被直接拒绝           | 内置 UA 头                                   |

> 如果你自己写脚本下载模型，记得也带上这两项，否则会在这两个地方卡住。

如仍失败，可换镜像：`set HF_ENDPOINT=https://你的镜像`。

> ⚠️ **镜像只能下载，不能上传。** 发布模型到 HF（[十二](#十二发布模型到-hugging-face)）  
> 必须用官方站 `https://huggingface.co`，且需要能访问它的网络。
>
> 💡 如果连镜像也不通，模型还有第二条路：本机 `python setup_model.py --rebuild`  
> （前提是已有 DAPT 底座 `ml/models/dapt_bert`）。

**Q2：我本机没有安装 MySQL，能跑吗？**

能。用 `quantmind\start_server_local.bat` 走 SQLite 模式，首次运行会自动建库。

代码通过 `BigInteger().with_variant(Integer, "sqlite")` 兼容两边，**原 MySQL 配置一行没改**。

**Q3：界面显示「模型不可用，已自动回退关键词规则」？**

1. **模型没准备好**（最常见）→ 跑 `cd ml && python setup_model.py`，  
   或先 `python setup_model.py --check` 看缺什么（报错会直接写清缺哪一项）；
2. `torch` / `transformers` 没装 → 见 [二、步骤 2](#步骤-2--建-python-环境)；
3. **底座路径不对**：adapter 必须挂在 `ml/models/dapt_bert` 上。若你只有  
   `bert-base-chinese`，请改用 HF 下载的合并模型（`--from-hf`）；
4. 启动用的是系统 Python 而非 `ml\.venv` → 启动脚本会打印实际使用的解释器，看一眼日志。

**Q4：模型在 CPU 上很慢？**

BERT-base 短文本在 CPU 上约 20~60ms/条。批量预测（`/api/sentiment_predict` 支持一次传多条）比逐条调用快得多。

**Q5：为什么仓库里没有模型文件？**

两个权重文件 390MB，超出 GitHub 单文件 100MB 上限，推送会被直接拒绝。

它们在 `.gitignore` 中并附了恢复方式注释；`ml\setup_model.py` 一条命令即可补齐  
（从 HF 下载，或用仓库自带 adapter 本地合并）。

**3.8MB 的 LoRA adapter 是真正的训练产物，已提交**（含 `extra_state.safetensors`，  
见 [五、情感引擎说明](#一个容易踩的坑adapter-必须带-extra_statesafetensors)）。

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

| 用途            | 数据集                                                                                                                          | 说明                                                            |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| **LoRA 监督训练** | [HikasaHana/eastmoney\_guba\_title](https://huggingface.co/datasets/HikasaHana/eastmoney_guba_title)                         | 东方财富上证50ETF股吧标题，**人工标注** 三分类，约 1.07 万条                        |
| **跨域评测（新闻域）** | [Kenpache/multilingual-financial-sentiment-v2](https://huggingface.co/datasets/Kenpache/multilingual-financial-sentiment-v2) | FLAME 中文子集，中文财经新闻标题 **人工标注** 三分类，约 1.48 万条，CC BY-NC 4.0（仅非商用） |
| DAPT 语料（可选）   | [mulfsa/CN-Market-Corpus](https://huggingface.co/datasets/mulfsa/CN-Market-Corpus)                                           | 2013–2023 中文金融市场文本，**无标签**，约 135 万条，Apache-2.0                |
| 底座模型          | [bert-base-chinese](https://huggingface.co/bert-base-chinese)                                                                | 中文 BERT-base，约 1.02 亿参数                                       |

**已知局限（会在论文/实训报告中说明）**：训练数据只有**上证50ETF 单一标的**，

因此模型对**个股**新闻（尤其是业绩公告类）泛化会打折。

这正是本项目保留 DAPT 环节、并做「关键词 vs 模型」对照实验的原因。

**⚠️ 这条局限已用实测数据验证**（见 [10.2 跨域泛化实测](#102-跨域泛化实测换了第二个域的数据集)）：  
股吧模型在财经新闻上从 **0.7460 掉到 0.5562（−19.0 pt）**，跨域后接近"全猜多数类"水准。

---

## 十、实测结果

### 10.0 线上模型（`lora_dapt_flame`，新闻测试集 2,936 条）

| 方法                      | 底座                  | 数据域  | Accuracy   | Macro-F1   |
| ----------------------- | ------------------- | ---- | ---------- | ---------- |
| 关键词规则基线                 | —                   | —    | 0.4545     | 0.3847     |
| BERT + LoRA             | `bert-base-chinese` | 股吧   | 0.7460     | 0.7417     |
| BERT + LoRA             | `bert-base-chinese` | 财经新闻 | 0.8273     | 0.8074     |
| **DAPT + LoRA（★ 线上模型）** | `dapt_bert`         | 财经新闻 | **0.8290** | **0.8075** |

- 可训练参数 **297,219 / 102,567,174 = 0.29%**（LoRA 的意义）
- 训练环境：RTX 3060 Laptop 6GB，bs=32 / seq=64 / 5 epoch 约 19 分钟
- 相对关键词基线：新闻域 **+37.5 pt**（Acc）/ **+42.3 pt**（Macro-F1）

### 10.1 股吧域对照（测试集 2,130 条，3 分类）

| 方法                  | 底座                  | Accuracy     | Macro-F1     |
| ------------------- | ------------------- | ------------ | ------------ |
| 关键词规则基线             | —                   | 0.4545       | 0.3847       |
| **BERT + LoRA**     | `bert-base-chinese` | **0.7460**   | **0.7417**   |
| DAPT + LoRA（超参完全一致） | `dapt_bert`         | 0.7446       | 0.7408       |
| 提升（基线 → LoRA）       | —                   | **+29.2 pt** | **+35.7 pt** |

> **DAPT 的结论是「无增益」**：换用 DAPT 底座后 Accuracy **−0.14 pt**、Macro-F1 **−0.09 pt**，  
> 三类 F1 差异都在 0.001 以内，属随机波动量级。原因：DAPT 语料是新闻/公告体（书面长句），  
> 下游是股吧标题（口语短句），**域不匹配**；且 1 epoch / 2,560 万 token 属轻量适应，  
> 不足以改变下游表征。这是一个有价值的负结果 —— 详见 `ml/out/baseline_vs_lora.md`。
>
> 该结论在**新闻域支线上做了对称验证**（见 [10.2](#102-跨域泛化实测换了第二个域的数据集)）：  
> 新闻支线 DAPT 版本 +0.17 pt，与股吧支线的 −0.14 pt **符号相反**，同样落在噪声内。  
> 两条独立支线都指向"无效"，比单支线更有说服力。
>
> ⚠️ **注意**：股吧域指标看起来比线上模型的新闻域指标低，是因为**两者测的是不同测试集**，  
> 不能直接比。同一测试集上的对比见 10.2 的跨域矩阵。

**DAPT 域自适应预训练（6,250 优化步，1 epoch）**

| 指标                      | 值                                               |
| ----------------------- | ----------------------------------------------- |
| 语料                      | CN-Market-Corpus 头部片段 **20 万条**（标题 + 正文前 150 字） |
| seq_len / batch / accum | 128 / 16 / 2                                    |
| 精度                      | bf16（fp16 在 MLM 上会出现 `grad_norm=inf`）           |
| 耗时                      | **3,560 s（约 59 分钟）**                            |
| MLM loss                | **2.23 → 1.59**（末 10 点均值 1.615）                 |
| 显存峰值                    | 4.34 GB / 6 GB                                  |

> ⚠️ 早期曾用股吧语料（LoRA 的训练+测试全集）跑 DAPT，那会造成测试集泄漏，
>
> 已废弃。现使用与监督数据**不同源**的 CN-Market-Corpus。

完整对照报告见 `ml/out/baseline_vs_lora.md`，DAPT 训练记录见 `ml/out/dapt_report.md`，

训练 curve 见 `ml/out/loss_curves/`。

**为什么基线会输得这么惨？** 关键词引擎把 2,130 条里的 1,603 条（75.3%）判成中性

（真实中性只占 40.9%），真实「看涨」的召回只有 0.16——股吧口语（「真牛」「绿盘」「怎么玩」）

根本不在词表里。这正是把规则引擎替换为模型的直接动机。

---

### 10.2 跨域泛化实测（换了第二个域的数据集）

前面的「已知局限」只是**推测**说"泛化到个股新闻会打折"。为了拿到实证，  
引入一份**完全独立**的中文财经新闻标注集做跨域评测。

| 数据集                  | 来源                                             | 条数     | 文本形态           |
| -------------------- | ---------------------------------------------- | ------ | -------------- |
| 股吧标题                 | `HikasaHana/eastmoney_guba_title`              | 10,646 | 散户口语，均值 20.6 字 |
| **财经新闻**（FLAME 中文子集） | `Kenpache/multilingual-financial-sentiment-v2` | 14,680 | 媒体书面，均值 29.1 字 |

> FLAME 的特点是**按本国投资者视角标注**（中文子集即"中国投资者怎么看"），与股吧场景口径一致；  
> 与股吧语料零重叠，与 DAPT 语料不同源。其中 38.7% 为繁体，已用 `zhconv` 统一转简体。

**跨域对照矩阵（Accuracy）—— 2×2：底座 × 数据域**

| 底座 \ 数据 → 模型                          | 股吧测试集 (2,130) | 新闻测试集 (2,936) |
| ------------------------------------- | ------------- | ------------- |
| `bert-base-chinese` + 股吧 `lora_guba`  | **0.7460**    | 0.5562        |
| `dapt_bert` + 股吧 `lora_dapt`          | 0.7446        | 0.5725        |
| `bert-base-chinese` + 新闻 `lora_flame` | 0.5549        | 0.8273        |
| `dapt_bert` + 新闻 `lora_dapt_flame`    | 0.5592        | **0.8290**    |

- **股吧 → 新闻：掉 19.0 pt**（0.7460 → 0.5562）；**新闻 → 股吧：掉 27.2 pt**（0.8273 → 0.5549）
- 跨域分数 0.55～0.57，**仅略高于"全猜多数类"**（股吧多数类 40.9%、新闻多数类 49.3%）

**三个稳定发现：**

1. **跨域时两个模型都塌向「中性」** —— 和关键词基线的失败模式一模一样。  
   股吧模型看新闻：中性 precision 仅 0.387、recall 却 0.761（真实"看涨"只召回 46.9%）；  
   新闻模型看股吧：中性 recall 0.848，而"看跌"召回崩到 **0.197**。  
   → **「域不匹配 ⇒ 退化成中性引擎」是这个任务的稳定失效模式。**
2. **迁移不对称且换模型救不了跨域** —— 股吧→新闻 比 新闻→股吧 容易（−19 vs −27 pt）；  
   新闻模型在本域（0.8273）反超股吧模型在本域（0.7460），  
   但它在股吧域上并不比股吧模型好（0.5549 vs 0.5562）。  
   → **没有单一模型能同时覆盖两个域**，应按业务场景选模型。
3. **DAPT 在两条支线上都无增益** —— 若 DAPT 真注入了金融域知识，应**同时**抬升两条支线；  
   实测却是股吧 −0.14 pt、新闻 +0.17 pt，**符号相反、幅度均在噪声内**。  
   → 这条 2×2 对照是对「DAPT 无用」的**对称验证**，比单支线更有说服力。

完整报告见 `ml/out/cross_domain_report.md`，原始指标见 `ml/out/cross_domain_matrix.json`  
与 `ml/out/cross_domain_matrix_dapt_flame.json`，  
构建与评测脚本为 `ml/prepare_flame_dataset.py`、`ml/eval_cross_domain.py`。

---

---

## 十一、复现训练链路

适合自己重训模型，或做实训报告里的实验复现。**先完成 [二、步骤 2](#步骤-2--建-python-环境) 的环境搭建**。

```bash
cd ml
.venv\Scripts\activate
```

```bash
# ① 股吧标注数据集（东方财富股吧标题，人工标注）——用于股吧域对照实验
python check_guba_dataset.py
#    → ml/data/guba_title_all.csv（10,646 条）
#    → ml/out/dataset_report.json / ml/out/sample_review.csv

# ② 财经新闻数据集（FLAME 中文子集）——线上模型的监督数据
python prepare_flame_dataset.py
#    → ml/data/flame_zh_all.csv（14,680 条，繁转简 + 8:2 分层切分）
```

```bash
# ③ DAPT 域自适应预训练（约 59 分钟，loss 2.23 → 1.59）
python prepare_dapt_corpus.py --max_records 200000
#    → ml/data/cn_market_dapt.txt（20 万条，约 91MB）

python train_dapt.py --model models/bert-base-chinese ^
                     --corpus data/cn_market_dapt.txt --max_samples 200000 ^
                     --seq_len 128 --batch_size 16 --grad_accum 2 ^
                     --lr 5e-5 --epochs 1 --out models/dapt_bert
#    → ml/models/dapt_bert/（约 409MB，已 gitignore）

#    ⚠️ 为什么不用 `--corpus hf:mulfsa/CN-Market-Corpus` 流式读？
#       该仓库是 d1.csv(13MB) + d2.json(13.8GB 单个 JSON 数组)，**不是 parquet**，
#       datasets 的 streaming 没有分片可切会退化成整文件下载而卡住（已实测）。
#       所以改用 HTTP Range 只取文件头部（镜像支持 206）。
```


```bash
# ④ LoRA 微调（2 底座 × 2 数据域 = 4 个变体，超参完全相同）
#    线上模型 = 变体 D
python train_lora.py --base models/bert-base-chinese --data data/guba_title_all.csv ^
                     --out models/lora_guba       --epochs 5 --merge     # A 股吧域
python train_lora.py --base models/dapt_bert       --data data/guba_title_all.csv ^
                     --out models/lora_dapt       --epochs 5 --merge     # B 对比 DAPT
python train_lora.py --base models/bert-base-chinese --data data/flame_zh_all.csv ^
                     --out models/lora_flame      --epochs 5 --merge     # C 新闻域
python train_lora.py --base models/dapt_bert       --data data/flame_zh_all.csv ^
                     --out models/lora_dapt_flame --epochs 5 --merge     # D ★ 线上模型
```

```bash
# ⑤ 对照评测
python eval_keyword_baseline.py            # 关键词基线
python eval_cross_domain.py ^
  --model "股吧模型=models/lora_guba/merged" ^
  --model "DAPT股吧模型=models/lora_dapt/merged" ^
  --model "新闻模型=models/lora_flame/merged" ^
  --model "DAPT新闻模型=models/lora_dapt_flame/merged" ^
  --data  "股吧测试集=data/guba_title_all.csv" ^
  --data  "新闻测试集=data/flame_zh_all.csv" ^
  --split test --out out/cross_domain_matrix.json
```

全部脚本都支持 `--smoke N`（只跑 N 步）用于快速验证链路是否通。

---

## 十二、发布模型到 Hugging Face

> ⚠️ **必须在能访问 `huggingface.co` 的机器上执行。**  
> `hf-mirror.com` 只支持下载，**不支持上传**；国内部分网络 `huggingface.co` 会被 DNS 污染而不可达。

```bash
cd ml
.venv\Scripts\activate

# 1. 在 https://huggingface.co/settings/tokens 建一个 write 权限的 token
set HF_TOKEN=hf_xxxxxxxx          # Windows
# export HF_TOKEN=hf_xxxxxxxx     # Linux / macOS

# 2. 先干跑，确认要传哪些文件
python upload_to_hf.py --repo Roeyyyyyy/quantmind-sentiment-dapt-flame --check

# 3. 正式上传（自动生成 Model Card，含真实指标）
python upload_to_hf.py --repo Roeyyyyyy/quantmind-sentiment-dapt-flame

#    连 DAPT 底座一起传（+390MB，便于他人完整复现）
python upload_to_hf.py --repo Roeyyyyyy/quantmind-sentiment-dapt-flame --include-dapt-base
```

> `--gh-owner` 决定 Model Card 里的 GitHub 链接指向谁，不指定时用项目默认
> `tt888777`（脚本内的 `DEFAULT_GH_OWNER`）。
> ⚠️ 本项目的 **HF 账号（`Roeyyyyyy`）与 GitHub 账号不同名**，所以这个默认值不能取 HF 用户名，
> 否则模型卡会指向一个不存在的仓库。要指向别的账号就显式写 `--gh-owner <用户名>`。
> 二次上传是**幂等**的：内容未变的文件（如 390MB 的 `model.safetensors`）会被跳过，
> 只重传 Model Card 只需几秒。

上传后仓库结构：

```
<repo>/
├── README.md            Model Card（自动生成：指标 / 数据 / 配置 / 复现命令）
├── config.json          ┐
├── model.safetensors    │ 合并后的完整模型，可直接 from_pretrained
├── vocab.txt / tokenizer* │
├── adapter/             LoRA adapter（可继续微调）
└── dapt_base/           DAPT 底座（仅 --include-dapt-base 时）
```

发布完成后，其他人在全新电脑上这样用（即 [二、步骤 4](#步骤-4--准备-ai-模型) 的方案 ①）：

```bash
cd ml && python setup_model.py --from-hf Roeyyyyyy/quantmind-sentiment-dapt-flame
```

### 已经发布的版本

| 项目    | 值                                                                          |
| ----- | -------------------------------------------------------------------------- |
| HF 仓库 | <https://huggingface.co/Roeyyyyyy/quantmind-sentiment-dapt-flame>          |
| 下载用法  | `python setup_model.py --from-hf Roeyyyyyy/quantmind-sentiment-dapt-flame` |
| 内容    | 合并模型（根目录）+ `adapter/` + `dapt_base/`                                       |
| 许可    | 训练数据含 CC BY-NC 4.0 → **仅供非商业/研究/教学使用**                                     |

国内网络直接用 `huggingface.co` 通常不通，`setup_model.py` 已默认走镜像下载，无需额外配置：

```bash
# 等价于设置 HF_ENDPOINT=https://hf-mirror.com
python setup_model.py --from-hf Roeyyyyyy/quantmind-sentiment-dapt-flame
```

**当作普通 transformers 模型用（脱离本项目也能跑）：**

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 国内；能直连 huggingface.co 则删掉
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo = "Roeyyyyyy/quantmind-sentiment-dapt-flame"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()

texts = ["公司发布业绩预告，净利润同比增长80%", "大股东拟减持不超过3%股份，股价承压"]
enc = tok(texts, truncation=True, max_length=64, padding=True, return_tensors="pt")
with torch.no_grad():
    probs = torch.softmax(model(**enc).logits, dim=-1)
for t, p in zip(texts, probs):
    i = int(p.argmax())
    print(model.config.id2label[i], round(float(p[i]), 4), t)
# 看跌 0.9978 公司发布业绩预告，净利润同比增长80%
```

> 标签顺序固定为 `0=看跌 / 1=看涨 / 2=中性`。若某条结果类别对不上，  
> 优先检查推理长度是否也是 64、以及输入是不是**财经新闻体**（该模型在股吧口语上会退化成中性）。

## 许可

本项目为学习/实训用途。引用第三方数据集与预训练模型时，请遵循其各自的许可协议。
