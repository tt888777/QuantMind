# 风险提示 LLM 生成 · 接口设计方案

> 目标：给报告的风险提示（`report.py` 中的 `_risks()`）接入大模型，支持**本地 Ollama / 远程 API** 双模式，
> 任何一环失败都**自动降级**到现有模板，保证报告永远能生成。
>
> 状态：**已实施（2026-10-18）——按用户要求移除本地 Ollama 模式，仅保留远程 API**　｜　模块：`quantmind/modules/risk_llm.py`

---

## 1. 设计原则

| 原则 | 说明 |
|---|---|
| **永不阻断** | LLM 失败绝不能让报告导出失败。最坏情况退回现有 `if/else` 模板，行为和今天完全一致。 |
| **零新增依赖** | 只用标准库 `urllib` 或项目**已有的 `requests`**，不装 `openai` SDK。Ollama 与 OpenAI 都是 HTTP + JSON。 |
| **双模式同构** | 本地 Ollama 与远程 API 都用 `/chat/completions` 协议，**同一份代码**，靠 `base_url` 切换。 |
| **防数字幻觉** | 提示词严禁模型输出数字；正文里的数值一律用 `{占位符}`，由后端用真实值替换。 |
| **可追溯** | 每次生成记录引擎、模型名、耗时、原始输出，写入日志；报告页标注「由大模型生成」。 |
| **默认关闭** | 环境变量不配置时**完全不启用**，不产生任何外部请求。 |

---

## 2. 架构与调用链

```
build_and_save()                    ← 唯一入口，不变
  └─ render_html()
       └─ _risks(data, tpl)
            ├─ engine = risk_llm.resolve_engine()      # template | llm
            ├─ if engine == "llm":
            │     risk_llm.generate(data)  ──► 成功：返回 list[str]（含归因与建议）
            │                              └─► 失败：返回 None 并记录原因
            └─ fallback: 现有 if/else 模板（永不失败）
```

`_risks()` 的签名与返回类型**保持不变**（`list[str]`），所以 `render_html` 里
`items = "".join(f"<li>{r}</li>" for r in _risks(data, tpl))` 这行**不用改**。

---

## 3. 引擎决策链

按优先级依次尝试，第一个可用的胜出：

| 顺位 | 引擎 | 触发条件 | 说明 |
|---|---|---|---|
| 1 | `ollama` | `QUANTMIND_LLM_ENGINE=ollama` 或 `auto` 且本地 11434 端口存活 | 本地离线，隐私最好 |
| 2 | `openai` | `QUANTMIND_LLM_ENGINE=openai` 或 `auto` 且配了 `api_key` | 远程，质量最高 |
| 3 | `template` | 以上都不可用 | 现有模板，零依赖 |

`auto` 模式下的探测顺序：先探 Ollama（本地，1 秒超时），不通再看有没有 API Key。

---

## 4. 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `QUANTMIND_LLM_ENGINE` | `template` | `template` / `ollama` / `openai` / `auto` |
| `QUANTMIND_LLM_BASE_URL` | 见下 | Ollama 默认 `http://127.0.0.1:11434/v1`；OpenAI 兼容服务填其 `/v1` |
| `QUANTMIND_LLM_API_KEY` | 空 | `openai` 模式必填；Ollama 模式留空 |
| `QUANTMIND_LLM_MODEL` | `qwen2.5:3b` | 模型名 |
| `QUANTMIND_LLM_TIMEOUT` | `60` | 单次请求超时（秒） |
| `QUANTMIND_LLM_MAX_TOKENS` | `600` | 输出上限，控制耗时 |
| `QUANTMIND_LLM_KEEP_ALIVE` | `30s` | **Ollama 专用**：用完尽快释放显存，否则 BERT 会 OOM |
| `QUANTMIND_LLM_TEMPERATURE` | `0.3` | 偏低，减少发挥与幻觉 |

> `keep_alive` 在 6GB 卡上是**必踩的坑**：默认 5 分钟常驻，跑完报告后模型仍占着显存，
> 紧接着做 BERT 批量推理就会 OOM。设为 `30s` 让它用完即走。

---

## 5. 输入：喂给模型的数据

现状 `_risks()` 只用了 5 个指标（`max_drawdown` / `trade_count` / `sharpe` / `sentiment.index` / `rows`），
`build_report_data()` 虽然返回 12 个字段，但大多是原样透传的字典，**缺少可供推理的派生指标**。
为了让 LLM 有东西可分析，需把 `build_report_data()` 扩展为约 **18 个字段**（这部分**不依赖 LLM，独立可做、纯增益**）：

```jsonc
{
  "stock_code": "600519",
  "period":   { "start": "2025-01-02", "end": "2025-10-17", "rows": 178,
                "span_days": 288 },
  "price":    { "close_first": 30.4, "close_last": 50.7, "change_pct": 66.92 },
  "backtest": {
    "total_return": 2.77, "annual_return": 3.5,
    "max_drawdown": 12.5, "win_rate": 42.1, "sharpe": 0.33,
    "trade_count": 8, "final_equity": 102770,
    "profit_loss_ratio": 1.8,        // 平均盈利 / 平均亏损
    "max_consec_loss": 3,            // 最大连续亏损次数
    "avg_hold_days": 11.4,           // 平均持仓天数
    "exposure_pct": 31.0             // 持仓时间占比
  },
  "sentiment": {
    "index": 52, "pos_pct": 34, "neg_pct": 22, "neu_pct": 44,
    "sample_size": 50,
    "top_events": [ { "name": "业绩预增", "type": "利好", "count": 3 } ],
    "other_ratio": 88.0              // "其他"类占比 → 事件识别粒度指标
  },
  "data_quality": { "rows_below_250": true, "has_gaps": false }
}
```

**关键设计**：`exposure_pct`、`other_ratio`、`profit_loss_ratio` 这几个字段是
「**模板写不出来、但 LLM 能据此推理**」的关键——例如
"持仓时间仅 31% 却取得 2.77% 收益" 能推出 "收益依赖少数交易"。

---

## 6. 提示词结构

三段式，**system 约束格式，user 给数据**：

**system**
```
你是量化策略风险分析师。基于给定 JSON 数据，写 4~6 条中文风险提示。
硬性要求：
1. 严禁编造任何数据中不存在的数字、日期、事件。需要引用数值时，
   必须原样使用 {字段名} 形式的占位符，例如 {max_drawdown}，不要自己写数字。
2. 每条 1~2 句，必须指明「哪个指标」+「说明什么问题」+「对应后果」。
3. 优先指出指标之间的矛盾（如高收益但低夏普、情绪偏空但仓位很高）。
4. 不要写"投资有风险"这类空话。只输出 JSON 数组，不要任何解释。
```

**user**：直接贴第 5 节的 JSON。

**输出约定**：
```json
["策略最大回撤达 {max_drawdown}%，而夏普仅 {sharpe}，说明…"]
```

后端拿到后按 `data` 做占位符替换 → 得到最终文案。**模型即使"想编数字"也没机会**，
因为它在数组里写死数字会被后处理规则拦截（见 §8）。

---

## 7. 对外接口

### 7.1 内部函数（`risk_llm.py`）

```python
class RiskLLMUnavailable(RuntimeError): ...

def resolve_engine() -> str:
    """返回 'ollama' | 'openai' | 'template'，纯本地判断，不发请求。"""

def probe(engine: str, timeout: float = 1.5) -> dict:
    """探测可用性。返回 {'ok': bool, 'engine': str, 'model': str,
    'latency_ms': int, 'error': str|None}"""

def generate(data: dict, *, engine: str | None = None) -> tuple[list[str] | None, dict]:
    """生成风险提示。
    返回 (items, meta)。items 为 None 表示失败，meta 含 engine/model/elapsed_ms/error/raw。
    本函数**不抛异常**（除参数错误），失败一律 (None, meta)。
    """

def status() -> dict:
    """给前端的状态快照，不触发实际生成。"""
```

### 7.2 HTTP 接口（`app.py` 新增）

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/api/risk_engine_status` | 返回当前引擎、是否可用、模型名、配置摘要（**不回显 api_key**） |
| `POST` | `/api/risk_preview` | 用当前/指定引擎**试生成一次**风险提示，供报告中心对比模板版与 LLM 版 |

`/api/risk_preview` 请求体：
```json
{ "stock_code": "600519", "engine": "llm", "start_date": "", "end_date": "" }
```
响应：
```json
{ "ok": true, "engine": "ollama", "model": "qwen2.5:3b", "elapsed_ms": 9420,
  "items": ["...", "..."], "template_items": ["...", "..."],
  "fallback": false, "error": null }
```

---

## 8. 后处理与安全

| 检查 | 规则 | 命中后 |
|---|---|---|
| **JSON 解析** | 模型可能带 ```json 围栏或多余解释 | 正则提取第一个 `[...]`；仍失败则判定为失败并降级 |
| **数量** | 少于 3 条或多于 8 条 | 截断/补齐模板条目 |
| **数字核实** | 条目里出现 `\d+\.?\d*%` 等裸数字，且该数字不在 `data` 的取值集合中 | **整条丢弃**（防幻觉） |
| **长度** | 单条 > 120 字 | 截断 |
| **禁用词** | 含"必然""保证""稳赚""推荐买入"等 | 整条丢弃 |
| **敏感** | 含具体买卖点位/荐股 | 整条丢弃 |

若过滤后剩余 < 2 条，整体降级为模板，并在 meta 里记录 `degraded_reason="filtered"`。

---

## 9. 报告中的呈现

在 `_risks()` 返回结果后追加一条来源标注（仅当 engine=llm）：

```
本段风险分析由本地大模型 qwen2.5:3b 生成，已通过数值校验；内容仅供参考，不构成投资建议。
```

模板模式则保持原样，不添加任何标注——**让读者能一眼看出这段是谁写的**。

HTML 中给 LLM 条目加一条不同样式的左边框（淡色），与模板条目视觉区分（可选）。

---

## 10. 前端改动（`prototype.html`）

报告中心「编辑模板」旁增加一个引擎指示器 + 对比开关：

```
风险提示引擎：[ 模板 ] [ 大模型 ]     ● qwen2.5:3b 就绪 (GPU)
```

- 点击「大模型」→ 调 `/api/risk_preview`，**左右并排显示模板版与 LLM 版**
- LLM 不可用时按钮置灰并显示原因（如"Ollama 未启动"），不弹错误框
- 与现有情感引擎 `#sentEngineSwitch` 的交互模式保持一致（`data-busy` 防连点）

---

## 11. 失败矩阵（必须全部覆盖）

| 场景 | 表现 | 处理 |
|---|---|---|
| Ollama 未安装/未启动 | 连接拒绝 | 降级模板，`status` 显示"本地模型未启动" |
| 模型未 pull | 404 | 降级模板，提示 `ollama pull qwen2.5:3b` |
| 超时（30s+） | socket timeout | 降级模板，**不重试**（避免报告卡死） |
| 显存不足 | OOM 错误码 | 降级模板，日志记录建议减小模型 |
| API Key 无效 / 余额不足 | 401 / 402 | 降级模板，`status` 显示"远程 API 鉴权失败" |
| 返回非法 JSON | 解析失败 | 降级模板，保留 raw 便于排查 |
| 全部条目被过滤 | 过滤后 < 2 条 | 降级模板，`degraded_reason` 记录 |
| 断网（远程模式） | 连接失败 | 降级模板 |

**任何一行触发，报告照常生成。**

---

## 12. 实施拆分（建议顺序）

| 阶段 | 内容 | 依赖外部服务 | 风险 |
|---|---|---|---|
| **1** | 扩展 `build_report_data()` 至 18 字段（纯数据，不改输出） | 否 | 无 |
| **2** | 新建 `risk_llm.py`：引擎解析 + 探测 + 生成 + 后处理，**先只接 Ollama** | 需装 Ollama | 低 |
| **3** | `_risks()` 接入 + 降级逻辑 + 状态接口 | — | 低（失败即回退） |
| **4** | 前端引擎切换与并排对比 | — | 低 |
| **5** | 增加 OpenAI 兼容远程模式（复用同一 HTTP 层，约 30 行） | 需 API Key | 低 |

阶段 1~4 全程不需要联网、不需要花钱，**且每一步都可独立验收**。

---

## 13. 一句话总结

> **把 LLM 当成"会写分析的实习生"，把 BERT 和回测引擎当成"给他提供事实的数据源"。**
> 实习生的产出必须经过数值校验才能进报告，他不在的时候活儿照样有人干（模板）。
