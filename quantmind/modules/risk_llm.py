"""风险提示生成 —— 远程大模型（OpenAI 兼容接口）。

只保留**远程**方案：本地 3B 级小模型在这个任务上输出偏套路，收益不明显。
支持 DeepSeek / 通义千问 / Kimi / 智谱 等一切 OpenAI 兼容的 ``/chat/completions``。

设计铁律
--------
1. **永不阻断报告**：本模块任何函数都不抛异常，失败一律返回 ``None``，
   由 ``report._risks()`` 降级到现有模板。报告永远能生成。
2. **防数字幻觉**：提示词禁止模型输出裸数字，只允许 ``{字段名}`` 占位符；
   后端用真实值替换，替换后再做一次"数字核实"——出现数据里不存在的数字，
   整条丢弃。
3. **零新增依赖**：用项目已有的 ``requests``，不装 openai SDK。
4. **默认关闭**：没有配置 API Key 时不发任何请求，行为与今天完全一致。
"""

from __future__ import annotations

import json
import os
import re
import threading
import time

import requests

# ---------------------------------------------------------------- 配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # quantmind/
CONFIG_FILE = os.path.join(BASE_DIR, ".llm_config.json")

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"

# 常见服务商的 base_url，方便用户只填服务商名
PROVIDERS = {
    "deepseek": ("https://api.deepseek.com/v1", "deepseek-chat"),
    "moonshot": ("https://api.moonshot.cn/v1", "moonshot-v1-8k"),
    "zhipu": ("https://open.bigmodel.cn/api/paas/v4", "glm-4-flash"),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
}

# 出现这些词的条目直接丢弃（荐股 / 收益承诺 / 绝对化表述）
BANNED_WORDS = [
    "稳赚", "必涨", "保证收益", "一定上涨", "必然上涨", "推荐买入",
    "建议买入", "满仓", "包赚", "无风险套利",
]

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _env(name, default=""):
    return os.environ.get(name, default).strip()


# ---------------------------------------------------------------- 配置文件
# 前端「配置」弹窗保存的 Key 落在 quantmind/.llm_config.json（已 gitignore，权限 0600）。
# 优先级：配置文件 > 环境变量 > 内置默认。
# 之所以让文件优先，是因为"用户在页面上显式填了"就应该覆盖掉启动脚本里的旧值，
# 否则改完不生效会让人以为功能坏了。
_LOCK = threading.RLock()
_FILE_CACHE: dict | None = None
_FILE_MTIME: float = 0.0

# 允许写进文件的字段（api_key 单独处理，避免被 status() 带出去）
_WRITABLE = ("provider", "base_url", "model", "timeout", "max_tokens", "temperature", "engine")


def _file_data() -> dict:
    """读取并缓存配置文件。文件被外部改动（mtime 变化）时自动重读。"""
    global _FILE_CACHE, _FILE_MTIME
    with _LOCK:
        try:
            mtime = os.path.getmtime(CONFIG_FILE)
        except OSError:
            _FILE_CACHE, _FILE_MTIME = {}, 0.0
            return {}
        if _FILE_CACHE is not None and mtime == _FILE_MTIME:
            return _FILE_CACHE
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            data = raw if isinstance(raw, dict) else {}
        except (OSError, ValueError):
            data = {}
        _FILE_CACHE, _FILE_MTIME = data, mtime
        return data


def _write_file(data: dict) -> None:
    """原子写入（先写临时文件再 replace），并尽量把权限收紧到 0600。"""
    global _FILE_CACHE, _FILE_MTIME
    with _LOCK:
        tmp = CONFIG_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, CONFIG_FILE)
        try:
            _FILE_MTIME = os.path.getmtime(CONFIG_FILE)
        except OSError:
            _FILE_MTIME = 0.0
        _FILE_CACHE = data


def _pick(key: str, file_data: dict, env_name: str, default: str = "") -> str:
    v = file_data.get(key)
    if v not in (None, ""):
        return str(v).strip()
    return _env(env_name, default)


def _pick_num(key: str, file_data: dict, env_name: str, default: float, cast=float):
    raw = _pick(key, file_data, env_name, str(default))
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return cast(default)


def config() -> dict:
    """读取配置。provider 只要填服务商名就能带出 base_url / model 默认值。"""
    fd = _file_data()
    provider = _pick("provider", fd, "QUANTMIND_LLM_PROVIDER", "deepseek").lower()
    p_base, p_model = PROVIDERS.get(provider, (DEFAULT_BASE_URL, DEFAULT_MODEL))
    return {
        "provider": provider,
        "base_url": _pick("base_url", fd, "QUANTMIND_LLM_BASE_URL", p_base).rstrip("/"),
        "model": _pick("model", fd, "QUANTMIND_LLM_MODEL", p_model),
        "api_key": _pick("api_key", fd, "QUANTMIND_LLM_API_KEY"),
        "timeout": _pick_num("timeout", fd, "QUANTMIND_LLM_TIMEOUT", 60.0),
        # 默认 1500：中文 JSON + 占位符大约 800~1200 token，700 会在数组中间被截断
        "max_tokens": _pick_num("max_tokens", fd, "QUANTMIND_LLM_MAX_TOKENS", 1500, int),
        "temperature": _pick_num("temperature", fd, "QUANTMIND_LLM_TEMPERATURE", 0.3),
    }


def engine_setting() -> str:
    """当前生效的引擎开关：配置文件 > 环境变量 > template。"""
    return _pick("engine", _file_data(), "QUANTMIND_LLM_ENGINE", "template").lower() or "template"


def enabled() -> bool:
    """只有显式开启且配了 Key 才启用。默认完全静默。"""
    if engine_setting() not in ("llm", "openai", "remote", "auto"):
        return False
    return bool(config()["api_key"])


# ---------------------------------------------------------------- 保存 / 清除
def masked_key(key: str) -> str:
    """回显用的脱敏形式：sk-abc…wxyz，绝不返回完整 Key。"""
    if not key:
        return ""
    if len(key) <= 10:
        return key[:2] + "*" * max(len(key) - 4, 1) + key[-2:]
    return f"{key[:6]}…{key[-4:]}"


def _coerce(patch: dict) -> dict:
    """校验并规整前端传来的字段，非法值直接报 ValueError（由接口转 400）。"""
    out = {}
    if "provider" in patch:
        p = str(patch.get("provider") or "").strip().lower()
        if p not in PROVIDERS and p != "custom":
            raise ValueError(f"不支持的服务商：{p}（可选 {', '.join(PROVIDERS)} / custom）")
        out["provider"] = p

    if "base_url" in patch:
        b = str(patch.get("base_url") or "").strip().rstrip("/")
        if b and not re.match(r"^https?://", b):
            raise ValueError("base_url 必须以 http:// 或 https:// 开头")
        out["base_url"] = b

    if "model" in patch:
        out["model"] = str(patch.get("model") or "").strip()

    if "timeout" in patch:
        try:
            t = float(patch.get("timeout"))
        except (TypeError, ValueError):
            raise ValueError("timeout 必须是数字")
        if not (5 <= t <= 300):
            raise ValueError("timeout 需在 5~300 秒之间")
        out["timeout"] = round(t, 1)

    if "max_tokens" in patch:
        try:
            mt = int(patch.get("max_tokens"))
        except (TypeError, ValueError):
            raise ValueError("max_tokens 必须是整数")
        if not (100 <= mt <= 4000):
            raise ValueError("max_tokens 需在 100~4000 之间")
        out["max_tokens"] = mt

    if "temperature" in patch:
        try:
            tp = float(patch.get("temperature"))
        except (TypeError, ValueError):
            raise ValueError("temperature 必须是数字")
        if not (0 <= tp <= 2):
            raise ValueError("temperature 需在 0~2 之间")
        out["temperature"] = round(tp, 2)

    if "enabled" in patch:
        out["engine"] = "llm" if patch.get("enabled") in (True, "1", 1, "true") else "template"
    if "engine" in patch:
        e = str(patch.get("engine") or "").strip().lower()
        if e not in ("llm", "template"):
            raise ValueError("engine 只能是 llm 或 template")
        out["engine"] = e

    return out


def save_config(patch: dict) -> dict:
    """局部保存。``api_key`` 为空串 / None 时**保留原 Key**（前端不回显 Key，
    用户只改模型名时不该把 Key 清掉）。要清除请用 ``reset_config()``。"""
    patch = dict(patch or {})
    fields = _coerce(patch)
    with _LOCK:
        data = dict(_file_data())

        if "api_key" in patch:
            key = str(patch.get("api_key") or "").strip()
            if key:
                if len(key) < 8:
                    raise ValueError("API Key 太短，请检查是否复制完整")
                data["api_key"] = key
        # 勾选启用但没有 Key、文件和环境变量都没有 → 直接报错，避免保存成"看起来能用"的状态
        if fields.get("engine") == "llm" or patch.get("enabled") in (True, "1", 1, "true"):
            if not data.get("api_key") and not _env("QUANTMIND_LLM_API_KEY"):
                raise ValueError("请先填写 API Key 再启用大模型引擎")

        for k in _WRITABLE:
            if k in fields:
                data[k] = fields[k]
        # 换服务商但没自定义 base_url/model → 跟随新服务商的默认值
        if "provider" in fields and fields["provider"] in PROVIDERS:
            if "base_url" not in fields:
                data["base_url"] = PROVIDERS[fields["provider"]][0]
            if "model" not in fields:
                data["model"] = PROVIDERS[fields["provider"]][1]

        _write_file(data)
    return status()


def reset_config() -> dict:
    """删除配置文件，回退到环境变量 / 内置默认。"""
    with _LOCK:
        try:
            if os.path.isfile(CONFIG_FILE):
                os.remove(CONFIG_FILE)
        except OSError as e:
            raise RuntimeError(f"配置文件删除失败：{e}")
        global _FILE_CACHE, _FILE_MTIME
        _FILE_CACHE, _FILE_MTIME = {}, 0.0
    return status()


def status() -> dict:
    """状态快照，**不发网络请求**、**不回显 api_key**。"""
    cfg = config()
    on = enabled()
    eng = engine_setting()
    return {
        "enabled": on,
        "engine": "llm" if on else "template",
        "engine_setting": eng,
        "provider": cfg["provider"],
        "model": cfg["model"],
        "base_url": cfg["base_url"],
        "has_key": bool(cfg["api_key"]),
        "key_hint": masked_key(cfg["api_key"]),
        "key_source": "file" if _file_data().get("api_key") else ("env" if cfg["api_key"] else None),
        "timeout": cfg["timeout"],
        "max_tokens": cfg["max_tokens"],
        "temperature": cfg["temperature"],
        "providers": {k: {"base_url": v[0], "model": v[1]} for k, v in PROVIDERS.items()},
        "has_config_file": os.path.isfile(CONFIG_FILE),
        "reason": None if on else (
            "未配置 API Key" if eng in ("llm", "openai", "remote", "auto")
            else "未启用（引擎开关为 template）"
        ),
    }



# ---------------------------------------------------------------- 提示词
_SYSTEM_PROMPT = """你是量化策略分析师。根据给定的 JSON 事实，写一段简短的中文策略分析 + 几条风险提示。

硬性要求：
1. **严禁编造任何数据中不存在的数字、日期、事件**。要引用数值时必须原样写
   {字段名} 形式的占位符（字段名取自 metrics），例如 {max_drawdown}。绝对不要自己写具体数字。
   **单位要跟在占位符后面写全**：百分比写 {max_drawdown}%，天数写 {avg_hold_days}天，
   笔数写 {trade_count}笔，不要写成"回撤 7.76"这种丢单位的表达。
2. `summary` 是**一段话**（80~140 字），整体评述这个策略：收益水平、风险暴露、
   收益来源的可靠性，以及最值得注意的一个问题。不要逐条罗列，要连贯成文。
3. `risks` 是 3~4 条风险提示，每条 1~2 句，结构为：哪个指标 → 反映什么问题 → 什么后果。
4. 优先指出指标之间的矛盾，例如：
   - 标的涨幅高但策略收益远低（踏空 / 择时失效）
   - 收益为正但夏普很低（风险调整后收益差）
   - 胜率低但盈亏比高（收益依赖少数大赚，稳定性差）
   - 持仓时间占比低却收益尚可（收益依赖少数交易，样本脆弱）
   - 舆情约束已启用但 sentiment_blocked 为 0（阈值形同虚设，从未拦住信号）——
     仅当 sentiment_dynamic 为"否"时适用；动态仓位下这是正常现象，不要套用
   - target_ratio 偏低导致收益被系统性压低（仓位设计所致，需与策略选股能力区分开）
   - 舆情事件识别覆盖率低 / "其他"类占比高（事件粒度粗，过滤的真实约束力被高估）
5. 不要写"投资有风险，入市需谨慎"这类空话套话。
6. 不得给出买卖建议、不得承诺收益、不得推荐个股。
7. 不要输出英文、true/false、None 之类的原始值；指标为“否”时用中文说明（如"未启用舆情过滤"）。
   若某个指标为 0 或无意义（如情绪样本为 0），不要硬凑一条，换别的指标。

输出格式：只输出一个 JSON 对象，形如
{"summary": "……", "risks": ["……", "……"]}
不要任何解释、不要 markdown 代码块。
"""

# 备用：只取条目数组（更省 token，模型若只给数组也能用）
_SYSTEM_PROMPT_ARRAY = """你是量化策略风险分析师。根据给定的 JSON 事实，写 3~4 条中文风险提示。

要求：
1. 严禁编造数据中不存在的数字。引用数值时只写 {字段名} 占位符，字段名取自 metrics。
2. 每条 1~2 句：哪个指标 → 反映什么问题 → 什么后果。
3. 优先指出指标之间的矛盾（如标的涨幅高但策略收益低、收益正但夏普低、胜率低但盈亏比高）。
4. 不写空话套话，不给买卖建议，不承诺收益。

只输出一个 JSON 数组，元素为字符串，不要解释、不要 markdown 代码块。
"""


# 字段含义说明 —— 必须给。否则模型看到 sentiment_active="是"、sentiment_blocked=0
# 这类孤立事实，会自己臆测出"舆情过滤未启用"这种与事实相反的结论（实测踩过）。
_FIELD_NOTES = """
字段含义（务必按此理解，不要臆测）：
- sentiment_active：回测是否真的把舆情纳入买卖决策。"是"=已启用，"否"=仅展示不参与。
- buy_signals：区间内出现的金叉买入信号总数（注意：这是信号数，不是成交笔数 trade_count）。
- sentiment_blocked：其中被舆情阈值拦下、因而没有买入的信号数。
- sentiment_active="是" 时：若 buy_signals>0 且 sentiment_blocked=0，说明**舆情已启用但一次都没拦住**，
  该阈值在本区间内没有实际过滤作用 —— 这是"阈值形同虚设"，**不是**"舆情未启用"。
  若 sentiment_blocked>0，说明舆情确实参与了决策并拦下过信号。
- coverage_pct：舆情事件被明确归类的比例，越低说明事件识别越粗。
- other_ratio：落入"其他"类的新闻占比，是 coverage_pct 的反面。
- event_kinds：识别出的具体事件类别数。
- rows_below_250：数据是否不足 250 交易日（"是"表示样本偏短）。
- sentiment_dynamic："是"表示采用**动态仓位**——金叉不再满仓，而是按情绪档位
  折算目标仓位（如情绪中性对应 40%）。此时 sentiment_blocked 通常为 0，
  因为舆情是通过"少买"而非"不买"起作用的，**不要**据此说舆情未启用或形同虚设。
- target_ratio：动态仓位下的目标仓位百分比。该值偏低会直接压低收益，
  这是设计使然，不是策略失效。
- avg_exposure_pct：日均实际仓位（含空仓期），通常明显低于 target_ratio。
"""


def _build_user_prompt(data: dict) -> str:
    metrics = dict(data.get("metrics") or {})
    # 只把非空字段给模型，减少噪声；布尔值转成"是/否"，避免模型把 true 当数字引用
    metrics = {
        k: ("是" if v else "否") if isinstance(v, bool) else v
        for k, v in metrics.items() if v is not None and v != []
    }
    payload = {
        "stock_code": data.get("stock_code"),
        "period": {"start": data.get("start"), "end": data.get("end")},
        "metrics": metrics,
    }
    return ("请基于以下事实写一段策略分析（summary）和 3~4 条风险提示（risks）。"
            "数字只能以 {字段名} 形式引用，字段名必须来自 metrics：\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + _FIELD_NOTES)


# ---------------------------------------------------------------- 调用
def _post_chat(messages: list[dict], cfg: dict) -> str:
    """调用 /chat/completions，返回文本内容。失败抛异常（由上层捕获）。"""
    resp = requests.post(
        f"{cfg['base_url']}/chat/completions",
        headers={
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        },
        json={
            "model": cfg["model"],
            "messages": messages,
            "temperature": cfg["temperature"],
            "max_tokens": cfg["max_tokens"],
            "stream": False,
        },
        timeout=cfg["timeout"],
    )
    if resp.status_code != 200:
        # 把服务商的错误信息带出来，便于判断是余额/Key/模型名问题
        detail = ""
        try:
            detail = resp.json().get("error", {}).get("message", "")
        except Exception:
            detail = resp.text[:200]
        raise RuntimeError(f"HTTP {resp.status_code}: {detail or '请求失败'}")
    body = resp.json()
    return body["choices"][0]["message"]["content"]


def _strip_fence(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _repair_truncated(text: str) -> list | None:
    """修复被 max_tokens 截断的 JSON 数组。

    截断通常发生在某个字符串中间，形如 ``["……完整项", "半截……``。
    做法：按顶层逗号切分，丢掉最后一个不完整的元素，再补齐闭合括号。
    """
    t = _strip_fence(text)
    start = t.find("[")
    if start == -1:
        return None
    body = t[start + 1:]
    # 逐步回退：每次去掉末尾残片，补上 ] 再试
    items, buf = [], ""
    in_str = esc = False
    for ch in body:
        if esc:
            buf += ch
            esc = False
            continue
        if ch == "\\":
            buf += ch
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            buf += ch
            continue
        if ch == "," and not in_str:
            items.append(buf)
            buf = ""
            continue
        if ch == "]" and not in_str:
            break
        buf += ch
    # buf 是否是一个"完整"的元素：不管怎样都尝试，失败就丢
    candidates = items + ([buf] if buf.strip() else [])
    out = []
    for raw in candidates:
        s = raw.strip().rstrip(",").strip()
        if not s:
            continue
        if not s.startswith('"'):
            continue
        if not s.endswith('"') or len(s) < 2:
            # 半截字符串：补引号试试，但只在后面没有更完整项时才有意义
            s = s + '"'
        try:
            v = json.loads(s)
        except Exception:
            continue
        if isinstance(v, str) and v.strip():
            out.append(v)
    return out or None


def _extract_result(text: str) -> tuple[str | None, list | None]:
    """从模型输出里提取 (summary, risks)。

    兼容四种形态：
      1. ``{"summary": "...", "risks": [...]}``
      2. ``["...", "..."]``
      3. 带 ```json 围栏的上面两种
      4. **被 max_tokens 截断**的上面两种（自动修复 / 至少保住已生成的部分）
    """
    t = _strip_fence(text)
    if not t:
        return None, None

    # ---- 1) 完整 JSON（对象或数组）
    sliced = t
    if not sliced.startswith("{") and not sliced.startswith("["):
        b, c = sliced.find("{"), sliced.find("[")
        pos = min([p for p in (b, c) if p != -1], default=-1)
        if pos != -1:
            sliced = sliced[pos:]
    try:
        v = json.loads(sliced)
        if isinstance(v, dict):
            return _dict_to_result(v)
        if isinstance(v, list):
            return None, [x for x in v if isinstance(x, str)] or None
    except Exception:
        pass

    # ---- 2) 截断修复
    if t.find("{") != -1 and (t.find("[") == -1 or t.find("{") < t.find("[")):
        obj = _repair_truncated_object(t)
        if obj:
            return _dict_to_result(obj)
    arr = _repair_truncated(t)
    if arr:
        return None, arr
    return None, None


def _dict_to_result(v: dict) -> tuple[str | None, list | None]:
    summary = v.get("summary")
    summary = summary.strip() if isinstance(summary, str) else None
    risks = v.get("risks")
    if risks is None:
        risks = v.get("risk")
    if isinstance(risks, str):
        risks = [risks]
    risks = [x for x in (risks or []) if isinstance(x, str) and x.strip()]
    return (summary or None), (risks or None)


def _repair_truncated_object(text: str) -> dict | None:
    """修复被截断的 JSON 对象：尽量至少保住 summary。"""
    t = _strip_fence(text)
    s_pos = t.find('"summary"')
    if s_pos == -1:
        s_pos = t.find('"summary')
    if s_pos != -1:
        colon = t.find(":", s_pos)
        if colon != -1:
            rest = t[colon + 1:].lstrip()
            if rest.startswith('"'):
                # 取出字符串内容（可能没闭合）
                i, out, esc = 1, [], False
                while i < len(rest):
                    ch = rest[i]
                    if esc:
                        out.append(ch)
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        break
                    else:
                        out.append(ch)
                    i += 1
                summary = "".join(out).strip()
                if len(summary) >= 20:
                    return {"summary": summary, "risks": []}
    return None


# ---------------------------------------------------------------- 校验
def _allowed_numbers(metrics: dict) -> set[str]:
    """数据中出现过的所有数字（含取整形式），用于核实模型有没有编数字。

    额外把**字段名里的数字**也放行：像 ``rows_below_250`` 这种布尔标记，
    模型写成"样本不足 250 行"是合理引用（250 来自字段名语义），不该判为编造。
    """
    nums = set()

    def add(v):
        try:
            f = float(v)
        except Exception:
            return
        nums.add(str(abs(f)))
        nums.add(f"{abs(f):.0f}")
        nums.add(f"{abs(f):.1f}")
        nums.add(f"{abs(f):.2f}")

    for k, v in metrics.items():
        for m in _NUM_RE.findall(str(k)):     # 字段名里的数字
            add(m)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            add(v)
        elif isinstance(v, str):
            for m in _NUM_RE.findall(v):
                add(m)
        elif isinstance(v, list):
            for it in v:
                if isinstance(it, dict):
                    for vv in it.values():
                        if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                            add(vv)
    return nums


def _fill_placeholders(text: str, metrics: dict) -> tuple[str, bool]:
    """把 {字段名} 替换成真实值。返回 (结果, 是否有未知占位符)。"""
    unknown = False

    def rep(m):
        nonlocal unknown
        key = m.group(1)
        if key in metrics and metrics[key] is not None:
            val = metrics[key]
            if isinstance(val, float):
                return f"{val:g}"
            return str(val)
        unknown = True
        return m.group(0)

    return _PLACEHOLDER_RE.sub(rep, text), unknown


def _validate(items: list, metrics: dict, max_len: int = 160) -> tuple[list[str], list[str]]:
    """过滤不合格条目。返回 (通过的条目, 丢弃原因列表)。

    ``max_len`` 是单条字数上限：风险条目传 160，整段分析传 320（否则段落会被腰斩）。
    """
    allowed = _allowed_numbers(metrics)
    passed, dropped = [], []

    for raw in items:
        if not isinstance(raw, str):
            dropped.append("非字符串")
            continue
        text = raw.strip().lstrip("-•·0123456789.、) ").strip()
        if not text:
            continue

        # 整段分析里可能出现正常的中英文混排数字（如"3 分类"），仍然要核实
        filled, unknown = _fill_placeholders(text, metrics)
        if unknown:
            dropped.append("引用了不存在的字段: " + text[:30])
            continue

        low = filled
        if any(w in low for w in BANNED_WORDS):
            dropped.append("含禁用词: " + text[:30])
            continue

        # 英文原始值（True/False/None/null）出现在中文报告里很难看，直接丢
        if re.search(r"\b(True|False|None|null|NaN|undefined)\b", filled):
            dropped.append("含原始值 True/False/None: " + text[:30])
            continue

        # 数字核实：替换后仍出现的裸数字，必须是数据里真实存在的
        suspicious = [n for n in _NUM_RE.findall(filled) if n.lstrip("-") not in allowed]
        if suspicious:
            dropped.append(f"疑似编造数字 {suspicious[:3]}: " + text[:30])
            continue

        if len(filled) > max_len:
            cut = filled[:max_len - 1]
            # 尽量在句号/分号处断，不要切在半句中间
            for sep in ("。", "；", "，", "、", " "):
                p = cut.rfind(sep)
                if p >= max_len * 0.6:
                    cut = cut[:p + (1 if sep in "。；" else 0)]
                    break
            filled = cut.rstrip("，、；") + "…"
        passed.append(filled)

    return passed, dropped


# ---------------------------------------------------------------- 主入口
def generate(data: dict, cfg: dict | None = None) -> tuple[list[str] | None, dict]:
    """生成风险提示。

    返回 ``(items, meta)``：
      * ``items``  成功时是已校验的字符串列表；失败是 ``None``
      * ``meta``   含 engine / model / elapsed_ms / error / dropped / summary / truncated

    **本函数不抛异常**（除参数类型错误），失败一律 ``(None, meta)``。
    """
    res, meta = generate_full(data, cfg)
    return (res.get("risks") if res else None), meta


def generate_full(data: dict, cfg: dict | None = None) -> tuple[dict | None, dict]:
    """生成「一段整体分析 + 风险提示」。

    返回 ``(result, meta)``，``result`` 形如
    ``{"summary": "……", "risks": ["……", "……"]}``；失败为 ``None``。
    meta 额外含 ``truncated``（是否被 max_tokens 截断）。
    """
    cfg = cfg or config()
    meta = {
        "engine": "llm" if cfg.get("api_key") else "template",
        "provider": cfg.get("provider"),
        "model": cfg.get("model"),
        "elapsed_ms": 0,
        "error": None,
        "dropped": [],
        "raw": None,
        "truncated": False,
    }

    if not enabled():
        meta["error"] = status()["reason"]
        return None, meta

    metrics = data.get("metrics") or {}
    if not metrics:
        meta["error"] = "缺少可用的指标数据"
        return None, meta

    t0 = time.time()
    try:
        content = _post_chat(
            [{"role": "system", "content": _SYSTEM_PROMPT},
             {"role": "user", "content": _build_user_prompt(data)}],
            cfg,
        )
        meta["raw"] = content
        summary, risks = _extract_result(content)

        # 首次没抽到 → 用「只要数组」的短提示词重试一次（省 token，更不容易截断）
        if not risks and not summary:
            meta["truncated"] = True
            content2 = _post_chat(
                [{"role": "system", "content": _SYSTEM_PROMPT_ARRAY},
                 {"role": "user", "content": _build_user_prompt(data)}],
                {**cfg, "max_tokens": max(int(cfg.get("max_tokens") or 700), 1200)},
            )
            meta["raw"] = content2
            s2, r2 = _extract_result(content2)
            summary, risks = summary or s2, risks or r2

        if not risks and not summary:
            meta["error"] = ("大模型输出被截断或格式不对（已尝试修复与重试），"
                             "可在「配置」里把最大输出 tokens 调到 1200 以上")
            meta["elapsed_ms"] = int((time.time() - t0) * 1000)
            return None, meta

        # 二者都过一遍数值核实；摘要单独校验
        passed, dropped = _validate(risks or [], metrics)
        meta["dropped"] = dropped

        clean_summary = None
        if summary:
            sm, sm_drop = _validate([summary], metrics, max_len=320)
            if sm:
                clean_summary = sm[0]
            else:
                meta["dropped"] = dropped + ["summary:" + (sm_drop[0] if sm_drop else "不合格")]

        if not passed and not clean_summary:
            meta["error"] = f"内容未通过数值校验（丢弃 {len(meta['dropped'])} 处）"
            meta["elapsed_ms"] = int((time.time() - t0) * 1000)
            return None, meta

        # 模型只给了数组、没给 summary 时，用前两条风险提示拼一段总述，
        # 保证报告里那段"分析"永远存在（内容仍全部来自已校验条目）
        if not clean_summary and passed:
            clean_summary = passed[0]
            if len(passed) > 1 and len(clean_summary) < 320:
                clean_summary = (clean_summary.rstrip("。") + "；" + passed[1]).strip("；") + "。"

        meta["elapsed_ms"] = int((time.time() - t0) * 1000)
        return {"summary": clean_summary, "risks": passed[:4]}, meta

    except requests.exceptions.Timeout:
        meta["error"] = f"请求超时（{cfg['timeout']}s）"
    except requests.exceptions.ConnectionError as e:
        meta["error"] = f"网络连接失败：{e.__class__.__name__}"
    except Exception as e:
        meta["error"] = f"{type(e).__name__}: {e}"
    meta["elapsed_ms"] = int((time.time() - t0) * 1000)
    return None, meta


def probe(cfg: dict | None = None) -> dict:
    """做一次极短的连通性测试（1 条 token 级别），供前端"测试连接"按钮使用。

    ``cfg`` 可传入**尚未保存**的临时配置，让用户先测通再点保存。
    传了 ``cfg`` 时不再检查 ``enabled()``（引擎开关未开也应该能测 Key 是否有效），
    只要求有 api_key。
    """
    if cfg:
        cfg = {**config(), **{k: v for k, v in cfg.items() if v not in (None, "")}}
        if not cfg.get("api_key"):
            return {"ok": False, "error": "未填写 API Key",
                    "model": cfg.get("model"), "provider": cfg.get("provider"),
                    "base_url": cfg.get("base_url")}
        cfg["base_url"] = str(cfg["base_url"]).rstrip("/")
    else:
        cfg = config()
        # 传了 probe=1 的「已保存配置测试」：有 Key 就测，不看引擎开关
        if not cfg.get("api_key"):
            return {"ok": False, "error": status()["reason"],
                    **{k: cfg[k] for k in ("provider", "model", "base_url")}}
    t0 = time.time()
    try:
        _post_chat([{"role": "user", "content": "回复两个字：正常"}],
                   {**cfg, "max_tokens": 16, "timeout": 15})
        return {"ok": True, "model": cfg["model"], "provider": cfg["provider"],
                "base_url": cfg["base_url"], "elapsed_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200], "model": cfg["model"],
                "provider": cfg["provider"], "base_url": cfg["base_url"]}
