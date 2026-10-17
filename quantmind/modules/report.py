"""报告生成：HTML 渲染 + 真 PDF 导出。

设计要点
--------
1. **单一数据源**：先算出 ``ReportData``（行情/绩效/舆情/风险），再由
   :func:`render_html` 渲染成 HTML。HTML 与 PDF 共用同一份 HTML，
   避免"导出 PDF 得到 HTML 内容"这类不一致。
2. **真 PDF**：本机装了 Chrome，用它的 headless ``--print-to-pdf`` 打印，
   中文零配置、排版与浏览器一致，**不新增任何 pip 依赖**。
   找不到 Chrome 时按 ``weasyprint`` → ``reportlab`` 依次降级；
   全部不可用则抛 :class:`PdfUnavailable`，由调用方明确告知用户，
   绝不把 HTML 内容写进 .pdf 文件。
3. **模板可编辑**：:func:`load_template` / :func:`save_template` /
   :func:`reset_template` 持久化到 ``exports/report_template.json``，
   支持标题、副标题、公司名、作者、主题色、章节开关、自定义免责声明。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent
EXPORT_DIR = ROOT_DIR / "exports"
TEMPLATE_FILE = EXPORT_DIR / "report_template.json"

EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# 最近一次风险提示的来源（单进程演示用；多进程部署时仅影响状态展示，不影响功能）
_last_risk_source: dict = {"engine": "template", "meta": {}}


class PdfUnavailable(RuntimeError):
    """没有任何可用的 PDF 渲染引擎。"""


# ---------------------------------------------------------------- 模板
DEFAULT_TEMPLATE = {
    "title": "{stock} 综合量化分析报告",
    "subtitle": "QuantMind 智能量化分析平台 · 自动生成",
    "company": "QuantMind",
    "author": "",
    "theme": "indigo",           # indigo / violet / emerald / amber
    "accent": "#6366F1",
    "sections": {
        "overview": True,
        "strategy": True,
        "performance": True,
        "sentiment": True,
        "risk": True,
    },
    "risk_items": [],            # 用户自定义风险条目，追加在内置条目之后
    "disclaimer": "本报告由 QuantMind 自动生成，仅供研究与教学演示使用，"
                  "不构成任何投资建议。据此操作，风险自负。",
    "footer": "{company} · {date} · 数据区间 {start} ~ {end}",
}

THEMES = {
    "indigo": "#6366F1",
    "violet": "#A855F7",
    "emerald": "#10B981",
    "amber": "#F59E0B",
    "rose": "#F43F5E",
    "slate": "#64748B",
}


def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_template() -> dict:
    """读取模板配置；文件缺失或损坏时回退默认值（绝不抛异常）。"""
    tpl = dict(DEFAULT_TEMPLATE)
    tpl["sections"] = dict(DEFAULT_TEMPLATE["sections"])
    tpl["risk_items"] = list(DEFAULT_TEMPLATE["risk_items"])
    try:
        if TEMPLATE_FILE.exists():
            raw = json.loads(TEMPLATE_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                tpl = _deep_merge(tpl, raw)
    except Exception:
        pass                                    # 损坏就用默认
    tpl["accent"] = THEMES.get(tpl.get("theme"), tpl.get("accent") or "#6366F1")
    return tpl


def save_template(patch: dict) -> dict:
    """合并并持久化模板配置。返回保存后的完整配置。"""
    tpl = _deep_merge(load_template(), patch or {})
    if tpl.get("theme") in THEMES:
        tpl["accent"] = THEMES[tpl["theme"]]
    # 只保留已知键，避免脏数据无限增长
    keep = set(DEFAULT_TEMPLATE)
    tpl = {k: v for k, v in tpl.items() if k in keep}
    if not isinstance(tpl.get("sections"), dict):
        tpl["sections"] = dict(DEFAULT_TEMPLATE["sections"])
    if not isinstance(tpl.get("risk_items"), list):
        tpl["risk_items"] = []
    TEMPLATE_FILE.write_text(json.dumps(tpl, ensure_ascii=False, indent=2), encoding="utf-8")
    return tpl


def reset_template() -> dict:
    try:
        if TEMPLATE_FILE.exists():
            TEMPLATE_FILE.unlink()
    except Exception:
        pass
    return load_template()


def theme_options() -> dict:
    return dict(THEMES)


# ---------------------------------------------------------------- 数据
def _trade_stats(bt: dict, rows: int) -> dict:
    """从回测交易明细推导派生指标。

    这些指标是"模板写不出来、但能支撑推理"的关键：
      * ``profit_loss_ratio`` 盈亏比 —— 配合胜率才能判断"靠大赚小亏还是靠高胜率"
      * ``max_consec_loss``   最大连续亏损 —— 衡量心理承受与策略稳定性
      * ``avg_hold_days``     平均持仓天数 —— 判断是短线频繁还是中长线
      * ``exposure_pct``      持仓时间占比 —— 低仓位却高收益 => 收益依赖少数交易
    """
    out = {
        "profit_loss_ratio": None, "max_consec_loss": 0,
        "avg_hold_days": None, "exposure_pct": None,
        "avg_win_pct": None, "avg_loss_pct": None,
    }
    if not bt:
        return out

    trades = bt.get("trades") or []
    # trades 是 [BUY, SELL, BUY, SELL, ...] 成对结构
    pairs, i = [], 0
    while i + 1 < len(trades):
        a, b = trades[i], trades[i + 1]
        if a.get("dir") == "BUY" and b.get("dir") == "SELL":
            pairs.append((a, b))
            i += 2
        else:
            i += 1

    if not pairs:
        return out

    wins, losses, hold_days = [], [], []
    for buy, sell in pairs:
        ret = (sell["price"] - buy["price"]) / buy["price"] * 100 if buy.get("price") else 0
        (wins if ret > 0 else losses).append(ret)
        try:
            d1 = pd.to_datetime(buy["date"])
            d2 = pd.to_datetime(sell["date"])
            hold_days.append(max((d2 - d1).days, 0))
        except Exception:
            pass

    avg_win = sum(wins) / len(wins) if wins else None
    avg_loss = abs(sum(losses) / len(losses)) if losses else None
    out["avg_win_pct"] = round(avg_win, 2) if avg_win is not None else None
    out["avg_loss_pct"] = round(avg_loss, 2) if avg_loss is not None else None
    if avg_win is not None and avg_loss:
        out["profit_loss_ratio"] = round(avg_win / avg_loss, 2)

    streak = best = 0
    for buy, sell in pairs:
        ret = (sell["price"] - buy["price"]) / buy["price"] if buy.get("price") else 0
        if ret <= 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    out["max_consec_loss"] = best

    if hold_days:
        out["avg_hold_days"] = round(sum(hold_days) / len(hold_days), 1)
        if rows:
            out["exposure_pct"] = round(sum(hold_days) / rows * 100, 1)
    return out


def build_report_data(stock_code: str, df: pd.DataFrame, bt: dict | None,
                      sent: dict | None, start: str = "", end: str = "") -> dict:
    """把各模块结果收拢成渲染所需的扁平结构。

    除渲染字段外，额外提供 ``metrics``（派生指标），
    模板版与大模型版风险提示共用同一份事实。
    """
    first = str(df.iloc[0]["date"]) if not df.empty else "-"
    last = str(df.iloc[-1]["date"]) if not df.empty else "-"
    close_first = float(df.iloc[0]["close"]) if not df.empty and "close" in df else None
    close_last = float(df.iloc[-1]["close"]) if not df.empty and "close" in df else None
    change = None
    if close_first and close_last is not None and close_first != 0:
        change = round((close_last - close_first) / close_first * 100, 2)

    rows = int(len(df))
    sent = sent or {}
    idx = sent.get("index")
    sent_label = ("偏乐观" if idx > 55 else "偏悲观" if idx < 45 else "中性") if idx is not None else "—"

    bt = bt or {}
    tstats = _trade_stats(bt, rows)

    # 舆情事件分布：'其他' 占比高 => 事件识别粒度粗，情绪过滤的实际约束力被高估
    events = sent.get("events") or []
    ev_total = sum(int(e.get("count") or 0) for e in events) or 0
    other = next((int(e.get("count") or 0) for e in events if e.get("name") == "其他"), 0)
    other_ratio = round(other / ev_total * 100, 1) if ev_total else None
    top_events = [
        {"name": e.get("name"), "type": e.get("type"), "count": e.get("count")}
        for e in sorted(events, key=lambda x: -(x.get("count") or 0))[:5]
        if e.get("name") != "其他"
    ]
    results = sent.get("results") or []

    span_days = None
    try:
        if first != "-" and last != "-":
            span_days = (pd.to_datetime(last) - pd.to_datetime(first)).days
    except Exception:
        pass

    metrics = {
        "rows": rows,
        "span_days": span_days,
        "price_change_pct": change,
        "total_return": bt.get("total_return"),
        "annual_return": bt.get("annual_return"),
        "max_drawdown": bt.get("max_drawdown"),
        "win_rate": bt.get("win_rate"),
        "sharpe": bt.get("sharpe"),
        "trade_count": bt.get("trade_count"),
        "sentiment_filtered": bt.get("sentiment_filtered"),
        "profit_loss_ratio": tstats["profit_loss_ratio"],
        "max_consec_loss": tstats["max_consec_loss"],
        "avg_hold_days": tstats["avg_hold_days"],
        "exposure_pct": tstats["exposure_pct"],
        "avg_win_pct": tstats["avg_win_pct"],
        "avg_loss_pct": tstats["avg_loss_pct"],
        "sentiment_index": idx,
        "sentiment_label": sent_label,
        "pos_pct": sent.get("pos_pct"),
        "neg_pct": sent.get("neg_pct"),
        "neu_pct": sent.get("neu_pct"),
        "sent_sample": len(results),
        "other_ratio": other_ratio,
        "top_events": top_events,
        "rows_below_250": rows < 250,
    }

    return {
        "stock_code": stock_code,
        "rows": rows,
        "start": first,
        "end": last,
        "close_first": close_first,
        "close_last": close_last,
        "change_pct": change,
        "backtest": bt,
        "sentiment": sent,
        "sentiment_index": idx,
        "sentiment_label": sent_label,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "range_note": f"{start or first} ~ {end or last}",
        "metrics": metrics,
    }


def template_risks(data: dict, tpl: dict) -> list[str]:
    """规则模板版风险提示（永不失败，作为大模型版的兜底）。

    相比最初版本，这里用上了 ``metrics`` 里的派生指标，
    因此即便不接大模型，提示也会更具体（而不是只报数字）。
    """
    bt = data["backtest"]
    sent = data["sentiment"]
    m = data.get("metrics") or {}
    out = [
        "本报告基于历史数据回测，历史表现不代表未来收益，实盘交易存在亏损风险。",
    ]
    if bt:
        out.append(f"策略最大回撤达 {bt.get('max_drawdown', 0)}%，请根据自身风险承受能力谨慎决策。")
        tc = bt.get("trade_count") or 0
        if tc < 10:
            out.append(f"回测交易笔数仅 {tc} 次，样本量不足，统计结果可能不具代表性。")
        if abs(bt.get("sharpe") or 0) < 0.5:
            out.append("夏普比率较低，策略风险调整后收益不理想，需谨慎参考。")

        # ---- 以下为派生指标带来的新洞察 ----
        plr = m.get("profit_loss_ratio")
        wr = m.get("win_rate")
        if plr is not None and wr is not None:
            if plr >= 1.5 and wr < 50:
                out.append(
                    f"胜率仅 {wr}% 而盈亏比达 {plr}，收益依赖少数大额盈利交易，"
                    "一旦少数关键行情缺失，整体表现可能显著恶化。")
            elif plr < 1.0 and wr >= 50:
                out.append(
                    f"胜率 {wr}% 但盈亏比仅 {plr}，属于赚小亏大结构，"
                    "单次大幅回撤即可吞掉多次盈利。")

        exp = m.get("exposure_pct")
        tr = m.get("total_return")
        if exp is not None and exp < 40 and tr is not None and tr > 0:
            out.append(
                f"持仓时间占比仅 {exp}%，却取得 {tr}% 收益，"
                "说明收益集中在少数交易，样本外可复制性存疑。")

        consec = m.get("max_consec_loss")
        if consec and consec >= 3:
            out.append(f"策略最大连续亏损达 {consec} 次，实际执行中容易因连续回撤而中断纪律。")

    if data["rows"] < 250:
        out.append(f"数据样本仅 {data['rows']} 条（不足一年），回测结果可能存在过拟合风险。")
    if sent:
        idx = sent.get("index")
        if (idx or 50) < 45:
            out.append("当前舆情情绪偏悲观，市场不确定性较高，建议观望或减仓。")
        results = sent.get("results") or []
        if len(results) < 5:
            out.append(f"舆情样本仅 {len(results)} 条，情绪研判可靠性有限。")
        other_ratio = m.get("other_ratio")
        if other_ratio is not None and other_ratio >= 70:
            out.append(
                f"舆情事件中「其他」类占比达 {other_ratio}%，事件识别粒度偏粗，"
                "情绪过滤对实际交易的约束力可能被高估。")
    out.append("舆情研判由 AI/关键词模型生成，无法保证信息完整性与判断准确性，仅供参考。")
    out.append("双均线策略为趋势跟踪型，在震荡行情中可能频繁产生虚假信号导致亏损。")
    for extra in (tpl.get("risk_items") or []):
        if isinstance(extra, str) and extra.strip():
            out.append(extra.strip())
    return out


def _risks(data: dict, tpl: dict, allow_llm: bool = True) -> list[str]:
    """风险提示入口：优先大模型，失败自动降级模板。

    保证 **永远返回非空列表**，报告生成不会因外部服务失败而中断。
    大模型成功时，第一条是它写的整段策略分析（以「分析：」开头）。
    """
    fallback = template_risks(data, tpl)
    if not allow_llm:
        return fallback

    try:
        from modules import risk_llm
        if not risk_llm.enabled():
            return fallback
        res, meta = risk_llm.generate_full(data)
        if res and (res.get("risks") or res.get("summary")):
            _last_risk_source["engine"] = "llm"
            _last_risk_source["meta"] = {
                "model": meta.get("model"), "provider": meta.get("provider"),
                "elapsed_ms": meta.get("elapsed_ms"), "dropped": len(meta.get("dropped") or []),
                "has_summary": bool(res.get("summary")),
            }
            out = []
            if res.get("summary"):
                out.append(res["summary"])
            out.extend(res.get("risks") or [])
            note = (f"上述分析由大模型 {meta.get('model')} 生成，"
                    "数字均已与回测结果核对；内容仅供参考，不构成投资建议。")
            return out + [note]
        _last_risk_source["engine"] = "template"
        _last_risk_source["meta"] = {"error": meta.get("error")}
    except Exception as e:                      # 任何意外都不影响报告
        _last_risk_source["engine"] = "template"
        _last_risk_source["meta"] = {"error": f"{type(e).__name__}: {e}"}
    return fallback


def last_risk_source() -> dict:
    """最近一次风险提示的来源，供接口回传。"""
    return dict(_last_risk_source)


# ---------------------------------------------------------------- 渲染
def _esc(s) -> str:
    """转义 HTML —— 大模型输出的文本必须转义后再嵌入报告。"""
    return (str(s if s is not None else "")
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _fmt(v, suffix="", nd=2):
    if v is None or v == "":
        return "—"
    try:
        return f"{float(v):,.{nd}f}{suffix}"
    except Exception:
        return f"{v}{suffix}"


def render_html(data: dict, tpl: dict | None = None) -> str:
    """渲染报告 HTML。PDF 导出也用这份 HTML，保证两者完全一致。"""
    tpl = tpl or load_template()
    accent = tpl.get("accent") or "#6366F1"
    sec = tpl.get("sections") or {}
    bt = data["backtest"] or {}
    sent = data["sentiment"] or {}

    title = (tpl.get("title") or "{stock} 综合量化分析报告").format(
        stock=data["stock_code"], date=data["generated_at"])
    subtitle = (tpl.get("subtitle") or "").replace("{stock}", data["stock_code"])
    company = tpl.get("company") or "QuantMind"
    author = tpl.get("author") or ""
    footer = (tpl.get("footer") or "").format(
        company=company, date=data["generated_at"],
        start=data["start"], end=data["end"])

    up, down = "#DC2626", "#16A34A"          # 中国习惯：涨红跌绿

    def tone(v):
        try:
            v = float(v)
        except Exception:
            return ""
        return ("color:" + up) if v > 0 else (("color:" + down) if v < 0 else "")

    parts = []
    parts.append(f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><title>{title}</title>
<style>
@page {{ size: A4; margin: 16mm 14mm; }}
* {{ box-sizing: border-box; }}
body {{ font-family: "Microsoft YaHei","PingFang SC","Segoe UI",sans-serif;
  color:#1F2430; background:#fff; margin:0; font-size:12.5px; line-height:1.75; }}
.head {{ border-bottom:3px solid {accent}; padding-bottom:14px; margin-bottom:22px; }}
.head h1 {{ font-size:24px; margin:0 0 6px; color:{accent}; letter-spacing:-.02em; }}
.head .sub {{ font-size:12px; color:#6B7280; }}
.meta {{ display:flex; flex-wrap:wrap; gap:8px 22px; font-size:11.5px; color:#4B5563;
  background:#F8FAFC; border:1px solid #E5E7EB; border-radius:8px; padding:10px 14px; margin-bottom:22px; }}
.meta b {{ color:#111827; }}
h2 {{ font-size:14.5px; color:{accent}; margin:26px 0 10px; padding-left:9px;
  border-left:4px solid {accent}; }}
table {{ border-collapse:collapse; width:100%; margin:8px 0; font-size:12px; }}
th,td {{ border:1px solid #E5E7EB; padding:7px 10px; text-align:left; }}
th {{ background:#F9FAFB; font-weight:600; color:#374151; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.kpi {{ display:flex; flex-wrap:wrap; gap:10px; margin:10px 0; }}
.kpi div {{ flex:1 1 108px; border:1px solid #E5E7EB; border-radius:8px; padding:10px 12px; background:#FCFCFD; }}
.kpi .k {{ font-size:10.5px; color:#6B7280; }}
.kpi .v {{ font-size:16px; font-weight:700; font-variant-numeric:tabular-nums; }}
ul.risk {{ margin:6px 0 0 0; padding-left:20px; }}
ul.risk li {{ margin-bottom:5px; }}
.analysis {{ margin:6px 0 12px; padding:11px 14px; background:#F8F9FB;
  border-left:3px solid {accent}; border-radius:0 6px 6px 0; line-height:1.85; }}
.src-note {{ margin-top:10px; font-size:10px; color:#9CA3AF; }}
.disc {{ margin-top:26px; border-top:1px dashed #D1D5DB; padding-top:12px;
  font-size:10.5px; color:#6B7280; line-height:1.7; }}
.foot {{ margin-top:10px; font-size:10px; color:#9CA3AF; }}
.empty {{ color:#9CA3AF; font-style:italic; }}
</style></head><body>
<div class="head">
  <h1>{title}</h1>
  <div class="sub">{subtitle}</div>
</div>
<div class="meta">
  <span>股票代码 <b>{data['stock_code']}</b></span>
  <span>生成时间 <b>{data['generated_at']}</b></span>
  <span>数据区间 <b>{data['start']} ~ {data['end']}</b></span>
  <span>样本数 <b>{data['rows']} 条</b></span>
  {'<span>作者 <b>' + author + '</b></span>' if author else ''}
</div>""")

    if sec.get("overview", True):
        chg = data["change_pct"]
        parts.append(f"""<h2>① 行情概览</h2>
<div class="kpi">
  <div><div class="k">起始收盘</div><div class="v">{_fmt(data['close_first'])}</div></div>
  <div><div class="k">最新收盘</div><div class="v">{_fmt(data['close_last'])}</div></div>
  <div><div class="k">区间涨跌</div><div class="v" style="{tone(chg)}">{_fmt(chg, '%')}</div></div>
  <div><div class="k">样本条数</div><div class="v">{data['rows']}</div></div>
</div>
<p>本报告覆盖 <b>{data['start']}</b> 至 <b>{data['end']}</b> 共 <b>{data['rows']}</b> 条日线数据。</p>""")

    if sec.get("strategy", True):
        allow = "允许买入" if (data["sentiment_index"] or 0) >= 45 else "拒绝买入"
        parts.append(f"""<h2>② 策略逻辑</h2>
<p>双均线（MA5 / MA20）金叉买入、死叉卖出，叠加<b>情绪指数过滤</b>：
当前情绪指数 <b>{data['sentiment_index'] if data['sentiment_index'] is not None else '—'}</b>，
{allow}；单次使用 90% 仓位，已计入手续费（0.03%）与滑点（0.01%）。</p>""")

    if sec.get("performance", True):
        parts.append("<h2>③ 回测绩效</h2>")
        if bt:
            parts.append(f"""<div class="kpi">
  <div><div class="k">总收益</div><div class="v" style="{tone(bt.get('total_return'))}">{_fmt(bt.get('total_return'), '%')}</div></div>
  <div><div class="k">年化收益</div><div class="v" style="{tone(bt.get('annual_return'))}">{_fmt(bt.get('annual_return'), '%')}</div></div>
  <div><div class="k">最大回撤</div><div class="v" style="color:{down}">{_fmt(bt.get('max_drawdown'), '%')}</div></div>
  <div><div class="k">胜率</div><div class="v">{_fmt(bt.get('win_rate'), '%')}</div></div>
  <div><div class="k">夏普比率</div><div class="v">{_fmt(bt.get('sharpe'))}</div></div>
  <div><div class="k">交易笔数</div><div class="v">{bt.get('trade_count', 0)}</div></div>
</div>
<table><tr><th>指标</th><th>数值</th></tr>
<tr><td>初始资金</td><td class="num">{_fmt(100000, ' 元', 0)}</td></tr>
<tr><td>期末权益</td><td class="num">{_fmt(bt.get('final_equity'), ' 元', 0)}</td></tr>
<tr><td>总收益率</td><td class="num" style="{tone(bt.get('total_return'))}">{_fmt(bt.get('total_return'), '%')}</td></tr>
<tr><td>年化收益率</td><td class="num" style="{tone(bt.get('annual_return'))}">{_fmt(bt.get('annual_return'), '%')}</td></tr>
<tr><td>最大回撤</td><td class="num" style="color:{down}">{_fmt(bt.get('max_drawdown'), '%')}</td></tr>
<tr><td>夏普比率</td><td class="num">{_fmt(bt.get('sharpe'))}</td></tr>
</table>""")
        else:
            parts.append('<p class="empty">回测未执行或无有效信号（需先计算技术指标）。</p>')

    if sec.get("sentiment", True):
        parts.append("<h2>④ 舆情摘要</h2>")
        if sent:
            results = sent.get("results") or []
            parts.append(f"""<div class="kpi">
  <div><div class="k">情绪指数</div><div class="v">{sent.get('index', '—')}<span style="font-size:11px;color:#6B7280">/100</span></div></div>
  <div><div class="k">研判</div><div class="v" style="font-size:14px">{data['sentiment_label']}</div></div>
  <div><div class="k">看涨占比</div><div class="v" style="color:{up}">{_fmt(sent.get('pos_pct'), '%')}</div></div>
  <div><div class="k">看跌占比</div><div class="v" style="color:{down}">{_fmt(sent.get('neg_pct'), '%')}</div></div>
  <div><div class="k">分析条数</div><div class="v">{len(results)}</div></div>
</div>""")
            events = sent.get("events") or []
            if events:
                rows = "".join(
                    f"<tr><td>{e.get('name', '')}</td><td>{e.get('type', '')}</td>"
                    f"<td class=\"num\">{e.get('count', 0)} 条</td>"
                    f"<td class=\"num\">{e.get('score', 0)}</td></tr>" for e in events[:6])
                parts.append("<table><tr><th>事件</th><th>类型</th><th>提及</th><th>情感分</th></tr>"
                             + rows + "</table>")
            if results:
                rows = "".join(
                    f"<tr><td>{r.get('title', '')}</td><td>{r.get('sentiment', '')}</td>"
                    f"<td class=\"num\">{r.get('confidence', 0)}</td></tr>" for r in results[:8])
                parts.append("<table><tr><th>标题（节选）</th><th>情感</th><th>置信度</th></tr>"
                             + rows + "</table>")
        else:
            parts.append('<p class="empty">未执行舆情分析。</p>')

    if sec.get("risk", True):
        risks = _risks(data, tpl)
        # 大模型成功时第一条是整段策略分析（不含这些尾部标记），单独排成一段话
        blocks = []
        note = ""
        if risks and risks[-1].startswith("上述"):
            note = risks[-1]
            risks = risks[:-1]
        src = _last_risk_source.get("engine")
        if src == "llm" and risks:
            summary, risks = risks[0], risks[1:]
            blocks.append(f'<p class="analysis">{_esc(summary)}</p>')
        if risks:
            lis = "".join(f"<li>{_esc(r)}</li>" for r in risks)
            blocks.append(f'<ul class="risk">{lis}</ul>')
        if note:
            blocks.append(f'<p class="src-note">{_esc(note)}</p>')
        parts.append('<h2>⑤ 策略分析与风险提示</h2>' + "".join(blocks))

    disc = (tpl.get("disclaimer") or "").strip()
    if disc:
        parts.append(f'<div class="disc">{disc}</div>')
    parts.append(f'<div class="foot">{footer}</div>')
    parts.append("</body></html>")
    return "".join(parts)


# ---------------------------------------------------------------- PDF
_CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
]


def find_browser() -> str | None:
    """按固定候选路径 + PATH 查找 Chrome/Edge。"""
    env = os.environ.get("QUANTMIND_CHROME") or os.environ.get("CHROME_PATH")
    if env and os.path.isfile(env):
        return env
    for p in _CHROME_CANDIDATES:
        if os.path.isfile(p):
            return p
    for name in ("google-chrome", "chrome", "chromium", "msedge", "microsoft-edge"):
        hit = shutil.which(name)
        if hit:
            return hit
    return None


def _html_to_pdf_chrome(html: str, out_pdf: Path) -> None:
    browser = find_browser()
    if not browser:
        raise PdfUnavailable("未找到 Chrome/Edge")

    # 临时 HTML 与 PDF 放同一临时目录；Windows 上 Chrome 需要 Windows 路径
    tmp_dir = Path(tempfile.mkdtemp(prefix="qm_pdf_"))
    tmp_html = tmp_dir / "report.html"
    tmp_pdf = tmp_dir / "report.pdf"
    tmp_html.write_text(html, encoding="utf-8")

    profile = tmp_dir / "profile"
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-first-run",
        "--no-default-browser-check",
        f"--user-data-dir={profile}",
        "--no-pdf-header-footer",
        f"--print-to-pdf={tmp_pdf}",
        tmp_html.resolve().as_uri(),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if not tmp_pdf.exists() or tmp_pdf.stat().st_size < 1000:
        err = (proc.stderr or b"").decode("utf-8", "ignore")[-400:]
        raise PdfUnavailable(f"Chrome 打印失败：{err.strip() or '未生成文件'}")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_pdf), str(out_pdf))
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _html_to_pdf_weasy(html: str, out_pdf: Path) -> None:
    try:
        from weasyprint import HTML           # type: ignore
    except Exception as e:
        raise PdfUnavailable(f"weasyprint 不可用：{e}") from e
    HTML(string=html).write_pdf(str(out_pdf))


def _html_to_pdf_reportlab(html: str, out_pdf: Path) -> None:
    """最后兜底：用 reportlab 生成纯文本版 PDF（排版简单但内容完整）。"""
    try:
        from reportlab.pdfgen import canvas          # type: ignore
        from reportlab.pdfbase import pdfmetrics     # type: ignore
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont  # type: ignore
    except Exception as e:
        raise PdfUnavailable(f"reportlab 不可用：{e}") from e
    import re

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    text = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", html)
    text = re.sub(r"<[^>]+>", "\n", text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    c = canvas.Canvas(str(out_pdf))
    c.setFont("STSong-Light", 10)
    y = 800
    for ln in lines:
        if y < 40:
            c.showPage()
            c.setFont("STSong-Light", 10)
            y = 800
        c.drawString(46, y, ln[:80])
        y -= 15
    c.save()


def render_pdf(html: str, out_pdf: Path) -> str:
    """把 HTML 转成真 PDF，返回实际使用的引擎名。

    任何引擎都不可用时抛 :class:`PdfUnavailable` —— **不写伪 PDF**。
    """
    errors = []
    for name, fn in (("chrome", _html_to_pdf_chrome),
                     ("weasyprint", _html_to_pdf_weasy),
                     ("reportlab", _html_to_pdf_reportlab)):
        try:
            fn(html, Path(out_pdf))
            data = Path(out_pdf).read_bytes()[:5]
            if data.startswith(b"%PDF"):
                return name
            errors.append(f"{name}: 输出不是合法 PDF")
        except Exception as e:
            errors.append(f"{name}: {e}")
    raise PdfUnavailable("；".join(errors))


# ---------------------------------------------------------------- 一致性
def build_and_save(stock_code: str, df: pd.DataFrame, bt: dict | None,
                   sent: dict | None, fmt: str = "html",
                   start: str = "", end: str = "",
                   tpl: dict | None = None) -> dict:
    """生成报告并落盘，返回 ``{ok, filename, url, format, engine}``。

    这是 API 与流水线共用的唯一入口，保证两条路径产物一致。
    """
    tpl = tpl or load_template()
    data = build_report_data(stock_code, df, bt, sent, start, end)
    html = render_html(data, tpl)
    fmt = (fmt or "html").lower()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if fmt == "pdf":
        fname = f"{stock_code}_report_{stamp}.pdf"
        path = EXPORT_DIR / fname
        engine = render_pdf(html, path)
        # 同时留一份 HTML，方便排查排版
        (EXPORT_DIR / f"{stock_code}_report_{stamp}.html").write_text(html, encoding="utf-8")
    else:
        fname = f"{stock_code}_report_{stamp}.html"
        path = EXPORT_DIR / fname
        path.write_text(html, encoding="utf-8")
        engine = "html"

    return {
        "ok": True,
        "filename": fname,
        "url": f"/exports/{fname}",
        "format": fmt,
        "engine": engine,
        "size": path.stat().st_size,
        "risk_source": last_risk_source(),   # 让调用方知道风险提示是 LLM 还是模板
        "data": data,
    }
