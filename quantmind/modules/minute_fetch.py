# -*- coding: utf-8 -*-
"""分钟行情采集（5 分钟 / 30 分钟）。

数据源选择依据见 `docs/minute-data-feasibility.md`：

- **主源**：新浪 `money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData`
  `?symbol=sh600519&scale=30&ma=5&datalen=5000`
  → 返回 **最多 5001 根**（`datalen` 传更大也是 5001），m30 可覆盖约 626 个交易日。

  🔴 **不要用 akshare 的 `stock_zh_a_minute`**：它把 `datalen` 硬编码成 `"1970"`，
  且用的老域名 `quotes.sina.cn` 在 `datalen > 1970` 时会直接返回 `null`。
  这是本项目此前「分钟数据只有 1970 条」误判的总根源。

- **备源**：腾讯 `ifzq.gtimg.cn/appstock/app/kline/mkline?param=code,mX,,count`
  → 只返回 **最多 482 根**，历史很短，仅作新浪失败时的兜底。

请求一律走 `requests` 且 `trust_env=False`（绕过系统代理）：
本机环境代理会让部分财经域名返回 502。
"""
import time
from datetime import datetime

import pandas as pd
import requests

from db import get_session, QuotationMinute, MinuteCollectLog

SESSION = requests.Session()
SESSION.trust_env = False          # 绕过环境代理，否则部分财经域名 502
SESSION.headers.update({
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "*/*",
    "Referer": "https://finance.sina.com.cn/",
})

SINA_URL = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
            "CN_MarketData.getKLineData")
TX_URL = "https://ifzq.gtimg.cn/appstock/app/kline/mkline"

# 新浪接口 datalen 传大于此值也只返回 5001 根（服务端滚动窗口硬顶）
SINA_MAX_ROWS = 5001

# 支持的周期：分钟数 → 新浪 scale 参数（两者一致，保留映射便于将来扩展）
SUPPORTED_PERIODS = (5, 30)

SOURCE_SINA = "SINA_MIN"
SOURCE_TENCENT = "TENCENT_MIN"


def to_sina_symbol(code):
    """6 位代码 → 新浪/腾讯格式：sh600519 / sz000001。

    已带前缀的原样返回；北交所（8/4 开头）暂不支持 —— 新浪该接口对其不返回数据。
    """
    code = str(code).strip().lower()
    if code[:2] in ("sh", "sz", "bj"):
        return code
    if len(code) != 6:
        return code
    return ("sh" if code[0] in "56" else "sz") + code


def _fetch_sina(symbol, period, count=SINA_MAX_ROWS, timeout=20):
    """新浪主源。返回 DataFrame[time, open, high, low, close, volume] 或空表。"""
    params = {"symbol": symbol, "scale": str(period), "ma": "5", "datalen": str(count)}
    r = SESSION.get(SINA_URL, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    need = {"day", "open", "high", "low", "close", "volume"}
    if not need.issubset(df.columns):
        return pd.DataFrame()

    df = df.rename(columns={"day": "time"})
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "close"])
    return df[["time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def _fetch_tencent(symbol, period, count=5000, timeout=20):
    """腾讯备源。只返回最多 482 根，历史短，仅兜底。"""
    params = {"param": f"{symbol},m{period},", "count": str(count)}
    r = SESSION.get(TX_URL, params=params, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    node = (payload.get("data") or {}).get(symbol) or {}
    rows = node.get(f"m{period}") or []
    if not rows:
        return pd.DataFrame()

    # 每行形如 ['202610161500', open, close, high, low, volume, {}, amount%]
    recs = []
    for row in rows:
        try:
            recs.append({
                "time": pd.to_datetime(str(row[0]), format="%Y%m%d%H%M"),
                "open": float(row[1]),
                "close": float(row[2]),
                "high": float(row[3]),
                "low": float(row[4]),
                "volume": float(row[5]),
            })
        except (ValueError, IndexError, TypeError):
            continue
    if not recs:
        return pd.DataFrame()
    return pd.DataFrame(recs)[["time", "open", "high", "low", "close", "volume"]]


def fetch_minute(code, period=30, count=SINA_MAX_ROWS, retries=2, sleep=0.2):
    """拉取单只标的的分钟 K 线。

    返回 ``(df, source)``；两个源都失败时返回 ``(空 DataFrame, None)``。
    ``df`` 列为 ``time / open / high / low / close / volume``，按时间升序。
    """
    if period not in SUPPORTED_PERIODS:
        raise ValueError(f"period 只支持 {SUPPORTED_PERIODS}，收到 {period}")

    symbol = to_sina_symbol(code)
    last_err = None

    for attempt in range(retries + 1):
        try:
            df = _fetch_sina(symbol, period, count=count)
            if not df.empty:
                return df, SOURCE_SINA
            last_err = "新浪返回空"
        except Exception as e:                                 # noqa: BLE001
            last_err = f"新浪 {type(e).__name__}: {e}"
        if attempt < retries:
            time.sleep(sleep * (attempt + 1))

    try:
        df = _fetch_tencent(symbol, period)
        if not df.empty:
            return df, SOURCE_TENCENT
        last_err = (last_err or "") + " | 腾讯返回空"
    except Exception as e:                                     # noqa: BLE001
        last_err = (last_err or "") + f" | 腾讯 {type(e).__name__}: {e}"

    return pd.DataFrame(), None if last_err is None else last_err


def save_minute(code, period, df, source=SOURCE_SINA, batch=2000):
    """幂等写入 ``quotation_minute``。

    已存在 ``(stock_code, period, trade_time)`` 的行直接跳过，只插入新 bar。
    这样采集脚本可以每天重复跑做增量更新，不会产生重复。

    返回 ``(新增行数, 跳过行数)``。
    """
    if df is None or df.empty:
        return 0, 0

    stock_code = str(code).strip().lower().replace("sh", "").replace("sz", "")
    s = get_session()
    try:
        existing = {
            r[0] for r in s.query(QuotationMinute.trade_time)
            .filter(QuotationMinute.stock_code == stock_code,
                    QuotationMinute.period == period)
            .all()
        }

        rows, skipped = [], 0
        for rec in df.itertuples(index=False):
            tt = pd.Timestamp(rec.time).to_pydatetime()
            if tt in existing:
                skipped += 1
                continue
            existing.add(tt)
            rows.append(QuotationMinute(
                stock_code=stock_code,
                period=period,
                trade_time=tt,
                open=float(rec.open),
                high=float(rec.high),
                low=float(rec.low),
                close=float(rec.close),
                volume=int(rec.volume or 0),
                data_source=source,
            ))

        for i in range(0, len(rows), batch):
            s.add_all(rows[i:i + batch])
            s.flush()
        s.commit()
        return len(rows), skipped
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def load_minute(code, period=30, start_time=None, end_time=None):
    """从库里读分钟行情，返回 DataFrame[time, open, high, low, close, volume]。

    列名与日线 ``load_quotation`` 对齐（``date`` 之外再加 ``time``），
    便于回测层复用同一套信号函数。
    """
    stock_code = str(code).strip().lower().replace("sh", "").replace("sz", "")
    s = get_session()
    try:
        q = (s.query(QuotationMinute)
             .filter(QuotationMinute.stock_code == stock_code,
                     QuotationMinute.period == period)
             .order_by(QuotationMinute.trade_time))
        if start_time is not None:
            q = q.filter(QuotationMinute.trade_time >= _as_dt(start_time))
        if end_time is not None:
            q = q.filter(QuotationMinute.trade_time <= _as_dt(end_time))
        rows = q.all()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{
            "time": r.trade_time,
            "date": r.trade_time.date(),
            "open": float(r.open), "high": float(r.high),
            "low": float(r.low), "close": float(r.close),
            "volume": int(r.volume or 0),
        } for r in rows])
    finally:
        s.close()


def _as_dt(value):
    if isinstance(value, datetime):
        return value
    if hasattr(value, "year") and not isinstance(value, str):
        return datetime(value.year, value.month, value.day)
    ts = pd.to_datetime(value)
    return ts.to_pydatetime()


def minute_stats(code=None):
    """库内分钟数据概况，供 CLI / 页面展示。"""
    s = get_session()
    try:
        q = s.query(QuotationMinute.stock_code, QuotationMinute.period,
                    QuotationMinute.trade_time)
        if code:
            q = q.filter(QuotationMinute.stock_code ==
                         str(code).strip().lower().replace("sh", "").replace("sz", ""))
        agg = {}
        for sc, period, tt in q.all():
            key = (sc, period)
            if key not in agg:
                agg[key] = [0, tt, tt]
            agg[key][0] += 1
            if tt < agg[key][1]:
                agg[key][1] = tt
            if tt > agg[key][2]:
                agg[key][2] = tt
        return [{"stock_code": k[0], "period": k[1], "rows": v[0],
                 "start": v[1], "end": v[2]} for k, v in sorted(agg.items())]
    finally:
        s.close()


def log_collect(code, period, ok, rows_fetched=0, rows_added=0, rows_skipped=0,
                source=None, message=None):
    """记一条采集日志。**绝不因为写日志失败而中断采集**。"""
    try:
        s = get_session()
        try:
            s.add(MinuteCollectLog(
                stock_code=str(code).strip().lower().replace("sh", "").replace("sz", ""),
                period=int(period),
                ok=1 if ok else 0,
                rows_fetched=int(rows_fetched or 0),
                rows_added=int(rows_added or 0),
                rows_skipped=int(rows_skipped or 0),
                data_source=(source or "")[:32] or None,
                message=(str(message)[:250] if message else None),
            ))
            s.commit()
        finally:
            s.close()
    except Exception:                                          # noqa: BLE001
        pass


def recent_logs(limit=20):
    """最近 N 条采集日志，供前端展示。"""
    s = get_session()
    try:
        rows = (s.query(MinuteCollectLog)
                .order_by(MinuteCollectLog.id.desc())
                .limit(int(limit))
                .all())
        return [{
            "id": r.id, "code": r.stock_code, "period": r.period,
            "ok": bool(r.ok), "fetched": r.rows_fetched, "added": r.rows_added,
            "skipped": r.rows_skipped, "source": r.data_source,
            "message": r.message,
            "time": str(r.created_at)[:19] if r.created_at else "",
        } for r in rows]
    finally:
        s.close()


def collect_and_save(code, period):
    """采集 + 写库 + 记日志，返回统一结构的结果 dict。

    把「拉取 → 落库 → 记日志」三步收在一个函数里，是为了让 CLI 与 Flask 接口
    **共用同一套逻辑**，避免两边对失败的处理不一致（之前 CLI 记日志、接口不记，
    排查问题时接口那一侧完全没有痕迹）。
    """
    df, src = fetch_minute(code, period)
    if df is None or len(df) == 0:
        msg = str(src) if src else "未知错误"
        log_collect(code, period, False, message=msg)
        return {"code": code, "period": period, "ok": False, "msg": msg}
    added, skipped = save_minute(code, period, df, source=src)
    log_collect(code, period, True, rows_fetched=len(df), rows_added=added,
                rows_skipped=skipped, source=src)
    return {"code": code, "period": period, "ok": True, "rows": len(df),
            "added": added, "skipped": skipped, "source": src}
