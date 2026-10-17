# -*- coding: utf-8 -*-
"""分钟级数据可得性探路。

只做只读探测：每个数据接口试一次，记录「能否拿到 / 拿到多少 / 覆盖多长」，
结果打印到 stdout 并写 JSON，不改动数据库与项目代码。
"""
import json
import sys
import time
import traceback

import akshare as ak

SYMBOL = "600519"          # 贵州茅台，与现有日线数据同标的
OUT = {}

def probe(name, fn):
    t0 = time.time()
    try:
        df = fn()
        dt = round(time.time() - t0, 2)
        if df is None or len(df) == 0:
            print(f"[空] {name:28s} {dt:>6}s  返回 0 行")
            OUT[name] = {"ok": False, "rows": 0, "elapsed": dt}
            return
        cols = list(df.columns)
        info = {"ok": True, "rows": len(df), "cols": cols, "elapsed": dt}
        # 尽量找出时间列与最早/最晚
        tcol = None
        for c in cols:
            if any(k in str(c) for k in ("时间", "日期", "time", "date")):
                tcol = c
                break
        if tcol is not None:
            info["tcol"] = tcol
            info["first"] = str(df[tcol].iloc[0])
            info["last"] = str(df[tcol].iloc[-1])
        OUT[name] = info
        print(f"[OK] {name:28s} {dt:>6}s  {len(df):>7} 行  列={cols}")
        if tcol is not None:
            print(f"     时间列 {tcol}: {info['first']}  →  {info['last']}")
        print(df.head(2).to_string(index=False)[:300])
    except Exception as e:
        dt = round(time.time() - t0, 2)
        msg = f"{type(e).__name__}: {e}"
        OUT[name] = {"ok": False, "error": msg, "elapsed": dt}
        print(f"[X ] {name:28s} {dt:>6}s  {msg[:140]}")


print("=" * 78)
print("分钟级数据探路  symbol=", SYMBOL)
print("=" * 78)

# ---- 1. 分钟线：东财 secid 系列（历史最长，但只返回近期） ----
probe("min_1m_em", lambda: ak.stock_zh_a_hist_min_em(
    symbol=SYMBOL, start_date="2026-09-01 09:30:00",
    end_date="2026-10-09 15:00:00", period="1", adjust=""))
probe("min_5m_em", lambda: ak.stock_zh_a_hist_min_em(
    symbol=SYMBOL, start_date="2026-06-01 09:30:00",
    end_date="2026-10-09 15:00:00", period="5", adjust=""))
probe("min_15m_em", lambda: ak.stock_zh_a_hist_min_em(
    symbol=SYMBOL, start_date="2026-01-01 09:30:00",
    end_date="2026-10-09 15:00:00", period="15", adjust=""))
probe("min_30m_em", lambda: ak.stock_zh_a_hist_min_em(
    symbol=SYMBOL, start_date="2025-01-01 09:30:00",
    end_date="2026-10-09 15:00:00", period="30", adjust=""))
probe("min_60m_em", lambda: ak.stock_zh_a_hist_min_em(
    symbol=SYMBOL, start_date="2024-01-01 09:30:00",
    end_date="2026-10-09 15:00:00", period="60", adjust=""))

# ---- 2. 新浪分钟线（滚动窗口，约近若干日） ----
probe("min_5m_sina", lambda: ak.stock_zh_a_minute(
    symbol="sh600519", period="5", adjust=""))

# ---- 3. 日内分时（当天/近几日 tick 级汇总） ----
probe("intraday_today", lambda: ak.stock_intraday_em(symbol=SYMBOL))
probe("bid_ask_tick", lambda: ak.stock_bid_ask_em(symbol=SYMBOL))

# ---- 4. 日线做对照（确认现有数据源的日频口径） ----
probe("daily_em", lambda: ak.stock_zh_a_hist(
    symbol=SYMBOL, period="daily", start_date="20260201", end_date="20261009", adjust=""))

with open("minute_probe.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=2)
print("\n结果已写入 minute_probe.json")
