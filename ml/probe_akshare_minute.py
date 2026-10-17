# -*- coding: utf-8 -*-
"""实测 akshare 所有 A 股分钟相关接口，和 Ashare 做横向对比。

目标：回答「akshare 到底能不能拿到和 Ashare 一样多的分钟历史」。
"""
import time
import warnings

warnings.filterwarnings("ignore")

import akshare as ak
import pandas as pd

CODE_EM = "600519"
CODE_SINA = "sh600519"


def probe(label, fn, **kw):
    t0 = time.time()
    try:
        df = fn(**kw)
        el = time.time() - t0
        if df is None or len(df) == 0:
            print(f"  {label:<46} 0 行  {el:.2f}s")
            return
        for cand in ("时间", "day", "datetime", "time", "date"):
            if cand in df.columns:
                col = cand
                break
        else:
            col = df.columns[0]
        s = pd.to_datetime(df[col], errors="coerce")
        days = s.dt.normalize().nunique() if s.notna().any() else 0
        print(f"  {label:<46} {len(df):>5} 行 / {days:>4} 天  {el:>5.2f}s  "
              f"{str(s.min())[:16]} ~ {str(s.max())[:16]}")
    except Exception as e:
        print(f"  {label:<46} ✗ {type(e).__name__}: {str(e)[:60]}")


print("=" * 118)
print("A. akshare 东财分钟接口（本机行情子域被 RST，预期全失败）")
for per in ("1", "5", "15", "30", "60"):
    probe(f"stock_zh_a_hist_min_em(period={per})",
          ak.stock_zh_a_hist_min_em, symbol=CODE_EM, period=per, adjust="qfq")
    time.sleep(0.6)

print("\n" + "=" * 118)
print("B. akshare 新浪分钟接口 stock_zh_a_minute（datalen 写死 1970）")
for per in ("1", "5", "15", "30", "60"):
    probe(f"stock_zh_a_minute(period={per})",
          ak.stock_zh_a_minute, symbol=CODE_SINA, period=per, adjust="qfq")
    time.sleep(0.6)

print("\n" + "=" * 118)
print("C. akshare 新浪分时接口 stock_intraday_sina")
probe("stock_intraday_sina", ak.stock_intraday_sina, symbol="sh600519", date="20261016")
time.sleep(0.6)

print("\n" + "=" * 118)
print("D. akshare ETF / 指数分钟（东财，验证是否同样被 RST）")
probe("fund_etf_hist_min_em(510300)", ak.fund_etf_hist_min_em,
      symbol="510300", period="5", adjust="qfq")
time.sleep(0.6)
probe("index_zh_a_hist_min_em(000001)", ak.index_zh_a_hist_min_em,
      symbol="000001", period="5")
time.sleep(0.6)

print("\n" + "=" * 118)
print("E. akshare 日线（对照，看是否有非东财通路）")
probe("stock_zh_a_daily(sina)", ak.stock_zh_a_daily,
      symbol="sh600519", start_date="20240101", end_date="20261016", adjust="qfq")
time.sleep(0.6)

print("\n" + "=" * 118)
print("F. 对照：直接用 money.finance.sina.com.cn（Ashare 同源）拿 5000 行")
import json
import subprocess

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
for scale in ("5", "15", "30", "60"):
    url = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"CN_MarketData.getKLineData?symbol=sh600519&scale={scale}&ma=5&datalen=5000")
    t0 = time.time()
    out = subprocess.run(["curl", "-s", "--noproxy", "*", "--max-time", "20",
                          "-H", "User-Agent: " + UA, url], capture_output=True)
    txt = out.stdout.decode("utf-8", "replace")
    try:
        d = json.loads(txt)
        days = len(set(str(r["day"])[:10] for r in d))
        print(f"  {'money.finance scale=' + scale:<46} {len(d):>5} 行 / {days:>4} 天  "
              f"{time.time()-t0:>5.2f}s  {d[0]['day'][:16]} ~ {d[-1]['day'][:16]}")
    except Exception as e:
        print(f"  {'money.finance scale=' + scale:<46} ✗ {str(e)[:50]}")
    time.sleep(0.6)
