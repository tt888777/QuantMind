# -*- coding: utf-8 -*-
"""新浪分钟线数据源边界探测。

东财 push2his 在本机网络不可达（直连被拒 / 走代理失败），
新浪 stock_zh_a_minute 可用，重点摸清它的真实能力边界。
"""
import json
import time

import akshare as ak

RESULT = {}

def stat(tag, period):
    t0 = time.time()
    try:
        df = ak.stock_zh_a_minute(symbol="sh600519", period=period, adjust="")
        dt = round(time.time() - t0, 2)
        if df is None or len(df) == 0:
            print(f"[空] {tag:12s} {dt:>5}s")
            return {"ok": False, "rows": 0}
        first, last = str(df["day"].iloc[0]), str(df["day"].iloc[-1])
        # 交易日数量
        days = sorted({str(x)[:10] for x in df["day"]})
        r = {"ok": True, "rows": len(df), "days": len(days),
             "first": first, "last": last, "elapsed": dt,
             "cols": list(df.columns)}
        print(f"[OK] {tag:12s} {dt:>5}s  {len(df):>7} 行  覆盖 {len(days):>3} 个交易日")
        print(f"     {first}  →  {last}")
        return r
    except Exception as e:
        print(f"[X ] {tag:12s} {type(e).__name__}: {str(e)[:90]}")
        return {"ok": False, "error": str(e)[:200]}

print("=" * 76)
print("新浪分钟线边界探测（sh600519）")
print("=" * 76)

for tag, p in (("5分钟", "5"), ("15分钟", "15"), ("30分钟", "30"), ("60分钟", "60")):
    RESULT[p] = stat(tag, p)
    time.sleep(1.2)      # 新浪对高频请求较敏感，留间隔

# ---- 多标的可获取性（判断是否要逐股抓取、耗时量级） ----
print("\n多标的 5 分钟线可获取性：")
symbols = [("sh600519", "贵州茅台"), ("sh601318", "中国平安"),
           ("sz000001", "平安银行"), ("sz300750", "宁德时代")]
lat = []
for sym, name in symbols:
    t0 = time.time()
    try:
        df = ak.stock_zh_a_minute(symbol=sym, period="5", adjust="")
        dt = round(time.time() - t0, 2)
        lat.append(dt)
        d = sorted({str(x)[:10] for x in df["day"]})
        print(f"  {sym} {name:6s} {len(df):>6} 行  {len(d):>3} 天  {dt:>5}s  "
              f"{str(df['day'].iloc[0])[:10]} → {str(df['day'].iloc[-1])[:10]}")
        RESULT[sym] = {"rows": len(df), "days": len(d), "elapsed": dt}
    except Exception as e:
        print(f"  {sym} {name:6s} 失败: {type(e).__name__}")
    time.sleep(1.2)

if lat:
    avg = sum(lat) / len(lat)
    print(f"\n平均单只耗时 {avg:.2f}s → 5000 只股票全量约 {avg * 5000 / 60:.0f} 分钟")

with open("minute_probe_sina.json", "w", encoding="utf-8") as f:
    json.dump(RESULT, f, ensure_ascii=False, indent=2)
print("已写入 minute_probe_sina.json")
