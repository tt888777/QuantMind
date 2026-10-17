# -*- coding: utf-8 -*-
"""实测 Ashare（mpquant/Ashare 单文件库）到底能拿多久的分钟数据。

Ashare 只用两个上游：
  - 新浪 money.finance.sina.com.cn  scale=周期(分钟) & datalen=N
  - 腾讯 ifzq.gtimg.cn              mkline param=code,mX,,count

本脚本直接复用官方源码 `_ashare_src.py` 里的函数，不修改其逻辑，
分别测试：① Ashare 默认行为 ② 强制新浪 ③ 强制腾讯 ④ 大 count 下的真实上限。
"""
import sys
import time

import pandas as pd

sys.path.insert(0, ".")
import importlib.util

spec = importlib.util.spec_from_file_location("Ashare", "_ashare_src.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)

CODE = "sh600519"
FREQS = [("1m", "1分钟"), ("5m", "5分钟"), ("15m", "15分钟"),
         ("30m", "30分钟"), ("60m", "60分钟")]


def desc(df):
    if df is None or len(df) == 0:
        return None
    idx = pd.to_datetime(df.index)
    days = idx.normalize().nunique()
    return {
        "rows": len(df),
        "days": days,
        "start": str(idx[0]),
        "end": str(idx[-1]),
        "cols": list(df.columns),
    }


def show(tag, r, el):
    if r is None:
        print(f"  {tag:<22} 失败")
        return
    print(f"  {tag:<22} {r['rows']:>5} 行 / {r['days']:>4} 交易日  {el:>5.2f}s  "
          f"{r['start']} ~ {r['end']}")


print("=" * 110)
print(f"Ashare 实测  标的 {CODE}")
print("=" * 110)

print("\n【1】Ashare 默认 get_price(frequency=..., count=50000)")
print("     （源码逻辑：分钟线先新浪，失败才腾讯）")
for fq, nm in FREQS:
    t0 = time.time()
    try:
        df = A.get_price(CODE, frequency=fq, count=50000)
        show(f"get_price {nm}", desc(df), time.time() - t0)
    except Exception as e:
        print(f"  get_price {nm:<16} ✗ {type(e).__name__}: {str(e)[:80]}")
    time.sleep(0.8)

print("\n【2】强制走新浪 get_price_sina(datalen 很大)")
print("     注意：Ashare 把 count 直接当 datalen 传给新浪")
for fq, nm in FREQS:
    t0 = time.time()
    try:
        df = A.get_price_sina(CODE, count=50000, frequency=fq)
        show(f"sina {nm}", desc(df), time.time() - t0)
    except Exception as e:
        print(f"  sina {nm:<21} ✗ {type(e).__name__}: {str(e)[:80]}")
    time.sleep(0.8)

print("\n【3】强制走腾讯 get_price_min_tx(count 很大)")
for fq, nm in FREQS:
    t0 = time.time()
    try:
        df = A.get_price_min_tx(CODE, count=50000, frequency=fq)
        show(f"tencent {nm}", desc(df), time.time() - t0)
    except Exception as e:
        print(f"  tencent {nm:<18} ✗ {type(e).__name__}: {str(e)[:80]}")
    time.sleep(0.8)

print("\n【4】新浪 datalen 逐档（找真实上限，30m）")
for n in (1000, 1970, 2000, 3000, 5000, 10000, 50000, 100000):
    t0 = time.time()
    try:
        df = A.get_price_sina(CODE, count=n, frequency="30m")
        el = time.time() - t0
        d = desc(df)
        print(f"  datalen={n:<7} -> {d['rows'] if d else 0:>6} 行 / "
              f"{d['days'] if d else 0:>5} 交易日  {el:.2f}s"
              f"  {d['start'] + ' ~ ' + d['end'] if d else '-'}")
    except Exception as e:
        print(f"  datalen={n:<7} -> ✗ {type(e).__name__}: {str(e)[:70]}")
    time.sleep(0.8)

print("\n【5】新浪 datalen 逐档（5m）")
for n in (1970, 3000, 10000, 50000):
    t0 = time.time()
    try:
        df = A.get_price_sina(CODE, count=n, frequency="5m")
        el = time.time() - t0
        d = desc(df)
        print(f"  datalen={n:<7} -> {d['rows'] if d else 0:>6} 行 / "
              f"{d['days'] if d else 0:>5} 交易日  {el:.2f}s"
              f"  {d['start'] + ' ~ ' + d['end'] if d else '-'}")
    except Exception as e:
        print(f"  datalen={n:<7} -> ✗ {type(e).__name__}: {str(e)[:70]}")
    time.sleep(0.8)

print("\n【6】日线 / 周线 / 月线能取多久（Ashare 强项）")
for fq, nm in (("1d", "日线"), ("1w", "周线"), ("1M", "月线")):
    t0 = time.time()
    try:
        df = A.get_price(CODE, frequency=fq, count=50000)
        show(f"get_price {nm}", desc(df), time.time() - t0)
    except Exception as e:
        print(f"  {nm:<22} ✗ {type(e).__name__}: {str(e)[:80]}")
    time.sleep(0.8)

print("\n【7】新股/短历史标的对照（sz301369 近两年上市）")
for fq, nm in (("5m", "5分钟"), ("30m", "30分钟"), ("1d", "日线")):
    t0 = time.time()
    try:
        df = A.get_price("sz301369", frequency=fq, count=50000)
        show(f"sz301369 {nm}", desc(df), time.time() - t0)
    except Exception as e:
        print(f"  sz301369 {nm:<12} ✗ {type(e).__name__}: {str(e)[:80]}")
    time.sleep(0.8)

print("\n" + "=" * 110)
print("实测结束")
