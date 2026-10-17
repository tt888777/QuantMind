# -*- coding: utf-8 -*-
"""东财 / 腾讯 分钟线重测（用 subprocess 调 curl，绕开 Python TLS 握手被掐断的问题）。

只读探测，不写数据库、不改项目代码。输出打印到标准输出。
"""
import json
import subprocess
import sys
import time

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

CURL = ["curl", "-s", "--noproxy", "*", "--max-time", "20",
        "-H", "User-Agent: " + UA,
        "-H", "Referer: https://quote.eastmoney.com/"]


def fetch(url):
    t0 = time.time()
    try:
        out = subprocess.run(CURL + [url], capture_output=True, timeout=25)
        return out.stdout.decode("utf-8", "replace"), time.time() - t0, out.returncode
    except Exception as e:
        return "", time.time() - t0, f"ERR:{e}"


EM = "https://push2his.eastmoney.com/api/qt/stock/kline/get"


def em_url(klt, lmt=50000, secid="1.600519", end="20500101"):
    return (f"{EM}?secid={secid}&fields1=f1,f2,f3,f4,f5,f6"
            f"&fields2=f51,f52,f53,f54,f55,f56&klt={klt}&fqt=1&lmt={lmt}&end={end}")


def parse_em(txt):
    try:
        j = json.loads(txt)
    except Exception:
        return None
    d = j.get("data") or {}
    return d.get("klines") or []


def report(label, kl, el, extra=""):
    if kl is None:
        print(f"  {label:<26} 解析失败 {extra}")
        return
    days = len(set(k[:10] for k in kl))
    span = f"{kl[0][:19]} ~ {kl[-1][:19]}" if kl else "-"
    print(f"  {label:<26} {len(kl):>6} 条 / {days:>5} 交易日  {el:.2f}s  {span}  {extra}")


print("=" * 108)
print("【1】东财 push2his 各周期（lmt=50000，curl 直连 + 浏览器 UA）")
for klt, nm in ((1, "1分钟"), (5, "5分钟"), (15, "15分钟"),
                (30, "30分钟"), (60, "60分钟"), (101, "日线")):
    txt, el, rc = fetch(em_url(klt))
    kl = parse_em(txt)
    report(f"klt={klt} {nm}", kl, el, f"rc={rc} bytes={len(txt)}")
    time.sleep(0.8)

print("=" * 108)
print("【2】东财 lmt 上限试探（klt=5）")
for lmt in (1000, 5000, 10000, 30000, 50000, 100000):
    txt, el, rc = fetch(em_url(5, lmt=lmt))
    kl = parse_em(txt)
    report(f"lmt={lmt}", kl, el, f"rc={rc} bytes={len(txt)}")
    time.sleep(0.8)

print("=" * 108)
print("【3】东财多标的耗时（klt=5, lmt=50000）")
for secid, nm in (("1.600519", "贵州茅台"), ("1.601318", "中国平安"),
                  ("0.000001", "平安银行"), ("0.300750", "宁德时代"),
                  ("1.000001", "上证指数")):
    txt, el, rc = fetch(em_url(5, secid=secid))
    kl = parse_em(txt)
    report(nm, kl, el, f"rc={rc}")
    time.sleep(0.8)

print("=" * 108)
print("【4】腾讯 ifzq.gtimg.cn 各周期（mkline）")
TX = "https://ifzq.gtimg.cn/appstock/app/kline/mkline"
for per in ("m1", "m5", "m15", "m30", "m60"):
    txt, el, rc = fetch(f"{TX}?param=sh600519,{per},&count=5000")
    try:
        d = (json.loads(txt).get("data") or {}).get("sh600519") or {}
        arr = d.get(per) or []
    except Exception:
        arr = []
    span = f"{arr[0][0]} ~ {arr[-1][0]}" if arr else "-"
    days = len(set(a[0][:8] for a in arr))
    print(f"  {per:<8} {len(arr):>5} 条 / {days:>4} 交易日  {el:.2f}s  {span}")
    time.sleep(0.5)

print("=" * 108)
print("【5】结论汇总")
print("  （数据见上，单位：条 / 交易日 / 秒）")
