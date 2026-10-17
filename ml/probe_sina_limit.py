# -*- coding: utf-8 -*-
"""定位新浪分钟接口的真实上限：到底是 datalen 限制，还是我此前的调用方式不对。

对比变量：
  A. 官方接口 money.finance.sina.com.cn + json_v2 + scale&datalen
  B. 我此前用的 quotes.sina.cn/jsonp_v2 + scale&datalen
  C. ma 参数有无影响
  D. http vs https
"""
import subprocess
import time

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


def curl(url, timeout=25):
    t0 = time.time()
    try:
        r = subprocess.run(["curl", "-s", "--noproxy", "*", "--max-time", str(timeout),
                            "-H", "User-Agent: " + UA, "-H", "Referer: https://finance.sina.com.cn/",
                            url], capture_output=True, timeout=timeout + 5)
        txt = r.stdout.decode("utf-8", "replace")
        return txt, time.time() - t0, r.returncode
    except Exception as e:
        return "", time.time() - t0, f"ERR:{e}"


def count_rows(txt):
    import json
    txt = txt.strip()
    if txt.startswith("/*"):
        txt = txt[txt.index("*/") + 2:]
    txt = txt.strip()
    if txt.startswith("x("):
        txt = txt[2:-2] if txt.endswith(");") else txt[2:]
    try:
        d = json.loads(txt)
    except Exception:
        return None
    return d if isinstance(d, list) else None


def report(label, rows, el):
    if rows is None:
        print(f"  {label:<62} 解析失败/空  {el:.2f}s")
        return
    if not rows:
        print(f"  {label:<62} 0 行  {el:.2f}s")
        return
    days = len(set(str(r.get("day", ""))[:10] for r in rows))
    print(f"  {label:<62} {len(rows):>5} 行 / {days:>4} 天  {el:.2f}s  "
          f"{str(rows[0].get('day'))[:16]} ~ {str(rows[-1].get('day'))[:16]}")


print("=" * 116)
print("A. 官方接口 money.finance.sina.com.cn  json_v2  （Ashare 用的就是这个）")
for scale in (5, 15, 30, 60):
    for dl in (5000, 10000):
        url = (f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
               f"CN_MarketData.getKLineData?symbol=sh600519&scale={scale}&ma=5&datalen={dl}")
        txt, el, rc = curl(url)
        report(f"scale={scale:<3} datalen={dl:<6} ma=5", count_rows(txt), el)
        time.sleep(0.4)

print("\n" + "=" * 116)
print("B. 我此前用的 quotes.sina.cn / jsonp_v2 （对比，看是不是接口版本差异）")
for scale in (5, 30):
    url = (f"https://quotes.sina.cn/cn/api/jsonp_v2.php/x/CN_MarketDataService.getKLineData"
           f"?symbol=sh600519&scale={scale}&ma=no&datalen=10000")
    txt, el, rc = curl(url)
    report(f"jsonp_v2 scale={scale} datalen=10000", count_rows(txt), el)
    time.sleep(0.4)

print("\n" + "=" * 116)
print("C. ma 参数影响（ma=no vs ma=5 vs 不传）")
for ma in ("ma=5", "ma=no", ""):
    url = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"CN_MarketData.getKLineData?symbol=sh600519&scale=30&{ma}&datalen=10000"
           if ma else
           "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           "CN_MarketData.getKLineData?symbol=sh600519&scale=30&datalen=10000")
    txt, el, rc = curl(url)
    report(f"scale=30  {ma or '(不传)'}", count_rows(txt), el)
    time.sleep(0.4)

print("\n" + "=" * 116)
print("D. http vs https")
for scheme in ("http", "https"):
    url = (f"{scheme}://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"CN_MarketData.getKLineData?symbol=sh600519&scale=30&ma=5&datalen=10000")
    txt, el, rc = curl(url)
    report(f"{scheme} scale=30 datalen=10000", count_rows(txt), el)
    time.sleep(0.4)

print("\n" + "=" * 116)
print("E. 极限试探：datalen 再往上（看 5001 是不是硬顶）")
for dl in (5001, 6000, 8000, 20000, 50000, 100000):
    url = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"CN_MarketData.getKLineData?symbol=sh600519&scale=30&ma=5&datalen={dl}")
    txt, el, rc = curl(url)
    report(f"datalen={dl}", count_rows(txt), el)
    time.sleep(0.4)

print("\n" + "=" * 116)
print("F. 日线（scale=240）极限")
for dl in (5000, 10000, 30000):
    url = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"CN_MarketData.getKLineData?symbol=sh600519&scale=240&ma=5&datalen={dl}")
    txt, el, rc = curl(url)
    report(f"scale=240 datalen={dl}", count_rows(txt), el)
    time.sleep(0.4)
