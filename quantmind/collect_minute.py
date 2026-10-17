# -*- coding: utf-8 -*-
"""分钟行情采集 CLI（5 分钟 / 30 分钟）。

默认采集 30 只核心标的（沪深 300 权重股），两档周期都采，写入 `quotation_minute`。
脚本**幂等**：重复运行只补新 bar，不会产生重复行。

用法：
  python collect_minute.py                          # 采默认 30 只，30m + 5m
  python collect_minute.py --period 30              # 只采 30 分钟
  python collect_minute.py --limit 5                # 只采前 5 只（快速演示）
  python collect_minute.py --codes 600519 000001    # 指定标的
  python collect_minute.py --stats                  # 只看库内现状，不采集
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 默认 30 只核心标的：行业分散的沪深权重股，覆盖消费/金融/科技/医药/能源/制造
DEFAULT_STOCKS = [
    ("600519", "贵州茅台", "食品饮料"),
    ("000858", "五粮液", "食品饮料"),
    ("601318", "中国平安", "非银金融"),
    ("600036", "招商银行", "银行"),
    ("601398", "工商银行", "银行"),
    ("600030", "中信证券", "非银金融"),
    ("300750", "宁德时代", "电力设备"),
    ("002594", "比亚迪", "汽车"),
    ("601012", "隆基绿能", "电力设备"),
    ("000333", "美的集团", "家用电器"),
    ("000651", "格力电器", "家用电器"),
    ("600276", "恒瑞医药", "医药生物"),
    ("300760", "迈瑞医疗", "医药生物"),
    ("002415", "海康威视", "电子"),
    ("002475", "立讯精密", "电子"),
    ("000725", "京东方A", "电子"),
    ("600887", "伊利股份", "食品饮料"),
    ("601899", "紫金矿业", "有色金属"),
    ("600028", "中国石化", "石油石化"),
    ("601857", "中国石油", "石油石化"),
    ("600900", "长江电力", "公用事业"),
    ("601088", "中国神华", "煤炭"),
    ("600019", "宝钢股份", "钢铁"),
    ("601668", "中国建筑", "建筑装饰"),
    ("600050", "中国联通", "通信"),
    ("000001", "平安银行", "银行"),
    ("002304", "洋河股份", "食品饮料"),
    ("600585", "海螺水泥", "建筑材料"),
    ("601888", "中国中免", "商贸零售"),
    ("000002", "万科A", "房地产"),
]


def cmd_stats():
    from modules.minute_fetch import minute_stats
    rows = minute_stats()
    if not rows:
        print("库内暂无分钟数据。先运行：python collect_minute.py")
        return
    print(f"{'代码':<8}{'周期':>6}{'行数':>8}  {'起始':<20}{'结束':<20}")
    print("-" * 68)
    by_period = {}
    for r in rows:
        print(f"{r['stock_code']:<8}{r['period']:>5}m{r['rows']:>8}  "
              f"{str(r['start'])[:19]:<20}{str(r['end'])[:19]:<20}")
        bp = by_period.setdefault(r["period"], [0, 0])
        bp[0] += 1
        bp[1] += r["rows"]
    print("-" * 68)
    for p, (cnt, total) in sorted(by_period.items()):
        print(f"  {p} 分钟：{cnt} 只 / {total} 行")


def cmd_collect(args):
    from db import Base, engine
    from modules.minute_fetch import collect_and_save
    from db import StockBasic, get_session

    Base.metadata.create_all(engine)     # 保证 quotation_minute / minute_collect_log 存在

    stocks = DEFAULT_STOCKS
    if args.codes:
        name_map = {c: (n, ind) for c, n, ind in DEFAULT_STOCKS}
        unknown = [c for c in args.codes if c not in name_map]
        if unknown:
            print(f"[提示] 以下代码不在默认清单中，名称留空：{', '.join(unknown)}")
        stocks = [(c, name_map.get(c, (c, ""))[0], name_map.get(c, ("", ""))[1])
                  for c in args.codes]
    if args.limit:
        stocks = stocks[:args.limit]

    periods = [int(p) for p in args.period.split(",") if p.strip()]

    print("=" * 78)
    print(f"分钟行情采集：{len(stocks)} 只 × {len(periods)} 档周期 "
          f"({', '.join(str(p) + 'm' for p in periods)})")
    print("源：新浪 money.finance（主）/ 腾讯 ifzq（备）")
    print("=" * 78)

    # 顺带把标的信息写进 stock_basic，前端下拉才有名字可用
    s = get_session()
    try:
        for code, name, ind in stocks:
            row = s.get(StockBasic, code)
            if row is None:
                s.add(StockBasic(stock_code=code, stock_name=name,
                                 market="SINA", industry=ind))
            elif not row.industry:
                row.industry = ind
        s.commit()
    finally:
        s.close()

    t_all = time.time()
    summary = []
    for p in periods:
        print(f"\n--- {p} 分钟 ---")
        print(f"{'代码':<8}{'名称':<10}{'新增':>7}{'已存在':>8}{'源':>14}{'耗时':>8}")
        print("-" * 62)
        ok, fail = 0, 0
        for idx, (code, name, _ind) in enumerate(stocks, 1):
            t0 = time.time()
            res = collect_and_save(code, p)
            el = time.time() - t0
            if not res["ok"]:
                fail += 1
                print(f"{code:<8}{name:<10}{'-':>7}{'-':>8}"
                      f"{str(res['msg'])[:13]:>14}{el:>7.2f}s")
                continue
            ok += 1
            summary.append({"code": code, "period": p, "rows": res["rows"],
                            "added": res["added"], "source": res["source"]})
            print(f"{code:<8}{name:<10}{res['added']:>7}{res['skipped']:>8}"
                  f"{res['source']:>14}{el:>7.2f}s")
            time.sleep(args.sleep)
        print("-" * 62)
        print(f"  成功 {ok} 只，失败 {fail} 只")

    print(f"\n总耗时 {time.time() - t_all:.1f}s")
    total_added = sum(x["added"] for x in summary)
    print(f"本次新增 {total_added} 行")
    print("\n库内现状：")
    cmd_stats()


def main():
    ap = argparse.ArgumentParser(description="采集 5/30 分钟行情写入 quotation_minute")
    ap.add_argument("--period", default="30,5", help="周期，逗号分隔，可选 5 / 30")
    ap.add_argument("--codes", nargs="*", help="指定股票代码（6 位）")
    ap.add_argument("--limit", type=int, default=0, help="只采前 N 只")
    ap.add_argument("--sleep", type=float, default=0.15, help="每只之间的间隔秒数（防限流）")
    ap.add_argument("--stats", action="store_true", help="只看库内现状")
    args = ap.parse_args()

    if args.stats:
        cmd_stats()
        return 0
    cmd_collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
