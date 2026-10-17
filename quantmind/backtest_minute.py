# -*- coding: utf-8 -*-
"""分钟策略横评 CLI（批量回测 + 结果汇总）。

对库内所有已采集的标的，遍历策略组合跑分钟级回测，输出三种视角：

  1. **标的榜**：每只标的的最佳策略，按扣费后收益排序 —— 回答"哪些标的适合分钟策略"
  2. **策略榜**：每个策略在全部标的上的平均表现 —— 回答"哪个策略在分钟线上有效"
  3. **明细**：标的 × 策略 全量结果 —— 供导出 CSV 做进一步分析

⚠️ 口径说明（务必理解，否则会误读数字）：
  - 默认 ``--lot-size 100``（真实 A 股一手 100 股）。高价股在小资金下买不起，
    ``skipped`` 列会显示有多少买入信号因资金不足被放弃。
  - ``--lot-size 1`` 是**模拟零股**，仅用于纯策略评估，不代表真实可执行。
  - ``未扣费`` 列是假设零交易成本的结果，``损耗`` = 未扣费 − 实际，衡量成本侵蚀。
    **分钟策略的关键就在于：未扣费赚钱、扣完变亏。**

用法：
  python backtest_minute.py                          # 30m，默认 100 万本金，一手 100 股
  python backtest_minute.py --period 5               # 5 分钟档
  python backtest_minute.py --capital 300000         # 30 万本金
  python backtest_minute.py --lot-size 1             # 零股模拟（纯策略评估）
  python backtest_minute.py --strategy boll          # 只跑布林带
  python backtest_minute.py --csv out/minutes.csv    # 导出全量明细
  python backtest_minute.py --top 15                 # 只显示前 15 名
"""
import argparse
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 默认策略组合。刻意包含"高频"（3/8）与"低频"（10/30）两端，
# 因为分钟策略的核心矛盾就是「信号太密→成本吃光收益」。
DEFAULT_COMBOS = [
    ("boll", 5, 20),
    ("dual_ma", 5, 20),
    ("dual_ma", 10, 30),
    ("dual_ma", 3, 8),
    ("momentum", 5, 20),
]


def _stock_names():
    """从 stock_basic 取代码→名称，取不到就退回代码本身。"""
    try:
        from db import StockBasic, get_session
        s = get_session()
        try:
            return {r.stock_code: (r.stock_name or r.stock_code)
                    for r in s.query(StockBasic).all()}
        finally:
            s.close()
    except Exception:                                          # noqa: BLE001
        return {}


def _available_codes(period):
    """库内该周期有哪些标的（按代码排序，保证输出稳定）。"""
    from modules.minute_fetch import minute_stats
    return sorted({r["stock_code"] for r in minute_stats() if r["period"] == period})


def run_matrix(period=30, combos=None, capital=1000000.0, lot_size=100,
               codes=None, start_time=None, end_time=None, verbose=True):
    """跑完整矩阵，返回 list[dict]（每个元素是一次回测的精简结果）。"""
    from modules.minute_backtest import run_minute_backtest

    combos = combos or DEFAULT_COMBOS
    names = _stock_names()
    codes = codes or _available_codes(period)
    if not codes:
        return []

    results = []
    total = len(codes) * len(combos)
    done = 0
    t0 = time.time()

    for code in codes:
        for strat, s_, l_ in combos:
            r = run_minute_backtest(code, period=period, strategy=strat,
                                    short=s_, long_=l_, init_capital=capital,
                                    start_time=start_time, end_time=end_time,
                                    lot_size=lot_size)
            done += 1
            if verbose and done % 20 == 0 or done == total:
                print(f"  进度 {done}/{total}  ({time.time() - t0:.1f}s)", flush=True)
            if not r:
                continue
            results.append({
                "code": r["stock_code"],
                "name": names.get(r["stock_code"], r["stock_code"]),
                "strategy": r["strategy"],
                "strategy_name": r["strategy_name"],
                "params": f"{s_}/{l_}",
                "bars": r["bars"],
                "days": r["days"],
                "round_trips": r["round_trips"],
                "trades_per_day": r["avg_trades_per_day"],
                "total_return": r["total_return"],
                "return_before_fee": r["return_before_fee"],
                "fee_drag": r["fee_drag"],
                "annual_return": r["annual_return"],
                "max_drawdown": r["max_drawdown"],
                "win_rate": r["win_rate"],
                "sharpe": r["sharpe"],
                "fees_total": r["fees_total"],
                "min_commission_hits": r["min_commission_hits"],
                "skipped": r["skipped_no_capital"],
                "start_time": r["start_time"],
                "end_time": r["end_time"],
            })
    return results


def print_stock_board(results, top=0, period=30):
    """视角一：每只标的的最佳策略（按扣费后收益降序）。"""
    best = {}
    for r in results:
        cur = best.get(r["code"])
        if cur is None or r["total_return"] > cur["total_return"]:
            best[r["code"]] = r
    rows = sorted(best.values(), key=lambda x: x["total_return"], reverse=True)
    if top:
        rows = rows[:top]

    print("\n" + "=" * 108)
    print(f"【视角一】各标的最佳策略（{period} 分钟线，按扣费后收益排序）")
    print("=" * 108)
    print(f"{'代码':<8}{'名称':<10}{'最佳策略':<16}{'轮次':>5}{'未扣费%':>9}"
          f"{'实际%':>8}{'损耗%':>8}{'年化%':>8}{'回撤%':>8}{'夏普':>7}{'买不起':>7}")
    print("-" * 108)
    for r in rows:
        flag = "" if r["total_return"] > 0 else ""
        print(f"{r['code']:<8}{r['name']:<10}"
              f"{(r['strategy_name'] + ' ' + r['params']):<16}"
              f"{r['round_trips']:>5}{r['return_before_fee']:>9.2f}"
              f"{r['total_return']:>8.2f}{r['fee_drag']:>8.2f}"
              f"{r['annual_return']:>8.2f}{r['max_drawdown']:>8.2f}"
              f"{r['sharpe']:>7.2f}{r['skipped']:>7}{flag}")
    print("-" * 108)
    pos = sum(1 for r in rows if r["total_return"] > 0)
    print(f"  扣费后正收益：{pos}/{len(rows)} 只")
    return rows


def print_strategy_board(results, period=30):
    """视角二：每个策略在全部标的上的平均表现。"""
    agg = {}
    for r in results:
        k = (r["strategy"], r["params"])
        a = agg.setdefault(k, {"n": 0, "ret": 0.0, "before": 0.0, "drag": 0.0,
                               "ann": 0.0, "rt": 0.0, "sharpe": 0.0, "pos": 0,
                               "dd": 0.0})
        a["n"] += 1
        a["ret"] += r["total_return"]
        a["before"] += r["return_before_fee"]
        a["drag"] += r["fee_drag"]
        a["ann"] += r["annual_return"]
        a["rt"] += r["round_trips"]
        a["sharpe"] += r["sharpe"]
        a["dd"] += r["max_drawdown"]
        a["pos"] += 1 if r["total_return"] > 0 else 0

    rows = sorted(agg.items(), key=lambda kv: kv[1]["ret"] / kv[1]["n"], reverse=True)
    print("\n" + "=" * 108)
    print(f"【视角二】策略横评（{period} 分钟线，全部标的平均）")
    print("=" * 108)
    print(f"{'策略':<24}{'样本':>5}{'均轮次':>8}{'均未扣费%':>11}"
          f"{'均实际%':>10}{'均损耗%':>10}{'均年化%':>10}{'正收益比':>10}{'均夏普':>9}")
    print("-" * 108)
    for (strat, params), a in rows:
        name = f"{strat} {params}"
        print(f"{name:<24}{a['n']:>5}{a['rt'] / a['n']:>8.1f}"
              f"{a['before'] / a['n']:>11.2f}{a['ret'] / a['n']:>10.2f}"
              f"{a['drag'] / a['n']:>10.2f}{a['ann'] / a['n']:>10.2f}"
              f"{a['pos']}/{a['n']}".rjust(10) +
              f"{a['sharpe'] / a['n']:>9.2f}")
    print("-" * 108)
    return rows


def print_insights(results, period=30):
    """视角三：把最关键的几条事实算出来，避免只给一堆数字。"""
    if not results:
        return
    n = len(results)
    pos = [r for r in results if r["total_return"] > 0]
    before_pos = [r for r in results if r["return_before_fee"] > 0]

    print("\n" + "=" * 108)
    print("【关键结论】")
    print("=" * 108)
    print(f"  回测组合总数：{n}")
    print(f"  未扣费正收益：{len(before_pos)}/{n} ({len(before_pos) / n * 100:.1f}%)")
    print(f"  扣费后正收益：{len(pos)}/{n} ({len(pos) / n * 100:.1f}%)")

    if before_pos:
        avg_before = sum(r["return_before_fee"] for r in before_pos) / len(before_pos)
        avg_drag = sum(r["fee_drag"] for r in before_pos) / len(before_pos)
        print(f"  未扣费为正的组合：均未扣费 +{avg_before:.2f}%，均被成本吃掉 {avg_drag:.2f} 个百分点"
              f" → 扣完剩 +{avg_before - avg_drag:.2f}%")

    # 轮次与损耗的相关性：这是分钟策略最硬的一条规律
    low = [r for r in results if r["round_trips"] <= 40]
    mid = [r for r in results if 40 < r["round_trips"] <= 120]
    high = [r for r in results if r["round_trips"] > 120]
    print("\n  轮次 vs 成本损耗（损耗随交易频率上升，几乎线性）：")
    for label, grp in (("≤40 轮", low), ("41~120 轮", mid), (">120 轮", high)):
        if grp:
            print(f"    {label:<12} 样本 {len(grp):>3}  均损耗 "
                  f"{sum(r['fee_drag'] for r in grp) / len(grp):>6.2f}%  "
                  f"均实际 {sum(r['total_return'] for r in grp) / len(grp):>7.2f}%")

    if pos:
        best = max(pos, key=lambda r: r["total_return"])
        print(f"\n  最佳组合：{best['name']}({best['code']}) "
              f"{best['strategy_name']} {best['params']} → "
              f"实际 +{best['total_return']:.2f}%，年化 {best['annual_return']:.2f}%，"
              f"夏普 {best['sharpe']:.2f}，{best['round_trips']} 轮")
    skipped = [r for r in results if r["skipped"] > 0]
    if skipped:
        print(f"\n  ⚠️ {len(skipped)} 个组合出现「买不起一手」："
              f"合计放弃 {sum(r['skipped'] for r in skipped)} 次买入信号")
        print("     → 这是真实 A 股约束（一手 100 股）。要看纯策略表现请加大本金或 --lot-size 1")


def export_csv(results, path):
    if not results:
        print("无可导出数据")
        return
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fields = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\n明细已导出：{path}（{len(results)} 行）")


def main():
    ap = argparse.ArgumentParser(description="分钟策略批量横评")
    ap.add_argument("--period", type=int, default=30, choices=[5, 30])
    ap.add_argument("--capital", type=float, default=1000000.0, help="初始资金（默认 100 万）")
    ap.add_argument("--lot-size", type=int, default=100, help="每手股数（100=真实，1=零股模拟）")
    ap.add_argument("--strategy", help="只跑某个策略（dual_ma / boll / momentum）")
    ap.add_argument("--codes", nargs="*", help="指定标的")
    ap.add_argument("--start", help="开始时间，如 2025-01-01")
    ap.add_argument("--end", help="结束时间")
    ap.add_argument("--top", type=int, default=0, help="只显示前 N 名")
    ap.add_argument("--csv", help="导出全量明细到此 csv")
    args = ap.parse_args()

    combos = DEFAULT_COMBOS
    if args.strategy:
        combos = [c for c in DEFAULT_COMBOS if c[0] == args.strategy]
        if not combos:
            print(f"未知策略：{args.strategy}")
            return 1

    print("=" * 108)
    print(f"分钟策略横评  周期={args.period}m  本金={args.capital:,.0f}  "
          f"每手={args.lot_size} 股  策略组合={len(combos)}")
    if args.lot_size == 1:
        print("⚠️ --lot-size 1 为「零股模拟」，只评估策略本身，不代表真实可执行")
    print("=" * 108)

    t0 = time.time()
    results = run_matrix(period=args.period, combos=combos, capital=args.capital,
                         lot_size=args.lot_size, codes=args.codes,
                         start_time=args.start, end_time=args.end)
    if not results:
        print("没有回测结果。请先运行：python collect_minute.py")
        return 1
    print(f"\n矩阵完成：{len(results)} 个组合，耗时 {time.time() - t0:.1f}s")

    print_stock_board(results, top=args.top, period=args.period)
    print_strategy_board(results, period=args.period)
    print_insights(results, period=args.period)

    if args.csv:
        export_csv(results, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
