# -*- coding: utf-8 -*-
"""分钟级回测引擎（5 分钟 / 30 分钟）。

与日线 ``backtest.run_backtest`` 的关系：
- **信号层完全复用** ``backtest.build_signals``（dual_ma / momentum / boll），
  这样分钟线与日线的策略语义一致，结果可横向对比；
- **成本层重新实现**：分钟级交易频率高，日线那套「佣金按比例、无最低 5 元」的算法
  会严重低估成本（详见 docs/minute-data-feasibility.md 第六点五节）。

A 股真实成本模型（本模块全部计入）：
    佣金      万三，**单笔最低 5 元**（小资金高频时这是主要成本）
    印花税    千一，**仅卖出**
    过户费    十万分之一，买卖双向
    滑点      万二（分钟级比日线更差）

指标口径注意：
- **年化收益按 bar 频率换算**，不是按 252 天。30 分钟线一天 8 根，5 分钟线一天 48 根，
  直接套 252 会把年化算成天文数字。
- **夏普同样按 bar 频率年化**，用 `sqrt(bars_per_year)`。
"""
import math

import pandas as pd

from modules.backtest import build_signals, STRATEGIES
from modules.minute_fetch import load_minute

# ---- A 股真实成本（与 docs/minute-data-feasibility.md 一致）----
COMMISSION_RATE = 0.0003      # 万三
MIN_COMMISSION = 5.0          # 单笔最低 5 元
STAMP_TAX = 0.001             # 千一，仅卖出
TRANSFER_FEE = 0.00001        # 十万分之一，双向
SLIPPAGE = 0.0002             # 万二

# 每个交易日有多少根 bar（A 股一天 4 小时 = 240 分钟）
BARS_PER_DAY = {5: 48, 30: 8, 15: 16, 60: 4}
TRADING_DAYS_PER_YEAR = 252


def _commission(amount):
    """佣金：按比例但不少于 5 元。"""
    return max(amount * COMMISSION_RATE, MIN_COMMISSION)


def bars_per_year(period):
    return BARS_PER_DAY.get(period, 8) * TRADING_DAYS_PER_YEAR


def run_minute_backtest(stock_code, period=30, strategy="dual_ma", short=5, long_=20,
                        init_capital=100000.0, start_time=None, end_time=None,
                        lot_size=100):
    """单标的分钟级回测。

    ``lot_size`` 是**每手股数**，默认 **100**（A 股一手 100 股），这是真实交易约束。

    ⚠️ 由此会产生一个必须理解的现象：**高价股在小资金下根本买不起**。
    例：茅台 30 分钟线价格区间 1165~1852 元，一手就要 11.6 万~18.5 万，
    10 万本金**一轮都做不了**，回测结果为全 0（``skipped_no_capital`` 会 > 0）。
    这不是 bug，是 A 股现实。要看策略本身的表现，请二选一：
      - 调大 ``init_capital``（如 50 万），或
      - 传 ``lot_size=1``（模拟"可买零股"，仅用于策略评估，不代表真实可执行）

    返回 dict；数据不足时返回 ``None``。
    """
    df = load_minute(stock_code, period, start_time=start_time, end_time=end_time)
    if df.empty:
        return None
    df = df.reset_index(drop=True)
    if len(df) < long_ + 5:
        return None

    signals = build_signals(df, strategy=strategy, short=short, long_=long_)

    cash = float(init_capital)
    position = 0
    entry = None                   # 当前持仓对应的买入记录，用于配对单笔盈亏
    entry_bar = 0
    trades = []
    equity_curve = []
    fees_total = 0.0
    min_commission_hits = 0        # 触发「最低 5 元」的次数 —— 衡量小资金劣势
    skipped_no_capital = 0         # 因买不起一手而放弃的买入信号数

    for i in range(1, len(df)):
        row = df.iloc[i]
        price = float(row["close"])
        sig = int(signals.iloc[i])
        tt = row["time"]

        if sig == 1 and position == 0:
            buy_price = price * (1 + SLIPPAGE)
            # 🔴 必须给费用预留余量，否则永远买不进：
            # 若直接按 cash/price 满仓算股数，再用 amount+fee<=cash 校验，
            # 因为 amount 已吃掉几乎全部现金，任何费用都会让校验失败 →
            # 每根买入信号都被判「买不起」，回测静默返回全 0。
            # 真实交易者下单时同样会留出手续费，这里按「含费后可用现金」反推股数。
            eff_cash = cash / (1 + COMMISSION_RATE + TRANSFER_FEE)
            shares = int(eff_cash / buy_price / lot_size) * lot_size
            if shares <= 0:
                # 现金不够买一手：真实的 A 股约束，必须如实记录，不能偷偷按零股成交
                skipped_no_capital += 1
            else:
                amount = shares * buy_price
                comm = _commission(amount)
                if comm == MIN_COMMISSION:
                    min_commission_hits += 1
                fee = comm + amount * TRANSFER_FEE
                if amount + fee <= cash:
                    cash -= amount + fee
                    fees_total += fee
                    position = shares
                    rec = {"time": str(tt)[:19], "dir": "BUY",
                           "price": round(buy_price, 3),
                           "shares": shares, "fee": round(fee, 2),
                           "amount": round(amount, 2)}
                    trades.append(rec)
                    entry, entry_bar = rec, i
                else:
                    # 含费后现金不够（临界情况），同样按"买不起"处理
                    skipped_no_capital += 1

        elif sig == -1 and position > 0:
            sell_price = price * (1 - SLIPPAGE)
            amount = position * sell_price
            comm = _commission(amount)
            if comm == MIN_COMMISSION:
                min_commission_hits += 1
            fee = comm + amount * STAMP_TAX + amount * TRANSFER_FEE
            cash += amount - fee
            fees_total += fee
            rec = {"time": str(tt)[:19], "dir": "SELL", "price": round(sell_price, 3),
                   "shares": position, "fee": round(fee, 2),
                   "amount": round(amount, 2), "close_cycle": True}
            if entry:
                # 单笔盈亏 = 卖出净额 - 买入总成本（含双边费用），才是真正落袋的钱。
                # 只算价差会把费用漏掉，分钟线上费用恰恰是决定性因素。
                cost = entry["price"] * entry["shares"] + entry["fee"]
                pnl = (amount - fee) - cost
                rec.update({"pnl": round(pnl, 2),
                            "pnl_pct": round(pnl / cost * 100, 2) if cost else 0.0,
                            "hold_bars": i - entry_bar,
                            "entry_time": str(entry["time"])[:19],
                            "entry_price": entry["price"]})
            trades.append(rec)
            position = 0
            entry = None

        equity_curve.append({"time": tt, "equity": cash + position * price})

    final_price = float(df.iloc[-1]["close"])
    final_equity = cash + position * final_price
    total_return = (final_equity - init_capital) / init_capital
    # 扣费前收益：把已付费用加回去，得到"假设零成本"的结果
    return_before_fee = (final_equity + fees_total - init_capital) / init_capital
    fee_drag = return_before_fee - total_return

    n_bars = len(df)
    bpy = bars_per_year(period)
    years = n_bars / bpy
    annual = ((1 + total_return) ** (1 / years) - 1) if years > 0 and total_return > -1 else -1.0

    eq = pd.DataFrame(equity_curve)
    eq["peak"] = eq["equity"].cummax()
    eq["dd"] = (eq["equity"] - eq["peak"]) / eq["peak"]
    max_dd = float(eq["dd"].min()) if len(eq) else 0.0

    rets = eq["equity"].pct_change().dropna()
    if len(rets) > 1 and rets.std() > 0:
        sharpe = float(rets.mean() / rets.std() * math.sqrt(bpy))
    else:
        sharpe = 0.0

    # 交易周期配对：BUY 开启，带 close_cycle 的 SELL 结束
    cycles, cur = [], None
    for t in trades:
        if t["dir"] == "BUY":
            cur = t
        elif t["dir"] == "SELL" and cur is not None and t.get("close_cycle"):
            cycles.append((t["price"] - cur["price"]) / cur["price"])
            cur = None
    wins = sum(1 for r in cycles if r > 0)
    win_rate = wins / len(cycles) if cycles else 0.0

    days = df["time"].dt.normalize().nunique()
    return {
        "stock_code": str(stock_code),
        "period": period,
        "strategy": strategy,
        "strategy_name": STRATEGIES.get(strategy, strategy),
        "short": short, "long": long_,
        "bars": n_bars,
        "days": days,
        "round_trips": len(cycles),
        "trade_count": len(trades),
        "total_return": round(total_return * 100, 2),
        "return_before_fee": round(return_before_fee * 100, 2),
        # 成本损耗：扣费前后收益之差，单位百分点。高频策略最该看的指标。
        "fee_drag": round(fee_drag * 100, 2),
        "annual_return": round(annual * 100, 2),
        "max_drawdown": round(abs(max_dd) * 100, 2),
        "win_rate": round(win_rate * 100, 2),
        "sharpe": round(sharpe, 2),
        "final_equity": round(final_equity, 2),
        "fees_total": round(fees_total, 2),
        "min_commission_hits": min_commission_hits,
        "skipped_no_capital": skipped_no_capital,
        "lot_size": lot_size,
        "avg_trades_per_day": round(len(cycles) / days, 2) if days else 0,
        "start_time": str(df["time"].iloc[0]),
        "end_time": str(df["time"].iloc[-1]),
        # 期末仍持仓时明细里没有对应卖出，单独标出来，避免"买入多于卖出"被当成漏记
        "open_position": ({"shares": position, "entry_time": str(entry["time"])[:19],
                           "entry_price": entry["price"]}
                          if position > 0 and entry else None),
        "trades": trades,
        "equity_curve": equity_curve,
    }


def compare_strategies(stock_code, period=30, combos=None, **kw):
    """同一标的上跑多组策略，返回结果列表（便于前端做对比表）。"""
    combos = combos or [
        ("dual_ma", 5, 20),
        ("momentum", 5, 20),
        ("boll", 5, 20),
        ("dual_ma", 3, 8),
        ("dual_ma", 10, 30),
    ]
    out = []
    for strat, s, l in combos:
        r = run_minute_backtest(stock_code, period=period, strategy=strat,
                                short=s, long_=l, **kw)
        if r:
            out.append(r)
    return out
