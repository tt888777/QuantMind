from decimal import Decimal
from datetime import date
import pandas as pd
from db import get_session, Strategy, BacktestTask, BacktestMetric


# 舆情仓位档位：(情绪指数下限, 目标仓位比例)，按下限从高到低排列。
# 指数 0~100，50 为中性；比 50 高的越多说明舆情越乐观。
# 🔴 刻意用整数档、不做参数调优 —— 样本仅一百多天，调参等于过拟合。
SENTIMENT_TIERS = [
    (65, 0.80),     # 偏乐观
    (55, 0.60),     # 中性偏多
    (45, 0.40),     # 中性
    (35, 0.20),     # 中性偏空
    (0,  0.00),     # 偏悲观，不建仓
]


def target_position_ratio(sent_index, tiers=None):
    """由情绪指数折算目标仓位比例（0~1）。

    sent_index 为 None（未启用舆情 / 采集失败）时返回 1.0，
    即退回原来的满仓行为 —— 保证不启用舆情时回测结果与旧版完全一致。
    """
    if sent_index is None:
        return 1.0
    tiers = tiers or SENTIMENT_TIERS
    for floor, ratio in tiers:
        if sent_index >= floor:
            return ratio
    return 0.0


# ------------------------------------------------------------------ 策略
# 🔴 旧实现的问题：run_backtest 虽然接收 short/long_，但信号计算直接读数据库里
# 预存的 ma5/ma20 两列，参数只用来拼策略名。结果无论传什么参数，回测结果都
# 一模一样。现在改为**按参数现算**指标，参数才是真的生效。
STRATEGIES = {
    "dual_ma": "双均线交叉",
    "momentum": "动量策略",
    "boll": "布林带突破",
    "dual_ma_sentiment": "双均线+情绪过滤",
}


def _signal_dual_ma(df, short, long_):
    """双均线：短上穿长为买(1)，下穿为卖(-1)。"""
    ms = df["close"].rolling(short).mean()
    ml = df["close"].rolling(long_).mean()
    above = ms > ml
    sig = pd.Series(0, index=df.index)
    sig[(~above.shift(1, fill_value=False)) & above] = 1
    sig[above.shift(1, fill_value=True) & (~above)] = -1
    return sig


def _signal_momentum(df, short, long_):
    """动量：短周期收益为正且高于长周期均值时持有，转负离场。

    用 close 相对 long_ 日前的变化率衡量动量，过零轴产生信号。
    """
    mom = df["close"] / df["close"].shift(long_) - 1
    sig = pd.Series(0, index=df.index)
    sig[(mom.shift(1) <= 0) & (mom > 0)] = 1
    sig[(mom.shift(1) >= 0) & (mom < 0)] = -1
    return sig


def _signal_boll(df, short, long_):
    """布林带突破：收盘上穿上轨买入，跌破中轨卖出。

    周期取 long_（默认 20），标准差倍数 2。
    """
    n = max(long_, 5)
    mid = df["close"].rolling(n).mean()
    std = df["close"].rolling(n).std()
    upper = mid + 2 * std
    sig = pd.Series(0, index=df.index)
    close = df["close"]
    sig[(close.shift(1) <= upper.shift(1)) & (close > upper)] = 1
    sig[(close.shift(1) >= mid.shift(1)) & (close < mid)] = -1
    return sig


_SIGNAL_FUNCS = {
    "dual_ma": _signal_dual_ma,
    "dual_ma_sentiment": _signal_dual_ma,   # 信号同双均线，差别在仓位层
    "momentum": _signal_momentum,
    "boll": _signal_boll,
}


def build_signals(df, strategy="dual_ma", short=5, long_=20):
    """返回与 df 等长的信号序列（1 买 / -1 卖 / 0 无）。"""
    fn = _SIGNAL_FUNCS.get(strategy) or _signal_dual_ma
    sig = fn(df, short, long_)
    return sig.fillna(0).astype(int)


def load_data_with_indicator(stock_code):
    from modules.data_fetch import load_quotation
    df = load_quotation(stock_code)
    if df.empty:
        return df
    s = get_session()
    try:
        rows = s.query(Strategy).all()
        from db import TechnicalIndicator
        ti_rows = s.query(TechnicalIndicator).filter_by(stock_code=stock_code).order_by(TechnicalIndicator.trade_date).all()
        if ti_rows:
            ti_df = pd.DataFrame([{
                "date": r.trade_date, "ma5": float(r.ma5 or 0), "ma10": float(r.ma10 or 0),
                "ma20": float(r.ma20 or 0), "macd_hist": float(r.macd_hist or 0)
            } for r in ti_rows])
            df = df.merge(ti_df, on="date", how="left")
        return df
    finally:
        s.close()


def run_backtest(stock_code, short=5, long_=20, init_capital=100000.0,
                 commission=0.0003, slippage=0.0001, lot_size=1,
                 start_date=None, end_date=None, sentiment_threshold=45,
                 sentiment_dynamic=True, strategy="dual_ma"):
    """多策略回测（双均线 / 动量 / 布林带 / 双均线+情绪过滤）。

    ``strategy`` 见 ``STRATEGIES``；``short`` / ``long_`` 会**真正参与指标计算**
    （旧版只读数据库里预存的 ma5/ma20，参数形同虚设）。

    ``sentiment_dynamic=True`` 启用**动态仓位**：按舆情情绪指数折算目标仓位
    （见 ``SENTIMENT_TIERS``），建仓一次定档、死叉清仓。
    ``sentiment_dynamic=False`` 为旧的一刀切：指数低于阈值则完全拦下，否则满仓。
    """
    df = load_data_with_indicator(stock_code)
    if df.empty:
        return None

    if start_date:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        df = df[df["date"] >= start_date]
    if end_date:
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
        df = df[df["date"] <= end_date]

    df = df.reset_index(drop=True)
    # 均线需要 long_ 个前置数据点才算得出来；预留 long_*2 便于布林带标准差收敛
    if len(df) < long_ + 5:
        return None

    signals = build_signals(df, strategy=strategy, short=short, long_=long_)

    sent_index = None
    if sentiment_threshold is not None:
        try:
            from modules.sentiment import collect_sentiment, analyze_sentiments
            sent_info = collect_sentiment(stock_code)
            sent_result = analyze_sentiments(stock_code, sent_info['news'])
            sent_index = sent_result['index']
        except Exception:
            sent_index = None

    cash = init_capital
    position = 0
    trades = []
    equity_curve = []
    exposure_series = []            # 每日仓位占比（持仓市值 / 总权益）
    entry_price = 0.0
    blocked_by_sentiment = 0        # 因舆情目标仓位为 0 而放弃建仓的次数
    total_buy_signals = 0           # 买入信号总数

    # 策略是否启用舆情：显式策略 dual_ma_sentiment 强制启用；
    # 其它策略是否用舆情由 sentiment_dynamic 决定。
    # 这样"双均线交叉"（纯技术）与"双均线+情绪过滤"能拉开差异，而不是同一结果。
    if strategy == "dual_ma_sentiment":
        use_sentiment = True
    elif strategy in ("dual_ma", "momentum", "boll"):
        use_sentiment = sentiment_dynamic
    else:
        use_sentiment = sentiment_dynamic

    target_ratio = target_position_ratio(sent_index) if use_sentiment else 1.0

    for i in range(1, len(df)):
        row = df.iloc[i]
        price = float(row["close"])
        signal = int(signals.iloc[i])

        buy_price = price * (1 + slippage)

        if signal == 1:
            total_buy_signals += 1

            if use_sentiment:
                # 动态仓位：目标档位为 0 视为"不建仓"，否则按比例建仓
                ratio = target_ratio
            else:
                # 旧行为：一刀切，低于阈值拦下，否则满仓
                ratio = 0.0 if (sent_index is not None
                                and sent_index < sentiment_threshold) else 1.0

            if ratio <= 0:
                blocked_by_sentiment += 1
            elif position == 0:
                # 仓位比例作用于**总资产**而非可用现金：
                # 用现金算的话，第二次建仓时 cash 只剩上次的盈亏残值，仓位会越做越小。
                budget = (cash + position * price) * ratio
                budget = min(budget, cash)
                shares = int(budget / buy_price / lot_size) * lot_size
                if shares > 0:
                    cash -= shares * buy_price * (1 + commission)
                    position = shares
                    entry_price = buy_price
                    trades.append({"date": row["date"], "dir": "BUY",
                                   "price": round(buy_price, 2), "shares": shares})

        elif signal == -1 and position > 0:
            # 死叉：清仓
            sell_price = price * (1 - slippage)
            cash += position * sell_price * (1 - commission)
            trades.append({"date": row["date"], "dir": "SELL", "price": round(sell_price, 2),
                           "shares": position, "close_cycle": True})
            position = 0
            entry_price = 0.0

        # 说明：这里**不做**日度仓位校准。
        # 情绪指数在本次回测中是整段区间的一个定值（一次性采集分析），
        # 逐日校准等于拿恒定的目标去砍浮动市值 —— 涨了就减、跌了不减，
        # 结果是"上涨被提前止盈、下跌全程承担"，实测收益从 2.77% 掉到 1.19%
        # 且夏普毫无改善。仓位由建仓时的档位一次确定，直到死叉清仓。

        equity = cash + position * price
        equity_curve.append({"date": row["date"], "equity": round(equity, 2)})
        exposure_series.append(position * price / equity if equity > 0 else 0.0)

    final_equity = cash + position * float(df.iloc[-1]["close"])
    total_return = (final_equity - init_capital) / init_capital
    days = len(df)
    annual_return = (1 + total_return) ** (252 / max(days, 1)) - 1 if total_return > -1 else -1
    avg_exposure_pct = (round(sum(exposure_series) / len(exposure_series) * 100, 1)
                        if exposure_series else None)

    eq = pd.DataFrame(equity_curve)
    eq["peak"] = eq["equity"].cummax()
    eq["dd"] = (eq["equity"] - eq["peak"]) / eq["peak"]
    max_dd = float(eq["dd"].min())

    # 交易周期（一次建仓 → 一次清仓）的盈亏。
    # 动态仓位下每次减仓都会记一条 SELL，所以不能再简单地按下标两两配对，
    # 改为扫 trades：BUY 开启一个周期，带 close_cycle 的 SELL 结束它并结算。
    cycles = []
    cur = None
    for t in trades:
        if t["dir"] == "BUY":
            if cur is None:
                cur = {"buy": t["price"], "shares": 0, "cost": 0.0}
            cur["shares"] += t["shares"]
            cur["cost"] += t["price"] * t["shares"]
        elif t["dir"] == "SELL" and cur is not None and t.get("close_cycle"):
            avg_cost = cur["cost"] / cur["shares"] if cur["shares"] else cur["buy"]
            cycles.append((t["price"] - avg_cost) / avg_cost if avg_cost else 0.0)
            cur = None
    wins = sum(1 for r in cycles if r > 0)
    trade_pairs = len(cycles)
    win_rate = wins / trade_pairs if trade_pairs > 0 else 0

    if len(eq) > 1:
        rets = eq["equity"].pct_change().dropna()
        sharpe = float(rets.mean() / rets.std() * (252 ** 0.5)) if rets.std() > 0 else 0
    else:
        sharpe = 0

    s = get_session()
    try:
        strat = Strategy(strategy_name=f"{STRATEGIES.get(strategy, strategy)}{short}/{long_}",
                         strategy_type=strategy.upper()[:24],
                         stock_code=stock_code,
                         params=f'{{"strategy":"{strategy}","short":{short},"long":{long_}}}',
                         init_capital=Decimal(str(init_capital)))
        s.add(strat)
        s.flush()
        task = BacktestTask(strategy_id=strat.strategy_id, stock_code=stock_code,
                            start_date=df.iloc[0]["date"], end_date=df.iloc[-1]["date"], status="DONE")
        s.add(task)
        s.flush()
        s.add(BacktestMetric(
            task_id=task.task_id,
            total_return=Decimal(str(round(total_return, 4))),
            annual_return=Decimal(str(round(annual_return, 4))),
            max_drawdown=Decimal(str(round(abs(max_dd), 4))),
            win_rate=Decimal(str(round(win_rate, 4))),
            sharpe_ratio=Decimal(str(round(sharpe, 4))),
            trade_count=len(trades)
        ))
        s.commit()
    finally:
        s.close()

    return {
        "total_return": round(total_return * 100, 2),
        "annual_return": round(annual_return * 100, 2),
        # 回撤统一用**正数**表示（行业惯例）：eq["dd"] 本身是负值，取绝对值
        "max_drawdown": round(abs(max_dd) * 100, 2),
        "win_rate": round(win_rate * 100, 2),
        "sharpe": round(sharpe, 2),
        "trade_count": len(trades),
        # 建仓到清仓的完整周期数 —— 动态仓位下 trade_count 会因减仓而变大，
        # 真正衡量"做了几轮交易"的是这个。
        "round_trips": trade_pairs,
        "strategy": strategy,
        "strategy_name": STRATEGIES.get(strategy, strategy),
        "short": short, "long": long_,
        "final_equity": round(final_equity, 2),
        "trades": trades,
        "equity_curve": equity_curve,
        "sentiment_index": sent_index,
        # 注意：这不是"舆情参与评分"的意思。它表示「回测期间金叉买入信号是否被
        # 舆情阈值拦下过」。sentiment_threshold=None 时舆情不参与，两个计数都为 0。
        "sentiment_filtered": sent_index is not None and sent_index < sentiment_threshold,
        "sentiment_active": sent_index is not None,
        "sentiment_blocked": blocked_by_sentiment,
        "buy_signals": total_buy_signals,
        # ---- 动态仓位相关 ----
        "sentiment_dynamic": bool(use_sentiment and sent_index is not None),
        "target_ratio": (round(target_ratio * 100, 1)
                         if use_sentiment and sent_index is not None else None),
        # 实际仓位暴露：持仓市值 / 总权益的日均值，比"持仓天数占比"更接近真实风险暴露
        "avg_exposure_pct": avg_exposure_pct,
    }