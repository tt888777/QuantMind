from decimal import Decimal
from datetime import date
import pandas as pd
from db import get_session, Strategy, BacktestTask, BacktestMetric


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
                 start_date=None, end_date=None, sentiment_threshold=45):
    df = load_data_with_indicator(stock_code)
    if df.empty or "ma5" not in df.columns:
        return None

    if start_date:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        df = df[df["date"] >= start_date]
    if end_date:
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
        df = df[df["date"] <= end_date]

    df = df.dropna(subset=["ma5", "ma20"]).reset_index(drop=True)
    if len(df) < long_ + 5:
        return None

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
    entry_price = 0.0
    blocked_by_sentiment = 0        # 被舆情拦下的买入信号数
    total_buy_signals = 0           # 金叉信号总数

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        price = float(row["close"])
        ma_s = float(row["ma5"])
        ma_l = float(row["ma20"])
        prev_ma_s = float(prev["ma5"])
        prev_ma_l = float(prev["ma20"])

        signal = 0
        if prev_ma_s <= prev_ma_l and ma_s > ma_l:
            signal = 1
        elif prev_ma_s >= prev_ma_l and ma_s < ma_l:
            signal = -1

        if signal == 1 and position == 0:
            total_buy_signals += 1
            if sent_index is not None and sent_index < sentiment_threshold:
                blocked_by_sentiment += 1
            else:
                buy_price = price * (1 + slippage)
                shares = int(cash / buy_price / lot_size) * lot_size
                if shares > 0:
                    cost = shares * buy_price * (1 + commission)
                    cash -= cost
                    position = shares
                    entry_price = buy_price
                    trades.append({"date": row["date"], "dir": "BUY", "price": round(buy_price, 2), "shares": shares})
        elif signal == -1 and position > 0:
            sell_price = price * (1 - slippage)
            revenue = position * sell_price * (1 - commission)
            cash += revenue
            trades.append({"date": row["date"], "dir": "SELL", "price": round(sell_price, 2), "shares": position})
            position = 0
            entry_price = 0.0

        equity = cash + position * price
        equity_curve.append({"date": row["date"], "equity": round(equity, 2)})

    final_equity = cash + position * float(df.iloc[-1]["close"])
    total_return = (final_equity - init_capital) / init_capital
    days = len(df)
    annual_return = (1 + total_return) ** (252 / max(days, 1)) - 1 if total_return > -1 else -1

    eq = pd.DataFrame(equity_curve)
    eq["peak"] = eq["equity"].cummax()
    eq["dd"] = (eq["equity"] - eq["peak"]) / eq["peak"]
    max_dd = float(eq["dd"].min())

    wins = 0
    for i in range(0, len(trades) - 1, 2):
        if i + 1 < len(trades):
            if trades[i + 1]["price"] > trades[i]["price"]:
                wins += 1
    trade_pairs = len(trades) // 2
    win_rate = wins / trade_pairs if trade_pairs > 0 else 0

    if len(eq) > 1:
        rets = eq["equity"].pct_change().dropna()
        sharpe = float(rets.mean() / rets.std() * (252 ** 0.5)) if rets.std() > 0 else 0
    else:
        sharpe = 0

    s = get_session()
    try:
        strat = Strategy(strategy_name=f"双均线{short}/{long_}", strategy_type="DUAL_MA",
                         stock_code=stock_code, params=f'{{"short":{short},"long":{long_}}}',
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
    }