# -*- coding: utf-8 -*-
"""用新浪 30/60 分钟线做一次「可行性试算」。

不写数据库、不接 Flask，只回答一个问题：
把现有双均线/动量/布林带逻辑搬到分钟线，交易频率和成本损耗各是什么量级。
"""
import json
import time

import akshare as ak
import pandas as pd

SYMBOL = "sh600519"
INIT_CAPITAL = 100000.0
# A 股真实成本：佣金万三（单笔最低 5 元）+ 卖出印花税千一 + 过户费万分之一
COMMISSION_RATE = 0.0003
MIN_COMMISSION = 5.0
STAMP_TAX = 0.001
TRANSFER_FEE = 0.00001
SLIPPAGE = 0.0002          # 分钟级滑点按万二估（比日线的万一更高）


def fetch(period):
    df = ak.stock_zh_a_minute(symbol=SYMBOL, period=period, adjust="qfq")
    df["day"] = pd.to_datetime(df["day"])
    for c in ("open", "high", "low", "close", "volume", "amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close"]).reset_index(drop=True)


def signals(df, kind, short=5, long_=20):
    c = df["close"]
    sig = pd.Series(0, index=df.index)
    if kind == "dual_ma":
        ms, ml = c.rolling(short).mean(), c.rolling(long_).mean()
        up = ms > ml
        sig[(~up.shift(1, fill_value=False)) & up] = 1
        sig[up.shift(1, fill_value=True) & (~up)] = -1
    elif kind == "boll":
        n = max(long_, 5)
        mid, std = c.rolling(n).mean(), c.rolling(n).std()
        upper = mid + 2 * std
        sig[(c.shift(1) <= upper.shift(1)) & (c > upper)] = 1
        sig[(c.shift(1) >= mid.shift(1)) & (c < mid)] = -1
    elif kind == "momentum":
        mom = c / c.shift(long_) - 1
        sig[(mom.shift(1) <= 0) & (mom > 0)] = 1
        sig[(mom.shift(1) >= 0) & (mom < 0)] = -1
    return sig.fillna(0).astype(int)


def run(df, sig):
    cash, pos = INIT_CAPITAL, 0
    trades, fees_total = 0, 0.0
    eq = []
    for i in range(1, len(df)):
        px = float(df["close"].iloc[i])
        s = int(sig.iloc[i])
        if s == 1 and pos == 0:
            bp = px * (1 + SLIPPAGE)
            shares = int(cash / bp)
            if shares > 0:
                amt = shares * bp
                fee = max(amt * COMMISSION_RATE, MIN_COMMISSION) + amt * TRANSFER_FEE
                if amt + fee <= cash:
                    cash -= amt + fee
                    fees_total += fee
                    pos = shares
                    trades += 1
        elif s == -1 and pos > 0:
            sp = px * (1 - SLIPPAGE)
            amt = pos * sp
            fee = max(amt * COMMISSION_RATE, MIN_COMMISSION) + amt * STAMP_TAX + amt * TRANSFER_FEE
            cash += amt - fee
            fees_total += fee
            pos = 0
            trades += 1
        eq.append(cash + pos * px)
    final = cash + pos * float(df["close"].iloc[-1])
    ret = (final - INIT_CAPITAL) / INIT_CAPITAL * 100
    ret_nofee = (final + fees_total - INIT_CAPITAL) / INIT_CAPITAL * 100
    days = len(df)
    ann = ((1 + ret / 100) ** (252 * (48 if days and False else 1) / max(days, 1)) - 1) * 100 if ret > -100 else -100
    s = pd.Series(eq)
    dd = ((s - s.cummax()) / s.cummax()).min() * 100 if len(s) > 1 else 0
    sd = s.pct_change().dropna().std()
    sharpe = (s.pct_change().dropna().mean() / sd * (252 ** 0.5)) if sd and sd > 0 else 0
    return {
        "bars": days, "trades": trades, "return": round(ret, 2),
        "return_before_fee": round(ret_nofee, 2),
        "fee_drag": round(ret_nofee - ret, 2),
        "max_dd": round(abs(dd), 2), "sharpe": round(sharpe, 2),
        "fees": round(fees_total, 2),
    }


print("=" * 88)
print("分钟线可行性试算  %s  初始资金 %.0f" % (SYMBOL, INIT_CAPITAL))
print("=" * 88)
out = {}
for period in ("30", "60"):
    df = fetch(period)
    days = df["day"].dt.date.nunique()
    print(f"\n--- {period} 分钟线：{len(df)} 根 bar，覆盖 {days} 个交易日 "
          f"({df['day'].iloc[0]} ~ {df['day'].iloc[-1]}) ---")
    print(f"{'策略':<14}{'成交笔数':>8}{'收益%':>9}{'未扣费%':>10}{'成本损耗%':>11}{'回撤%':>8}{'夏普':>7}")
    for kind, name in (("dual_ma", "双均线5/20"), ("boll", "布林带20"), ("momentum", "动量20")):
        r = run(df, signals(df, kind))
        r["days"] = days
        out[f"{period}m_{kind}"] = r
        print(f"{name:<14}{r['trades']:>8}{r['return']:>9}{r['return_before_fee']:>10}"
              f"{r['fee_drag']:>11}{r['max_dd']:>8}{r['sharpe']:>7}")
    time.sleep(1.5)

with open("minute_feasibility.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\n结果已写入 minute_feasibility.json")
