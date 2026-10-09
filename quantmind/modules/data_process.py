from decimal import Decimal
import pandas as pd
from db import get_session, QuotationDaily, TechnicalIndicator


def clean(stock_code):
    s = get_session()
    try:
        rows = s.query(QuotationDaily).filter_by(stock_code=stock_code).order_by(QuotationDaily.trade_date).all()
        if not rows:
            return {"removed": 0, "filled": 0, "fixed": 0}
        seen = set()
        removed = 0
        fixed = 0
        clean_rows = []
        for r in rows:
            if r.trade_date in seen:
                removed += 1
                continue
            seen.add(r.trade_date)
            if float(r.close or 0) <= 0:
                fixed += 1
                r.close = r.open
            clean_rows.append(r)
        s.commit()
        return {"removed": removed, "filled": 0, "fixed": fixed, "total": len(clean_rows)}
    finally:
        s.close()


def calc_indicators(stock_code):
    from modules.data_fetch import load_quotation
    df = load_quotation(stock_code)
    if df.empty:
        return 0

    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_dif"] = ema12 - ema26
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = 2 * (df["macd_dif"] - df["macd_dea"])

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    s = get_session()
    try:
        s.query(TechnicalIndicator).filter_by(stock_code=stock_code).delete()
        count = 0
        def _dec(val):
            if pd.isna(val):
                return None
            return Decimal(str(round(val, 4)))

        for _, row in df.iterrows():
            if pd.isna(row["ma5"]) or pd.isna(row["ma10"]) or pd.isna(row["ma20"]):
                continue
            s.add(TechnicalIndicator(
                stock_code=stock_code, trade_date=row["date"].date() if hasattr(row["date"], "date") else row["date"],
                ma5=_dec(row["ma5"]),
                ma10=_dec(row["ma10"]),
                ma20=_dec(row["ma20"]),
                ma60=_dec(row["ma60"]),
                macd_dif=_dec(row["macd_dif"]),
                macd_dea=_dec(row["macd_dea"]),
                macd_hist=_dec(row["macd_hist"]),
                rsi=_dec(row["rsi"]),
            ))
            count += 1
        s.commit()
        return count
    finally:
        s.close()