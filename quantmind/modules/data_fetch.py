import os
import random
from datetime import date, timedelta
from decimal import Decimal
import pandas as pd
from sqlalchemy import text
from db import get_session, QuotationDaily, StockBasic


def generate_sample_data(stock_code="600519", stock_name=None, days=120):
    s = get_session()
    try:
        if stock_name is None:
            stock_name = stock_code
        s.query(StockBasic).filter_by(stock_code=stock_code).delete()
        s.add(StockBasic(stock_code=stock_code, stock_name=stock_name, market="SH", industry=""))
        s.query(QuotationDaily).filter_by(stock_code=stock_code).delete()

        price = 1600.0
        today = date.today()
        records = []
        for i in range(days):
            d = today - timedelta(days=days - i)
            if d.weekday() >= 5:
                continue
            change = random.gauss(0, 0.02)
            open_p = price
            close_p = round(price * (1 + change), 2)
            high_p = round(max(open_p, close_p) * (1 + abs(random.gauss(0, 0.005))), 2)
            low_p = round(min(open_p, close_p) * (1 - abs(random.gauss(0, 0.005))), 2)
            vol = random.randint(500000, 3000000)
            amt = round(vol * close_p, 2)
            records.append(QuotationDaily(
                stock_code=stock_code, trade_date=d,
                open=open_p, high=high_p, low=low_p, close=close_p,
                volume=vol, amount=amt, data_source="SAMPLE"
            ))
            price = close_p
        s.add_all(records)
        s.commit()
        return len(records)
    finally:
        s.close()


def import_csv(file_path, stock_code):
    df = pd.read_csv(file_path)
    col_map = {c.lower(): c for c in df.columns}
    s = get_session()
    try:
        s.query(QuotationDaily).filter_by(stock_code=stock_code).delete()
        count = 0
        for _, row in df.iterrows():
            td = row[col_map.get("date", col_map.get("trade_date"))]
            if isinstance(td, str):
                td = pd.to_datetime(td).date()
            s.add(QuotationDaily(
                stock_code=stock_code, trade_date=td,
                open=Decimal(str(row[col_map.get("open")])),
                high=Decimal(str(row[col_map.get("high")])),
                low=Decimal(str(row[col_map.get("low")])),
                close=Decimal(str(row[col_map.get("close")])),
                volume=int(row[col_map.get("volume")]),
                amount=Decimal(str(row.get(col_map.get("amount", "0"), 0))),
                data_source="CSV"
            ))
            count += 1
        s.commit()
        return count
    finally:
        s.close()


def import_tdx_day(file_path, stock_code):
    import struct
    RECORD_SIZE = 32
    s = get_session()
    try:
        s.query(QuotationDaily).filter_by(stock_code=stock_code).delete()
        count = 0
        with open(file_path, 'rb') as f:
            while True:
                buf = f.read(RECORD_SIZE)
                if len(buf) < RECORD_SIZE:
                    break
                raw_date, raw_open, raw_high, raw_low, raw_close, raw_amount, raw_vol, _ = struct.unpack('IIIIIfII', buf)
                year = raw_date // 10000
                month = (raw_date % 10000) // 100
                day = raw_date % 100
                td = date(year, month, day)
                s.add(QuotationDaily(
                    stock_code=stock_code, trade_date=td,
                    open=Decimal(str(raw_open / 100.0)),
                    high=Decimal(str(raw_high / 100.0)),
                    low=Decimal(str(raw_low / 100.0)),
                    close=Decimal(str(raw_close / 100.0)),
                    volume=raw_vol,
                    amount=Decimal(str(raw_amount)),
                    data_source="TDX"
                ))
                count += 1
        s.add(StockBasic(stock_code=stock_code, stock_name=stock_code, market="TDX", industry=""))
        s.commit()
        return count
    finally:
        s.close()


def import_tdx_dir(dir_path):
    import os, glob
    results = []
    for fp in glob.glob(os.path.join(dir_path, '*.day')):
        fname = os.path.basename(fp)
        sc = os.path.splitext(fname)[0]
        n = import_tdx_day(fp, sc)
        results.append({'file': fname, 'stock_code': sc, 'count': n})
    return results


def load_quotation(stock_code):
    s = get_session()
    try:
        rows = s.query(QuotationDaily).filter_by(stock_code=stock_code).order_by(QuotationDaily.trade_date).all()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([{
            "date": r.trade_date, "open": float(r.open), "high": float(r.high),
            "low": float(r.low), "close": float(r.close),
            "volume": int(r.volume), "amount": float(r.amount or 0)
        } for r in rows])
        return df
    finally:
        s.close()