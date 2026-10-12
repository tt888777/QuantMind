"""
本地 SQLite 快速初始化（本机没有 MySQL 时用于跑通演示）
======================================================
作用：
  1. 创建全部数据表；
  2. 为若干示例股票生成日线数据 → 清洗 → 计算技术指标。

用法：
  python init_local_db.py
  python init_local_db.py --db my.db --days 300 --stocks 600519 000001

说明：本脚本只写入本地 SQLite 文件，**不会碰 MySQL**（默认 DB_URL 仍是 MySQL）。
生成数据库后，设置环境变量再启动服务即可连上：
  QUANTMIND_DB_URL=sqlite:///<绝对路径>/quantmind_local.db  python app.py
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

STOCK_NAMES = {
    "600519": "贵州茅台",
    "000001": "平安银行",
    "300750": "宁德时代",
    "601318": "中国平安",
    "000858": "五粮液",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(HERE, "quantmind_local.db"),
                    help="SQLite 文件路径")
    ap.add_argument("--days", type=int, default=250, help="每只股票生成的日线天数")
    ap.add_argument("--stocks", nargs="*", default=["600519", "000001", "300750"],
                    help="要生成的股票代码")
    args = ap.parse_args()

    # 必须在导入 config/db 之前设置，让 db.py 连到 SQLite
    os.environ["QUANTMIND_DB_URL"] = "sqlite:///" + args.db.replace("\\", "/")
    sys.path.insert(0, HERE)

    from db import Base, engine, get_session, QuotationDaily, StockBasic
    from modules.data_fetch import generate_sample_data
    from modules.data_process import clean, calc_indicators

    if os.path.exists(args.db):
        os.remove(args.db)
        print(f"已删除旧库：{args.db}")

    Base.metadata.create_all(engine)
    print(f"数据库：{args.db}")
    print("已建表：" + ", ".join(sorted(Base.metadata.tables)))

    for sc in args.stocks:
        sn = STOCK_NAMES.get(sc, sc)
        n = generate_sample_data(stock_code=sc, stock_name=sn, days=args.days)
        info = clean(sc)
        k = calc_indicators(sc)
        print(f"  {sc} {sn}：日线 {n} 条，清洗 去重{info.get('removed', 0)}/"
              f"修复{info.get('fixed', 0)}，指标 {k} 条")

    s = get_session()
    try:
        print(f"\n校验：stock_basic {s.query(StockBasic).count()} 行，"
              f"quotation_daily {s.query(QuotationDaily).count()} 行")
    finally:
        s.close()

    print("\n完成。启动服务：")
    print(f'  QUANTMIND_DB_URL="sqlite:///{args.db.replace(chr(92), "/")}" python app.py')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
