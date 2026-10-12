from sqlalchemy import create_engine, Column, String, Date, DateTime, DECIMAL, Integer, BigInteger, Text, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from config import DB_URL

_engine_kwargs = {"echo": False}
if DB_URL.startswith("sqlite"):
    # SQLite 用于本地无 MySQL 时的演示：Flask 是多线程的，必须放开同线程检查
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    _engine_kwargs["pool_recycle"] = 3600

engine = create_engine(DB_URL, **_engine_kwargs)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# SQLite 只把 INTEGER PRIMARY KEY 视为自增主键，BIGINT 不会自增；
# 用 with_variant 让同一份模型在 MySQL 上仍保持 BIGINT。
PK_TYPE = BigInteger().with_variant(Integer, "sqlite")


class StockBasic(Base):
    __tablename__ = "stock_basic"
    stock_code = Column(String(16), primary_key=True)
    stock_name = Column(String(64), nullable=False)
    market = Column(String(16))
    industry = Column(String(64))


class QuotationDaily(Base):
    __tablename__ = "quotation_daily"
    id = Column(PK_TYPE, primary_key=True, autoincrement=True)
    stock_code = Column(String(16), nullable=False)
    trade_date = Column(Date, nullable=False)
    open = Column(DECIMAL(12, 4))
    high = Column(DECIMAL(12, 4))
    low = Column(DECIMAL(12, 4))
    close = Column(DECIMAL(12, 4))
    volume = Column(BigInteger, default=0)
    amount = Column(DECIMAL(18, 2), default=0)
    data_source = Column(String(32))


class TechnicalIndicator(Base):
    __tablename__ = "technical_indicator"
    id = Column(PK_TYPE, primary_key=True, autoincrement=True)
    stock_code = Column(String(16), nullable=False)
    trade_date = Column(Date, nullable=False)
    ma5 = Column(DECIMAL(12, 4))
    ma10 = Column(DECIMAL(12, 4))
    ma20 = Column(DECIMAL(12, 4))
    ma60 = Column(DECIMAL(12, 4))
    macd_dif = Column(DECIMAL(12, 4))
    macd_dea = Column(DECIMAL(12, 4))
    macd_hist = Column(DECIMAL(12, 4))
    rsi = Column(DECIMAL(12, 4))


class Strategy(Base):
    __tablename__ = "strategy"
    strategy_id = Column(PK_TYPE, primary_key=True, autoincrement=True)
    strategy_name = Column(String(128), nullable=False)
    strategy_type = Column(String(32), nullable=False)
    stock_code = Column(String(16), nullable=False)
    params = Column(Text)
    init_capital = Column(DECIMAL(18, 2), default=100000)


class BacktestTask(Base):
    __tablename__ = "backtest_task"
    task_id = Column(PK_TYPE, primary_key=True, autoincrement=True)
    strategy_id = Column(BigInteger, nullable=False)
    stock_code = Column(String(16), nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    status = Column(String(16), default="PENDING")


class BacktestMetric(Base):
    __tablename__ = "backtest_metric"
    id = Column(PK_TYPE, primary_key=True, autoincrement=True)
    task_id = Column(BigInteger, nullable=False)
    total_return = Column(DECIMAL(8, 4))
    annual_return = Column(DECIMAL(8, 4))
    max_drawdown = Column(DECIMAL(8, 4))
    win_rate = Column(DECIMAL(6, 4))
    sharpe_ratio = Column(DECIMAL(8, 4))
    trade_count = Column(Integer, default=0)


def get_session():
    return Session()