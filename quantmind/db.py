from datetime import datetime

from sqlalchemy import (create_engine, Column, String, Date, DateTime, DECIMAL,
                        Integer, BigInteger, Text, JSON, Boolean,
                        UniqueConstraint, Index)
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


class QuotationMinute(Base):
    """分钟级行情（5 分钟 / 30 分钟两档）。

    与 `quotation_daily` 分开建表而不是加一列 `period`，原因是：
    日线表被回测、指标计算、报告多处按 `(stock_code, trade_date)` 读取，
    混进分钟数据会污染这些已有路径。独立表最安全。

    唯一索引 `(stock_code, period, trade_time)`：采集脚本可重复运行，
    重复采到同一根 bar 时用 upsert 覆盖，不会产生重复行。
    """
    __tablename__ = "quotation_minute"
    __table_args__ = (
        UniqueConstraint("stock_code", "period", "trade_time",
                         name="uq_minute_code_period_time"),
        Index("ix_minute_code_period", "stock_code", "period"),
    )
    id = Column(PK_TYPE, primary_key=True, autoincrement=True)
    stock_code = Column(String(16), nullable=False)
    period = Column(Integer, nullable=False)        # 5 / 30，单位分钟
    trade_time = Column(DateTime, nullable=False)   # 该根 bar 的结束时刻
    open = Column(DECIMAL(12, 4))
    high = Column(DECIMAL(12, 4))
    low = Column(DECIMAL(12, 4))
    close = Column(DECIMAL(12, 4))
    volume = Column(BigInteger, default=0)
    data_source = Column(String(32))


class MinuteCollectLog(Base):
    """分钟数据采集日志。

    每次采集（含失败）落一条，用于：
    - 前端「分钟数据」页展示最近采集记录，让演示时能说清数据是哪来的；
    - 排查「某只标的为什么没有数据」—— 失败原因在这里有原文（新浪超时/返回空等）。

    高频写入但量很小（一天几十条），不建额外索引，按 id 倒序取最近 N 条即可。
    """
    __tablename__ = "minute_collect_log"
    id = Column(PK_TYPE, primary_key=True, autoincrement=True)
    stock_code = Column(String(16), nullable=False)
    period = Column(Integer, nullable=False)
    ok = Column(Integer, default=1)                 # 1 成功 / 0 失败
    rows_fetched = Column(Integer, default=0)       # 接口返回根数
    rows_added = Column(Integer, default=0)         # 实际新增（去重后）
    rows_skipped = Column(Integer, default=0)       # 已存在被跳过
    data_source = Column(String(32))
    message = Column(String(255))                   # 失败原因 / 摘要
    created_at = Column(DateTime, default=datetime.now)


class ActivityLog(Base):
    """全站操作活动日志（主页「最近活动」的数据源）。

    为什么存数据库而不是 localStorage：
    - 以前存在浏览器 localStorage 里，只保留 1 小时，刷新/换浏览器就没了，
      演示时经常出现「刚跑完回测，主页空白」；
    - 分钟回测是纯后端计算，前端不写日志就完全没有记录；
    - 落库后可以按模块回溯，也能解释「这条数据是怎么来的」。

    写入频率低（一次操作一条），只在 created_at 上建索引。
    """
    __tablename__ = "activity_log"
    __table_args__ = (Index("ix_activity_created", "created_at"),)
    id = Column(PK_TYPE, primary_key=True, autoincrement=True)
    module = Column(String(32), nullable=False)      # 量化回测 / 分钟回测 / 数据导入 ...
    event = Column(String(500), nullable=False)      # 一句话描述
    status = Column(String(16), default="ok")        # ok / fail / info
    stock_code = Column(String(16))
    user_id = Column(BigInteger)
    created_at = Column(DateTime, default=datetime.now)


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


# ---------------------------------------------------------------
# 账号体系（注册 / 登录 / 个人中心）
# 表名刻意加 qm_ 前缀：MySQL 里 user 是个容易撞名的词。
# ---------------------------------------------------------------
class User(Base):
    __tablename__ = "qm_user"
    id = Column(PK_TYPE, primary_key=True, autoincrement=True)
    username = Column(String(32), nullable=False, unique=True, index=True)
    email = Column(String(190), unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    display_name = Column(String(64))
    role = Column(String(16), default="user")          # user / admin
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    last_login_at = Column(DateTime)
    last_login_ip = Column(String(64))


class LoginLog(Base):
    """登录 / 注册 / 改密等账号事件流水，用于个人中心的「登录记录」。"""
    __tablename__ = "qm_login_log"
    id = Column(PK_TYPE, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, index=True)
    username = Column(String(32))
    action = Column(String(16), nullable=False)        # register / login / logout / password / profile
    success = Column(Boolean, default=True, nullable=False)
    ip = Column(String(64))
    user_agent = Column(String(255))
    created_at = Column(DateTime, default=datetime.now)


def get_session():
    return Session()