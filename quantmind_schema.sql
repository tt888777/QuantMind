-- ============================================================
-- QuantMind · AI 智能体辅助股票量化分析系统
-- 数据库建库脚本（MySQL 8.0+）
-- 依据：user-stories.md / 需求跟踪矩阵（14 功能点 / 5 模块）
-- 说明：可直接执行本脚本形成数据库
-- ============================================================

DROP DATABASE IF EXISTS quantmind;
CREATE DATABASE quantmind DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE quantmind;

-- ------------------------------------------------------------
-- 模块一：数据获取（M1 / S1）
-- ------------------------------------------------------------

-- 1. 股票基本信息表（标的字典）
CREATE TABLE stock_basic (
    stock_code      VARCHAR(16)    NOT NULL COMMENT '股票代码',
    stock_name      VARCHAR(64)    NOT NULL COMMENT '股票名称',
    market          VARCHAR(16)    DEFAULT NULL COMMENT '市场(SH/SZ/BJ)',
    industry        VARCHAR(64)    DEFAULT NULL COMMENT '所属行业',
    list_date       DATE           DEFAULT NULL COMMENT '上市日期',
    is_active       TINYINT(1)     DEFAULT 1 COMMENT '是否启用',
    created_at      DATETIME       DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (stock_code)
) ENGINE=InnoDB COMMENT='股票基本信息表(标的字典)';

-- 2. 日线行情表（OHLCV）—— US-001/002/012 落库
CREATE TABLE quotation_daily (
    id              BIGINT         NOT NULL AUTO_INCREMENT,
    stock_code      VARCHAR(16)    NOT NULL COMMENT '股票代码',
    trade_date      DATE           NOT NULL COMMENT '交易日',
    open            DECIMAL(12,4)  DEFAULT NULL COMMENT '开盘价',
    high            DECIMAL(12,4)  DEFAULT NULL COMMENT '最高价',
    low             DECIMAL(12,4)  DEFAULT NULL COMMENT '最低价',
    close           DECIMAL(12,4)  DEFAULT NULL COMMENT '收盘价',
    volume          BIGINT         DEFAULT 0 COMMENT '成交量(股)',
    amount          DECIMAL(18,2)  DEFAULT 0 COMMENT '成交额(元)',
    data_source     VARCHAR(32)    DEFAULT NULL COMMENT '数据来源(TDX/TUSHARE/AKSHARE/UPLOAD)',
    import_batch    VARCHAR(64)    DEFAULT NULL COMMENT '导入批次号',
    created_at      DATETIME       DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uk_stock_date (stock_code, trade_date),
    KEY idx_trade_date (trade_date),
    KEY idx_source (data_source)
) ENGINE=InnoDB COMMENT='日线行情表(OHLCV)';

-- 3. 数据导入日志表 —— US-001/012/013
CREATE TABLE data_import_log (
    id              BIGINT         NOT NULL AUTO_INCREMENT,
    batch_no        VARCHAR(64)    NOT NULL COMMENT '导入批次号',
    file_name       VARCHAR(255)   DEFAULT NULL COMMENT '文件名',
    file_format     VARCHAR(16)    DEFAULT NULL COMMENT '文件格式(CSV/XML/EXCEL/JSON/DAY)',
    stock_code      VARCHAR(16)    DEFAULT NULL COMMENT '股票代码',
    record_count    INT            DEFAULT 0 COMMENT '导入记录数',
    date_range_start DATE          DEFAULT NULL COMMENT '数据起始日期',
    date_range_end   DATE          DEFAULT NULL COMMENT '数据结束日期',
    column_mapping  TEXT           DEFAULT NULL COMMENT 'AI列名映射结果(JSON)',
    status          VARCHAR(16)    DEFAULT 'SUCCESS' COMMENT '状态(SUCCESS/FAILED/PARTIAL)',
    error_msg       TEXT           DEFAULT NULL COMMENT '错误信息',
    created_at      DATETIME       DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_batch (batch_no),
    KEY idx_stock (stock_code)
) ENGINE=InnoDB COMMENT='数据导入日志表';

-- ------------------------------------------------------------
-- 模块二：数据处理与报表（M2 / S2 S3 S9）
-- ------------------------------------------------------------

-- 4. 数据清洗日志表 —— US-003
CREATE TABLE data_cleaning_log (
    id              BIGINT         NOT NULL AUTO_INCREMENT,
    batch_no        VARCHAR(64)    NOT NULL COMMENT '清洗批次号',
    stock_code      VARCHAR(16)    NOT NULL COMMENT '股票代码',
    strategy        VARCHAR(64)    DEFAULT NULL COMMENT '清洗策略(FILL_FWD/CLIP/...)',
    dup_removed     INT            DEFAULT 0 COMMENT '删除重复记录数',
    gap_filled      INT            DEFAULT 0 COMMENT '填充缺口数',
    outlier_fixed   INT            DEFAULT 0 COMMENT '修正异常值数',
    passed_check    TINYINT(1)     DEFAULT 1 COMMENT '是否通过完整性校验',
    created_at      DATETIME       DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_batch (batch_no),
    KEY idx_stock (stock_code)
) ENGINE=InnoDB COMMENT='数据清洗日志表';

-- 5. 技术指标表（宽表）—— US-004
CREATE TABLE technical_indicator (
    id              BIGINT         NOT NULL AUTO_INCREMENT,
    stock_code      VARCHAR(16)    NOT NULL COMMENT '股票代码',
    trade_date      DATE           NOT NULL COMMENT '交易日',
    ma5             DECIMAL(12,4)  DEFAULT NULL COMMENT 'MA5',
    ma10            DECIMAL(12,4)  DEFAULT NULL COMMENT 'MA10',
    ma20            DECIMAL(12,4)  DEFAULT NULL COMMENT 'MA20',
    ma60            DECIMAL(12,4)  DEFAULT NULL COMMENT 'MA60',
    ema12           DECIMAL(12,4)  DEFAULT NULL COMMENT 'EMA12',
    ema26           DECIMAL(12,4)  DEFAULT NULL COMMENT 'EMA26',
    macd_dif        DECIMAL(12,4)  DEFAULT NULL COMMENT 'MACD DIF',
    macd_dea        DECIMAL(12,4)  DEFAULT NULL COMMENT 'MACD DEA',
    macd_hist       DECIMAL(12,4)  DEFAULT NULL COMMENT 'MACD HIST',
    boll_upper      DECIMAL(12,4)  DEFAULT NULL COMMENT '布林带上轨',
    boll_mid        DECIMAL(12,4)  DEFAULT NULL COMMENT '布林带中轨',
    boll_lower      DECIMAL(12,4)  DEFAULT NULL COMMENT '布林带下轨',
    rsi             DECIMAL(12,4)  DEFAULT NULL COMMENT 'RSI',
    created_at      DATETIME       DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uk_stock_date (stock_code, trade_date),
    KEY idx_trade_date (trade_date)
) ENGINE=InnoDB COMMENT='技术指标表(MA/EMA/MACD/布林带/RSI)';

-- ------------------------------------------------------------
-- 模块三：AI 智能体舆情决策（M3 / S6 S7）
-- ------------------------------------------------------------

-- 6. 舆情文本表 —— US-006
CREATE TABLE sentiment_text (
    id              BIGINT         NOT NULL AUTO_INCREMENT,
    stock_code      VARCHAR(16)    NOT NULL COMMENT '关联股票代码',
    title           VARCHAR(512)   DEFAULT NULL COMMENT '标题',
    content         MEDIUMTEXT     DEFAULT NULL COMMENT '正文',
    source_type     VARCHAR(16)    DEFAULT NULL COMMENT '来源类型(NEWS/GUBA)',
    source_name     VARCHAR(128)   DEFAULT NULL COMMENT '来源名称',
    publish_time    DATETIME       DEFAULT NULL COMMENT '发布时间',
    fetch_time      DATETIME       DEFAULT CURRENT_TIMESTAMP COMMENT '采集时间',
    content_hash    VARCHAR(64)    DEFAULT NULL COMMENT '内容哈希(去重用)',
    is_analyzed     TINYINT(1)     DEFAULT 0 COMMENT '是否已分析',
    PRIMARY KEY (id),
    KEY idx_stock (stock_code),
    KEY idx_publish (publish_time),
    UNIQUE KEY uk_hash (content_hash)
) ENGINE=InnoDB COMMENT='舆情文本表(新闻/股吧)';

-- 7. 情感分析结果表 —— US-007
CREATE TABLE sentiment_analysis (
    id              BIGINT         NOT NULL AUTO_INCREMENT,
    text_id         BIGINT         NOT NULL COMMENT '舆情文本ID',
    sentiment_label VARCHAR(16)    NOT NULL COMMENT '情感标签(STRONG_BUY/BUY/NEUTRAL/SELL/STRONG_SELL)',
    confidence      DECIMAL(5,4)   DEFAULT NULL COMMENT '置信度(0-1)',
    event_summary   VARCHAR(512)   DEFAULT NULL COMMENT '关键事件摘要',
    model_version   VARCHAR(32)    DEFAULT NULL COMMENT '模型版本',
    analyzed_at     DATETIME       DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uk_text (text_id),
    KEY idx_label (sentiment_label)
) ENGINE=InnoDB COMMENT='情感分析结果表';

-- 8. 市场情绪指数表（日级聚合）—— US-007
CREATE TABLE market_sentiment (
    id              BIGINT         NOT NULL AUTO_INCREMENT,
    stock_code      VARCHAR(16)    NOT NULL COMMENT '股票代码',
    trade_date      DATE           NOT NULL COMMENT '交易日',
    sentiment_index DECIMAL(6,2)   DEFAULT NULL COMMENT '当日情绪指数(-100~100)',
    positive_cnt    INT            DEFAULT 0 COMMENT '看涨文本数',
    negative_cnt    INT            DEFAULT 0 COMMENT '看跌文本数',
    neutral_cnt     INT            DEFAULT 0 COMMENT '中性文本数',
    created_at      DATETIME       DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uk_stock_date (stock_code, trade_date)
) ENGINE=InnoDB COMMENT='市场情绪指数表(日级聚合)';

-- 9. 热点事件表 —— US-007
CREATE TABLE hot_event (
    id              BIGINT         NOT NULL AUTO_INCREMENT,
    stock_code      VARCHAR(16)    NOT NULL COMMENT '股票代码',
    event_name      VARCHAR(128)   NOT NULL COMMENT '事件名称',
    event_type      VARCHAR(32)    DEFAULT NULL COMMENT '事件类型(利好/利空/中性)',
    event_desc      TEXT           DEFAULT NULL COMMENT '事件描述',
    occur_date      DATE           DEFAULT NULL COMMENT '发生日期',
    impact_score    DECIMAL(5,2)   DEFAULT NULL COMMENT '影响分值',
    created_at      DATETIME       DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_stock_date (stock_code, occur_date)
) ENGINE=InnoDB COMMENT='热点事件表';

-- ------------------------------------------------------------
-- 模块四：量化回测（M4 / S5 S8 S10）
-- ------------------------------------------------------------

-- 10. 策略表 —— US-009
CREATE TABLE strategy (
    strategy_id     BIGINT         NOT NULL AUTO_INCREMENT,
    strategy_name   VARCHAR(128)   NOT NULL COMMENT '策略名称',
    strategy_type   VARCHAR(32)    NOT NULL COMMENT '策略类型(DUAL_MA/MOMENTUM/...)',
    stock_code      VARCHAR(16)    NOT NULL COMMENT '标的股票代码',
    params          TEXT           DEFAULT NULL COMMENT '策略参数(JSON: 周期/仓位/...)',
    init_capital    DECIMAL(18,2)  DEFAULT 100000 COMMENT '初始资金',
    commission_rate DECIMAL(6,4)   DEFAULT 0.0003 COMMENT '手续费率',
    slippage        DECIMAL(6,4)   DEFAULT 0.0001 COMMENT '滑点',
    signal_logic    TEXT           DEFAULT NULL COMMENT '信号逻辑说明',
    status          VARCHAR(16)    DEFAULT 'DRAFT' COMMENT '状态(DRAFT/READY/ARCHIVED)',
    created_at      DATETIME       DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME       DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (strategy_id),
    KEY idx_stock (stock_code),
    KEY idx_type (strategy_type)
) ENGINE=InnoDB COMMENT='策略表';

-- 11. 策略情绪过滤条件表 —— US-008 (情绪融合)
CREATE TABLE strategy_sentiment_filter (
    id              BIGINT         NOT NULL AUTO_INCREMENT,
    strategy_id     BIGINT         NOT NULL COMMENT '策略ID',
    filter_type     VARCHAR(32)    NOT NULL COMMENT '过滤类型(BUY_FILTER/SELL_FILTER)',
    condition_expr  VARCHAR(255)   NOT NULL COMMENT '条件表达式(如 sentiment_index>50)',
    threshold       DECIMAL(8,2)   DEFAULT NULL COMMENT '阈值',
    created_at      DATETIME       DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_strategy (strategy_id)
) ENGINE=InnoDB COMMENT='策略情绪过滤条件表(舆情融合)';

-- 12. 回测任务表 —— US-010
CREATE TABLE backtest_task (
    task_id         BIGINT         NOT NULL AUTO_INCREMENT,
    strategy_id     BIGINT         NOT NULL COMMENT '策略ID',
    stock_code      VARCHAR(16)    NOT NULL COMMENT '回测标的',
    start_date      DATE           NOT NULL COMMENT '回测起始日期',
    end_date        DATE           NOT NULL COMMENT '回测结束日期',
    status          VARCHAR(16)    DEFAULT 'PENDING' COMMENT '状态(PENDING/RUNNING/DONE/FAILED)',
    started_at      DATETIME       DEFAULT NULL,
    finished_at     DATETIME       DEFAULT NULL,
    created_at      DATETIME       DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (task_id),
    KEY idx_strategy (strategy_id),
    KEY idx_status (status)
) ENGINE=InnoDB COMMENT='回测任务表';

-- 13. 回测交易明细表 —— US-010
CREATE TABLE backtest_trade (
    id              BIGINT         NOT NULL AUTO_INCREMENT,
    task_id         BIGINT         NOT NULL COMMENT '回测任务ID',
    trade_date      DATE           NOT NULL COMMENT '交易日期',
    direction       VARCHAR(8)    NOT NULL COMMENT '方向(BUY/SELL)',
    price           DECIMAL(12,4)  NOT NULL COMMENT '成交价',
    volume          INT            NOT NULL COMMENT '成交数量(股)',
    amount          DECIMAL(18,2)  NOT NULL COMMENT '成交金额',
    commission      DECIMAL(18,2)  DEFAULT 0 COMMENT '手续费',
    cash_after      DECIMAL(18,2)  DEFAULT NULL COMMENT '交易后现金',
    position_after  INT            DEFAULT NULL COMMENT '交易后持仓',
    PRIMARY KEY (id),
    KEY idx_task (task_id),
    KEY idx_date (trade_date)
) ENGINE=InnoDB COMMENT='回测交易明细表';

-- 14. 回测绩效指标表 —— US-010
CREATE TABLE backtest_metric (
    id              BIGINT         NOT NULL AUTO_INCREMENT,
    task_id         BIGINT         NOT NULL COMMENT '回测任务ID',
    total_return    DECIMAL(8,4)   DEFAULT NULL COMMENT '总收益率',
    annual_return   DECIMAL(8,4)   DEFAULT NULL COMMENT '年化收益率',
    max_drawdown    DECIMAL(8,4)   DEFAULT NULL COMMENT '最大回撤',
    win_rate        DECIMAL(6,4)   DEFAULT NULL COMMENT '胜率',
    sharpe_ratio    DECIMAL(8,4)   DEFAULT NULL COMMENT '夏普比率',
    trade_count     INT            DEFAULT 0 COMMENT '交易笔数',
    created_at      DATETIME       DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uk_task (task_id)
) ENGINE=InnoDB COMMENT='回测绩效指标表';

-- ------------------------------------------------------------
-- 模块五：可配置分析流水线（M5 / S0）
-- ------------------------------------------------------------

-- 15. 流水线执行记录表 —— US-014
CREATE TABLE pipeline_run (
    run_id          BIGINT         NOT NULL AUTO_INCREMENT,
    stock_code      VARCHAR(16)    NOT NULL COMMENT '标的股票代码',
    selected_steps  TEXT           NOT NULL COMMENT '勾选步骤(JSON: [S3,S5,S7,...])',
    run_params      TEXT           DEFAULT NULL COMMENT '各步骤参数(JSON)',
    status          VARCHAR(16)    DEFAULT 'PENDING' COMMENT '状态(PENDING/RUNNING/DONE/FAILED)',
    started_at      DATETIME       DEFAULT NULL,
    finished_at     DATETIME       DEFAULT NULL,
    created_at      DATETIME       DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id),
    KEY idx_stock (stock_code),
    KEY idx_status (status)
) ENGINE=InnoDB COMMENT='流水线执行记录表';

-- 16. 流水线步骤执行日志表 —— US-014
CREATE TABLE pipeline_step_log (
    id              BIGINT         NOT NULL AUTO_INCREMENT,
    run_id          BIGINT         NOT NULL COMMENT '流水线执行ID',
    step_code       VARCHAR(8)     NOT NULL COMMENT '步骤编码(S1~S10)',
    step_name       VARCHAR(32)    NOT NULL COMMENT '步骤名称',
    status          VARCHAR(16)    DEFAULT 'PENDING' COMMENT '状态(PENDING/RUNNING/DONE/SKIPPED/FAILED)',
    cost_ms         INT            DEFAULT NULL COMMENT '耗时(毫秒)',
    output_ref      VARCHAR(255)   DEFAULT NULL COMMENT '产物引用(表/图表/报告链接)',
    error_msg       TEXT           DEFAULT NULL,
    started_at      DATETIME       DEFAULT NULL,
    finished_at     DATETIME       DEFAULT NULL,
    PRIMARY KEY (id),
    KEY idx_run (run_id),
    KEY idx_step (step_code)
) ENGINE=InnoDB COMMENT='流水线步骤执行日志表';

-- ============================================================
-- 模块六：账号与登录（QQ-01 / QQ-02）
-- 说明：应用启动时 SQLAlchemy 也会自动 create_all 这两张表，
--       这里单独给出 DDL，便于直接用 MySQL 脚本建库的场合。
-- ============================================================

-- 17. 用户表
CREATE TABLE qm_user (
    id              BIGINT       NOT NULL AUTO_INCREMENT,
    username        VARCHAR(32)  NOT NULL COMMENT '登录名，3~20 位',
    email           VARCHAR(190) DEFAULT NULL COMMENT '邮箱（可空，可用于登录）',
    password_hash   VARCHAR(255) NOT NULL COMMENT 'scrypt 加盐哈希，绝不存明文',
    display_name    VARCHAR(64)  DEFAULT NULL COMMENT '昵称',
    role            VARCHAR(16)  DEFAULT 'user' COMMENT 'user / admin',
    is_active       TINYINT(1)   NOT NULL DEFAULT 1 COMMENT '1=启用 0=停用',
    created_at      DATETIME     DEFAULT CURRENT_TIMESTAMP,
    last_login_at   DATETIME     DEFAULT NULL,
    last_login_ip   VARCHAR(64)  DEFAULT NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uk_username (username),
    UNIQUE KEY uk_email (email)
) ENGINE=InnoDB COMMENT='用户账号表';

-- 18. 登录/操作流水表
CREATE TABLE qm_login_log (
    id              BIGINT       NOT NULL AUTO_INCREMENT,
    user_id         BIGINT       DEFAULT NULL,
    username        VARCHAR(32)  DEFAULT NULL,
    action          VARCHAR(16)  NOT NULL COMMENT 'register/login/logout/password/profile',
    success         TINYINT(1)   NOT NULL DEFAULT 1,
    ip              VARCHAR(64)  DEFAULT NULL,
    user_agent      VARCHAR(255) DEFAULT NULL,
    created_at      DATETIME     DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_user (user_id),
    KEY idx_time (created_at)
) ENGINE=InnoDB COMMENT='登录与账号操作流水表';

-- ============================================================
-- 建库完成：共 18 张表，覆盖 6 模块（含账号体系）
-- ============================================================