import os

# 默认沿用原 MySQL 配置（不改动原有行为）。
# 若本机没有 MySQL，可用环境变量临时切换，例如：
#   QUANTMIND_DB_URL=sqlite:///C:/Users/xxx/source/repos/QuantMind/quantmind/quantmind_local.db
# 初始化本地 SQLite 数据：python init_local_db.py
DB_URL = os.environ.get(
    "QUANTMIND_DB_URL",
    "mysql+pymysql://root:root123456@localhost:3306/quantmind?charset=utf8mb4",
)
