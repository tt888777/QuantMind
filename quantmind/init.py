# init_db.py
import db  # 这里的 models 换成你刚才那个文件的名字（不带.py）
from db import engine, Base

if __name__ == '__main__':
    print("正在创建数据库表，请稍候...")
    Base.metadata.create_all(engine)
    print(" 所有数据库表创建成功！")