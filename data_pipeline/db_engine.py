import os
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

load_dotenv()

def get_engine():
    cloud_url = os.getenv("DATABASE_URL")
    
    # 尝试连接云端
    try:
        # 设置一个短一点的超时时间，免得在公司等半天
        engine = create_engine(cloud_url, connect_args={'connect_timeout': 5})
        # 尝试做一个极简的查询来测试连接
        with engine.connect() as conn:
            print("🌐 成功连接到 Neon 云端数据库！")
            return engine
    except Exception:
        print("🏢 检测到公司内网限制，已自动切换到本地 SQLite 模式...")
        # 在项目目录下创建一个本地数据库文件
        return create_engine('sqlite:///us_market_local.db')

# 之后你所有的代码都调用这个 engine
#engine = get_engine()