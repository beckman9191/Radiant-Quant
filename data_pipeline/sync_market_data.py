import os
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# 加载配置
load_dotenv()

def get_engine():
    # 使用你刚才测试成功的逻辑：优先云端，失败则本地
    db_url = os.getenv("DATABASE_URL")
    try:
        engine = create_engine(db_url, connect_args={'connect_timeout': 5})
        with engine.connect() as conn:
            return engine
    except:
        return create_engine('sqlite:///us_market_data.db')

def download_and_save_smart(symbols):
    engine = get_engine()
    
    for symbol in symbols:
        table_name = symbol.lower()
        print(f"🔄 正在检查 {symbol} 的增量更新...")

        # 1. 获取本地数据库中该股票的最新日期
        try:
            query = f"SELECT MAX(date) FROM {table_name}"
            last_date_str = pd.read_sql(query, engine).iloc[0, 0]
            # 处理不同数据库返回的日期格式
            last_date = pd.to_datetime(last_date_str)
            print(f"📅 本地最后记录日期: {last_date.date()}")
        except Exception:
            print(f"🆕 找不到本地表 {table_name}，将进行全量下载")
            last_date = None

        # 2. 确定下载范围
        if last_date:
            # 只下载从最后一天之后到今天的数据
            # yf.download 的 start 是包含关系，所以加 1 天
            start_date = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            df = yf.download(symbol, start=start_date, progress=False)
        else:
            # 全量下载
            df = yf.download(symbol, period="5y", progress=False)

        if df.empty:
            print(f"☕ {symbol} 已经是最新，无需更新。")
            continue
            
        # 3. 清洗数据
        df = df.reset_index()
        # 处理 yfinance 偶尔出现的 MultiIndex 列名
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]

        # 4. 存入数据库：如果是新表则 replace，如果是更新则 append
        mode = 'replace' if last_date is None else 'append'
        df.to_sql(table_name, engine, if_exists=mode, index=False)
        print(f"✅ {symbol} 成功增量更新 {len(df)} 行数据 (Mode: {mode})")

if __name__ == "__main__":
    # 你感兴趣的美股清单
    my_stocks = ['NVDA', 'AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'QQQ']
    download_and_save_smart(my_stocks)