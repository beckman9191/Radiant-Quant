import os
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# 加载配置
load_dotenv()
db_url = os.getenv("DATABASE_URL")

# 创建数据库引擎
# Neon 要求必须使用 SSL，连接字符串后面已经带了 ?sslmode=require
engine = create_engine(db_url)

def sync_stock_to_cloud(symbol):
    print(f"正在从 Yahoo Finance 获取 {symbol} 的数据...")
    
    # 获取历史数据 (例如过去2年)
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="2y")
    
    # 简单的清洗：重置索引让日期变成一列
    df = df.reset_index()
    
    # 存入 Neon 数据库
    # if_exists='replace' 表示如果表存在就覆盖，'append' 表示追加
    df.to_sql(symbol.lower(), engine, if_exists='replace', index=False)
    
    print(f"✅ {symbol} 数据已成功同步到云端数据库！")

if __name__ == "__main__":
    # 试着同步几只你感兴趣的股票
    stocks = ['NVDA', 'AAPL', 'TSLA', 'MSFT']
    for s in stocks:
        sync_stock_to_cloud(s)