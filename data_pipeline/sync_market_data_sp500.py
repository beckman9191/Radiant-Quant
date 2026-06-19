import os
import requests
import pandas as pd
import yfinance as yf
from io import StringIO
from sqlalchemy import text
from dotenv import load_dotenv
import warnings
from db_engine import get_engine  # 导入您之前定义的 engine 获取函数

# 忽略 yfinance 产生的一些版本警告
warnings.filterwarnings('ignore')
load_dotenv()

def get_sp500_tickers():
    """从维基百科抓取标普500成分股列表，处理403限制"""
    print("🌐 正在从维基百科获取标普500最新成分股列表...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # 解析 HTML 表格
        table = pd.read_html(StringIO(response.text))
        df = table[0]
        
        # 转换代码格式：yfinance 使用 '-' 替代 '.' (如 BRK.B -> BRK-B)
        tickers = df['Symbol'].str.replace('.', '-').tolist()
        print(f"✅ 成功获取 {len(tickers)} 只股票代码")
        return tickers
    except Exception as e:
        print(f"❌ 获取标普500列表失败: {e}")
        return []

def download_and_save_smart(symbols):
    """增量同步股票数据到单表 sp500"""
    engine = get_engine()
    table_name = 'sp500'
    
    for i, symbol in enumerate(symbols):
        symbol = symbol.upper()
        print(f"[{i+1}/{len(symbols)}] 🔄 正在同步 {symbol}...")

        # 1. 检查本地数据库中该股票的最新日期
        last_date = None
        try:
            query = text(f"SELECT MAX(date) FROM {table_name} WHERE symbol = :sym")
            with engine.connect() as conn:
                result = conn.execute(query, {"sym": symbol}).scalar()
                if result:
                    last_date = pd.to_datetime(result)
                    print(f"   📅 本地最后记录: {last_date.date()}")
        except Exception:
            # 首次运行或表不存在
            print(f"   🆕 准备全量下载 (最近5年)")

        # 2. 确定下载范围并抓取数据
        try:
            if last_date:
                start_date = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                # 关键：auto_adjust=False 确保获取原始 Adj Close
                df = yf.download(symbol, start=start_date, progress=False, auto_adjust=False)
            else:
                df = yf.download(symbol, period="5y", progress=False, auto_adjust=False)

            if df.empty:
                print(f"   ☕ {symbol} 已经是最新，跳过。")
                continue

            # 3. 强力清洗数据列名
            df = df.reset_index()
            
            clean_cols = []
            for col in df.columns:
                # 处理 yfinance 的 MultiIndex 或简单字符串
                col_name = str(col[0] if isinstance(col, tuple) else col)
                # 转换：小写、去空格、空格换下划线 (解决 adj close -> adj_close)
                clean_name = col_name.strip().lower().replace(' ', '_')
                clean_cols.append(clean_name)
            
            df.columns = clean_cols

            # 4. 补救 adj_close 缺失
            if 'adj_close' not in df.columns:
                if 'close' in df.columns:
                    df['adj_close'] = df['close']
                    print("   💡 使用 close 填充缺失的 adj_close")
                else:
                    print(f"   ❌ {symbol} 数据不完整 (缺收盘价)，跳过")
                    continue

            # 5. 添加标签并整理列顺序
            df['symbol'] = symbol
            cols_order = ['date', 'symbol', 'adj_close', 'close', 'open', 'high', 'low', 'volume']
            # 仅保留我们需要的列，防止 yfinance 多余列干扰
            df = df[[c for c in cols_order if c in df.columns]]

            # 6. 存入数据库
            df.to_sql(table_name, engine, if_exists='append', index=False)
            print(f"   ✅ 成功同步 {len(df)} 行数据")

        except Exception as e:
            print(f"   ❌ {symbol} 同步过程中出错: {e}")

if __name__ == "__main__":
    # 获取代码
    sp500_tickers = get_sp500_tickers()
    
    if sp500_tickers:
        # 执行同步
        download_and_save_smart(sp500_tickers)
        print("\n🎉 标普500数据同步任务圆满完成！")
    else:
        print("🚫 未能获取股票列表，同步终止。")