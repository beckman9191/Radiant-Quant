import pandas as pd
import yfinance as yf
import requests
import io
import os
from sqlalchemy import text, create_engine
from datetime import datetime, timedelta
from db_engine import get_engine

class MacroDataSyncer:
    def __init__(self):
        self.engine = get_engine()
        self.table_name = "macro_features_store"
        # 对应关系
        self.ticker_map = {
            '^VIX': 'vix_index',
            'BZ=F': 'brent_oil_price',
            '^TNX': 'us_10y_yield'
        }

    def get_latest_date(self):
        try:
            with self.engine.connect() as conn:
                query = text(f"SELECT MAX(date) FROM {self.table_name}")
                result = conn.execute(query).scalar()
                return pd.to_datetime(result) if result else None
        except Exception:
            return None

    def fetch_market_data(self, start_date):
        print(f"📈 正在抓取数据 (从: {start_date.date()})...")
        data_frames = []
        
        for ticker, col_name in self.ticker_map.items():
            # 获取单只数据
            df = yf.download(ticker, start=start_date, end=datetime.now(), progress=False)
            
            if not df.empty:
                # 关键修复点：处理 MultiIndex
                # 如果 yfinance 返回的是多级索引，只取 Close 这一级
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.xs('Close', axis=1, level=0)
                else:
                    df = df[['Close']]
                
                # 转换为普通的 DataFrame 并重命名列
                df = df.rename(columns={df.columns[0]: col_name})
                data_frames.append(df)
        
        if not data_frames:
            return pd.DataFrame()

        # 合并所有宏观因子
        combined = pd.concat(data_frames, axis=1)
        # 将日期索引转为列
        return combined.reset_index()

    def sync(self):
        last_date = self.get_latest_date()
        start_date = last_date + timedelta(days=1) if last_date else pd.to_datetime("2024-01-01")
        
        # 避免重复跑当天的
        if start_date.date() > datetime.now().date():
            print("✨ 数据库已是最新。")
            return

        df = self.fetch_market_data(start_date)
        if df.empty:
            print("📭 未发现新数据。")
            return

        # 确保列名全小写且为字符串
        df.columns = [str(col).lower() for col in df.columns]
        
        # 地缘政治风险补丁 (针对 2026-02-28 军事行动)
        if 'gpr_index' not in df.columns:
            df['gpr_index'] = 100.0
            
        war_mask = (df['date'] >= '2026-02-28')
        df.loc[war_mask, 'gpr_index'] = 450.0

        # 数据清洗：填充缺失值
        df = df.ffill().bfill()

        # 写入数据库
        print(f"💾 正在写入 {len(df)} 条新记录到 {self.table_name}...")
        try:
            df.to_sql(self.table_name, self.engine, if_exists='append', index=False)
            print("✅ 宏观因子库同步成功！")
        except Exception as e:
            print(f"❌ 写入失败: {e}")

if __name__ == "__main__":
    MacroDataSyncer().sync()