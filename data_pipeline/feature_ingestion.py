import pandas as pd
import numpy as np
import yfinance as yf
import time
import os
import requests
import gc
import psutil
from datetime import datetime, timedelta
from sqlalchemy import inspect, text

# 1. 导入你的数据流水线和 Alpha 引擎
from db_engine import get_engine
from alphas.smart_momentum import SmartRetailMomentum
from alphas.nlp_sentiment import FinBERTSentiment
from alphas.mean_reversion import MeanReversionFactor

class FeatureIngestorYF:
    def __init__(self):
        self.engine = get_engine()
        self.table_name = 'alpha_feature_store'
        
        # 初始化因子引擎
        self.mom_engine = SmartRetailMomentum()
        self.nlp_engine = FinBERTSentiment(decay_span=20)
        self.rev_engine = MeanReversionFactor()

    def _print_status(self, message):
        process = psutil.Process(os.getpid())
        mem_gb = process.memory_info().rss / (1024 ** 3)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message} | RAM: {mem_gb:.2f}GB")

    def get_sp500_tickers(self):
        """从维基百科获取 S&P 500 列表"""
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            response = requests.get(url, headers=headers)
            table = pd.read_html(response.text)
            df = table[0]
            # yfinance 喜欢用 '-' 代替 '.' (如 BRK-B)
            tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
            return tickers
        except Exception as e:
            print(f"❌ 标的获取失败: {e}")
            return []

    def check_and_create_table(self):
        """数据库表结构初始化"""
        inspector = inspect(self.engine)
        if not inspector.has_table(self.table_name):
            print(f"📡 正在 Neon 中创建 {self.table_name}...")
            create_query = f"""
            CREATE TABLE {self.table_name} (
                date DATE NOT NULL,
                symbol TEXT NOT NULL,
                momentum FLOAT,
                sentiment FLOAT,
                tick_imbalance FLOAT,
                mean_reversion FLOAT,
                target_return FLOAT,
                PRIMARY KEY (date, symbol)
            );
            """
            with self.engine.connect() as conn:
                conn.execute(text(create_query))
                conn.commit()

    def sync(self, lookback_days=365):
        self.check_and_create_table()
        tickers = self.get_sp500_tickers()
        if not tickers: return

        self._print_status(f"🚀 yfinance 引擎启动。标的数量: {len(tickers)}")

        # --- 核心改进：批量抓取所有数据 ---
        # 32GB RAM 完全可以一次性 hold 住 500 只票 2 年的日线
        # --- 核心修复：移除不兼容的 proxy 参数 ---
        print(f"📥 正在从 Yahoo Finance 批量下载数据 (过去 {lookback_days + 365} 天)...")
        
        data = yf.download(
            tickers, 
            period="2y", 
            interval="1d", 
            group_by='column', 
            threads=True
            # proxy=None  <-- 删掉这一行
        )

        if data.empty:
            print("❌ 未能获取到数据，请检查网络连接。")
            return

        # 提取 Close 和 Volume 矩阵
        price_df = data['Close'].ffill()
        volume_df = data['Volume'].ffill()
        
        # --- 核心修复：确保 date 永远是 index 中存在的真实交易日 ---
        # 按照周五频率分组，提取每一组（每一周）中实际存在的最后一个交易日索引
        rebalance_dates = price_df.index.to_series().groupby(
            pd.Grouper(freq='W-FRI')
        ).max().dropna()

        # 仅保留 lookback_days 范围内的数据
        rebalance_dates = rebalance_dates[rebalance_dates > (datetime.now() - timedelta(days=lookback_days))]

        self._print_status(f"📊 待处理节点: {len(rebalance_dates)} 个周五（或当周最后交易日）")

        for date in rebalance_dates:
            date_str = date.strftime('%Y-%m-%d')
            
            # --- 计算 Alpha 载荷 ---
            f_mom = self.mom_engine.score(price_df, date)
            f_rev = self.rev_engine.score(price_df, date)
            
            # 日线级资金流代理 (基于量价配合)
            # 计算逻辑：当日收益率 * (当日成交量 / 20日均成交量)
            returns = price_df.pct_change().loc[date]
            vol_ratio = volume_df.loc[date] / volume_df.rolling(20).mean().loc[date]
            f_tick = (returns * vol_ratio).rank(pct=True) # 归一化到 0-1

            # 目标收益率 (Target Label: 下周五收益)
            future_date = date + timedelta(days=7)
            idx = price_df.index.get_indexer([future_date], method='nearest')[0]
            actual_future_date = price_df.index[idx]
            target_ret = np.log(price_df.loc[actual_future_date] / price_df.loc[date])

            # 组装横截面 DataFrame
            batch_df = pd.DataFrame({
                'date': date.date(),
                'symbol': price_df.columns,
                'momentum': f_mom,
                'sentiment': 0.0, # 留空给 NLP 推理
                'tick_imbalance': f_tick,
                'mean_reversion': f_rev,
                'target_return': target_ret
            }).dropna(subset=['momentum', 'mean_reversion'])

            # 同步至 Neon
            try:
                # 写入数据库
                batch_df.to_sql(self.table_name, self.engine, if_exists='append', index=False)
                self._print_status(f"✅ {date_str} 同步完成 ({len(batch_df)} 条记录)")
            except Exception as e:
                print(f"❌ {date_str} 写入失败: {e}")
            
            # 及时释放内存
            del batch_df
            gc.collect()

if __name__ == "__main__":
    ingestor = FeatureIngestorYF()
    ingestor.sync(lookback_days=365)