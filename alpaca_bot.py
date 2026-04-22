import os
import torch
import pandas as pd
import numpy as np
import datetime
import alpaca_trade_api as tradeapi
from models.lstm_model import LSTMQuantModel
from data_pipeline.feature_eng import StockDatasetBinary

# === 1. 配置 Alpaca API (模拟盘) ===
API_KEY = "PK4NNX3MY5XBA6LSRYFYPAFSOU"
SECRET_KEY = "6f5L5zr4FgJZokwnmMk2WML5r8re9hLBcuqHGAWX6qST"
BASE_URL = "https://paper-api.alpaca.markets" # ⚠️ 模拟盘专用地址

# 策略参数 (与 V3 保持高度一致)
TARGET_STOCKS = ['NVDA', 'AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'META']
PROB_THRESHOLD = 0.45
SL_STOP = 0.04

class AlpacaBot:
    def __init__(self):
        self.api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
        self.model = self._load_model()
        
    def _load_model(self):
        # 强制使用 input_dim=6
        model = LSTMQuantModel(input_dim=6, hidden_dim=64, num_layers=2, num_stocks=7, embed_dim=8)
        # 自动获取最强 checkpoint
        import glob
        files = glob.glob("checkpoints/*.pth")
        best_path = sorted(files, key=lambda x: float(x.split('valLoss_')[-1].replace('.pth', '')))[0]
        model.load_state_dict(torch.load(best_path))
        model.eval()
        print(f"✅ 模型加载成功: {best_path}")
        return model

    def fetch_data(self, symbols):
        """抓取最新 K 线数据 (强制拉取历史版)"""
        
        # 计算 500 天前的日期，确保均线和指标有足够数据
        start_date = (datetime.datetime.now() - datetime.timedelta(days=500)).strftime('%Y-%m-%d')
        
        data = {}
        for s in symbols:
            print(f"📡 正在从 Alpaca 同步 {s} 历史数据...")
            # 增加 start 参数，显式要求拉取从 500 天前开始的数据
            bars = self.api.get_bars(
                s, 
                tradeapi.rest.TimeFrame.Day, 
                start=start_date, 
                adjustment='all'
            ).df
            
            if bars.empty or len(bars) < 200:
                print(f"⚠️ {s} 数据量不足 ({len(bars)} 条)，跳过")
                continue
                
            # 重点：加一个 .copy() 解决你看到的 SettingWithCopyWarning
            df = bars.copy()
            df.columns = [c.lower() for c in df.columns]
            data[s] = df
        return data

    def get_signals(self):
        """核心推理逻辑"""
        all_data = self.fetch_data(TARGET_STOCKS + ['QQQ'])
        
        # 1. 🔍 大盘滤网诊断
        if 'QQQ' not in all_data or all_data['QQQ'].empty:
            print("❌ 无法获取 QQQ 数据，请检查网络或 API Key")
            return {}, False

        qqq_series = all_data['QQQ']['close'].ffill() # 填充可能存在的空值
        current_price = qqq_series.iloc[-1]
        
        # 使用 min_periods=150 降低门槛，防止因为数据稍微少一点就返回 NaN
        ma200_series = qqq_series.rolling(window=200, min_periods=150).mean()
        current_ma = ma200_series.iloc[-1]

        # 【核心 Debug 打印】：这行能告诉你真相
        print(f"📊 滤网诊断 -> QQQ 当前价: {current_price:.2f} | 200MA: {current_ma:.2f} | 样本量: {len(qqq_series)}")

        # 判定逻辑：加入 NaN 检查
        if pd.isna(current_ma):
            print("⚠️ 警告：200MA 计算结果为 NaN，可能是数据量不足，强行通过滤网以进行交易")
            market_bullish = True
        else:
            market_bullish = current_price > current_ma
        
        if not market_bullish:
            print(f"🛑 大盘趋势走弱 (QQQ {current_price:.2f} < 200MA {current_ma:.2f})，今日保持观望")
            return {}, False

        # 2. 标的推理
        probs = {}
        for idx, s in enumerate(TARGET_STOCKS):
            df = all_data[s]
            # 调用你完美的精英特征提取逻辑
            dataset = StockDatasetBinary(df, stock_id=idx, window_size=30)
            
            # 取最后一行数据进行预测
            last_x = dataset.X[-1:] 
            last_id = dataset.stock_ids[-1:]
            
            with torch.no_grad():
                out = self.model(last_x, last_id)
                prob = torch.sigmoid(out).item()
                probs[s] = prob
        
        return probs, True

    def execute_trades(self):
        """执行调仓逻辑"""
        probs, is_market_ok = self.get_signals()
        
        # 如果大盘不好，执行清仓逻辑
        if not is_market_ok:
            self.api.close_all_positions()
            return

        # 获取当前胜率排名
        sorted_probs = pd.Series(probs).sort_values(ascending=False)
        top_2_stocks = sorted_probs[sorted_probs >= PROB_THRESHOLD].head(2)

        print(f"📊 今日预测结果:\n{sorted_probs}")
        print(f"🎯 拟定入场标的: {top_2_stocks.index.tolist()}")

        # 动态仓位计算
        account = self.api.get_account()
        equity = float(account.equity)
        
        # 清除不在 Top 2 里的旧持仓
        positions = self.api.list_positions()
        for p in positions:
            if p.symbol not in top_2_stocks.index:
                print(f"📉 卖出止损/轮换: {p.symbol}")
                self.api.submit_order(symbol=p.symbol, qty=p.qty, side='sell', type='market', time_in_force='day')

        # 按照信心加权买入
        if len(top_2_stocks) > 0:
            confidences = top_2_stocks - PROB_THRESHOLD
            total_conf = confidences.sum() if confidences.sum() > 0 else 0.001
            
            for symbol, conf in confidences.items():
                weight = (conf / total_conf) * 0.95 # 总计占用 95% 资金
                target_value = equity * weight
                
                # 获取最新价并下单
                price = self.api.get_latest_trade(symbol).price
                qty = int(target_value / price)
                
                if qty > 0:
                    print(f"🚀 买入/补仓: {symbol} | 权重: {weight:.2%}")
                    self.api.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='day')

if __name__ == "__main__":
    bot = AlpacaBot()
    bot.execute_trades()