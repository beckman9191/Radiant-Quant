import os
import torch
import pandas as pd
import numpy as np
import datetime
import glob
import alpaca_trade_api as tradeapi
from models.lstm_model import LSTMQuantModel
from data_pipeline.feature_eng import StockDatasetBinary

# === 1. 配置 Alpaca API (模拟盘) ===
API_KEY = "PK4NNX3MY5XBA6LSRYFYPAFSOU"
SECRET_KEY = "6f5L5zr4FgJZokwnmMk2WML5r8re9hLBcuqHGAWX6qST"
BASE_URL = "https://paper-api.alpaca.markets" # ⚠️ 模拟盘专用地址

# 策略参数
TARGET_STOCKS = ['NVDA', 'AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'META']
PROB_THRESHOLD = 0.45
SL_STOP = 0.04  # 4% 硬止损线

class AlpacaBot:
    def __init__(self):
        self.api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
        self.model = self._load_model()
        
    def _load_model(self):
        # 1. 找到 checkpoints 下所有的日期文件夹并排序，取最新的一个
        date_dirs = sorted(glob.glob("checkpoints/[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]"), reverse=True)
        
        if not date_dirs:
            files = glob.glob("checkpoints/*.pth")
        else:
            latest_dir = date_dirs[0]
            print(f"📂 发现最新训练目录: {latest_dir}")
            files = glob.glob(os.path.join(latest_dir, "*.pth"))

        if not files:
            raise FileNotFoundError("😱 找不到任何模型文件，请先运行 train.py 进行训练！")

        # 2. 按 valLoss 升序排列，取最小值
        best_path = sorted(files, key=lambda x: float(x.split('valLoss_')[-1].replace('.pth', '')))[0]
        
        # 3. 加载模型
        model = LSTMQuantModel(input_dim=6, hidden_dim=64, num_layers=2, num_stocks=7, embed_dim=8)
        model.load_state_dict(torch.load(best_path))
        model.eval()
        
        print(f"🎯 已自动锁定最新日期最强模型: {best_path}")
        return model

    def fetch_data(self, symbols):
        """抓取最新 K 线数据 (强制拉取历史版)"""
        start_date = (datetime.datetime.now() - datetime.timedelta(days=500)).strftime('%Y-%m-%d')
        data = {}
        for s in symbols:
            print(f"📡 正在从 Alpaca 同步 {s} 历史数据...")
            bars = self.api.get_bars(
                s, 
                tradeapi.rest.TimeFrame.Day, 
                start=start_date, 
                adjustment='all'
            ).df
            
            if bars.empty or len(bars) < 200:
                print(f"⚠️ {s} 数据量不足 ({len(bars)} 条)，跳过")
                continue
                
            df = bars.copy()
            df.columns = [c.lower() for c in df.columns]
            data[s] = df
        return data

    def get_signals(self):
        """核心推理逻辑 - 【补丁3：EWM平滑对齐】"""
        all_data = self.fetch_data(TARGET_STOCKS + ['QQQ'])
        
        # 1. 🔍 大盘滤网诊断
        if 'QQQ' not in all_data or all_data['QQQ'].empty:
            print("❌ 无法获取 QQQ 数据，请检查网络或 API Key")
            return {}, False

        qqq_series = all_data['QQQ']['close'].ffill() 
        current_price = qqq_series.iloc[-1]
        
        ma200_series = qqq_series.rolling(window=200, min_periods=150).mean()
        current_ma = ma200_series.iloc[-1]

        print(f"📊 滤网诊断 -> QQQ 当前价: {current_price:.2f} | 200MA: {current_ma:.2f} | 样本量: {len(qqq_series)}")

        if pd.isna(current_ma):
            market_bullish = True
        else:
            market_bullish = current_price > current_ma
        
        if not market_bullish:
            print(f"🛑 大盘趋势走弱 (QQQ {current_price:.2f} < 200MA {current_ma:.2f})，今日保持观望")
            return {}, False

        # 2. 标的推理 (带 3天 EWM 平滑)
        probs = {}
        print("\n🧠 正在执行 LSTM 双核推理与 EWM 概率平滑...")
        for idx, s in enumerate(TARGET_STOCKS):
            df = all_data[s]
            dataset = StockDatasetBinary(df, stock_id=idx, window_size=30)
            
            # 提取过去 10 天的特征，以供 EWM 平滑预热
            lookback_days = min(10, len(dataset.X)) 
            if lookback_days == 0:
                continue
                
            recent_x = dataset.X[-lookback_days:] 
            recent_id = dataset.stock_ids[-lookback_days:]
            
            with torch.no_grad():
                out = self.model(recent_x, recent_id)
                raw_probs = torch.sigmoid(out).numpy().flatten()
            
            # 核心对齐：指数加权移动平均，剔除单日噪音
            smoothed_series = pd.Series(raw_probs).ewm(span=3).mean()
            final_prob = smoothed_series.iloc[-1]
            probs[s] = final_prob
            
            print(f"  👉 {s:<5} | 原始概率: {raw_probs[-1]:.3f} -> EWM平滑: {final_prob:.3f}")
        
        return probs, True
    
    def get_scale_factor(self, prob):
        """根据概率决定仓位级别（分批逻辑）"""
        if prob < 0.45: return 0.0
        if prob < 0.50: return 0.3  # 轻仓试探
        if prob < 0.60: return 0.7  # 中仓确认
        return 1.0                  # 满仓冲锋

    def execute_trades(self):
        """执行调仓逻辑 - 【终极对齐版】"""
        probs, is_market_ok = self.get_signals()
        
        # 1. 大盘滤网清仓
        if not is_market_ok:
            print("🚨 大盘风险！执行清仓。")
            self.api.close_all_positions()
            return

        sorted_probs = pd.Series(probs).sort_values(ascending=False)
        print(f"\n📊 今日最终平滑预测概率:\n{sorted_probs}")

        # 买入目标池：前 2 名
        top_2_stocks = sorted_probs[sorted_probs >= PROB_THRESHOLD].head(2)
        # 【补丁2：排名宽容度】 离场缓冲池：前 3 名
        top_3_stocks = sorted_probs[sorted_probs >= PROB_THRESHOLD].head(3)
        
        account = self.api.get_account()
        equity = float(account.equity)
        
        # --- 2. 检查现有持仓：执行止损与退场逻辑 ---
        positions = self.api.list_positions()
        valid_current_positions = {}
        
        print("\n🛡️ 开始风控与离场检查...")
        for p in positions:
            symbol = p.symbol
            qty = int(p.qty)
            avg_entry = float(p.avg_entry_price)
            current_price = float(p.current_price)
            
            # 【补丁1：4% 硬止损】
            loss_pct = (current_price - avg_entry) / avg_entry
            if loss_pct <= -SL_STOP:
                print(f"  🩸 触发硬止损: {symbol} (亏损 {loss_pct:.2%} | 成本 {avg_entry:.2f} -> 现价 {current_price:.2f})")
                self.api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='day')
                continue # 已卖出，不再参与后续计算

            # 【补丁2：排名宽容度】 只要还在前 3 名，或者因为止损等原因还没达到，就不强行清仓
            if symbol not in top_3_stocks.index:
                print(f"  📉 跌出前三 / 概率过低，执行离场换仓: {symbol}")
                self.api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='day')
                continue
                
            valid_current_positions[symbol] = qty

        # --- 3. 计算目标持仓 ---
        target_quantities = {}
        if len(top_2_stocks) > 0:
            # 每只股票最大占用 47.5% 本金
            base_weight = 0.95 / len(top_2_stocks)
            
            for symbol, prob in top_2_stocks.items():
                scale = self.get_scale_factor(prob)
                target_weight = base_weight * scale
                target_value = equity * target_weight
                
                price = self.api.get_latest_trade(symbol).price
                target_quantities[symbol] = int(target_value / price)

        # --- 4. 仓位动态平衡 (多退少补) ---
        print("\n⚖️ 开始执行仓位动态平衡...")
        
        # A. 减仓逻辑
        for symbol, qty in valid_current_positions.items():
            target_qty = target_quantities.get(symbol, 0)
            if qty > target_qty:
                diff = qty - target_qty
                print(f"  🔻 动态减仓: {symbol} | 卖出数量: {diff}")
                self.api.submit_order(symbol=symbol, qty=diff, side='sell', type='market', time_in_force='day')

        # B. 加仓/买入逻辑
        for symbol, target_qty in target_quantities.items():
            current_qty = valid_current_positions.get(symbol, 0)
            if target_qty > current_qty:
                diff = target_qty - current_qty
                print(f"  🚀 动态加仓: {symbol} | 目标概率: {probs[symbol]:.2f} | 买入数量: {diff}")
                self.api.submit_order(symbol=symbol, qty=diff, side='buy', type='market', time_in_force='day')

        print(f"\n🏁 调仓指令下达完毕。目标应持仓股数: {target_quantities}")

if __name__ == "__main__":
    bot = AlpacaBot()
    bot.execute_trades()