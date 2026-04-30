import os
import torch
import pandas as pd
import numpy as np
import datetime
import glob
import alpaca_trade_api as tradeapi
import yfinance as yf  # 🆕 新增：用于拉取财报日历

from models.lstm_model_15 import LSTMQuantModel
from data_pipeline.feature_eng import StockDatasetBinary

# === 1. 配置 Alpaca API (模拟盘) ===
API_KEY = "PK4NNX3MY5XBA6LSRYFYPAFSOU"
SECRET_KEY = "6f5L5zr4FgJZokwnmMk2WML5r8re9hLBcuqHGAWX6qST"
BASE_URL = "https://paper-api.alpaca.markets"

TARGET_STOCKS = [
    'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA',
    'AMD', 'SMCI', 'ARM', 'AVGO', 'TSM', 
    'PLTR', 'CRWD', 'COIN'
]

# 策略参数
PROB_ENTRY_THRESHOLD = 0.50 
PROB_EXIT_THRESHOLD = 0.45  
SL_STOP = 0.04              
EARNINGS_BLACKOUT_DAYS = 2  # 🆕 财报前 2 天内强制清仓/禁止买入

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

class AlpacaBotV7:
    def __init__(self):
        self.api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
        self.model = self._load_model()
        
    def _load_model(self):
        date_dirs = sorted(glob.glob("checkpoints/[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]"), reverse=True)
        latest_dir = date_dirs[0] if date_dirs else "checkpoints"
        files = glob.glob(os.path.join(latest_dir, "*.pth"))

        if not files:
            raise FileNotFoundError("😱 找不到模型文件！")

        best_path = sorted(files, key=lambda x: float(x.split('valLoss_')[-1].replace('.pth', '')))[0]
        
        model = LSTMQuantModel(input_dim=6, hidden_dim=64, num_layers=2, num_stocks=15, embed_dim=8)
        model.load_state_dict(torch.load(best_path))
        model.eval()
        print(f"🎯 已自动锁定最新 V7 模型: {best_path}")
        return model

    def _is_earnings_approaching(self, symbol, days_threshold):
        """🆕 检查某只股票是否即将发布财报"""
        try:
            ticker = yf.Ticker(symbol)
            earnings_dates = ticker.earnings_dates
            if earnings_dates is None or earnings_dates.empty:
                return False
                
            now = pd.Timestamp.now(tz='UTC')
            # 过滤出未来的财报日期
            future_earnings = earnings_dates[earnings_dates.index > now]
            
            if future_earnings.empty:
                return False
                
            next_earnings_date = future_earnings.index[0]
            days_to_earnings = (next_earnings_date - now).days
            
            if 0 <= days_to_earnings <= days_threshold:
                print(f"⚠️ 财报警报: {symbol} 将在 {days_to_earnings} 天后发财报 (日期: {next_earnings_date.strftime('%Y-%m-%d')})")
                return True
                
            return False
        except Exception as e:
            return False

    def fetch_data(self, symbols):
        start_date = (datetime.datetime.now() - datetime.timedelta(days=500)).strftime('%Y-%m-%d')
        data = {}
        for s in symbols:
            bars = self.api.get_bars(s, tradeapi.rest.TimeFrame.Day, start=start_date, adjustment='all').df
            if not bars.empty and len(bars) >= 200:
                df = bars.copy()
                df.columns = [c.lower() for c in df.columns]
                data[s] = df
        return data

    def get_signals(self):
        all_data = self.fetch_data(TARGET_STOCKS + ['QQQ'])
        
        if 'QQQ' not in all_data:
            return {}, {}, False

        qqq_series = all_data['QQQ']['close'].ffill() 
        current_price = qqq_series.iloc[-1]
        ma200_series = qqq_series.rolling(window=200, min_periods=150).mean()
        current_ma = ma200_series.iloc[-1]

        market_bullish = current_price > current_ma if not pd.isna(current_ma) else True
        if not market_bullish:
            return {}, {}, False

        probs, rsi_dict = {}, {}
        for idx, s in enumerate(TARGET_STOCKS):
            if s not in all_data: continue
            df = all_data[s]
            rsi_dict[s] = calculate_rsi(df['close'], period=14).iloc[-1]
            
            dataset = StockDatasetBinary(df, stock_id=idx, window_size=30)
            lookback_days = min(10, len(dataset.X)) 
            if lookback_days == 0: continue
                
            with torch.no_grad():
                out = self.model(dataset.X[-lookback_days:], dataset.stock_ids[-lookback_days:])
                raw_probs = torch.sigmoid(out).numpy().flatten()
            
            probs[s] = pd.Series(raw_probs).ewm(span=3).mean().iloc[-1]
            
        return probs, rsi_dict, True

    def execute_trades(self):
        probs, rsi_dict, is_market_ok = self.get_signals()
        
        if not is_market_ok:
            print("🚨 大盘护盾触发！清空所有头寸。")
            self.api.close_all_positions()
            return

        sorted_probs = pd.Series(probs).sort_values(ascending=False)
        top_2_stocks = sorted_probs[sorted_probs >= PROB_ENTRY_THRESHOLD].head(2)
        top_3_stocks = sorted_probs[sorted_probs >= PROB_EXIT_THRESHOLD].head(3)
        
        # 🆕 1. 启动全市场财报雷达，收集高危名单
        print("\n📅 正在扫描 15 股财报日历，启动财报避雷针...")
        earnings_blackout = set()
        for symbol in TARGET_STOCKS:
            if self._is_earnings_approaching(symbol, EARNINGS_BLACKOUT_DAYS):
                earnings_blackout.add(symbol)
        
        account = self.api.get_account()
        equity = float(account.equity)
        
        positions = self.api.list_positions()
        valid_current_positions = {}
        
        # 🆕 2. 当日黑名单（包含：刚触发止损的股票 + 即将发财报的股票）
        blacklisted_symbols = set(earnings_blackout)
        
        print("\n🔪 开始持仓风控扫描...")
        for p in positions:
            symbol = p.symbol
            qty = int(p.qty)
            avg_entry = float(p.avg_entry_price)
            current_price = float(p.current_price)
            
            # 🆕 A. 财报避雷针拦截 (最高优先级)
            if symbol in earnings_blackout:
                print(f"  🛡️ 财报避雷针触发: {symbol} 即将发财报，为了规避跳空风险，执行强制清仓！")
                self.api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='day')
                continue

            # B. 4% 熔断硬止损
            loss_pct = (current_price - avg_entry) / avg_entry
            if loss_pct <= -SL_STOP:
                print(f"  🩸 触发止损离场: {symbol} (现跌 {loss_pct:.2%})")
                self.api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='day')
                blacklisted_symbols.add(symbol) # 🆕 加入黑名单，防止洗仓
                continue 

            # C. 跌落神坛换仓
            if symbol not in top_3_stocks.index:
                print(f"  📉 动能衰退，执行清仓: {symbol}")
                self.api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='day')
                continue
                
            valid_current_positions[symbol] = qty

        target_quantities = {}
        if len(top_2_stocks) > 0:
            base_weight = 0.95 / len(top_2_stocks)
            
            for symbol, prob in top_2_stocks.items():
                # 🆕 D. 买入拦截器
                if symbol in blacklisted_symbols:
                    if symbol in earnings_blackout:
                        print(f"  🚫 建仓拦截: {symbol} 位于财报静默期，系统拒绝买入指令！")
                    else:
                        print(f"  🚫 冷却期拦截: {symbol} 今日刚触发止损，系统防洗仓机制拒绝重新买入！")
                    continue
                
                current_rsi = rsi_dict.get(symbol, 50.0)
                if current_rsi < 35: multiplier = 1.0 
                elif current_rsi > 75: multiplier = 0.3
                else: multiplier = 0.6 
                
                target_value = equity * (base_weight * multiplier)
                price = self.api.get_latest_trade(symbol).price
                target_quantities[symbol] = int(target_value / price)

        print("\n⚖️ 启动高频做 T 动态平衡...")
        for symbol, qty in valid_current_positions.items():
            target_qty = target_quantities.get(symbol, 0)
            if qty > target_qty:
                diff = qty - target_qty
                print(f"  🔻 高抛/减仓: {symbol} | 卖出: {diff} 股")
                self.api.submit_order(symbol=symbol, qty=diff, side='sell', type='market', time_in_force='day')

        for symbol, target_qty in target_quantities.items():
            current_qty = valid_current_positions.get(symbol, 0)
            if target_qty > current_qty:
                diff = target_qty - current_qty
                print(f"  🚀 抄底/加仓: {symbol} | 买入: {diff} 股")
                self.api.submit_order(symbol=symbol, qty=diff, side='buy', type='market', time_in_force='day')

        print(f"\n🏁 V7 实盘调仓执行完毕。系统终极持仓目标: {target_quantities}")

if __name__ == "__main__":
    bot = AlpacaBotV7()
    bot.execute_trades()