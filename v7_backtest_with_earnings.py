import os
import glob
import pandas as pd
import numpy as np
import torch
import vectorbt as vbt
import yfinance as yf # 🆕 引入 yfinance 获取历史财报
from data_pipeline.db_engine import get_engine
from data_pipeline.feature_eng import StockDatasetBinary
from models.lstm_model_15 import LSTMQuantModel
import warnings
warnings.filterwarnings("ignore")

def get_best_checkpoint(checkpoint_dir='checkpoints'):
    date_pattern = os.path.join(checkpoint_dir, "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    date_dirs = sorted(glob.glob(date_pattern), reverse=True)
    if not date_dirs:
        search_path = os.path.join(checkpoint_dir, "*.pth")
    else:
        search_path = os.path.join(date_dirs[0], "*.pth")
    files = glob.glob(search_path)
    files.sort(key=lambda x: float(x.split('valLoss_')[-1].replace('.pth', '')))
    return files[0]

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def build_earnings_mask(target_stocks, target_index, blackout_days=2):
    """
    【核心补丁】：构建历史财报静默期遮罩 (True 表示处于静默期，禁止做多并强制清仓)
    """
    print("\n📅 正在从 yfinance 拉取历史财报日历，构建回测避雷针...")
    mask_df = pd.DataFrame(False, index=target_index, columns=target_stocks)
    
    for stock in target_stocks:
        try:
            ticker = yf.Ticker(stock)
            # 获取历史财报日期 (limit=40 大约覆盖过去 10 年)
            earnings_dates = ticker.get_earnings_dates(limit=40)
            
            if earnings_dates is None or earnings_dates.empty:
                continue
                
            # 清洗时间区，统一为 tz-naive
            hist_dates = earnings_dates.index.tz_localize(None).normalize()
            
            # 遍历每一个财报日，将前 blackout_days 天设为 True
            for e_date in hist_dates:
                # 财报静默期范围：财报前 N 天 到 财报当天
                start_blackout = e_date - pd.Timedelta(days=blackout_days)
                
                # 找到在 target_index 中落入该范围的日期
                in_blackout = (target_index >= start_blackout) & (target_index <= e_date)
                mask_df.loc[in_blackout, stock] = True
                
        except Exception as e:
            print(f"  ⚠️ 无法获取 {stock} 的历史财报: {e}")
            
    return mask_df

def run_v7_shield_swing_backtest():
    engine = get_engine()
    target_stocks = [
        'nvda', 'aapl', 'msft', 'googl', 'amzn', 'meta', 'tsla',
        'amd', 'smci', 'arm', 'avgo', 'tsm', 
        'pltr', 'crwd', 'coin'
    ]
    
    print("1. 加载模型与提取特征...")
    # 🆕 捕获最佳模型的完整路径和文件名
    best_model_path = get_best_checkpoint()
    model_filename = os.path.basename(best_model_path)
    
    model = LSTMQuantModel(input_dim=6, hidden_dim=64, num_layers=2, num_stocks=15, embed_dim=8)
    model.load_state_dict(torch.load(best_model_path)) # 🆕 使用捕获的路径加载
    model.eval()

    try:
        qqq_df = pd.read_sql("SELECT date, close FROM qqq ORDER BY date", engine)
        qqq_df['date'] = pd.to_datetime(qqq_df['date']).dt.tz_localize(None).dt.normalize()
        qqq_df = qqq_df.set_index('date')['close']
        qqq_ma200 = qqq_df.rolling(window=200).mean()
        market_bullish = qqq_df > qqq_ma200
        print("✅ 大盘护盾（QQQ 200MA）加载成功")
    except Exception as e:
        print(f"❌ 无法加载 QQQ 数据: {e}")
        return

    all_probs = {}
    all_closes = {}
    
    for idx, stock in enumerate(target_stocks):
        try:
            df = pd.read_sql(f"SELECT * FROM {stock} ORDER BY date", engine)
            if len(df) < 100: continue
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.normalize()
            dates = df['date'].values
            
            dataset = StockDatasetBinary(df, stock_id=idx, window_size=30, atr_multiplier=1.5)
            valid_dates = dates[30 : 30 + len(dataset.X)]
            
            with torch.no_grad():
                outputs = model(dataset.X, dataset.stock_ids)
                probs = torch.sigmoid(outputs).numpy().flatten()
                
            all_probs[stock] = pd.Series(probs, index=valid_dates)
            all_closes[stock] = pd.Series(df['close'].values[30 : 30 + len(dataset.X)], index=valid_dates)
        except Exception as e:
            continue

    prob_df = pd.DataFrame(all_probs).fillna(0)
    close_df = pd.DataFrame(all_closes).ffill()
    prob_df.index = pd.to_datetime(prob_df.index).tz_localize(None).normalize()
    close_df.index = pd.to_datetime(close_df.index).tz_localize(None).normalize()

    # --- 获取财报静默期遮罩 ---
    earnings_blackout_df = build_earnings_mask(prob_df.columns, prob_df.index, blackout_days=2)

    m_filter = market_bullish.reindex(prob_df.index, method='ffill').fillna(False)
    prob_df = prob_df.ewm(span=3).mean()
    rank_df = prob_df.rank(axis=1, ascending=False)
    rsi_df = close_df.apply(lambda x: calculate_rsi(x, period=14))

    print("2. 执行 V7: LSTM + RSI做T + QQQ护盾 + 财报避雷针...")
    
    m_filter_expanded = m_filter.values[:, np.newaxis]
    
    # 【核心逻辑升级：引入 earnings_blackout_df 拦截】
    # 入场条件附加：绝对不能在财报静默期买入 (~earnings_blackout_df)
    entries = (rank_df <= 2) & (prob_df >= 0.50) & m_filter_expanded & (~earnings_blackout_df)
    
    # 出场条件附加：如果进入财报静默期，强制触发出场！
    exits = (rank_df > 3) | (prob_df < 0.45) | (~m_filter_expanded) | earnings_blackout_df

    size_df = pd.DataFrame(0.0, index=prob_df.index, columns=prob_df.columns)
    
    for date, row in entries.iterrows():
        active_stocks = row[row == True].index
        if len(active_stocks) == 0: continue
            
        base_alloc = 0.96 / len(active_stocks) 
        
        for stock in active_stocks:
            # 如果某天不小心被标记为要买入，但处于财报期，强制清零（双重保险）
            if earnings_blackout_df.loc[date, stock]:
                size_df.loc[date, stock] = 0.0
                continue
                
            rsi_val = rsi_df.loc[date, stock]
            if pd.isna(rsi_val):
                size_df.loc[date, stock] = base_alloc * 0.5
                continue

            if rsi_val < 35:
                size_df.loc[date, stock] = base_alloc * 1.0 
            elif rsi_val > 75:
                size_df.loc[date, stock] = base_alloc * 0.3 
            else:
                size_df.loc[date, stock] = base_alloc * 0.6 

    # ==========================================
    # ⏳ 新增：时间轴截断（Regime Shift Filtering）
    # ==========================================
    print("\n✂️ 正在截断时间轴：切除疫情与加息周期，仅保留过去两年...")
    # 计算两年前的日期
    start_date = pd.Timestamp.now().normalize() - pd.DateOffset(years=2)
    
    # 生成时间遮罩 (只保留 start_date 之后的数据)
    time_mask = close_df.index >= start_date
    
    # 挥下快刀，同步切断所有矩阵
    close_df = close_df[time_mask]
    entries = entries[time_mask]
    exits = exits[time_mask]
    size_df = size_df[time_mask]

    print(f"📅 实际回测执行区间: {close_df.index[0].strftime('%Y-%m-%d')} 至 {close_df.index[-1].strftime('%Y-%m-%d')}")

    print("3. 启动 Vectorbt 策略对齐回测...")
    # 注意：在 Vectorbt 中，我们无法简单用纯 Pandas 矩阵模拟“当天止损后第二天拉黑”的动态防洗仓逻辑。
    # 因为 vbt 的 sl_stop 是在底层 C/Numba 引擎里运算的。
    # 但由于回测按日运行，且止损后一般需要重新满足 rank<=2 和 prob>=0.5 才会建仓，
    # 洗仓影响在日线回测中相对实盘极短期的分钟级跳动要小很多，重点在于解决【财报缺口】问题。
    pf = vbt.Portfolio.from_signals(
        close=close_df,
        entries=entries,
        exits=exits,
        size=size_df,              
        size_type='percent',        
        cash_sharing=True,          
        init_cash=100000,
        fees=0.001,
        freq='1D',
        sl_stop=0.04,  
    )
    
    print("\n" + "="*55)
    print("📈 V7 终极一致性报告 (财报免疫版)")
    print(f"🧠 驱动大脑: {model_filename}") # 🆕 在报告头部打印模型名称
    print("="*55)
    print(pf.stats())
    
    pf.value().vbt.plot(trace_kwargs=dict(name='V7 Perfect Parity 净值曲线')).show()

if __name__ == "__main__":
    run_v7_shield_swing_backtest()