import os
import glob
import pandas as pd
import numpy as np
import torch
import vectorbt as vbt
from data_pipeline.db_engine import get_engine
from data_pipeline.feature_eng import StockDatasetBinary
from models.lstm_model import LSTMQuantModel

def get_best_checkpoint(checkpoint_dir='checkpoints'):
    """寻找最新日期文件夹，挑选 valLoss 最低的模型"""
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
    """原生 Pandas 实现 RSI (Wilder 平滑法)"""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def run_v7_shield_swing_backtest():
    engine = get_engine()
    target_stocks = ['nvda', 'aapl', 'tsla', 'msft', 'googl', 'amzn', 'meta']
    
    print("1. 加载模型与提取特征...")
    model = LSTMQuantModel(input_dim=6, hidden_dim=64, num_layers=2, num_stocks=7, embed_dim=8)
    model.load_state_dict(torch.load(get_best_checkpoint()))
    model.eval()

    # --- 读取 QQQ 数据作为大盘护盾 ---
    try:
        qqq_df = pd.read_sql("SELECT date, close FROM qqq ORDER BY date", engine)
        qqq_df['date'] = pd.to_datetime(qqq_df['date']).dt.tz_localize(None).dt.normalize()
        qqq_df = qqq_df.set_index('date')['close']
        # 计算 200 日均线
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

    # 对齐大盘滤网
    m_filter = market_bullish.reindex(prob_df.index, method='ffill').fillna(False)

    # 核心 LSTM 平滑与排名
    prob_df = prob_df.ewm(span=3).mean()
    rank_df = prob_df.rank(axis=1, ascending=False)

    # 计算 RSI
    rsi_df = close_df.apply(lambda x: calculate_rsi(x, period=14))

    print("2. 执行 V7: LSTM + RSI做T + QQQ护盾 逻辑...")
    
    # 入场：大盘走牛 & 排名靠前 & 概率达标
    # 使用 np.newaxis 对齐维度
    m_filter_expanded = m_filter.values[:, np.newaxis]
    entries = (rank_df <= 2) & (prob_df >= 0.50) & m_filter_expanded
    
    # 出场：跌出排名 或 概率转弱 或 大盘走熊
    exits = (rank_df > 3) | (prob_df < 0.45) | (~m_filter_expanded)

    # 仓位控制矩阵
    size_df = pd.DataFrame(0.0, index=prob_df.index, columns=prob_df.columns)
    
    for date, row in entries.iterrows():
        active_stocks = row[row == True].index
        if len(active_stocks) == 0: continue
            
        base_alloc = 0.96 / len(active_stocks) 
        
        for stock in active_stocks:
            rsi_val = rsi_df.loc[date, stock]
            if pd.isna(rsi_val):
                size_df.loc[date, stock] = base_alloc * 0.5
                continue

            # 融合做 T 逻辑
            if rsi_val < 30:
                size_df.loc[date, stock] = base_alloc * 1.0  # 超跌加码
            elif rsi_val > 80:
                size_df.loc[date, stock] = base_alloc * 0.3  # 超买减仓
            else:
                size_df.loc[date, stock] = base_alloc * 0.6  # 标准持有

    print("3. 启动 Vectorbt 策略对齐回测...")
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
        sl_stop=0.04,  # 硬止损
    )
    
    print("\n" + "="*45)
    print("📈 V7: 护盾+轮动+做T 终极回测报告")
    print("="*45)
    print(pf.stats())
    
    pf.value().vbt.plot(trace_kwargs=dict(name='V7 Shield+Swing 净值曲线')).show()

if __name__ == "__main__":
    run_v7_shield_swing_backtest()