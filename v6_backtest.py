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
    """优先寻找最新日期文件夹，挑选 valLoss 最低的模型"""
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
    """
    【方案A核心】：纯 Pandas 手写 RSI 计算公式，完美绕开 numba 依赖报错。
    采用 Wilder 的指数平滑法，与专业炒股软件计算结果完全一致。
    """
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def run_core_satellite_backtest_v6():
    engine = get_engine()
    target_stocks = ['nvda', 'aapl', 'tsla', 'msft', 'googl', 'amzn', 'meta']
    
    print("1. 加载 LSTM 模型与提取特征...")
    model = LSTMQuantModel(input_dim=6, hidden_dim=64, num_layers=2, num_stocks=7, embed_dim=8)
    model.load_state_dict(torch.load(get_best_checkpoint()))
    model.eval()

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

    # --- 核心 LSTM 平滑与排名 ---
    prob_df = prob_df.ewm(span=3).mean()
    rank_df = prob_df.rank(axis=1, ascending=False)

    print("2. 正在计算全市场 RSI (使用原生 Pandas 加速)...")
    rsi_df = close_df.apply(lambda x: calculate_rsi(x, period=14))

    print("3. 执行 V6: Core-Satellite 动态做 T 逻辑...")
    # ==========================================
    # 🎯 选股逻辑 (LSTM 负责大方向，找龙头)
    # ==========================================
    core_entries = (rank_df <= 2) & (prob_df >= 0.50)
    core_exits = (rank_df > 3) | (prob_df < 0.45)

    size_df = pd.DataFrame(0.0, index=prob_df.index, columns=prob_df.columns)
    
    # ==========================================
    # 🧠 仓位控制器 (RSI 负责高抛低吸)
    # ==========================================
    for date, row in core_entries.iterrows():
        active_stocks = row[row == True].index
        if len(active_stocks) == 0: continue
            
        # 给每只选出的龙头股分配基准资金 (假设选出 2 只，每只最大可用资金 48%)
        base_allocation = 0.96 / len(active_stocks) 
        
        for stock in active_stocks:
            current_rsi = rsi_df.loc[date, stock]
            
            if pd.isna(current_rsi):
                size_df.loc[date, stock] = base_allocation * 0.5 # RSI 数据不足时给半仓
                continue

            # --- 动态做 T 核心 ---
            if current_rsi < 30:
                # 【超跌抄底】：满仓游击队！(占用分配给该股的 100% 资金)
                size_df.loc[date, stock] = base_allocation * 1.0 
                
            elif current_rsi > 80:
                # 【高抛止盈】：收割游击队！(只保留底仓，即占用该股资金的 30%)
                size_df.loc[date, stock] = base_allocation * 0.3 
                
            else:
                # 【稳健持有】：正常震荡区间，保持 60% 中等仓位
                size_df.loc[date, stock] = base_allocation * 0.6 

    print("4. 启动 Vectorbt 波段交易回测引擎...")
    pf = vbt.Portfolio.from_signals(
        close=close_df,
        entries=core_entries,
        exits=core_exits,
        size=size_df,              
        size_type='percent',        
        cash_sharing=True,          
        init_cash=100000,
        fees=0.001,    # 万分之十的手续费，检验做 T 的真实盈利能力
        freq='1D',
        sl_stop=0.04,  # 依然保持 4% 铁血硬止损
    )
    
    print("\n" + "="*45)
    print("📈 V6: LSTM(趋势) + RSI(做T) 增强策略 (纯血 Pandas 版)")
    print("="*45)
    print(pf.stats())
    
    # 画出净值曲线图
    pf.value().vbt.plot(trace_kwargs=dict(name='V6 Swing 净值曲线')).show()

if __name__ == "__main__":
    run_core_satellite_backtest_v6()