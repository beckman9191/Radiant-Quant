import os
import glob
import pandas as pd
import torch
import vectorbt as vbt
import numpy as np

from data_pipeline.db_engine import get_engine
from data_pipeline.feature_eng import StockDatasetBinary
from models.lstm_model import LSTMQuantModel

def get_best_checkpoint(checkpoint_dir='checkpoints'):
    """从文件夹中自动寻找文件名中 Loss 最低的那个模型"""
    files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not files:
        raise FileNotFoundError("❌ checkpoints 文件夹下没有找到任何 .pth 模型文件")
    
    files.sort(key=lambda x: float(x.split('valLoss_')[-1].replace('.pth', '')))
    best_model = files[0]
    print(f"🏆 自动选择表现最好的模型: {best_model}")
    return best_model

def run_single_backtest(target_stock='nvda'):
    print(f"1. 连接数据库，获取 {target_stock.upper()} 历史数据...")
    engine = get_engine()
    df = pd.read_sql(f"SELECT * FROM {target_stock} ORDER BY date", engine)
    
    dates = pd.to_datetime(df['date'].values)
    
    print("2. 提取特征矩阵 (已适配 ATR 自适应引擎)...")
    # 【核心匹配】：使用 atr_multiplier=1.5
    dataset = StockDatasetBinary(df, window_size=30, atr_multiplier=1.5)
    
    valid_dates = dates[30 : 30 + len(dataset.X)]
    valid_close = df['close'].values[30 : 30 + len(dataset.X)]
    
    print("3. 加载深度学习模型 (特征维度升级为 input_dim=10)...")
    # 【核心匹配】：input_dim=10
    model = LSTMQuantModel(input_dim=10, hidden_dim=64, num_layers=2)
    best_path = get_best_checkpoint()
    model.load_state_dict(torch.load(best_path))
    model.eval()
    
    print("4. 生成交易信号 (有限状态机 5 天强制平仓)...")
    with torch.no_grad():
        outputs = model(dataset.X)
        probs = torch.sigmoid(outputs).numpy().flatten()
    
    # 依然使用 0.40 作为开仓阈值
    raw_entries = probs > 0.40 
    
    clean_entries = np.zeros(len(raw_entries), dtype=bool)
    clean_exits = np.zeros(len(raw_entries), dtype=bool)
    
    in_position = False 
    days_held = 0       
    
    for i in range(len(raw_entries)):
        if in_position:
            days_held += 1
            if days_held == 5:
                clean_exits[i] = True
                in_position = False
                days_held = 0
        else:
            if raw_entries[i]:
                clean_entries[i] = True
                in_position = True
                days_held = 0
                
    print("\n5. 启动 vectorbt 引擎计算收益...")
    price_series = pd.Series(valid_close, index=valid_dates)
    entries_series = pd.Series(clean_entries, index=valid_dates)
    exits_series = pd.Series(clean_exits, index=valid_dates)
    
    pf = vbt.Portfolio.from_signals(
        close=price_series,
        entries=entries_series,
        exits=exits_series,
        init_cash=10000,
        fees=0.001,
        freq='1D',
        # 保留硬风控
        sl_stop=0.05,  
        tp_stop=0.15   
    )
    
    print("\n" + "="*40)
    print(f"📈 {target_stock.upper()} 单标的深度学习回测报告 (ATR版)")
    print("="*40)
    stats = pf.stats()
    print(f"总收益率 (Total Return): {stats['Total Return [%]']:.2f}%")
    print(f"夏普比率 (Sharpe Ratio): {stats['Sharpe Ratio']:.2f}")
    print(f"最大回撤 (Max Drawdown): {stats['Max Drawdown [%]']:.2f}%")
    print(f"总交易次数: {stats['Total Trades']}")
    
    pf.plot().show()

if __name__ == "__main__":
    run_single_backtest('aapl')