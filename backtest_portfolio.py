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
    """从文件夹中自动寻找文件名中 Loss 最低的那个模型"""
    files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not files:
        raise FileNotFoundError("❌ checkpoints 文件夹下没有找到任何 .pth 模型文件")
    
    files.sort(key=lambda x: float(x.split('valLoss_')[-1].replace('.pth', '')))
    best_model = files[0]
    print(f"🏆 自动选择表现最好的模型: {best_model}")
    return best_model

def run_cross_sectional_backtest():
    engine = get_engine()
    # 【注意】：这里的股票顺序必须和跑 run_train_lstm.py 时一模一样！
    # 这样才能保证 NVDA 永远是 0 号，AAPL 永远是 1 号。
    target_stocks = ['nvda', 'aapl', 'tsla', 'msft', 'googl', 'amzn', 'meta']
    
    print("1. 加载全局自适应模型 (带 Entity Embeddings)...")
    # 【适配 1】：模型初始化加入 num_stocks 和 embed_dim
    model = LSTMQuantModel(input_dim=10, hidden_dim=64, num_layers=2, num_stocks=7, embed_dim=8)
    best_path = get_best_checkpoint()
    model.load_state_dict(torch.load(best_path)) 
    model.eval()

    all_probs = {}
    all_closes = {}
    
    print("2. 正在生成全市场预测矩阵...")
    # 【适配 2】：用 enumerate 提取每只股票的专属 ID (idx)
    for idx, stock in enumerate(target_stocks):
        try:
            print(f"   📊 正在扫描: {stock.upper()} (ID: {idx})")
            df = pd.read_sql(f"SELECT * FROM {stock} ORDER BY date", engine)
            if len(df) < 100: continue
            
            dates = pd.to_datetime(df['date'].values)
            
            # 【适配 3】：把股票 ID 传给特征工厂
            dataset = StockDatasetBinary(df, stock_id=idx, window_size=30, atr_multiplier=1.5)
            valid_dates = dates[30 : 30 + len(dataset.X)]
            
            with torch.no_grad():
                # 【适配 4】：推理解码时，左脑喂盘面(X)，右脑喂身份(stock_ids)
                outputs = model(dataset.X, dataset.stock_ids)
                probs = torch.sigmoid(outputs).numpy().flatten()
                
            all_probs[stock] = pd.Series(probs, index=valid_dates)
            all_closes[stock] = pd.Series(df['close'].values[30 : 30 + len(dataset.X)], index=valid_dates)
        except Exception as e:
            print(f"   ⚠️ 跳过 {stock.upper()}，原因: {e}")

    # 组合成 DataFrame 矩阵
    prob_df = pd.DataFrame(all_probs).fillna(0)
    close_df = pd.DataFrame(all_closes).ffill()
    
    print("\n3. 执行纯截面轮动矩阵 (废弃手工FSM，解决风控脱节Bug)...")
    # 计算每天所有股票的横截面排名 (1 是最高概率)
    rank_df = prob_df.rank(axis=1, ascending=False)

    # 极其优雅的矩阵化条件 (直接生成连续的布尔矩阵)
    # 买入信号：只要排名前 2，就一直亮绿灯 (True)
    clean_entries = rank_df <= 2
    # 卖出信号：只要掉出前 3 名，就亮红灯强制换车 (True)
    clean_exits = rank_df > 3

    print("\n3.5 正在生成动态资金分配矩阵...")
    size_df = pd.DataFrame(0.0, index=clean_entries.index, columns=clean_entries.columns)

    for date, row in clean_entries.iterrows():
        # 找出当天亮绿灯的股票 (最多 2 只)
        active_stocks = row[row == True].index
        
        if len(active_stocks) == 1:
            size_df.loc[date, active_stocks[0]] = 0.99 
        elif len(active_stocks) >= 2:
            size_df.loc[date, active_stocks[0]] = 0.50
            size_df.loc[date, active_stocks[1]] = 0.99

    print("\n4. 启动 Vectorbt 投资组合级回测 (全量资金池动态分配模式)...")
    pf = vbt.Portfolio.from_signals(
        close=close_df,
        entries=clean_entries,
        exits=clean_exits,
        size=size_df,               # 🔑 传入我们刚刚做好的动态资金比例矩阵
        size_type='percent',        # 🔑 退回引擎支持的 Percent 模式
        cash_sharing=True,          # 🔑 强迫 7 只股票共享同一个 10 万美金池子
        init_cash=100000,
        fees=0.001,
        freq='1D',
        sl_stop=0.05,
        tp_stop=0.15
    )
    
    print("\n" + "="*40)
    print("📈 全市场截面轮动策略回测报告 (ATR + FSM + 实体嵌入)")
    print("="*40)
    print(pf.stats())
    
    # 绘制策略总净值曲线
    fig = pf.value().vbt.plot(trace_kwargs=dict(name='策略总净值 (Portfolio Value)'))
    fig.show()

if __name__ == "__main__":
    run_cross_sectional_backtest()