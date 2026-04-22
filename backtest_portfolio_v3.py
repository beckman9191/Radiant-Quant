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
    """
    升级版：优先寻找最新日期文件夹，并在其中挑选 valLoss 最低的模型
    """
    # 1. 查找所有形如 YYYY-MM-DD 的日期文件夹
    date_pattern = os.path.join(checkpoint_dir, "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    date_dirs = sorted(glob.glob(date_pattern), reverse=True)
    
    if not date_dirs:
        # 兼容逻辑：如果没有日期文件夹，则在根目录下查找
        print(f"⚠️ 未发现日期子目录，尝试从 {checkpoint_dir} 根目录查找...")
        search_path = os.path.join(checkpoint_dir, "*.pth")
    else:
        # 锁定最新日期文件夹
        latest_dir = date_dirs[0]
        print(f"📅 锁定最新训练日期: {os.path.basename(latest_dir)}")
        search_path = os.path.join(latest_dir, "*.pth")

    # 2. 查找目标文件夹下的所有模型文件
    files = glob.glob(search_path)
    
    if not files:
        raise FileNotFoundError(f"❌ 在 {search_path} 路径下没有找到任何 .pth 模型文件")

    # 3. 按照文件名中的 valLoss 数值升序排列 (取最小值)
    try:
        files.sort(key=lambda x: float(x.split('valLoss_')[-1].replace('.pth', '')))
    except (IndexError, ValueError) as e:
        raise ValueError(f"❌ 模型文件名格式不正确，无法提取 valLoss: {e}")

    best_model = files[0]
    print(f"🏆 最终选择模型: {best_model}")
    return best_model

def run_cross_sectional_backtest_v3():
    engine = get_engine()
    target_stocks = ['nvda', 'aapl', 'tsla', 'msft', 'googl', 'amzn', 'meta']
    
    print("1. 加载双核模型与大盘滤网...")
    model = LSTMQuantModel(input_dim=6, hidden_dim=64, num_layers=2, num_stocks=7, embed_dim=8)
    best_path = get_best_checkpoint()
    model.load_state_dict(torch.load(best_path)) 
    model.eval()

    # --- 🛡️ 优化点 1：从本地数据库读取 QQQ 趋势 ---
    try:
        qqq_df = pd.read_sql("SELECT date, close FROM qqq ORDER BY date", engine)
        qqq_df['date'] = pd.to_datetime(qqq_df['date']).dt.tz_localize(None).dt.normalize()
        qqq_df = qqq_df.set_index('date')['close']
        market_bullish = qqq_df > qqq_df.rolling(200).mean()
        print("✅ 成功加载本地 QQQ 大盘滤网")
    except Exception as e:
        print(f"⚠️ 无法加载 QQQ 数据，请先运行下载脚本增加 QQQ。错误: {e}")
        return

    all_probs = {}
    all_closes = {}
    
    print("2. 正在生成全市场预测矩阵...")
    for idx, stock in enumerate(target_stocks):
        try:
            df = pd.read_sql(f"SELECT * FROM {stock} ORDER BY date", engine)
            if len(df) < 100: continue
            
            # 统一日期格式：无时区，0点
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
            print(f"   ⚠️ 跳过 {stock.upper()}，原因: {e}")

    # 3. 矩阵对齐与平滑
    prob_df = pd.DataFrame(all_probs).fillna(0)
    close_df = pd.DataFrame(all_closes).ffill()
    
    # 【核心修复】：强行同步所有 DataFrame 的索引格式
    prob_df.index = pd.to_datetime(prob_df.index).tz_localize(None).normalize()
    close_df.index = pd.to_datetime(close_df.index).tz_localize(None).normalize()

    # --- 🛡️ 优化点 2：概率平滑，减少剧烈调仓 ---
    prob_df = prob_df.ewm(span=3).mean() 

    print("\n3. 执行带滤网的截面轮动逻辑...")
    rank_df = prob_df.rank(axis=1, ascending=False)
    # 将大盘滤网对齐到交易日
    m_filter = market_bullish.reindex(prob_df.index, method='ffill').fillna(False)

    # --- 🛡️ 优化点 3：严苛入场条件 ---
    # 逻辑：排名前2 + 概率高于0.45 + 大盘必须是牛市
    clean_entries = (rank_df <= 2) & (prob_df >= 0.45) & m_filter.to_frame().reindex_like(prob_df, method='ffill').iloc[:,0].values[:,None]
    # 出场：掉出前3 或 大盘转熊
    clean_exits = (rank_df > 3) | (~m_filter.to_frame().reindex_like(prob_df, method='ffill').iloc[:,0].values[:,None])

    # 4. 生成资金分配矩阵 (保持 50/50 分配)
    size_df = pd.DataFrame(0.0, index=clean_entries.index, columns=clean_entries.columns)
    for date, row in clean_entries.iterrows():
        active_stocks = row[row == True].index
        if len(active_stocks) == 1:
            size_df.loc[date, active_stocks[0]] = 0.99 
        elif len(active_stocks) >= 2:
            size_df.loc[date, active_stocks[0]] = 0.50
            size_df.loc[date, active_stocks[1]] = 0.99

    print("\n4. 启动 Vectorbt 投资组合级回测 (V3 攻守兼备版)...")
    pf = vbt.Portfolio.from_signals(
        close=close_df,
        entries=clean_entries,
        exits=clean_exits,
        size=size_df,               
        size_type='percent',        
        cash_sharing=True,          
        init_cash=100000,
        fees=0.001,
        freq='1D',
        sl_stop=0.04,  # 收紧止损
        tp_stop=0.15
    )
    
    print("\n" + "="*40)
    print("📈 全市场截面轮动策略 v3 战报")
    print("="*40)
    print(pf.stats())
    pf.value().vbt.plot(trace_kwargs=dict(name='V3 净值曲线')).show()

if __name__ == "__main__":
    run_cross_sectional_backtest_v3()