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
    files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not files:
        raise FileNotFoundError("❌ checkpoints 文件夹下没有找到任何 .pth 模型文件")
    files.sort(key=lambda x: float(x.split('valLoss_')[-1].replace('.pth', '')))
    best_model = files[0]
    print(f"🏆 自动选择表现最好的模型: {best_model}")
    return best_model

def run_cross_sectional_backtest_v3():
    engine = get_engine()
    target_stocks = ['nvda', 'aapl', 'tsla', 'msft', 'googl', 'amzn', 'meta']
    
    print("1. 加载双核模型与大盘滤网...")
    model = LSTMQuantModel(input_dim=6, hidden_dim=64, num_layers=2, num_stocks=7, embed_dim=8)
    best_path = get_best_checkpoint()
    model.load_state_dict(torch.load(best_path)) 
    model.eval()

    # --- 🛡️ 优化点 1：读取本地 QQQ 趋势 ---
    try:
        qqq_df = pd.read_sql("SELECT date, close FROM qqq ORDER BY date", engine)
        qqq_df['date'] = pd.to_datetime(qqq_df['date']).dt.tz_localize(None).dt.normalize()
        qqq_df = qqq_df.set_index('date')['close']
        market_bullish = qqq_df > qqq_df.rolling(200).mean()
        print("✅ 成功加载本地 QQQ 大盘滤网")
    except Exception as e:
        print(f"⚠️ 无法加载 QQQ 数据，请确保本地 DB 有 qqq 表。错误: {e}")
        return

    all_probs = {}
    all_closes = {}
    
    print("2. 正在生成全市场预测矩阵...")
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
            print(f"   ⚠️ 跳过 {stock.upper()}，原因: {e}")

    # 3. 矩阵对齐与平滑
    prob_df = pd.DataFrame(all_probs).fillna(0)
    close_df = pd.DataFrame(all_closes).ffill()
    
    prob_df.index = pd.to_datetime(prob_df.index).tz_localize(None).normalize()
    close_df.index = pd.to_datetime(close_df.index).tz_localize(None).normalize()

    # --- 🛡️ 优化点 2：概率平滑 ---
    prob_df = prob_df.ewm(span=3).mean() 

    print("\n3. 执行带滤网的截面轮动逻辑...")
    rank_df = prob_df.rank(axis=1, ascending=False)
    m_filter = market_bullish.reindex(prob_df.index, method='ffill').fillna(False)

    # 入场信号：排名<=2 & 概率>=0.45 & 大盘牛市
    PROB_THRESHOLD = 0.45
    clean_entries = (rank_df <= 2) & (prob_df >= PROB_THRESHOLD) & m_filter.to_frame().reindex_like(prob_df, method='ffill').iloc[:,0].values[:,None]
    # 出场信号：排名>3 或 大盘熊市
    clean_exits = (rank_df > 3) | (~m_filter.to_frame().reindex_like(prob_df, method='ffill').iloc[:,0].values[:,None])

    # =========================================================
    # 🚀 4. 【动态仓位推进器】(Confidence-Weighted Sizing)
    # =========================================================
    print("\n3.5 启动【动态仓位推进器】(概率加权模式)...")
    size_df = pd.DataFrame(0.0, index=clean_entries.index, columns=clean_entries.columns)
    
    for date, row in clean_entries.iterrows():
        active_stocks = row[row == True].index
        if len(active_stocks) == 0: continue
            
        # 获取这些股票对应的模型平滑后概率
        active_probs = prob_df.loc[date, active_stocks]
        
        # 计算“超额信心值” (超出 0.45 门槛的部分)
        # 加上微小常数防止分母为 0
        confidences = (active_probs - PROB_THRESHOLD).clip(lower=0.001)
        
        if len(active_stocks) == 1:
            # 单兵作战：如果概率极高(如0.6以上)，直接重仓(98%)；刚及格则轻仓
            # 缩放因子 0.15 表示：如果概率比门槛高出 0.15，则认为极其自信
            single_size = (confidences.iloc[0] / 0.15) * 0.98
            size_df.loc[date, active_stocks[0]] = min(single_size, 0.98)
            
        elif len(active_stocks) >= 2:
            # 双兵作战：按信心比例瓜分总头寸
            # 将 active_stocks 按概率从高到低排序，确保 vectorbt 顺序执行时逻辑正确
            sorted_stocks = active_probs.sort_values(ascending=False).index
            total_conf = confidences.sum()
            
            # 第一只股票 (最强信心)：占据总现金的 (权重 * 98%)
            weight_0 = confidences[sorted_stocks[0]] / total_conf
            size_df.loc[date, sorted_stocks[0]] = weight_0 * 0.98
            
            # 第二只股票 (次强信心)：占据【剩余现金】的几乎全部
            # 在 vectorbt cash_sharing 模式下，第二个 size 填 0.99 会买入剩余现金的 99%
            size_df.loc[date, sorted_stocks[1]] = 0.99

    print("\n4. 启动 Vectorbt 投资组合级回测 (全动态分配版)...")
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
        sl_stop=0.04,  # 严格止损
        tp_stop=0.15
    )
    
    print("\n" + "="*40)
    print("📈 全市场截面轮动策略 v3 + 动态仓位")
    print("="*40)
    print(pf.stats())
    
    # 展示价值曲线
    pf.value().vbt.plot(trace_kwargs=dict(name='V3 Dynamic 净值曲线')).show()

if __name__ == "__main__":
    run_cross_sectional_backtest_v3()