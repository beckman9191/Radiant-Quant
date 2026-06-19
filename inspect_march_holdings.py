import pandas as pd
import joblib
import os
from datetime import datetime
from data_pipeline.db_engine import get_engine

def inspect_march_top_picks(checkpoint_dir, sentiment_threshold=0.3):
    engine = get_engine()
    feature_cols = ['momentum', 'sentiment', 'tick_imbalance', 'mean_reversion']
    
    # 1. 筛选 3 月份的交易日（周五）
    march_dates = ['20260306', '20260313', '20260320', '20260327']
    
    # 2. 读取并预处理全量数据
    print("📥 正在提取 3 月份特征数据...")
    query = "SELECT * FROM alpha_feature_store WHERE date >= '2026-03-01' AND date <= '2026-03-31'"
    df = pd.read_sql(query, engine)
    df['date'] = pd.to_datetime(df['date'])

    # 3. 同步预处理逻辑
    df.loc[df['sentiment'].between(0.00000, 0.00005), 'sentiment'] = 0.0
    df.loc[df['sentiment'].abs() < sentiment_threshold, 'sentiment'] = 0.0
    
    def zscore(x):
        return (x - x.mean()) / (x.std() + 1e-8)
    for col in feature_cols:
        df[col] = df.groupby('date')[col].transform(zscore)

    # 4. 逐周提取 Q5 名单
    for d_str in march_dates:
        model_path = os.path.join(checkpoint_dir, f"ridge_fold_{d_str}.joblib")
        if not os.path.exists(model_path):
            continue
            
        model = joblib.load(model_path)
        week_data = df[df['date'] == datetime.strptime(d_str, '%Y%m%d')].copy()
        
        if week_data.empty:
            continue
            
        # 预测并分层
        week_data['prediction'] = model.predict(week_data[feature_cols])
        week_data['quintile'] = pd.qcut(week_data['prediction'], 5, labels=False)
        
        # 提取 Q5 (预测最看好的股票)
        q5_holdings = week_data[week_data['quintile'] == 4].sort_values('prediction', ascending=False)
        
        print(f"\n📅 报告日期: {d_str}")
        print(f"✅ Q5 持仓数量: {len(q5_holdings)}")
        print(f"📊 权重因子平均值: \n{q5_holdings[feature_cols].mean().to_string()}")
        
        # 输出前 20 只重点持仓及其实际收益率
        print("\n🔝 Top 20 Symbols in Q5:")
        print(q5_holdings[['symbol', 'prediction', 'target_return', 'sentiment', 'momentum']].head(20).to_string(index=False))
        print("-" * 50)

if __name__ == "__main__":
    CKPT_PATH = "checkpoints/RidgeRegression/2026-05-02"
    inspect_march_top_picks(CKPT_PATH)