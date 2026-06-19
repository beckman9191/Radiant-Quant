import pandas as pd
import numpy as np
import os
import joblib
import time
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sqlalchemy import text

# 1. 导入你的数据库配置
from data_pipeline.db_engine import get_engine

class RidgeTuner:
    def __init__(self, train_window_weeks=20, sentiment_threshold=0.3):
        self.engine = get_engine()
        self.train_window = train_window_weeks
        self.sentiment_threshold = sentiment_threshold
        self.feature_cols = ['momentum', 'sentiment', 'tick_imbalance', 'mean_reversion']
        self.target_col = 'target_return'
        
        # 调试用的参数列表
        self.alphas = [1, 10, 100, 500, 1000, 5000]

    def load_and_preprocess(self):
        print("📥 正在从数据库读取历史数据...")
        query = f"SELECT * FROM alpha_feature_store ORDER BY date ASC"
        df = pd.read_sql(query, self.engine)
        df['date'] = pd.to_datetime(df['date'])

        # 情绪去噪逻辑
        # 处理我们之前设定的微小占位符
        placeholder_mask = df['sentiment'].between(0.00000, 0.00005)
        df.loc[placeholder_mask, 'sentiment'] = 0.0
        
        # 阈值去噪
        noise_mask = df['sentiment'].abs() < self.sentiment_threshold
        df.loc[noise_mask, 'sentiment'] = 0.0
        
        print(f"🧹 情绪去噪完成 (Threshold: {self.sentiment_threshold})")
        return df

    def cross_sectional_scaling(self, df):
        """横截面 Z-Score 标准化"""
        def zscore(x):
            std = x.std()
            return (x - x.mean()) / (std + 1e-9)
        
        for col in self.feature_cols:
            df[col] = df.groupby('date')[col].transform(zscore)
        return df

    def tune_alpha(self):
        raw_df = self.load_and_preprocess()
        df = self.cross_sectional_scaling(raw_df).dropna(subset=[self.target_col])
        unique_dates = sorted(df['date'].unique())
        
        print(f"🚀 开始在 {len(unique_dates)} 周数据上进行 Alpha 搜索...")
        
        # 存储结果：{alpha: [ic_fold1, ic_fold2, ...]}
        alpha_perf = {a: [] for a in self.alphas}

        for i in range(self.train_window, len(unique_dates)):
            train_dates = unique_dates[i - self.train_window : i]
            test_date = unique_dates[i]
            
            train_set = df[df['date'].isin(train_dates)]
            test_set = df[df['date'] == test_date]
            
            X_train, y_train = train_set[self.feature_cols], train_set[self.target_col]
            X_test, y_test = test_set[self.feature_cols], test_set[self.target_col]

            for a in self.alphas:
                model = Ridge(alpha=a)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                # 计算 Rank IC (Spearman 相关系数)
                # 使用 Series 处理防止 ConstantInputWarning 导致的 NaN
                ic = pd.Series(preds).corr(pd.Series(y_test.values), method='spearman')
                if not np.isnan(ic):
                    alpha_perf[a].append(ic)
            
            if i % 5 == 0:
                print(f"  进度: 已完成至 {test_date.date()}")

        # 汇总评估指标
        summary = []
        for a in self.alphas:
            ics = alpha_perf[a]
            if not ics: continue
            
            mean_ic = np.mean(ics)
            std_ic = np.std(ics)
            # 信息比率 (IR) = 平均 IC / IC 标准差
            ir = mean_ic / std_ic if std_ic > 0 else 0
            
            summary.append({
                'alpha': a,
                'mean_rank_ic': mean_ic,
                'std_rank_ic': std_ic,
                'IR': ir,
                'positive_weeks': sum(1 for x in ics if x > 0) / len(ics)
            })
            
        return pd.DataFrame(summary).sort_values('IR', ascending=False)

if __name__ == "__main__":
    # 作为华为工程师，你可以在这里根据你的 NPU/GPU 环境调整 alphas
    tuner = RidgeTuner(train_window_weeks=20, sentiment_threshold=0.3)
    results = tuner.tune_alpha()
    
    print("\n" + "="*50)
    print("📊 策略：Alpha 调优最终报告 (按 IR 排序)")
    print("="*50)
    print(results.to_string(index=False))
    print("="*50)
    
    best_alpha = results.iloc[0]['alpha']
    print(f"\n💡 建议选择 alpha = {best_alpha}")
    print(f"接下来，请将 train_wfa_ridge.py 中的 Ridge(alpha=1.0) 改为 Ridge(alpha={best_alpha})")