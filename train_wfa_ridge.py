import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.linear_model import Ridge
from sqlalchemy import text
import warnings

# 导入你的配置
from data_pipeline.db_engine import get_engine

class WFATrainer:
    def __init__(self, train_window_weeks=20, alpha=500.0, sentiment_threshold=0.3):
        self.engine = get_engine()
        self.train_window = train_window_weeks
        self.alpha = alpha 
        self.sentiment_threshold = sentiment_threshold
        
        # 基础特征 + 宏观特征
        self.stock_features = ['momentum', 'sentiment', 'tick_imbalance', 'mean_reversion']
        self.macro_features = ['vix_index', 'oil_ret', 'gpr_index']
        self.feature_cols = self.stock_features + self.macro_features
        
        self.target_col = 'target_return'
        
        now_str = datetime.now().strftime('%Y-%m-%d')
        self.ckpt_path = f"checkpoints/RidgeRegression/{now_str}"
        os.makedirs(self.ckpt_path, exist_ok=True)

    def load_data(self):
        """
        通过 SQL JOIN 将个股特征与宏观因子拼接
        """
        print("📥 正在执行多表关联查询 (个股 + 宏观)...")
        query = """
        SELECT 
            a.*, 
            m.vix_index, 
            m.brent_oil_price, 
            m.gpr_index 
        FROM alpha_feature_store a
        LEFT JOIN macro_features_store m ON a.date = m.date
        ORDER BY a.date ASC
        """
        df = pd.read_sql(query, self.engine)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def preprocess_data(self, df):
        """
        预处理：包含个股去噪和宏观因子平稳化
        """
        # 1. 情绪去噪
        df.loc[df['sentiment'].between(0.00000, 0.00005), 'sentiment'] = 0.0
        df.loc[df['sentiment'].abs() < self.sentiment_threshold, 'sentiment'] = 0.0

        # 2. 宏观因子处理：将原油价格转换为周收益率 (平稳化)
        # 注意：由于同一天内宏观值相同，我们取 date 分组后的第一个值计算 pct_change
        macro_oil = df.groupby('date')['brent_oil_price'].first().pct_change()
        df['oil_ret'] = df['date'].map(macro_oil).fillna(0)

        # 3. 收益率缩尾
        df[self.target_col] = df[self.target_col].clip(
            lower=df[self.target_col].quantile(0.01), 
            upper=df[self.target_col].quantile(0.99)
        )
        
        return df

    def cross_sectional_scaling(self, df):
        """
        横截面标准化：$$ \hat{X} = \frac{X - \mu}{\sigma + \epsilon} $$
        """
        def zscore(x):
            std = x.std()
            if std == 0 or np.isnan(std):
                return x - x.mean()
            return (x - x.mean()) / (std + 1e-8)
        
        # 对所有特征进行标准化
        for col in self.feature_cols:
            df[col] = df.groupby('date')[col].transform(zscore)
            
        df[self.target_col] = df.groupby('date')[self.target_col].transform(zscore)
        return df

    def run_training(self):
        raw_df = self.load_data()
        df_cleaned = self.preprocess_data(raw_df)
        df = self.cross_sectional_scaling(df_cleaned).dropna(subset=[self.target_col])
        
        unique_dates = sorted(df['date'].unique())
        print(f"📊 数据就绪。总周数: {len(unique_dates)} | 包含特征: {self.feature_cols}")

        results = []

        for i in range(self.train_window, len(unique_dates)):
            train_dates = unique_dates[i - self.train_window : i]
            test_date = unique_dates[i]
            
            train_set = df[df['date'].isin(train_dates)]
            test_set = df[df['date'] == test_date]
            
            X_train, y_train = train_set[self.feature_cols], train_set[self.target_col]
            X_test, y_test = test_set[self.feature_cols], test_set[self.target_col]

            # 5. 训练 Ridge 模型
            model = Ridge(alpha=self.alpha)
            model.fit(X_train, y_train)

            # 保存模型
            model_filename = f"ridge_fold_{test_date.strftime('%Y%m%d')}.joblib"
            joblib.dump(model, os.path.join(self.ckpt_path, model_filename))

            # 6. 计算 Rank IC
            preds = model.predict(X_test)
            
            # --- 动态防御逻辑 ---
            # 检查宏观风险：如果当前 VIX 或 GPR 异常，我们虽然记录 IC，但可以在实盘中通过此逻辑标记“不建议操作”
            curr_vix = test_set['vix_index'].iloc[0]
            curr_gpr = test_set['gpr_index'].iloc[0]
            risk_flag = " [RISK]" if curr_vix > 35 or curr_gpr > 300 else ""
            
            rank_ic = pd.Series(preds).corr(pd.Series(y_test.values), method='spearman')
            results.append({'date': test_date, 'rank_ic': rank_ic})
            
            status = "✅" if rank_ic > 0 else "⚠️"
            print(f"{status} Fold {i-self.train_window+1}: {test_date.date()} | Rank IC: {rank_ic:.4f}{risk_flag}")

        # 8. 总结
        res_df = pd.DataFrame(results).dropna()
        print(f"\n📈 策略训练完成！平均 Rank IC: {res_df['rank_ic'].mean():.4f}")
        print(f"模型保存在: {self.ckpt_path}")

if __name__ == "__main__":
    trainer = WFATrainer(train_window_weeks=20, alpha=500.0)
    trainer.run_training()