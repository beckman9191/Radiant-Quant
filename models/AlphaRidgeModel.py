import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import os

# 导入你的 Alpha 因子库
from alphas.smart_momentum import SmartRetailMomentum
from alphas.nlp_sentiment import FinBERTSentiment
from alphas.tick_imbalance import TickImbalanceFactor
from alphas.mean_reversion import MeanReversionFactor

class AlphaRidgeModel:
    def __init__(self, alpha=10.0):
        """
        alpha: Ridge 正则化强度。
        由于因子间存在共线性（如动量和超买超卖），建议设为 10.0 以上起步。
        """
        self.model = Ridge(alpha=alpha, fit_intercept=True)
        self.scaler = StandardScaler()
        
        # 初始化四大因子引擎
        self.mom_engine = SmartRetailMomentum()
        self.nlp_engine = FinBERTSentiment(decay_span=20) # 适配周频
        self.rev_engine = MeanReversionFactor()
        
        self.factor_names = ['momentum', 'sentiment', 'mean_reversion']

    def prepare_training_data(self, price_df, raw_sentiment_df, intraday_dict):
        """
        构建训练矩阵 X 和 标签 y
        price_df: 日线收盘价 (Dates x Stocks)
        raw_sentiment_df: 原始情绪分 (Dates x Stocks)
        intraday_dict: 1min线字典 {'NVDA': df, ...}
        """
        # 1. 确定周五调仓日
        rebalance_dates = price_df.resample('W-FRI').last().index
        
        all_samples = []
        all_targets = []
        
        print("🏗️ 正在拼接 Alpha 特征矩阵...")
        
        # 2. 计算目标变量 (未来 5 天的对数收益率)
        # 物理意义：我们希望模型预测的是“从本周五到下周五”能涨多少
        target_returns = np.log(price_df.shift(-5) / price_df)
        
        for date in rebalance_dates:
            if date not in price_df.index or date not in target_returns.index:
                continue
            
            # 确保有足够数据计算长周期动量
            if len(price_df.loc[:date]) < 252: continue
            
            # --- 调用四大金刚的 score 方法 ---
            f_mom = self.mom_engine.score(price_df, date)
            f_nlp = self.nlp_engine.score(raw_sentiment_df, date)
            f_tick = self.tick_engine.score(intraday_dict, date)
            f_rev = self.rev_engine.score(price_df, date)
            
            # --- 拼接横截面特征 ---
            # 将 15 只股票的四个因子拼成一个 DataFrame
            df_step = pd.DataFrame({
                'momentum': f_mom,
                'sentiment': f_nlp,
                'imbalance': f_tick,
                'mean_reversion': f_rev
            })
            
            # 获取这 15 只股票对应的下周真实收益率
            y_step = target_returns.loc[date]
            
            # 对齐数据（过滤掉 NaN）
            combined = pd.concat([df_step, y_step.rename('target')], axis=1).dropna()
            
            if not combined.empty:
                all_samples.append(combined[self.factor_names].values)
                all_targets.append(combined['target'].values)

        # 将列表转换为 NumPy 矩阵
        X = np.vstack(all_samples)
        y = np.concatenate(all_targets)
        
        return X, y

    def train(self, X, y):
        """
        训练模型并打印因子重要性
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        # 打印各因子的 Beta 系数，看看谁最能赚钱
        weights = dict(zip(self.factor_names, self.model.coef_))
        print("\n" + "="*40)
        print("🧠 V7 Alpha 合并完成")
        for f, w in weights.items():
            print(f"因子 [{f:15}]: 权重 {w:+.6f}")
        print("="*40)

    def predict_expected_returns(self, current_prices, current_sentiment, current_intraday, target_date):
        """
        实盘预测：输出池子里每只票下周的预期收益率
        """
        f_mom = self.mom_engine.score(current_prices, target_date)
        f_nlp = self.nlp_engine.score(current_sentiment, target_date)
        f_tick = self.tick_engine.score(current_intraday, target_date)
        f_rev = self.rev_engine.score(current_prices, target_date)
        
        X_current = pd.DataFrame({
            'momentum': f_mom,
            'sentiment': f_nlp,
            'imbalance': f_tick,
            'mean_reversion': f_rev
        }).fillna(0)
        
        X_scaled = self.scaler.transform(X_current.values)
        exp_returns = self.model.predict(X_scaled)
        
        return pd.Series(exp_returns, index=X_current.index)