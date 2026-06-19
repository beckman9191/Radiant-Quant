import pandas as pd
import numpy as np
from transformers import pipeline
import warnings

# 忽略模型加载时的冗余警告
warnings.filterwarnings("ignore")

class FinBERTSentiment:
    """
    机构级 NLP 情绪因子 (周频优化版)
    针对每周调仓 (Weekly Rebalancing) 进行了特征聚合优化。
    """
    def __init__(self, decay_span=20):
        """
        decay_span: 情绪信号的衰减周期。
        周频策略建议设为 20 (约一个月)，以过滤短期高频噪音，保留中期趋势。
        """
        self.decay_span = decay_span
        self.nlp_model = None

    def _load_model(self):
        """延迟加载 FinBERT 模型，仅在需要推理时占用内存"""
        if self.nlp_model is None:
            print("⏳ 正在加载 FinBERT 金融情绪模型 (ProsusAI/finbert)...")
            # 该模型专门针对金融领域文本（新闻、财报）进行过微调
            self.nlp_model = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            print("✅ 模型加载完毕！")

    def get_raw_score(self, texts):
        """
        [离线/数据清洗模块] 
        将文本转化为原始数值：Positive(1.0), Negative(-1.0), Neutral(0.0)
        建议在每日数据回填 (Data Backfill) 时运行。
        """
        self._load_model()
        if not texts or (isinstance(texts, list) and len(texts) == 0):
            return []
            
        if isinstance(texts, str):
            texts = [texts]
            
        results = self.nlp_model(texts)
        scores = []
        for res in results:
            if res['label'] == 'positive':
                scores.append(res['score'])       # 极性分
            elif res['label'] == 'negative':
                scores.append(-res['score'])      # 负极性分
            else:
                scores.append(0.0)                # 中性记为 0
        return scores

    def score(self, raw_sentiment_df, target_date):
        """
        [因子生成模块] 适配周频调仓。
        
        参数:
        raw_sentiment_df: 原始分 DataFrame (Index=日期, Columns=股票代码)
        target_date: 调仓日 (通常是周五)
        
        核心逻辑:
        1. 5日滚动平均: 捕捉这一周内市场对该股的共识情绪。
        2. EWMA 衰减: 给予近期新闻更高权重，同时保留历史记忆。
        3. 横截面 Z-Score: 消除大盘整体情绪，寻找“情绪超额”标的。
        """
        # 1. 截取至目标日期，防止未来函数
        history = raw_sentiment_df.loc[:target_date]
        if history.empty:
            return pd.Series(0.0, index=raw_sentiment_df.columns)

        # 2. 【周频特供】计算 5 日滚动均值 (Weekly Sum/Mean)
        # 理由：周五调仓时，周二或周三的重大利好不应被遗忘
        weekly_aggregation = history.rolling(window=5, min_periods=1).mean()
        
        # 3. 计算指数加权移动平均 (时效性处理)
        smoothed_sentiment = weekly_aggregation.ewm(span=self.decay_span, adjust=False).mean()
        
        # 4. 提取调仓日当天的因子载荷
        current_scores = smoothed_sentiment.iloc[-1]
        
        # 5. 横截面标准化 (Cross-sectional Standardization)
        # 目的：让因子符合 Ridge 回归的输入正态分布假设
        mean_s = current_scores.mean()
        std_s = current_scores.std()
        
        if std_s == 0 or pd.isna(std_s):
            # 若全场无新闻或得分完全一致，返回 0 向量
            return pd.Series(0.0, index=current_scores.index)
            
        z_scores = (current_scores - mean_s) / std_s
        
        # 6. 极值处理 (Winsorization)
        # 将偏离度超过 3 倍标准差的极值截断，增强回归模型的鲁棒性
        return z_scores.clip(-3, 3)