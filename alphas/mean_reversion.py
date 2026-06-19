import pandas as pd
import numpy as np

class MeanReversionFactor:
    """
    机构级均值回归因子：周度乖离度与超买超卖 (Weekly Bias & RSI)
    适配周五调仓，捕捉价格相对于中期均线的极端偏离。
    """
    def __init__(self, ma_period=20, rsi_period=14):
        self.ma_period = ma_period
        self.rsi_period = rsi_period

    def _calculate_rsi(self, series, period):
        """原生实现 RSI，避免引入额外库"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def score(self, prices, target_date):
        """
        计算周五当天的均值回归得分
        逻辑：
        1. 乖离度 (Bias)：当前价格偏离 20 日均线的百分比
        2. 相对强弱 (RSI)：14 日 RSI 的当前水平
        3. 综合两者进行反向打分（偏离越高，得分越低，代表未来回归压力越大）
        """
        # 1. 提取所需长度的历史数据
        history = prices.loc[:target_date].tail(self.ma_period + 10)
        
        # 2. 计算乖离度 (Bias)
        # 物理意义：衡量价格是否跑得太快，脱离了 20 日（约一个月）的成本中轴
        ma20 = history.rolling(window=self.ma_period).mean()
        bias = (history - ma20) / ma20
        current_bias = bias.iloc[-1]
        
        # 3. 计算 RSI
        rsi = history.apply(lambda x: self._calculate_rsi(x, self.rsi_period))
        current_rsi = rsi.iloc[-1]
        
        # 4. 因子合成：将两个超买超卖指标结合
        # 注意：均值回归是反向指标。Bias 越高，预期收益率应该越低
        # 我们对它们取负值，使其与收益率正相关（便于 Ridge 回归理解）
        combined_signal = -(current_bias.rank(pct=True) + current_rsi.rank(pct=True))
        
        # 5. 横截面 Z-Score 标准化
        if combined_signal.std() == 0 or pd.isna(combined_signal.std()):
            return pd.Series(0.0, index=combined_signal.index)
            
        z_scores = (combined_signal - combined_signal.mean()) / combined_signal.std()
        return z_scores.clip(-3, 3)

    def get_factor_loading(self, prices, rebalance_dates):
        """批量生成周调仓日期的因子载荷"""
        loading_matrix = {}
        for date in rebalance_dates:
            if len(prices.loc[:date]) < self.ma_period: continue
            loading_matrix[date] = self.score(prices, date)
        return pd.DataFrame(loading_matrix).T