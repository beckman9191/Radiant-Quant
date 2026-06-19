import pandas as pd
import numpy as np

class SmartRetailMomentum:
    """
    机构级动量因子 (周频优化版)
    用于为 Ridge 回归提供横截面动量特征。
    """
    def __init__(self):
        # 保持多时间窗口以捕捉不同尺度的趋势
        self.lookbacks = [21, 63, 126, 252]
        self.cur_w = pd.Series(dtype=float)

    def score(self, prices, target_date):
        """
        计算特定日期的横截面动量得分 (经过周度平滑)
        """
        # 1. 获取目标日期前一周的数据，用于平滑排名
        # 这样可以防止周五单日的异常波动干扰信号
        history = prices.loc[:target_date].tail(5)
        
        daily_ranks = []
        for d in history.index:
            scores = pd.DataFrame()
            for lb in self.lookbacks:
                # 计算过去 lb 天的收益率
                r = prices.pct_change(lb).loc[d]
                # 横截面排名 (0到1之间)
                scores[f'{lb}'] = r.rank(pct=True)
            daily_ranks.append(scores.mean(axis=1))
            
        # 2. 取过去一周排名的平均值，作为最终因子载荷
        avg_rank = pd.concat(daily_ranks, axis=1).mean(axis=1)
        
        # 3. Z-Score 标准化 (适配 Ridge 回归)
        return (avg_rank - avg_rank.mean()) / avg_rank.std()

    def get_factor_loading(self, prices, freq='W-FRI'):
        """
        【优化版】仅在调仓日生成因子载荷，极大提升训练集构建速度
        """
        # 筛选出所有的周五
        rebalance_dates = prices.resample(freq).last().index
        all_scores = {}
        
        print(f"⏳ 正在生成动量因子载荷 (频率: {freq})...")
        for date in rebalance_dates:
            # 确保有足够的数据计算最长窗口
            if len(prices.loc[:date]) < max(self.lookbacks): 
                continue
            all_scores[date] = self.score(prices, date)
    
        return pd.DataFrame(all_scores).T

    def portfolio(self, scores, n_stocks=3):
        """
        独立策略逻辑：根据得分构建简单的多空组合
        注：在使用 Ridge 模型时，通常不直接调用此函数
        """
        w = pd.Series(0.0, index=scores.index)
        # 做多前 n 只，做空后 n 只
        w[scores.nlargest(n_stocks).index] = 1.0 / n_stocks
        w[scores.nsmallest(n_stocks).index] = -1.0 / n_stocks
        
        # 换手率控制
        if not self.cur_w.empty:
            chg = (w - self.cur_w).abs().sum()
            if chg > 0.5:
                w = self.cur_w + 0.5 * (w - self.cur_w)
                
        self.cur_w = w
        return w