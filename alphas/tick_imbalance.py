import pandas as pd
import numpy as np

class TickImbalanceFactor:
    """
    机构级微观结构因子：周度订单流失衡 (Weekly Order Flow Imbalance)
    适配周五调仓节奏，通过 1 分钟线模拟一周内的资金净流入压力。
    """
    def __init__(self):
        # 无需存储复杂状态，主要负责计算逻辑
        pass

    def _compute_daily_raw_imbalance(self, min_bars_df):
        """
        [内部方法] 基于 1 分钟数据应用 Tick Rule 判定单日失衡
        """
        if min_bars_df.empty:
            return 0.0
        
        df = min_bars_df.copy()
        
        # 1. 价格变动判定方向：上涨=+1 (买入), 下跌=-1 (卖出)
        # 使用 np.sign 捕捉价格变化的物理方向
        direction = np.sign(df['close'].diff()).replace(0, method='ffill').fillna(0)
        
        # 2. 计算有方向的成交量 (Signed Volume)
        signed_vol = direction * df['volume']
        
        # 3. 返回单日净流出比例
        total_vol = df['volume'].sum()
        return signed_vol.sum() / total_vol if total_vol > 0 else 0.0

    def score(self, intraday_data_dict, target_date):
        """
        [因子生成模块] 计算目标日期（周五）及其过去一周的累计失衡得分
        
        参数:
        intraday_data_dict: 字典 { 'NVDA': df_min_bars, ... } 包含过去一周的 1min 数据
        target_date: 调仓日 (周五)
        """
        raw_weekly_scores = {}
        
        # 确定过去 5 个交易日的范围
        start_date = pd.to_datetime(target_date) - pd.Timedelta(days=6)
        
        for symbol, df in intraday_data_dict.items():
            # 1. 截取本周内的数据
            week_data = df.loc[start_date:target_date]
            if week_data.empty:
                raw_weekly_scores[symbol] = 0.0
                continue
            
            # 2. 计算本周内每天的失衡值并取平均
            # 物理意义：这一周内资金进入的平均“积极程度”
            daily_groups = week_data.groupby(week_data.index.date)
            daily_imb = [self._compute_daily_raw_imbalance(group) for _, group in daily_groups]
            
            raw_weekly_scores[symbol] = np.mean(daily_imb) if daily_imb else 0.0
            
        scores_ser = pd.Series(raw_weekly_scores)
        
        # 3. 横截面 Z-Score 标准化
        # 找出哪些股票本周的资金流入强度显著高于池子里的其他股票
        if scores_ser.std() == 0 or pd.isna(scores_ser.std()):
            return pd.Series(0.0, index=scores_ser.index)
            
        z_scores = (scores_ser - scores_ser.mean()) / scores_ser.std()
        
        # 4. 极值处理
        return z_scores.clip(-3, 3)

    def get_factor_loading(self, all_min_bars_dict, rebalance_dates):
        """
        批量生成训练集所需的因子矩阵
        """
        loading_matrix = {}
        print("⏳ 正在聚合周度订单流失衡因子...")
        
        for date in rebalance_dates:
            loading_matrix[date] = self.score(all_min_bars_dict, date)
            
        return pd.DataFrame(loading_matrix).T