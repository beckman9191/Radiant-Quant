# 逻辑：(收盘价 - 最低价) - (最高价 - 收盘价) / (最高价 - 最低价) * 成交量
# 这是一个经典的日线级资金流向指标 (ADL 变体)
def score(self, price_df, volume_df, date):
    # 简单的日内强弱：收盘价在当日振幅的位置
    # 计算当前成交量相对于过去 20 天平均成交量的倍数
    relative_vol = volume_df.loc[date] / volume_df.rolling(20).mean().loc[date]
    
    # 结合价格涨跌幅
    returns = price_df.pct_change().loc[date]
    money_flow_proxy = returns * relative_vol
    
    # 横截面 Z-Score
    return (money_flow_proxy - money_flow_proxy.mean()) / money_flow_proxy.std()