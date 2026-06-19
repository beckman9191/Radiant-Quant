import pandas as pd
from sqlalchemy import create_engine
from alphas.smart_momentum import SmartRetailMomentum
import matplotlib.pyplot as plt
from data_pipeline.db_engine import get_engine

def load_and_prepare_data(engine):
    print("📥 正在从数据库提取 S&P 500 数据...")
    # 提取回测所需的最小数据集
    query = "SELECT date, symbol, adj_close FROM sp500"
    df = pd.read_sql(query, engine)
    
    # 数据清洗：确保日期格式，并透视为宽表（横截面矩阵）
    df['date'] = pd.to_datetime(df['date'])
    prices = df.pivot(index='date', columns='symbol', values='adj_close')
    
    # 填充缺失值（处理停牌）并剔除全空股票
    prices = prices.ffill().dropna(axis=1, how='all')
    print(f"✅ 数据准备就绪：包含 {prices.shape[1]} 只股票，时间范围 {prices.index.min().date()} 至 {prices.index.max().date()}")
    return prices

def execute_backtest(prices):
    strategy = SmartRetailMomentum()
    
    # --- 修改这里：获取每个月实际存在的最后一个交易日 ---
    # 逻辑：按年和月分组，取该组内 index 的最大值
    rebalance_dates = prices.index.to_series().groupby(
        [prices.index.year, prices.index.month]
    ).max()
    
    # 转换为 DatetimeIndex 格式，方便后续计算
    rebalance_dates = pd.DatetimeIndex(rebalance_dates.values)
    
    results = []
    dates = []

    print("🚀 开始执行回测循环...")
    for i in range(len(rebalance_dates) - 1):
        curr_date = rebalance_dates[i]
        next_date = rebalance_dates[i+1]
        
        # 确保有足够的回溯期数据（252天）
        if curr_date < prices.index[252]:
            continue

        # 1. 生成信号 (计算排名得分)
        scores = strategy.score(prices, curr_date)
        
        # 2. 构建组合 (权重分配：前20%多，后20%空)
        weights = strategy.portfolio(scores)
        
        # 3. 计算收益率：下个月所有股票的变动
        # 向量化计算：(下月价格 / 本月价格) - 1
        monthly_returns = (prices.loc[next_date] / prices.loc[curr_date]) - 1
        
        # 4. 组合收益 = 权重点乘收益
        strategy_return = (weights * monthly_returns).sum()
        
        results.append(strategy_return)
        dates.append(next_date)

    return pd.Series(results, index=dates)

if __name__ == "__main__":
    # 执行流程
    engine = get_engine()
    prices_matrix = load_and_prepare_data(engine)
    strategy_returns = execute_backtest(prices_matrix)
    
    # 计算累计净值
    cumulative_nav = (1 + strategy_returns).cumprod()
    
    # --- 可视化 ---
    plt.figure(figsize=(12, 6))
    cumulative_nav.plot(label="Smart Momentum (Long/Short)", color="#2ecc71", lw=2)
    plt.axhline(1, color='red', linestyle='--', alpha=0.5)
    plt.title("S&P 500 Cross-Sectional Momentum Backtest", fontsize=14)
    plt.ylabel("Cumulative Returns")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # 输出核心指标
    total_ret = cumulative_nav.iloc[-1] - 1
    # 计算简单夏普比率 (假设无风险利率为0)
    sharpe = (strategy_returns.mean() / strategy_returns.std()) * (12**0.5)
    print(f"\n📊 回测报告:")
    print(f"总累计收益率: {total_ret:.2%}")
    print(f"年化夏普比率: {sharpe:.2f}")