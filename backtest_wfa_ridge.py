import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

# 导入你的数据库引擎
from data_pipeline.db_engine import get_engine

warnings.filterwarnings('ignore')

class V7HyperNeutralBacktester:
    def __init__(self, checkpoint_dir, sentiment_threshold=0.3, target_vol=0.10):
        self.engine = get_engine()
        self.checkpoint_dir = checkpoint_dir
        self.sentiment_threshold = sentiment_threshold
        self.target_vol = target_vol 
        
        # 核心特征
        self.feature_cols = ['momentum', 'sentiment', 'tick_imbalance', 'mean_reversion', 
                             'vix_index', 'oil_ret', 'gpr_index']
        self.target_col = 'target_return'
        self.models = self._load_checkpoints()

    def _load_checkpoints(self):
        models = {}
        for file in os.listdir(self.checkpoint_dir):
            if file.endswith('.joblib'):
                date_part = file.split('_')[-1].split('.')[0]
                models[date_part] = joblib.load(os.path.join(self.checkpoint_dir, file))
        return models

    def load_and_preprocess(self):
        print("📥 正在执行高灵敏度 Beta 估计与特征预处理...")
        query = """
        SELECT a.*, m.vix_index, m.brent_oil_price, m.gpr_index 
        FROM alpha_feature_store a 
        LEFT JOIN macro_features_store m ON a.date = m.date
        ORDER BY a.date ASC
        """
        df = pd.read_sql(query, self.engine)
        df['date'] = pd.to_datetime(df['date'])

        # 1. 宏观平稳化
        macro_oil = df.groupby('date')['brent_oil_price'].first().pct_change()
        df['oil_ret'] = df['date'].map(macro_oil).fillna(0)

        # 2. 调优建议 2：高灵敏度动态 Beta (从 12 周缩短至 4 周)
        # 针对 2026-03 这种突变行情，缩短窗口能更快感知风险
        df['mkt_ret'] = df.groupby('date')[self.target_col].transform('mean')
        
        def get_fast_beta(group):
            ret = group[self.target_col]
            mkt = group['mkt_ret']
            # 使用 4 周快速窗口
            cov = ret.rolling(4).cov(mkt)
            var = mkt.rolling(4).var()
            group['beta_est'] = (cov / (var + 1e-8)).fillna(1.0)
            return group

        df = df.groupby('symbol', group_keys=False).apply(get_fast_beta)

        # 3. 特征标准化
        def zscore(x):
            return (x - x.mean()) / (x.std() + 1e-8)
        
        for col in self.feature_cols:
            df[col] = df.groupby('date')[col].transform(zscore)
        df[self.target_col] = df.groupby('date')[self.target_col].transform(zscore)
        
        return df

    def apply_asymmetric_beta_matching(self, long_pool, short_pool, total_exp):
        """
        调优建议 1：不对称权重约束
        多头 Cap 2%, 空头放宽至 4% 以便有足够的权重去对冲高 Beta 的科技多头
        """
        if long_pool.empty or short_pool.empty:
            return pd.Series(), pd.Series()

        avg_beta_l = long_pool['beta_est'].mean()
        avg_beta_s = short_pool['beta_est'].mean()
        
        # 贝塔平衡公式: W_l * Beta_l = W_s * Beta_s
        hedge_ratio = avg_beta_l / (avg_beta_s + 1e-8)
        hedge_ratio = np.clip(hedge_ratio, 0.4, 2.5) # 适当放宽对冲比率

        # 计算理论总权重分配
        w_l_total = total_exp / (1 + hedge_ratio)
        w_s_total = total_exp - w_l_total
        
        # 不对称 Cap
        def get_weights(pool, target_total, cap):
            n = len(pool)
            base_w = target_total / n
            w = pd.Series(min(base_w, cap), index=pool.index)
            return w

        weights_l = get_weights(long_pool, w_l_total, 0.02) # 多头保持 2%
        weights_s = get_weights(short_pool, w_s_total, 0.04) # 空头放宽至 4%
        
        return weights_l, -weights_s

    def run_backtest(self):
        df = self.load_and_preprocess()
        sorted_dates = sorted(self.models.keys())
        results = []
        hist_returns = []

        print(f"🚀 启动 V7 终极对冲版回测 (Fast-Beta + Asymmetric Cap)...")

        for d_str in sorted_dates:
            curr_date = datetime.strptime(d_str, '%Y%m%d')
            week_data = df[df['date'] == curr_date].copy()
            if week_data.empty: continue
            
            model = self.models[d_str]
            week_data['prediction'] = model.predict(week_data[self.feature_cols])
            week_data['quintile'] = pd.qcut(week_data['prediction'], 5, labels=False)

            # 风险缩放 (VIX + Vol Target)
            curr_vix = week_data['vix_index'].iloc[0]
            vix_scale = np.clip(20.0 / curr_vix, 0.1, 1.0)
            
            vol_adj = 1.0
            if len(hist_returns) >= 4:
                realized_vol = np.std(hist_returns[-4:]) * np.sqrt(52)
                vol_adj = np.clip(self.target_vol / (realized_vol + 1e-8), 0.2, 1.0)
            
            initial_exp = vix_scale * vol_adj

            # 对称性对冲计算
            long_pool = week_data[week_data['quintile'] == 4]
            short_pool = week_data[week_data['quintile'] == 0]
            
            # # 🆕 先按预测收益率 (prediction) 从高到低排序，然后再提取股票代码
            # sorted_q5_symbols = long_pool.sort_values(by='prediction', ascending=False)['symbol'].tolist()
            
            # # 打印排序后的名单
            # print(f"📅 日期: {curr_date.strftime('%Y-%m-%d')} | Q5 做多标的 ({len(long_pool)}只): {sorted_q5_symbols}")

            w_l, w_s = self.apply_asymmetric_beta_matching(long_pool, short_pool, initial_exp)
            
            # 调优建议 3：安全开关 (Beta Thresholding)
            net_beta = (w_l * long_pool['beta_est']).sum() + (w_s * short_pool['beta_est']).sum()
            
            # 如果 Net Beta 依然无法压低到 0.15 以下，说明对冲完全失效，降低暴露度
            safety_scale = 1.0
            if abs(net_beta) > 0.15:
                safety_scale = 0.5 # 强制砍半
            
            final_p_ret = ((w_l * long_pool[self.target_col]).sum() + 
                           (w_s * short_pool[self.target_col]).sum()) * safety_scale
            
            hist_returns.append(final_p_ret)
            
            results.append({
                'date': curr_date,
                'strategy_ret': final_p_ret,
                'benchmark': week_data['mkt_ret'].iloc[0],
                'net_beta': net_beta * safety_scale,
                'exposure': (w_l.sum() + abs(w_s.sum())) * safety_scale
            })

        return pd.DataFrame(results).set_index('date')

    def plot_performance(self, res_df):
        # 1. 计算累计净值 (Wealth Curves)
        # 统一使用 'benchmark' 键名，确保黑色虚线反映真实的指数涨跌
        cum_strategy = (1 + res_df['strategy_ret']).cumprod()
        cum_benchmark = (1 + res_df['benchmark']).cumprod()
        
        # 2. 计算回撤 (Drawdown)
        dd_strat = (cum_strategy - cum_strategy.cummax()) / cum_strategy.cummax()

        # 开始绘图
        plt.style.use('seaborn-v0_8-muted')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

        # --- 图表 1: 策略 vs 动态基准 (展示市场涨跌) ---
        ax1.plot(cum_strategy, label='V7 Strategy (Beta Neutral)', color='crimson', linewidth=2.5)
        ax1.plot(cum_benchmark, label='Market Benchmark (Dynamic)', color='black', linestyle='--', alpha=0.6)
        
        # 填充超额收益区域 (Alpha 可视化)
        ax1.fill_between(res_df.index, cum_strategy, cum_benchmark, 
                         where=(cum_strategy > cum_benchmark), color='green', alpha=0.1)
        
        ax1.set_title('Strategic Wealth Curve vs. Dynamic Market Benchmark', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Wealth')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # --- 图表 2: 风险透视 (回撤与仓位暴露) ---
        # 展示 MDD 路径与系统在压力下的仓位缩减反应
        ax2.fill_between(res_df.index, 0, dd_strat, color='red', alpha=0.3, label='Strategy Drawdown')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(res_df['exposure'], color='gray', alpha=0.5, label='Exposure', linewidth=1)
        
        ax2.set_title('Risk Profile: Maximum Drawdown & Portfolio Exposure', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown %')
        ax2_twin.set_ylabel('Market Exposure')
        ax2.legend(loc='lower left')
        ax2_twin.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)

        # 重点标注：2026年3月压力测试点
        war_date = pd.to_datetime('2026-02-28')
        for ax in [ax1, ax2]:
            ax.axvline(war_date, color='orange', linestyle=':', linewidth=2)
            # 动态定位标注位置
            y_limit = ax.get_ylim()[1]
            ax.annotate('2026-03 Stress Event', xy=(war_date, y_limit * 0.8), 
                        color='orange', rotation=90, fontweight='bold', size=10,
                        xytext=(5, 0), textcoords='offset points')

        plt.tight_layout()
        plt.show()

        # 打印分析报告摘要
        print(f"\n📈 V7 终极优化版分析报告 (2026-05-02):")
        print(f"   - 累计 Alpha 收益: {(cum_strategy.iloc[-1]-1)*100:.2f}%")
        print(f"   - 年化夏普 (Sharpe): {(res_df['strategy_ret'].mean() / res_df['strategy_ret'].std()) * np.sqrt(52):.2f}")
        print(f"   - 最大回撤 (MDD): {dd_strat.min()*100:.2f}%")
        print(f"   - 平均 Net Beta 暴露: {res_df['net_beta'].abs().mean():.6f}")

if __name__ == "__main__":
    CKPT = "checkpoints/RidgeRegression/2026-05-02"
    tester = V7HyperNeutralBacktester(CKPT, target_vol=0.10)
    res = tester.run_backtest()
    tester.plot_performance(res)

    