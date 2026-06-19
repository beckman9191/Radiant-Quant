import pandas as pd
import numpy as np
import joblib
import alpaca_trade_api as tradeapi
import yfinance as yf
from datetime import datetime, timedelta
import logging

from data_pipeline.sync_macro_features import MacroDataSyncer
from data_pipeline.feature_ingestion import FeatureIngestorYF


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlpacaTraderV7:
    def __init__(self, api_key, secret_key, base_url, model_path):
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        self.model = joblib.load(model_path)
        self.feature_cols = ['momentum', 'sentiment', 'tick_imbalance', 'mean_reversion', 
                             'vix_index', 'oil_ret', 'gpr_index']
        self.target_vol = 0.10 # 维持终极版的 10% 波动率目标
        
    def get_market_context(self):
        """获取实盘真实的 VIX 数据"""
        try:
            # 下载最近 5 天的 VIX 数据，取最后一条有效收盘价
            vix_data = yf.download('^VIX', period='5d', progress=False)
            if vix_data.empty:
                raise ValueError("未获取到 VIX 数据")
            # 处理 yf 多重索引的情况
            vix = float(vix_data['Close'].iloc[-1].item() if isinstance(vix_data['Close'].iloc[-1], pd.Series) else vix_data['Close'].iloc[-1])
            logger.info(f"📈 成功获取当前市场 VIX: {vix:.2f}")
            return vix
        except Exception as e:
            logger.error(f"❌ 获取 VIX 失败: {e} | 降级使用安全默认值 20.0")
            return 20.0

    def get_historical_returns(self):
        """
        通过 Alpaca 接口获取账户真实的历史净值，计算过去 4 周的实际收益率
        """
        try:
            # 获取过去 30 天的历史净值（日线）
            history = self.api.get_portfolio_history(period='1M', timeframe='1D')
            
            if not history.timestamp or len(history.timestamp) < 2:
                logger.warning("⚠️ 账户历史数据不足（可能是新账户），无法计算真实历史收益。")
                return []

            # 转换为 DataFrame
            equity_df = pd.DataFrame({
                'timestamp': history.timestamp,
                'equity': history.equity
            })
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'], unit='s')
            equity_df.set_index('timestamp', inplace=True)
            
            # 剔除可能存在的 None 或空值净值
            equity_df = equity_df.dropna()

            # 按周重采样（每周五的最终净值）
            weekly_equity = equity_df['equity'].resample('W-FRI').last().dropna()
            
            # 计算周收益率
            weekly_returns = weekly_equity.pct_change().dropna().tolist()
            
            logger.info(f"🏦 获取到真实历史周收益率: {[round(r, 4) for r in weekly_returns]}")
            
            if len(weekly_returns) < 4:
                logger.warning(f"⚠️ 账户历史收益不足 4 期 (当前 {len(weekly_returns)} 期)，波动率调节可能受限。")
                
            return weekly_returns
            
        except Exception as e:
            logger.error(f"❌ 获取账户历史数据失败: {e}")
            return []

    def calculate_target_weights(self, current_data):
        """
        核心逻辑复刻：Beta 匹配 + 不对称权重约束 + 动态波动调节 + 安全开关
        """
        if 'beta_est' not in current_data.columns:
            raise ValueError("🚨 传入数据缺失 'beta_est'！请确保 Data Pipeline 实现了 4 周快速 Beta 的计算逻辑。")

        # 1. 预测排名
        current_data['prediction'] = self.model.predict(current_data[self.feature_cols])
        current_data['quintile'] = pd.qcut(current_data['prediction'], 5, labels=False)

        # 2. 风险缩放计算 (基于真实的 VIX)
        vix = self.get_market_context()
        vix_scale = np.clip(20.0 / vix, 0.1, 1.0)
        
        # 3. 动态波动率调节 (基于账户真实收益)
        hist_returns = self.get_historical_returns()
        vol_adj = 1.0
        if len(hist_returns) >= 4:
            # 计算过去四周的年化已实现波动率
            realized_vol = np.std(hist_returns[-4:]) * np.sqrt(52)
            vol_adj = np.clip(self.target_vol / (realized_vol + 1e-8), 0.2, 1.0)
        else:
            logger.warning("⚠️ 启动初始安全缓冲模式 (Vol Adj = 0.8)")
            vol_adj = 0.8 # 初始实盘略微降仓
            
        initial_exp = vix_scale * vol_adj

        # 4. 提取多空池 (Q5 vs Q1)
        long_pool = current_data[current_data['quintile'] == 4].copy()
        short_pool = current_data[current_data['quintile'] == 0].copy()

        # 5. 贝塔对冲计算 (Asymmetric Beta Matching)
        avg_beta_l = long_pool['beta_est'].mean()
        avg_beta_s = short_pool['beta_est'].mean()
        hedge_ratio = np.clip(avg_beta_l / (avg_beta_s + 1e-8), 0.4, 2.5)

        w_l_total = initial_exp / (1 + hedge_ratio)
        w_s_total = initial_exp - w_l_total

        # 6. 应用 2% (多) / 4% (空) Cap
        long_pool['target_w'] = (w_l_total / len(long_pool)).clip(upper=0.02)
        short_pool['target_w'] = -(w_s_total / len(short_pool)).clip(upper=0.04)

        # 7. 安全开关 (Beta Thresholding)
        net_beta = (long_pool['target_w'] * long_pool['beta_est']).sum() + \
                   (short_pool['target_w'] * short_pool['beta_est']).sum()
        
        safety_scale = 1.0
        if abs(net_beta) > 0.15:
            logger.warning(f"🚨 警告: 预期 Net Beta ({net_beta:.4f}) 超过 0.15 阈值，触发安全开关，总敞口强制砍半！")
            safety_scale = 0.5
            
        # 应用最终风控缩放
        long_pool['target_w'] *= safety_scale
        short_pool['target_w'] *= safety_scale
        
        final_net_beta = net_beta * safety_scale
        final_exposure = long_pool['target_w'].sum() + abs(short_pool['target_w'].sum())
        logger.info(f"📊 风控计算完成 | VIX: {vix:.2f} | Vol Adj: {vol_adj:.2f} | Net Beta: {final_net_beta:.4f} | Total Exposure: {final_exposure:.2f}")

        return pd.concat([long_pool[['target_w']], short_pool[['target_w']]])

    def execute_rebalance(self, target_weights):
        """
        执行调仓：Alpaca 订单管理
        """
        account = self.api.get_account()
        equity = float(account.equity)
        logger.info(f"💰 当前账户真实净值: {equity}")

        # 获取当前持仓
        positions = {p.symbol: float(p.qty) for p in self.api.list_positions()}
        
        # 1. 首先平掉不在目标列表中的持仓
        for symbol in list(positions.keys()):
            if symbol not in target_weights.index:
                logger.info(f"🚫 清仓不再符合策略的股票: {symbol}")
                try:
                    self.api.submit_order(symbol, abs(positions[symbol]), 'sell' if positions[symbol]>0 else 'buy', 'market', 'day')
                except Exception as e:
                    logger.error(f"❌ 清仓失败 {symbol}: {e}")

        # 2. 执行目标权重调仓
        for symbol, weight in target_weights['target_w'].items():
            try:
                last_price = self.get_last_price(symbol)
                target_qty = int((equity * weight) / last_price)
                current_qty = positions.get(symbol, 0)
                diff_qty = target_qty - current_qty

                if diff_qty == 0: continue

                side = 'buy' if diff_qty > 0 else 'sell'
                logger.info(f"🔄 调仓 {symbol}: 当前 {current_qty} -> 目标 {target_qty} (Side: {side})")
                
                self.api.submit_order(symbol, abs(diff_qty), side, 'market', 'day')
            except Exception as e:
                logger.error(f"❌ 订单发送失败 {symbol}: {e}")

    def get_last_price(self, symbol):
        # 添加重试或回退机制，防止偶发的数据拉取失败导致调仓中断
        try:
            return self.api.get_latest_trade(symbol).price
        except Exception as e:
            logger.warning(f"⚠️ 无法获取 {symbol} 最新 trade 价格，尝试获取 snapshot 价格: {e}")
            snapshot = self.api.get_snapshot(symbol)
            return snapshot.latest_trade.price

if __name__ == "__main__":
    # === 配置 Alpaca API (请根据需要切换 Paper 或 Live URL) ===
    API_KEY = "PK4NNX3MY5XBA6LSRYFYPAFSOU"
    SECRET_KEY = "6f5L5zr4FgJZokwnmMk2WML5r8re9hLBcuqHGAWX6qST"
    BASE_URL = "https://paper-api.alpaca.markets"

    ALPACA_CONFIG = {
        'api_key': API_KEY,
        'secret_key': SECRET_KEY,
        'base_url': BASE_URL, 
        'model_path': 'checkpoints/RidgeRegression/2026-05-02/ridge_fold_20260501.joblib'
    }

    trader = AlpacaTraderV7(**ALPACA_CONFIG)
    
    # === 运行主逻辑 ===
    # 示例传入：从数据库/特征引擎拉取的最新包含 beta_est 的真实截面数据
    # current_market_data = trader.data_engine.fetch_latest_features() 
    # weights = trader.calculate_target_weights(current_market_data)
    # trader.execute_rebalance(weights)