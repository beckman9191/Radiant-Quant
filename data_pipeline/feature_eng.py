import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

class StockDatasetBinary:
    # 【修改 1】：强行要求传入 stock_id
    def __init__(self, df, stock_id, window_size=30, atr_multiplier=1.5):
        self.window_size = window_size
        self.atr_multiplier = atr_multiplier 
        self.stock_id = stock_id  # 记录这批数据属于哪只股票
        
        print(f"🛠️ 正在提取技术指标池 | 股票身份 ID: {self.stock_id}...")
        # 1. 先运行 extract_features 把所有指标（包括新的 dist_to_ma20 等）都算出来
        df = self._extract_features(df)
        
        # 2. 生成标签逻辑 (保持不变)
        df['future_5d_return'] = df['close'].shift(-5) / df['close'] - 1
        df['target'] = (df['future_5d_return'] > self.atr_multiplier * df['atr_pct']).astype(float)
        df.dropna(inplace=True)
        
        pos_ratio = df['target'].mean()
        print(f"📊 标签分布: 正样本占比 {pos_ratio:.2%}")

        # ==========================================
        # 🎯 【精英特征选拔：10 -> 6】
        # 这里我们通过之前的排行榜，剔除了冗余和噪音：
        # - 踢掉了 close, ma5, ma20 (由 dist 替代)
        # - 踢掉了 macd, macd_hist, volume (由更敏锐的指标替代)
        # ==========================================
        self.feature_cols = [
            'dist_to_ma20', # 精英 1：价格离 20 日均线多远 (代替绝对价格)
            'ma5_dist',     # 精英 2：价格离 5 日均线多远 (捕捉短线超买超卖)
            'atr_pct',      # 精英 3：波动率环境 (核心自适应特征)
            'bb_width',     # 精英 4：布林带挤压 (识别突破前夜)
            'rsi',          # 精英 5：经典超买超卖
            'vol_change'    # 精英 6：成交量异动 (资金流向指标)
        ]
        
        print(f"🚀 精英特征入队，当前输入维度: {len(self.feature_cols)}")

        # 3. 特征归一化 (只对选出的 6 个精英进行归一化)
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(df[self.feature_cols])
        
        # 4. 生成时间序列窗口
        print("🪓 正在切割时间序列窗口...")
        self.X, self.y = self._create_sequences(scaled_data, df['target'].values)
        
        # 5. 生成实体嵌入身份 ID
        self.stock_ids = torch.full((len(self.X),), self.stock_id, dtype=torch.long)


    def _extract_features(self, df):
        # --- 保持原有逻辑不变 ---
        df['rsi'] = self._calc_rsi(df['close'], 14)
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal'] 
        
        # 布林带
        df['std20'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['ma20'] + 2 * df['std20']
        df['bb_lower'] = df['ma20'] - 2 * df['std20']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['ma20'] 
        
        # 量能
        df['vol_change'] = df['volume'] / df['volume'].rolling(5).mean() - 1

        # ATR
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']

        # --- ✨ 新增：我们要提取的“精英特征” ---
        # 1. 价格偏离 20 日均线的比例 (代替 close 和 ma20)
        df['dist_to_ma20'] = (df['close'] - df['ma20']) / df['ma20']
        # 2. 价格偏离 5 日均线的比例 (代替 ma5)
        df['ma5_dist'] = (df['close'] - df['ma5']) / df['ma5']
        
        return df.dropna()

    def _calc_rsi(self, prices, period):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _create_sequences(self, data, target):
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i : i + self.window_size])
            y.append(target[i + self.window_size])
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)