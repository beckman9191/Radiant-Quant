import torch
import torch.nn as nn

class LSTMQuantModel(nn.Module):
    # 【新增参数】：num_stocks (股票总数，默认7只), embed_dim (性格向量的维度，默认8维)
    def __init__(self, input_dim=10, hidden_dim=64, num_layers=2, num_stocks=7, embed_dim=8):
        super(LSTMQuantModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # ==========================================
        # 🧠 右脑：股票身份翻译器 (Entity Embedding)
        # 作用：将 0~6 的整数 ID，映射为 8 维的浮点数向量 (专门学习不同股票的性格)
        # ==========================================
        self.stock_embedding = nn.Embedding(num_embeddings=num_stocks, embedding_dim=embed_dim)
        
        # ==========================================
        # 🧠 左脑：盘面特征提取器 (LSTM)
        # 作用：处理 input_dim=10 的技术指标时间序列，提取趋势规律
        # ==========================================
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # ==========================================
        # 🧠 小脑：最终决策层 (Fully Connected Layer)
        # 输入维度 = LSTM提取的盘面特征(64) + 股票性格特征(8) = 72
        # ==========================================
        self.fc = nn.Linear(hidden_dim + embed_dim, 1)

    def forward(self, x, stock_ids):
        # x shape: (batch_size, sequence_length, input_dim)
        # stock_ids shape: (batch_size,) 或 (batch_size, 1)

        # 1. 过 LSTM 左脑提取时间序列特征
        out, _ = self.lstm(x)
        # 我们只需要时间序列最后一天(最后一个 time step)的隐藏状态
        lstm_features = out[:, -1, :]  # shape: (batch_size, hidden_dim)

        # 2. 过 Embedding 右脑提取股票的性格向量
        # .view(-1) 是为了确保一维结构，防止 shape 不匹配报错
        stock_ids = stock_ids.view(-1) 
        embed_features = self.stock_embedding(stock_ids) # shape: (batch_size, embed_dim)

        # 3. 灵魂融合：将盘面特征和股票性格在特征维度(dim=1)上拼接在一起！
        # concat 之后 shape: (batch_size, hidden_dim + embed_dim)
        fused_features = torch.cat((lstm_features, embed_features), dim=1)

        # 4. 小脑做出最终的涨跌判断
        final_output = self.fc(fused_features)
        
        return final_output