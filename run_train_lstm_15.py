import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import datetime
from sklearn.metrics import classification_report

from data_pipeline.db_engine import get_engine
from data_pipeline.feature_eng import StockDatasetBinary
# 【关键修改 1】：引入适配 15 只股票的新模型定义
from models.lstm_model_15 import LSTMQuantModel

def load_and_split_multi_stock_data():
    engine = get_engine()
    target_stocks = [
        'nvda', 'aapl', 'msft', 'googl', 'amzn', 'meta', 'tsla',
        'amd', 'smci', 'arm', 'avgo', 'tsm', 
        'pltr', 'crwd', 'coin'
    ]
    
    X_train_list, ids_train_list, y_train_list = [], [], []
    X_test_list, ids_test_list, y_test_list = [], [], []
    
    # 🔪 定义时序隔离的“绝对切割点”
    CUTOFF_DATE = pd.to_datetime('2024-01-01')
    
    print(f"📥 开始拉取数据，执行严格的绝对时序切割 (切割点: {CUTOFF_DATE.strftime('%Y-%m-%d')})...")
    for idx, stock in enumerate(target_stocks):
        try:
            df = pd.read_sql(f"SELECT * FROM {stock} ORDER BY date", engine)
            if len(df) < 100: continue
            
            # 必须把 date 转成 datetime 对象方便对比
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            
            dataset = StockDatasetBinary(df, stock_id=idx, window_size=30, atr_multiplier=1.5)
            
            # 因为经过了 window_size(30) 的滑动，我们需要对齐目标 Y 的真实日期
            # X 和 Y 的长度比原始 df 少了 window_size，所以 dates 也要从 30 开始取
            valid_dates = df['date'].values[30 : 30 + len(dataset.X)]
            
            # 生成布尔掩码 (Mask)
            train_mask = valid_dates < CUTOFF_DATE
            test_mask = valid_dates >= CUTOFF_DATE
            
            X_train_list.append(dataset.X[train_mask])
            ids_train_list.append(dataset.stock_ids[train_mask])
            y_train_list.append(dataset.y[train_mask])
            
            X_test_list.append(dataset.X[test_mask])
            ids_test_list.append(dataset.stock_ids[test_mask])
            y_test_list.append(dataset.y[test_mask])
            
            train_count = train_mask.sum()
            test_count = test_mask.sum()
            print(f"   ✅ {stock.upper():<5} | 训练集: {train_count:4d} | 测试集: {test_count:4d}")
            
        except Exception as e:
            print(f"   ⚠️ 跳过 {stock.upper()}，原因: {e}")

    # 将 15 只股票的数据融合成超级大张量
    X_train = torch.cat(X_train_list, dim=0)
    ids_train = torch.cat(ids_train_list, dim=0)
    y_train = torch.cat(y_train_list, dim=0)
    
    X_test = torch.cat(X_test_list, dim=0)
    ids_test = torch.cat(ids_test_list, dim=0)
    y_test = torch.cat(y_test_list, dim=0)
    
    print(f"\n🌍 15股大联盟组装完毕! 总训练集: {len(y_train)} | 总测试集: {len(y_test)}")
    return X_train, ids_train, y_train, X_test, ids_test, y_test

def train_model(X_train, ids_train, y_train, X_test, ids_test, y_test, epochs=200):
    # 【关键修改 3】：初始化模型，num_stocks 设为 15
    # embed_dim 保持 8，或者根据需要微调至 10-12 以增强区分度
    model = LSTMQuantModel(input_dim=6, hidden_dim=64, num_layers=2, num_stocks=15, embed_dim=8)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005) # 稍微调低学习率，因为数据量变大，需要更稳健的学习过程
    
    # 类别平衡权重 (正样本在妖股中可能依然稀缺)
    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    
    y_train = y_train.view(-1, 1)
    y_test = y_test.view(-1, 1)

    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    checkpoint_dir = os.path.join('checkpoints', current_date)
    os.makedirs(checkpoint_dir, exist_ok=True) 
    
    best_val_loss = float('inf') 

    print(f"\n🚀 开始训练 15 核 LSTM 模型 (Alpha Pool 进阶版)...")
    print(f"📁 模型将保存至: {checkpoint_dir}")
    
    for epoch in range(epochs):
        # 1. 训练
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train, ids_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # 2. 验证
        model.eval() 
        with torch.no_grad():
            test_outputs = model(X_test, ids_test)
            val_loss = criterion(test_outputs, y_test).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                file_name = f'lstm_v15_{current_date}_epoch_{epoch+1:03d}_valLoss_{val_loss:.4f}.pth'
                save_path = os.path.join(checkpoint_dir, file_name)
                torch.save(model.state_dict(), save_path)
                print(f"🌟 发现新低！Epoch [{epoch+1:3d}] Val Loss: {val_loss:.4f} -> 已保存")
        
        # 3. 日志
        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                train_preds = (torch.sigmoid(outputs) > 0.5).float()
                train_acc = (train_preds == y_train).float().mean()
                
                test_preds = (torch.sigmoid(test_outputs) > 0.5).float()
                test_acc = (test_preds == y_test).float().mean()
                
            print(f'➡️ Epoch [{epoch+1:3d}/{epochs}] | Loss: {loss.item():.4f} | T-Acc: {train_acc.item():.2%} | V-Acc: {test_acc.item():.2%}')
            
    print(f"\n✅ 15股深度炼丹完成！模型存入 {checkpoint_dir}")
    return model

def evaluate_trading_edge(model, X_test, ids_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test, ids_test)
        probs = torch.sigmoid(outputs).numpy()
        preds = (probs > 0.5).astype(float)
        y_true = y_test.numpy()

    print("\n" + "="*40)
    print("📊 15股 Alpha Pool：量化交易评估报告")
    print("="*40)
    print(classification_report(y_true, preds, target_names=["震荡/下跌 (0)", "大涨突破 (1)"]))

if __name__ == "__main__":
    X_train, ids_train, y_train, X_test, ids_test, y_test = load_and_split_multi_stock_data()
    best_model = train_model(X_train, ids_train, y_train, X_test, ids_test, y_test, epochs=250) # 妖股逻辑更复杂，增加到 250 epoch
    evaluate_trading_edge(best_model, X_test, ids_test, y_test)