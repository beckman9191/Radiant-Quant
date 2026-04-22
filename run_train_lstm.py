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
from models.lstm_model import LSTMQuantModel

def load_and_split_multi_stock_data():
    engine = get_engine()
    target_stocks = ['nvda', 'aapl', 'tsla', 'msft', 'googl', 'amzn', 'meta']
    
    X_train_list, ids_train_list, y_train_list = [], [], []
    X_test_list, ids_test_list, y_test_list = [], [], []
    
    print("📥 开始拉取全市场数据，并执行严格的时序切分...")
    for idx, stock in enumerate(target_stocks):
        try:
            df = pd.read_sql(f"SELECT * FROM {stock} ORDER BY date", engine)
            if len(df) < 100:
                continue
                
            # 【关键 1】：传入 stock_id，让特征工程知道当前在处理谁
            dataset = StockDatasetBinary(df, stock_id=idx, window_size=30, atr_multiplier=1.5)
            
            # 单只股票内部按时间切分 (前 80% 训练，后 20% 测试)
            split_idx = int(len(dataset.X) * 0.8)
            
            X_train_list.append(dataset.X[:split_idx])
            ids_train_list.append(dataset.stock_ids[:split_idx])
            y_train_list.append(dataset.y[:split_idx])
            
            X_test_list.append(dataset.X[split_idx:])
            ids_test_list.append(dataset.stock_ids[split_idx:])
            y_test_list.append(dataset.y[split_idx:])
            
            print(f"   ✅ {stock.upper()} | ID: {idx} | 训练样本: {split_idx} | 测试样本: {len(dataset.X)-split_idx}")
        except Exception as e:
            print(f"   ⚠️ 跳过 {stock.upper()}，原因: {e}")

    # 【关键 2】：将 7 只股票的数据融合成超级大张量
    X_train = torch.cat(X_train_list, dim=0)
    ids_train = torch.cat(ids_train_list, dim=0)
    y_train = torch.cat(y_train_list, dim=0)
    
    X_test = torch.cat(X_test_list, dim=0)
    ids_test = torch.cat(ids_test_list, dim=0)
    y_test = torch.cat(y_test_list, dim=0)
    
    print(f"\n🌍 数据组装完毕! 总训练集: {len(y_train)} | 总测试集: {len(y_test)}")
    return X_train, ids_train, y_train, X_test, ids_test, y_test

def train_model(X_train, ids_train, y_train, X_test, ids_test, y_test, epochs=150):
    # 【关键 3】：初始化带 Embedding 层的双核模型
    model = LSTMQuantModel(input_dim=6, hidden_dim=64, num_layers=2, num_stocks=7, embed_dim=8)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 类别平衡权重
    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    
    y_train = y_train.view(-1, 1)
    y_test = y_test.view(-1, 1)

    # --- 修改部分：按日期创建文件夹 ---
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    # 路径变为 checkpoints/2026-04-22
    checkpoint_dir = os.path.join('checkpoints', current_date)
    os.makedirs(checkpoint_dir, exist_ok=True) 
    # -------------------------------
    
    best_val_loss = float('inf') 

    print(f"\n🚀 开始训练双核 LSTM 模型 (Entity Embeddings 技术)...")
    print(f"📁 模型将保存至: {checkpoint_dir}")
    
    for epoch in range(epochs):
        # --- 1. 训练阶段 ---
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train, ids_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # --- 2. 验证阶段 ---
        model.eval() 
        with torch.no_grad():
            test_outputs = model(X_test, ids_test)
            val_loss = criterion(test_outputs, y_test).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # --- 修改部分：文件名也加入日期前缀 ---
                file_name = f'lstm_{current_date}_epoch_{epoch+1:03d}_valLoss_{val_loss:.4f}.pth'
                # -----------------------------------
                save_path = os.path.join(checkpoint_dir, file_name)
                torch.save(model.state_dict(), save_path)
                print(f"🌟 发现新低！Epoch [{epoch+1:3d}] Val Loss: {val_loss:.4f} -> 已保存至日期目录")
        
        # --- 3. 打印日志 ---
        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                train_preds = (torch.sigmoid(outputs) > 0.5).float()
                train_acc = (train_preds == y_train).float().mean()
                
                test_preds = (torch.sigmoid(test_outputs) > 0.5).float()
                test_acc = (test_preds == y_test).float().mean()
                
            print(f'➡️ Epoch [{epoch+1:3d}/{epochs}] | Train Loss: {loss.item():.4f} | Train Acc: {train_acc.item():.2%} | Test Acc: {test_acc.item():.2%}')
            
    print(f"\n✅ 炼丹完成！极品模型已存入 {checkpoint_dir} 文件夹。")
    return model

def evaluate_trading_edge(model, X_test, ids_test, y_test):
    model.eval()
    with torch.no_grad():
        # 同样需要传入 IDs
        outputs = model(X_test, ids_test)
        probs = torch.sigmoid(outputs).numpy()
        preds = (probs > 0.5).astype(float)
        y_true = y_test.numpy()

    print("\n" + "="*40)
    print("📊 实体嵌入版：量化交易评估报告")
    print("="*40)
    print(classification_report(y_true, preds, target_names=["震荡/下跌 (0)", "大涨突破 (1)"]))

if __name__ == "__main__":
    # 1. 组装数据
    X_train, ids_train, y_train, X_test, ids_test, y_test = load_and_split_multi_stock_data()
    
    # 2. 训练模型 (稍微增加了 epoch 到 150，因为模型变复杂了，需要更多时间学习)
    best_model = train_model(X_train, ids_train, y_train, X_test, ids_test, y_test, epochs=150)
    
    # 3. 打印最终的分类报告，验收“脑域手术”成果
    evaluate_trading_edge(best_model, X_test, ids_test, y_test)