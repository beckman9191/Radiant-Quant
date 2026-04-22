import os
import glob
import torch
import pandas as pd
import numpy as np
from data_pipeline.db_engine import get_engine
from data_pipeline.feature_eng import StockDatasetBinary
from models.lstm_model import LSTMQuantModel

def get_best_model_path(checkpoint_dir='checkpoints'):
    files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not files:
        raise FileNotFoundError("❌ 未找到模型文件，请先确保已经训练过模型")
    files.sort(key=lambda x: float(x.split('valLoss_')[-1].replace('.pth', '')))
    return files[0]

def load_eval_data():
    """加载全市场测试数据用于评估"""
    engine = get_engine()
    target_stocks = ['nvda', 'aapl', 'tsla', 'msft', 'googl', 'amzn', 'meta']
    
    X_list, ids_list, y_list = [], [], []
    
    print("📥 正在加载全市场样本用于特征分析...")
    for idx, stock in enumerate(target_stocks):
        df = pd.read_sql(f"SELECT * FROM {stock} ORDER BY date", engine)
        if len(df) < 100: continue
        
        # 使用你之前的 10 维特征提取逻辑
        dataset = StockDatasetBinary(df, stock_id=idx, window_size=30, atr_multiplier=1.5)
        X_list.append(dataset.X)
        ids_list.append(dataset.stock_ids)
        y_list.append(dataset.y)

    return (torch.cat(X_list, dim=0), 
            torch.cat(ids_list, dim=0), 
            torch.cat(y_list, dim=0).view(-1, 1))

def run_importance_analysis():
    # 1. 准备环境
    X, stock_ids, y = load_eval_data()
    
    # 2. 加载模型 (注意：这里 input_dim 必须与你目前保存的模型一致，如果是旧模型请填 10)
    model = LSTMQuantModel(input_dim=10, hidden_dim=64, num_layers=2, num_stocks=7, embed_dim=8)
    best_path = get_best_model_path()
    print(f"🏆 正在分析模型: {best_path}")
    model.load_state_dict(torch.load(best_path))
    model.eval()

    # 3. 计算基准 Loss (Baseline)
    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        base_out = model(X, stock_ids)
        base_loss = criterion(base_out, y).item()
    
    print(f"📉 基准 Loss: {base_loss:.4f}")

    # 特征列表 (请确保顺序与你 feature_eng.py 中 feature_cols 一致)
    feature_names = [
        'close', 'volume', 'rsi', 'ma5', 'ma20', 
        'macd', 'macd_hist', 'bb_width', 'vol_change', 'atr_pct'
    ]
    
    importance_results = []

    print("\n🕵️ 正在进行特征压力测试 (Permutation Importance)...")
    for i in range(len(feature_names)):
        # 拷贝数据
        X_shuffled = X.clone()
        
        # 核心：打乱第 i 个特征的顺序 (在 Batch 维度打乱)
        # 我们打乱所有样本在该特征上的分布，看模型是否会“抓狂”
        perm = torch.randperm(X_shuffled.size(0))
        X_shuffled[:, :, i] = X_shuffled[perm, :, i]

        with torch.no_grad():
            shuffled_out = model(X_shuffled, stock_ids)
            shuffled_loss = criterion(shuffled_out, y).item()
        
        # 重要性评分 = 破坏后 Loss 的增量
        # 增量越大，说明该特征对模型越不可或缺
        importance_score = shuffled_loss - base_loss
        importance_results.append({
            'Feature': feature_names[i],
            'Importance': round(importance_score, 6)
        })
        print(f"   [{i+1}/{len(feature_names)}] 分析完成: {feature_names[i]}")

    # 4. 结果展示
    df_imp = pd.DataFrame(importance_results).sort_values(by='Importance', ascending=False)
    
    print("\n" + "="*40)
    print("🏆 特征贡献度排行榜 (由高到低)")
    print("="*40)
    print(df_imp.to_string(index=False))
    print("="*40)
    
    # 给出建议
    low_imp = df_imp[df_imp['Importance'] <= 0]['Feature'].tolist()
    if low_imp:
        print(f"🚩 建议剪枝 (贡献为负或为零): {low_imp}")
    else:
        print("💡 所有特征均有正向贡献，建议保留前 5-6 名。")

if __name__ == "__main__":
    run_importance_analysis()