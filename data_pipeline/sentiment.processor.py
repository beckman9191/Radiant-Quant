import pandas as pd
import torch
from transformers import pipeline
from sqlalchemy import text
from alpaca_trade_api.rest import REST
from datetime import timedelta, datetime
import time
import gc

# 数据库与 Alpaca 配置
from db_engine import get_engine


API_KEY = "PK4NNX3MY5XBA6LSRYFYPAFSOU"
SECRET_KEY = "6f5L5zr4FgJZokwnmMk2WML5r8re9hLBcuqHGAWX6qST"
BASE_URL = "https://paper-api.alpaca.markets" # ⚠️ 模拟盘专用地址

# 配置 (建议从 .env 获取)
ALPACA_CONFIG = {
    'key': API_KEY,
    'secret': SECRET_KEY,
    'base_url': BASE_URL
}

class HistoricalSentimentFiller:
    def __init__(self):
        self.api = REST(ALPACA_CONFIG['key'], ALPACA_CONFIG['secret'], ALPACA_CONFIG['base_url'])
        self.engine = get_engine()
        self.table_name = 'alpha_feature_store'
        
        # 针对 32GB RAM 加速
        device = 0 if torch.cuda.is_available() else -1
        print(f"📡 初始化 FinBERT 推理引擎 (Device: {'GPU' if device==0 else 'CPU'})...")
        self.nlp = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)

    def process_batch(self, limit=1000):
        query = f"""
            SELECT date, symbol FROM {self.table_name} 
            WHERE sentiment = 0.0 
            ORDER BY date ASC LIMIT {limit}
        """
        with self.engine.connect() as conn:
            tasks = conn.execute(text(query)).fetchall()

        if not tasks:
            return 0

        success_count = 0
        batch_start_time = time.time()

        for idx, row in enumerate(tasks):
            target_date, symbol = row[0], row[1]
            end_time = target_date.isoformat()
            start_time = (target_date - timedelta(days=3)).isoformat()

            try:
                # 1. 抓取新闻
                news = self.api.get_news(symbol=symbol, start=start_time, end=end_time, limit=5)
                headlines = [n.headline for n in news if len(n.headline) > 5]

                # 2. 计算与标记逻辑
                status_msg = ""
                final_val = 0.0
                
                if not headlines:
                    final_val = 0.00001 # 标记：无新闻
                    status_msg = "📭 No News"
                else:
                    results = self.nlp(headlines)
                    score_map = {'positive': 1, 'negative': -1, 'neutral': 0}
                    scores = [score_map[r['label']] * r['score'] for r in results]
                    raw_score = sum(scores) / len(scores)
                    
                    if raw_score == 0.0:
                        final_val = 0.00002 # 标记：纯中性
                        status_msg = f"😐 Neutral (Raw: {raw_score:.4f})"
                    else:
                        final_val = raw_score
                        # 根据得分正负显示颜色/符号
                        icon = "📈" if final_val > 0 else "📉"
                        status_msg = f"{icon} Score: {final_val:.4f}"

                # 3. 更新数据库
                self._update_db(target_date, symbol, final_val)
                success_count += 1

                # --- 实时进度 Print ---
                # 显示格式：[1/1000] 2025-05-09 | AAPL | 📈 Score: 0.8542
                print(f"[{idx+1}/{len(tasks)}] {target_date} | **{symbol:5}** | {status_msg}")

            except Exception as e:
                print(f"  ❌ {symbol} Error: {str(e)[:50]}...")
                self._update_db(target_date, symbol, 0.00003) # 标记：异常
            
            # 你的 Alpaca 权限如果较高可以调低这个 sleep
            time.sleep(0.05) 
            
        return success_count

    def _update_db(self, date, symbol, score):
        sql = f"UPDATE {self.table_name} SET sentiment = :score WHERE date = :date AND symbol = :symbol"
        with self.engine.connect() as conn:
            conn.execute(text(sql), {"score": score, "date": date, "symbol": symbol})
            conn.commit()

# --- 你的主循环逻辑 ---
if __name__ == "__main__":
    filler = HistoricalSentimentFiller()
    total_processed = 0
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 开始自动化回填...")

    try:
        while True:
            # 现在 HistoricalSentimentFiller 有了 process_batch 方法
            count = filler.process_batch(limit=1000)
            
            if count == 0:
                print("\n🎉 任务全部完成，数据库已清理干净！")
                break
                
            total_processed += count
            print(f"✅ 已完成 {total_processed} 条记录 | 休息 5 秒继续...")
            
            gc.collect()
            time.sleep(5)
            
    except KeyboardInterrupt:
        print(f"\n🛑 手动停止。当前进度：{total_processed}")