@echo off
:: 切换到脚本所在目录，确保模型和 .env 能被找到
cd /d D:\quant

:: 使用绝对路径启动 Python，并把输出记录到日志
"C:\Users\t84389223\AppData\Local\Python\pythoncore-3.14-64\python.exe" alpaca_bot.py >> trade_log.txt 2>&1

:: 运行完后自动退出
exit