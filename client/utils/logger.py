# 日志管理类
from datetime import datetime
import tkinter as tk

class Logger:
    def __init__(self, log_area):
        self.log_area = log_area
        
    def add_log(self, message):
        """添加消息到日志区域"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        self.log_area.config(state=tk.NORMAL)
        self.log_area.insert(tk.END, log_entry)
        self.log_area.see(tk.END)  # 自动滚动到底部
        self.log_area.config(state=tk.DISABLED)