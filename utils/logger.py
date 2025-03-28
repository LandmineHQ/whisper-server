#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys
import os
from datetime import datetime

def setup_logger(level=logging.INFO, log_file=None):
    """
    配置日志记录器
    
    参数:
        level: 日志级别
        log_file: 日志文件路径，如果为None则仅输出到控制台
    """
    logger = logging.getLogger("whisper-server")
    logger.setLevel(level)
    
    # 清除已有的处理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 日志文件处理器(如果指定)
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_default_log_file():
    """获取默认日志文件路径"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    now = datetime.now()
    log_file = os.path.join(
        log_dir, 
        f"whisper_server_{now.strftime('%Y%m%d_%H%M%S')}.log"
    )
    return log_file