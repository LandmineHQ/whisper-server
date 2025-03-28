#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import os
import websockets
import logging
from threading import Thread

from models.whisper_model import load_whisper_model
from websocket.handler import handle_websocket, set_whisper_model
from websocket.session import cleanup_inactive_sessions
from utils.logger import setup_logger, get_default_log_file

# 全局变量
whisper_model = None


def load_model_in_background():
    """在后台线程加载模型"""
    global whisper_model
    # whisper_model = load_whisper_model("large-v3-turbo")
    whisper_model = load_whisper_model("tiny")
    # 设置全局模型
    set_whisper_model(whisper_model)


async def main():
    # 配置日志
    log_file = get_default_log_file()
    logger = setup_logger(level=logging.INFO, log_file=log_file)

    logger.info("启动Whisper WebSocket转录服务器...")

    # 启动模型加载线程
    model_thread = Thread(target=load_model_in_background, daemon=True)
    model_thread.start()

    # 启动WebSocket服务器
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8765))

    logger.info(f"启动WebSocket服务器: {host}:{port}")

    # 创建cleanup任务
    cleanup_task = asyncio.create_task(cleanup_inactive_sessions())

    # 启动WebSocket服务器
    async with websockets.serve(handle_websocket, host, port):
        await asyncio.Future()  # 无限运行


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.getLogger("whisper-server").info("检测到中断信号，程序退出")
    except Exception as e:
        logger = logging.getLogger("whisper-server")
        logger.error(f"程序错误: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
