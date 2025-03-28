#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import asyncio
import time

logger = logging.getLogger("whisper-server")

# 会话信息存储
active_sessions = {}

async def cleanup_inactive_sessions():
    """定期清理不活跃的会话"""
    while True:
        try:
            current_time = time.time()
            inactive_sessions = []

            # 查找超过5分钟不活跃的会话
            for session_id, session_data in active_sessions.items():
                if current_time - session_data["last_activity"] > 300:  # 5分钟
                    inactive_sessions.append(session_id)

            # 删除不活跃会话
            for session_id in inactive_sessions:
                logger.info(f"清理不活跃会话: {session_id}")
                del active_sessions[session_id]

        except Exception as e:
            logger.error(f"清理会话时出错: {str(e)}")

        finally:
            # 每60秒检查一次
            await asyncio.sleep(60)

def get_session(session_id):
    """获取会话信息"""
    return active_sessions.get(session_id)

def create_session(session_id, processor, config):
    """创建新会话"""
    active_sessions[session_id] = {
        "processor": processor,
        "last_activity": time.time(),
        "config": config,
    }

def update_session_activity(session_id):
    """更新会话活动时间"""
    if session_id in active_sessions:
        active_sessions[session_id]["last_activity"] = time.time()

def remove_session(session_id):
    """移除会话"""
    if session_id in active_sessions:
        del active_sessions[session_id]