#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import uuid
import json
import logging
import time
import websockets
import traceback

from ..audio.processor import AudioProcessor
from .session import (
    active_sessions, 
    update_session_activity, 
    create_session, 
    remove_session
)

logger = logging.getLogger("whisper-server")

# 全局变量
whisper_model = None

def set_whisper_model(model):
    """设置全局Whisper模型"""
    global whisper_model
    whisper_model = model

async def handle_websocket(websocket):
    """处理WebSocket连接"""
    client_address = websocket.remote_address
    logger.info(f"新的连接: {client_address}")

    session_id = None
    audio_processor = None
    last_activity = time.time()

    try:
        async for message in websocket:
            last_activity = time.time()

            # 检查消息类型 - 是二进制音频数据还是文本控制消息
            if isinstance(message, bytes) and session_id and audio_processor:
                # 处理二进制音频数据 - 直接使用，无需解码
                try:
                    # 直接将二进制数据添加到处理队列
                    audio_processor.add_audio_chunk(message)

                    # 检查是否有新的转录结果
                    result = audio_processor.get_latest_result()
                    if result:
                        # 发送转录结果
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "transcription",
                                    "text": result["text"],
                                    "is_final": result["is_final"],
                                    "language": result["language"],
                                    "processing_time": result.get("processing_time", 0),
                                }
                            )
                        )
                except Exception as e:
                    logger.error(f"处理音频数据时出错: {str(e)}")
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "code": "audio_processing_error",
                                "message": f"处理音频数据时出错: {str(e)}",
                            }
                        )
                    )

            elif isinstance(message, str):
                # 处理控制消息 (JSON)
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "init":
                        # 处理初始化请求
                        session_id = str(uuid.uuid4())
                        config = data.get("config", {})

                        language = config.get("language", "zh")
                        sample_rate = config.get("sample_rate", 16000)

                        logger.info(
                            f"会话初始化: {session_id}, 语言: {language}, 采样率: {sample_rate}"
                        )

                        # 创建音频处理器
                        audio_processor = AudioProcessor(
                            sample_rate, 
                            language,
                            whisper_model=whisper_model
                        )

                        # 保存会话
                        create_session(
                            session_id, 
                            audio_processor, 
                            config
                        )

                        # 响应初始化确认
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "init_ack",
                                    "session_id": session_id,
                                    "status": "ready",
                                }
                            )
                        )

                    elif msg_type == "end" and session_id:
                        # 结束会话
                        logger.info(f"会话结束: {session_id}")
                        remove_session(session_id)

                        await websocket.send(
                            json.dumps({"type": "end_ack", "session_id": session_id})
                        )
                        break

                    else:
                        # 未知消息类型或会话未初始化
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "error",
                                    "code": "invalid_request",
                                    "message": "无效的请求或会话未初始化",
                                }
                            )
                        )

                except json.JSONDecodeError:
                    logger.error("JSON解析错误")
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "code": "json_parse_error",
                                "message": "无法解析JSON消息",
                            }
                        )
                    )

                except Exception as e:
                    logger.error(f"处理消息时出错: {str(e)}")
                    logger.error(traceback.format_exc())
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "code": "server_error",
                                "message": f"服务器处理错误: {str(e)}",
                            }
                        )
                    )

            else:
                # 未知消息格式
                await websocket.send(
                    json.dumps(
                        {
                            "type": "error",
                            "code": "invalid_message_format",
                            "message": "无效的消息格式",
                        }
                    )
                )

            # 更新会话活动时间
            if session_id:
                update_session_activity(session_id)

    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"连接关闭: {client_address}, code: {e.code}, reason: {e.reason}")

    except Exception as e:
        logger.error(f"未处理的异常: {str(e)}")
        logger.error(traceback.format_exc())

    finally:
        # 清理会话
        if session_id:
            remove_session(session_id)
        logger.info(f"连接结束: {client_address}")