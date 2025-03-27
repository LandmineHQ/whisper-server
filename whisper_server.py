#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import websockets
import json
import numpy as np
import logging
import uuid
import time
import whisper
import torch
from queue import Queue
from threading import Thread
import os
from collections import deque
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("whisper-server")

# 会话信息存储
active_sessions = {}


# 加载whisper模型
def load_whisper_model(model_name="large-v3-turbo"):
    logger.info(f"正在加载 Whisper 模型 {model_name}...")
    model = whisper.load_model(model_name)
    logger.info(f"Whisper 模型 {model_name} 加载完成")
    return model


# 全局模型实例
whisper_model = None


class AudioProcessor:
    def __init__(self, sample_rate, language):
        self.sample_rate = sample_rate
        self.language = language
        self.audio_buffer = []
        self.total_duration = 0.0
        self.last_transcript = ""
        self.last_full_transcript = ""
        self.audio_queue = Queue()
        self.result_queue = Queue()
        self.processing_thread = Thread(target=self._process_audio_thread, daemon=True)
        self.processing_thread.start()

        # 保持一个最近的音频缓冲区，用于连续转录
        self.audio_history = deque(maxlen=5)  # 约5秒的历史记录

    def add_audio_chunk(self, audio_data):
        """添加16位PCM音频数据块到缓冲区"""
        # 计算块持续时间(秒)
        chunk_duration = len(audio_data) / 2 / self.sample_rate
        self.total_duration += chunk_duration

        # 将音频数据转换为float32数组并规范化到[-1, 1]范围
        float_data = (
            np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        )

        # 保存到当前音频块和历史记录
        self.audio_buffer.append(float_data)
        self.audio_history.append(float_data)

        # 如果累积了足够的音频(约1秒)，进行处理
        if self.total_duration >= 1.0:
            # 合并音频数据
            combined_data = np.concatenate(self.audio_buffer)
            self.audio_queue.put(combined_data)

            # 重置缓冲区
            self.audio_buffer = []
            self.total_duration = 0.0

    def get_latest_result(self):
        """获取最新的转录结果(非阻塞)"""
        if not self.result_queue.empty():
            result = self.result_queue.get()
            self.last_transcript = result["text"]
            if result["is_final"]:
                self.last_full_transcript = self.last_transcript
            return result
        return None

    def _process_audio_thread(self):
        """后台音频处理线程"""
        global whisper_model

        while True:
            try:
                # 获取下一个音频块
                audio_data = self.audio_queue.get()

                # 如果whisper模型还没加载完成，等待
                if whisper_model is None:
                    time.sleep(0.1)
                    self.result_queue.put(
                        {
                            "text": "正在加载语音识别模型...",
                            "is_final": False,
                            "language": self.language,
                        }
                    )
                    self.audio_queue.task_done()
                    continue

                # 准备历史音频记录(可选，用于更好的连续性)
                context_audio = None
                if len(self.audio_history) > 1:
                    # 从历史记录重建一段上下文音频
                    history_items = list(self.audio_history)[:-1]  # 除了最后添加的块
                    if history_items:
                        context_audio = np.concatenate(history_items)

                # 将音频发送到Whisper模型
                with torch.no_grad():
                    # 优先使用具有上下文的历史音频(如果有)
                    if context_audio is not None:
                        input_audio = np.concatenate([context_audio, audio_data])
                    else:
                        input_audio = audio_data

                    # 进行转录
                    result = whisper_model.transcribe(
                        input_audio,
                        language=self.language,
                        initial_prompt=(
                            self.last_full_transcript
                            if self.last_full_transcript
                            else None
                        ),
                        verbose=False,
                    )

                    # 获取文本
                    transcribed_text = result["text"].strip()

                    # 发送中间结果
                    self.result_queue.put(
                        {
                            "text": transcribed_text,
                            "is_final": True,  # 每个音频块我们都当作最终结果
                            "language": result.get("language", self.language),
                        }
                    )

            except Exception as e:
                logger.error(f"音频处理错误: {str(e)}")
                logger.error(traceback.format_exc())

            finally:
                self.audio_queue.task_done()


async def handle_websocket(websocket, path):
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
                        audio_processor = AudioProcessor(sample_rate, language)

                        # 保存会话
                        active_sessions[session_id] = {
                            "processor": audio_processor,
                            "last_activity": last_activity,
                            "config": config,
                        }

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
                        if session_id in active_sessions:
                            del active_sessions[session_id]

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

    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"连接关闭: {client_address}, code: {e.code}, reason: {e.reason}")

    except Exception as e:
        logger.error(f"未处理的异常: {str(e)}")
        logger.error(traceback.format_exc())

    finally:
        # 清理会话
        if session_id and session_id in active_sessions:
            del active_sessions[session_id]
        logger.info(f"连接结束: {client_address}")


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


def load_model_in_background():
    """在后台线程加载模型"""
    global whisper_model
    whisper_model = load_whisper_model("large-v3-turbo")


async def main():
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
    logger.info("启动Whisper WebSocket转录服务器...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("检测到中断信号，程序退出")
    except Exception as e:
        logger.error(f"程序错误: {str(e)}")
        logger.error(traceback.format_exc())
