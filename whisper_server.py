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
from queue import Queue, Empty  # 修改这里，导入Empty异常
from threading import Thread
import os
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
        self.audio_queue = Queue(maxsize=10)  # 限制队列大小，防止积压
        self.result_queue = Queue()
        self.processing_thread = Thread(target=self._process_audio_thread, daemon=True)
        self.processing_thread.start()

        # 优化：保持一个最近的音频缓冲区，限制实际秒数而不仅是数量
        # 计算每秒音频的大致样本数
        self.samples_per_second = self.sample_rate  # 对于单声道float32音频
        # 只保留最近2秒的音频作为上下文
        self.max_history_seconds = 2.0
        self.audio_history = []
        self.history_duration = 0.0

    def add_audio_chunk(self, audio_data):
        """添加16位PCM音频数据块到缓冲区"""
        # 计算块持续时间(秒)
        chunk_duration = len(audio_data) / 2 / self.sample_rate
        self.total_duration += chunk_duration

        # 将音频数据转换为float32数组并规范化到[-1, 1]范围
        float_data = (
            np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        )

        # 保存到当前音频块
        self.audio_buffer.append(float_data)

        # 优化：更新历史音频，使用基于实际秒数的滑动窗口
        self._update_audio_history(float_data, chunk_duration)

        # 如果累积了足够的音频(约1秒)，进行处理
        if self.total_duration >= 1.0:
            # 合并音频数据
            combined_data = np.concatenate(self.audio_buffer)

            # 使用非阻塞方式添加到队列，避免因队列满而阻塞
            try:
                self.audio_queue.put_nowait(combined_data)
            except asyncio.QueueFull:
                logger.warning("音频处理队列已满，丢弃当前音频块")

            # 重置缓冲区
            self.audio_buffer = []
            self.total_duration = 0.0

    def _update_audio_history(self, audio_data, duration):
        """更新音频历史，保持滑动窗口"""
        # 添加新音频到历史
        self.audio_history.append((audio_data, duration))
        self.history_duration += duration

        # 移除过旧的音频，保持历史不超过最大秒数
        while (
            self.history_duration > self.max_history_seconds
            and len(self.audio_history) > 1
        ):
            old_data, old_duration = self.audio_history.pop(0)
            self.history_duration -= old_duration

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
            got_item = False  # 添加标志，跟踪是否成功获取了队列项目
            try:
                # 获取下一个音频块，设置超时防止无限等待
                try:
                    audio_data = self.audio_queue.get(timeout=5)
                    got_item = True  # 标记成功获取项目
                except Empty:  # 使用正确的Empty异常
                    continue

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

                # 优化：准备历史音频记录，只使用有限的上下文
                context_audio = None
                if self.audio_history:
                    # 从历史记录中提取音频数据
                    history_audio_data = [data for data, _ in self.audio_history]
                    if history_audio_data:
                        context_audio = np.concatenate(history_audio_data)
                        # 记录使用的历史音频大小
                        logger.debug(
                            f"使用历史音频上下文: {len(context_audio)/self.samples_per_second:.2f}秒"
                        )

                # 添加超时机制，确保单次转录不会无限消耗时间
                start_time = time.time()
                timeout = 5.0  # 5秒超时

                # 将音频发送到Whisper模型
                try:
                    with torch.no_grad():
                        # 优先使用有限的上下文音频
                        if context_audio is not None:
                            # 使用最近的历史作为上下文
                            input_audio = np.concatenate([context_audio, audio_data])
                        else:
                            input_audio = audio_data

                        # 检查处理是否超时
                        if time.time() - start_time > timeout:
                            logger.warning("音频处理准备阶段已超时")
                            raise TimeoutError("音频处理准备超时")

                        # 进行转录，限制输入音频长度
                        # 如果输入音频太长，只取最后30秒
                        max_audio_length = 30 * self.sample_rate  # 30秒的样本数
                        if len(input_audio) > max_audio_length:
                            logger.warning(
                                f"输入音频过长 ({len(input_audio)/self.sample_rate:.1f}秒)，截取最后30秒"
                            )
                            input_audio = input_audio[-max_audio_length:]

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

                        # 检查处理是否超时
                        if time.time() - start_time > timeout:
                            logger.warning("音频转录已超时，但仍完成了处理")

                        # 发送结果
                        self.result_queue.put(
                            {
                                "text": transcribed_text,
                                "is_final": True,  # 每个音频块我们都当作最终结果
                                "language": result.get("language", self.language),
                                "processing_time": time.time() - start_time,
                            }
                        )

                except TimeoutError:
                    logger.error("音频处理超时")
                    self.result_queue.put(
                        {
                            "text": "转录处理超时，请稍后再试",
                            "is_final": False,
                            "language": self.language,
                        }
                    )
                except Exception as e:
                    logger.error(f"转录处理错误: {str(e)}")
                    self.result_queue.put(
                        {
                            "text": "转录处理出错，请稍后再试",
                            "is_final": False,
                            "language": self.language,
                        }
                    )

            except Exception as e:
                logger.error(f"音频处理错误: {str(e)}")
                logger.error(traceback.format_exc())

            finally:
                # 只有在成功从队列中获取了项目时才标记任务完成
                if got_item:
                    self.audio_queue.task_done()


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
    # whisper_model = load_whisper_model("large-v3-turbo")
    whisper_model = load_whisper_model("tiny")


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
