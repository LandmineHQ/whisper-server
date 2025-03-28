#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import uuid
import numpy as np
import logging
import time
import websockets
import whisper
import torch
from queue import Queue, Empty
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor
import os
import traceback
import webrtcvad
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import psutil
from enum import Enum
import json

# 配置日志
logger = logging.getLogger("whisper-server")
# 会话信息存储
active_sessions = {}


# 定义处理状态枚举
class ProcessingState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    WAITING_MODEL = "waiting_model"


# 音频片段数据类
@dataclass
class AudioSegment:
    data: np.ndarray
    duration: float
    timestamp: float = field(default_factory=time.time)
    contains_speech: bool = True


# 转录结果数据类
@dataclass
class TranscriptionResult:
    text: str
    is_final: bool
    language: str
    processing_time: float = 0.0
    confidence: float = 1.0
    segments: List[Dict[str, Any]] = field(default_factory=list)


# 加载whisper模型
def load_whisper_model(
    model_name="large-v3-turbo", device=None, quantize=True, amd_optimization=False
):
    """
    加载并优化Whisper模型

    参数:
        model_name: 模型名称
        device: 设备 (None=自动选择, "cpu", "cuda", "mps", "xpu" 等)
        quantize: 是否量化模型
        amd_optimization: 是否使用AMD优化
    """
    logger.info(f"正在加载 Whisper 模型 {model_name}...")

    # 自动选择最佳设备
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = "xpu"  # Intel XPU
        else:
            device = "cpu"

    # 加载基础模型
    model = whisper.load_model(model_name)

    # 设备特定优化
    if device == "cuda":
        model = model.to(device)
        if quantize:
            model = model.half()  # FP16量化
    elif device == "cpu":
        # CPU优化
        if amd_optimization:
            # AMD CPU特定优化
            try:
                import torch_directml  # 需要安装: pip install torch-directml

                dml = torch_directml.device()
                model = model.to(dml)
                logger.info("已应用AMD DirectML优化")
            except ImportError:
                logger.warning("未找到torch_directml，使用标准CPU模式")

                # 尝试使用ONNX Runtime优化
                try:
                    import onnxruntime as ort

                    # 检查是否有AMD优化的执行提供程序
                    providers = ort.get_available_providers()
                    if "ROCMExecutionProvider" in providers:
                        logger.info("使用ROCMExecutionProvider进行ONNX优化")
                        # 这里可以添加ONNX模型转换和优化代码
                    else:
                        logger.info(f"可用ONNX提供程序: {providers}")
                except ImportError:
                    logger.warning("未找到onnxruntime，使用标准PyTorch")

                # 使用PyTorch内置优化
                torch.set_num_threads(psutil.cpu_count(logical=True))
                if hasattr(torch, "set_num_interop_threads"):
                    torch.set_num_interop_threads(psutil.cpu_count(logical=False))
    elif device == "xpu":
        # Intel XPU (如果可用)
        model = model.to(device)
    else:
        model = model.to(device)

    # 模型预热，减少首次推理延迟
    logger.info("进行模型预热...")
    dummy_audio = np.zeros(16000, dtype=np.float32)  # 1秒静音
    with torch.no_grad():
        model.transcribe(dummy_audio, verbose=False)

    logger.info(f"Whisper 模型 {model_name} 加载完成，使用设备: {device}")
    return model


class AudioProcessor:
    def __init__(
        self,
        sample_rate,
        language,
        use_vad=True,
        min_history_seconds=2.0,
        max_history_seconds=10.0,
        max_workers=2,
        vad_aggressiveness=3,
        custom_vocabulary=None,
    ):
        """
        初始化音频处理器

        参数:
            sample_rate: 音频采样率
            language: 转录语言
            use_vad: 是否使用语音活动检测
            min_history_seconds: 最小历史记录时长
            max_history_seconds: 最大历史记录时长
            max_workers: 处理线程池最大线程数
            vad_aggressiveness: VAD灵敏度(0-3)
            custom_vocabulary: 自定义词汇表
        """
        self.sample_rate = sample_rate
        self.language = language
        self.audio_buffer = []
        self.total_duration = 0.0
        self.last_transcript = ""
        self.last_full_transcript = ""

        # 队列和线程池
        self.audio_queue = Queue(maxsize=20)  # 增加队列大小，提高缓冲能力
        self.result_queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []
        self.processing_lock = Lock()  # 防止并发处理冲突

        # 状态管理
        self.state = ProcessingState.IDLE
        self.last_state_change = time.time()

        # 历史记录管理
        self.samples_per_second = self.sample_rate
        self.min_history_seconds = min_history_seconds
        self.max_history_seconds = max_history_seconds
        self.current_history_seconds = min_history_seconds
        self.audio_history = []
        self.history_duration = 0.0

        # VAD配置
        self.use_vad = use_vad
        if use_vad:
            self.vad = webrtcvad.Vad(vad_aggressiveness)
            self.vad_frame_duration = 30  # 毫秒
            self.vad_frame_size = int(sample_rate * self.vad_frame_duration / 1000)
            self.vad_window = []  # 保存最近的VAD结果用于平滑
            self.vad_window_size = 5

        # 自定义词汇
        self.custom_vocabulary = custom_vocabulary

        # 性能监控
        self.processing_stats = {
            "processed_chunks": 0,
            "total_processing_time": 0,
            "max_processing_time": 0,
            "min_processing_time": float("inf"),
            "timeouts": 0,
            "errors": 0,
            "vad_filtered": 0,
            "memory_usage": [],
            "queue_sizes": [],
        }

        # 启动处理线程
        self.processing_thread = Thread(target=self._process_audio_thread, daemon=True)
        self.processing_thread.start()

        # 启动监控线程
        self.monitoring_thread = Thread(target=self._monitoring_thread, daemon=True)
        self.monitoring_thread.start()

        logger.info(
            f"AudioProcessor 初始化完成: 采样率={sample_rate}, 语言={language}, VAD={use_vad}"
        )

    def add_audio_chunk(self, audio_data):
        """添加16位PCM音频数据块到缓冲区，应用VAD和智能批处理"""
        # 计算块持续时间(秒)
        chunk_duration = len(audio_data) / 2 / self.sample_rate

        # 将音频数据转换为float32数组并规范化到[-1, 1]范围
        float_data = (
            np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        )

        # 应用预处理
        float_data = self._preprocess_audio(float_data)

        # 使用VAD检查是否包含语音
        contains_speech = True
        if self.use_vad:
            contains_speech = self._is_speech(float_data)

            # 更新VAD窗口用于平滑决策
            self.vad_window.append(contains_speech)
            if len(self.vad_window) > self.vad_window_size:
                self.vad_window.pop(0)

            # 平滑VAD决策，减少抖动
            speech_ratio = sum(self.vad_window) / len(self.vad_window)
            contains_speech = (
                speech_ratio > 0.3
            )  # 如果30%以上帧检测到语音，则认为有语音

            if not contains_speech:
                self.processing_stats["vad_filtered"] += 1

        # 保存到当前音频块
        self.audio_buffer.append(float_data)
        self.total_duration += chunk_duration

        # 更新历史音频
        self._update_audio_history(float_data, chunk_duration, contains_speech)

        # 动态调整处理触发阈值，根据队列大小和是否检测到语音
        process_threshold = 1.0  # 默认1秒
        queue_size = self.audio_queue.qsize()

        # 队列负载自适应
        if queue_size > self.audio_queue.maxsize * 0.7:
            # 队列较满，提高阈值减少处理频率
            process_threshold = 2.0
            logger.debug(
                f"队列负载较高 ({queue_size}/{self.audio_queue.maxsize})，调整处理阈值为 {process_threshold}秒"
            )
        elif queue_size < self.audio_queue.maxsize * 0.3:
            # 队列较空，降低阈值增加处理频率
            process_threshold = 0.5

        # 如果检测到语音结束或累积了足够的音频，进行处理
        if self.total_duration >= process_threshold or (
            self.use_vad and not contains_speech and self.total_duration >= 0.5
        ):

            # 只有当缓冲区有内容时才处理
            if self.audio_buffer:
                # 合并音频数据
                combined_data = np.concatenate(self.audio_buffer)

                # 创建音频段对象
                segment = AudioSegment(
                    data=combined_data,
                    duration=self.total_duration,
                    contains_speech=contains_speech,
                )

                # 使用非阻塞方式添加到队列
                try:
                    self.audio_queue.put_nowait(segment)
                    logger.debug(
                        f"添加音频块到队列: {self.total_duration:.2f}秒, 队列大小: {self.audio_queue.qsize()}/{self.audio_queue.maxsize}"
                    )
                except Queue.Full:
                    logger.warning(
                        f"音频处理队列已满({self.audio_queue.qsize()}/{self.audio_queue.maxsize})，丢弃当前音频块"
                    )

                # 记录队列大小统计
                self.processing_stats["queue_sizes"].append(self.audio_queue.qsize())
                if len(self.processing_stats["queue_sizes"]) > 100:
                    self.processing_stats["queue_sizes"] = self.processing_stats[
                        "queue_sizes"
                    ][-100:]

            # 重置缓冲区
            self.audio_buffer = []
            self.total_duration = 0.0

    def _preprocess_audio(self, audio_data):
        """音频预处理函数"""
        # 应用简单的噪声门限（低于阈值的信号视为噪声）
        noise_threshold = 0.005  # 阈值设为最大振幅的0.5%
        audio_data = np.where(np.abs(audio_data) < noise_threshold, 0, audio_data)

        # 应用简单的音频规范化
        if np.max(np.abs(audio_data)) > 0:
            normalized = audio_data / np.max(np.abs(audio_data)) * 0.9
            return normalized

        return audio_data

    def _is_speech(self, audio_data):
        """检查音频片段是否包含语音"""
        if not self.use_vad:
            return True

        # 将float32转回int16以供VAD使用
        pcm_data = (audio_data * 32768).astype(np.int16).tobytes()

        # 分帧检测语音
        frames = [
            pcm_data[i : i + self.vad_frame_size * 2]
            for i in range(0, len(pcm_data), self.vad_frame_size * 2)
        ]

        if not frames:
            return False

        speech_frames = sum(
            1
            for frame in frames
            if len(frame) == self.vad_frame_size * 2
            and self.vad.is_speech(frame, self.sample_rate)
        )

        # 如果超过25%的帧包含语音，则认为这段音频含有语音
        speech_ratio = speech_frames / len(frames) if frames else 0
        return speech_ratio > 0.25

    def _update_audio_history(self, audio_data, duration, contains_speech=True):
        """更新音频历史，保持滑动窗口"""
        # 添加新音频到历史
        self.audio_history.append((audio_data, duration, contains_speech))
        self.history_duration += duration

        # 移除过旧的音频，保持历史不超过最大秒数
        while (
            self.history_duration > self.current_history_seconds
            and len(self.audio_history) > 1
        ):
            old_data, old_duration, _ = self.audio_history.pop(0)
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

    def _adapt_history_size(self, transcription_quality):
        """根据转录质量动态调整历史大小"""
        # transcription_quality是衡量转录质量的指标，0-1之间

        if transcription_quality < 0.5:
            # 转录质量差，增加历史上下文长度
            self.current_history_seconds = min(
                self.max_history_seconds, self.current_history_seconds * 1.5
            )
            logger.debug(
                f"转录质量较低，增加历史上下文至 {self.current_history_seconds:.1f}秒"
            )
        elif transcription_quality > 0.8:
            # 转录质量好，可以减少历史上下文
            self.current_history_seconds = max(
                self.min_history_seconds, self.current_history_seconds * 0.8
            )
            logger.debug(
                f"转录质量良好，减少历史上下文至 {self.current_history_seconds:.1f}秒"
            )

    def _prepare_transcription_prompt(self):
        """准备转录提示，智能使用先前的转录结果"""
        if not self.last_full_transcript:
            return None

        # 只使用上一个转录的最后几个句子作为提示
        sentences = self.last_full_transcript.split("。")
        if len(sentences) > 3:
            # 只使用最后3个句子
            prompt = "。".join(sentences[-3:]) + "。"
        else:
            prompt = self.last_full_transcript

        return prompt

    def _prepare_whisper_options(self):
        """准备Whisper模型选项"""
        options = {
            "language": self.language,
            "initial_prompt": self._prepare_transcription_prompt(),
            "verbose": False,
        }

        # 添加自定义词汇（如果有）
        if self.custom_vocabulary:
            # 根据Whisper API添加自定义词汇选项
            # 注意：具体实现可能需要根据Whisper版本调整
            options["vocab"] = self.custom_vocabulary

        return options

    def _process_audio_thread(self):
        """后台音频处理线程"""
        global whisper_model

        consecutive_errors = 0
        max_consecutive_errors = 3
        backoff_time = 1.0  # 初始回退时间，秒

        while True:
            got_item = False
            try:
                # 更新状态
                self.state = ProcessingState.IDLE
                self.last_state_change = time.time()

                # 获取下一个音频块，设置超时防止无限等待
                try:
                    segment = self.audio_queue.get(timeout=5)
                    got_item = True

                    # 更新状态
                    self.state = ProcessingState.PROCESSING
                    self.last_state_change = time.time()

                except Empty:
                    continue

                # 如果whisper模型还没加载完成，等待
                if whisper_model is None:
                    self.state = ProcessingState.WAITING_MODEL
                    self.last_state_change = time.time()

                    time.sleep(0.1)
                    self.result_queue.put(
                        {
                            "text": "正在加载语音识别模型...",
                            "is_final": False,
                            "language": self.language,
                            "confidence": 1.0,
                        }
                    )
                    self.audio_queue.task_done()
                    continue

                # 如果该段不包含语音且我们使用VAD，则跳过处理
                if self.use_vad and not segment.contains_speech:
                    logger.debug("跳过不包含语音的音频段")
                    self.audio_queue.task_done()
                    continue

                # 准备历史音频上下文
                context_audio = self._prepare_audio_context()

                # 添加超时机制，确保单次转录不会无限消耗时间
                start_time = time.time()
                timeout = 5.0  # 5秒超时

                # 将音频发送到Whisper模型
                try:
                    with torch.no_grad():
                        # 准备输入音频
                        input_audio = self._prepare_input_audio(
                            segment.data, context_audio
                        )

                        # 检查处理是否超时
                        if time.time() - start_time > timeout:
                            logger.warning("音频处理准备阶段已超时")
                            raise TimeoutError("音频处理准备超时")

                        # 准备Whisper选项
                        whisper_options = self._prepare_whisper_options()

                        # 进行转录
                        result = whisper_model.transcribe(
                            input_audio, **whisper_options
                        )

                        # 获取文本和置信度
                        transcribed_text = result["text"].strip()
                        confidence = self._calculate_confidence(result)

                        # 动态调整历史大小
                        self._adapt_history_size(confidence)

                        # 检查处理是否超时
                        processing_time = time.time() - start_time
                        if processing_time > timeout:
                            logger.warning("音频转录已超时，但仍完成了处理")

                        # 更新统计信息
                        self._update_processing_stats(processing_time)

                        # 发送结果
                        self.result_queue.put(
                            {
                                "text": transcribed_text,
                                "is_final": True,
                                "language": result.get("language", self.language),
                                "processing_time": processing_time,
                                "confidence": confidence,
                            }
                        )

                        # 成功处理，重置错误计数
                        consecutive_errors = 0
                        backoff_time = 1.0

                except TimeoutError:
                    self.processing_stats["timeouts"] += 1
                    logger.error("音频处理超时")
                    self.result_queue.put(
                        {
                            "text": "转录处理超时，请稍后再试",
                            "is_final": False,
                            "language": self.language,
                            "confidence": 0.0,
                        }
                    )
                except Exception as e:
                    self.processing_stats["errors"] += 1
                    logger.error(f"转录处理错误: {str(e)}")
                    self.result_queue.put(
                        {
                            "text": "转录处理出错，请稍后再试",
                            "is_final": False,
                            "language": self.language,
                            "confidence": 0.0,
                        }
                    )

            except Exception as e:
                consecutive_errors += 1
                self.state = ProcessingState.ERROR
                self.last_state_change = time.time()

                logger.error(
                    f"音频处理错误 ({consecutive_errors}/{max_consecutive_errors}): {str(e)}"
                )
                logger.error(traceback.format_exc())

                if consecutive_errors >= max_consecutive_errors:
                    logger.critical("达到最大连续错误次数，线程暂停恢复...")
                    time.sleep(backoff_time)
                    backoff_time = min(30, backoff_time * 2)  # 指数回退，最多30秒

            finally:
                # 只有在成功从队列中获取了项目时才标记任务完成
                if got_item:
                    self.audio_queue.task_done()

    def _prepare_audio_context(self):
        """准备历史音频上下文"""
        context_audio = None

        # 从历史记录中提取包含语音的音频数据
        speech_history = [
            data for data, _, has_speech in self.audio_history if has_speech
        ]

        if speech_history:
            context_audio = np.concatenate(speech_history)
            logger.debug(
                f"使用历史音频上下文: {len(context_audio)/self.samples_per_second:.2f}秒"
            )

        return context_audio

    def _prepare_input_audio(self, current_audio, context_audio):
        """准备输入音频，合并上下文和当前音频"""
        # 如果有上下文音频，合并它
        if context_audio is not None:
            input_audio = np.concatenate([context_audio, current_audio])
        else:
            input_audio = current_audio

        # 限制输入音频长度
        max_audio_length = 30 * self.sample_rate  # 30秒的样本数
        if len(input_audio) > max_audio_length:
            logger.warning(
                f"输入音频过长 ({len(input_audio)/self.sample_rate:.1f}秒)，截取最后30秒"
            )
            input_audio = input_audio[-max_audio_length:]

        return input_audio

    def _calculate_confidence(self, result):
        """计算转录结果的置信度"""
        # 如果结果包含段级别的置信度分数
        if "segments" in result and result["segments"]:
            # 计算所有段的平均置信度
            confidences = [seg.get("confidence", 1.0) for seg in result["segments"]]
            return sum(confidences) / len(confidences)
        return 1.0  # 默认置信度

    def _update_processing_stats(self, processing_time):
        """更新处理统计信息"""
        self.processing_stats["processed_chunks"] += 1
        self.processing_stats["total_processing_time"] += processing_time
        self.processing_stats["max_processing_time"] = max(
            self.processing_stats["max_processing_time"], processing_time
        )
        if processing_time < self.processing_stats["min_processing_time"]:
            self.processing_stats["min_processing_time"] = processing_time

    def _monitoring_thread(self):
        """监控线程，收集系统资源使用情况"""
        while True:
            try:
                # 获取当前进程
                process = psutil.Process(os.getpid())

                # 记录内存使用
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)  # 转换为MB

                # 添加到统计信息
                self.processing_stats["memory_usage"].append(memory_mb)

                # 只保留最近100个数据点
                if len(self.processing_stats["memory_usage"]) > 100:
                    self.processing_stats["memory_usage"] = self.processing_stats[
                        "memory_usage"
                    ][-100:]

                # 检查内存使用是否过高
                if memory_mb > 1000:  # 超过1GB
                    logger.warning(f"内存使用较高: {memory_mb:.1f} MB")

            except Exception as e:
                logger.error(f"监控线程错误: {str(e)}")

            finally:
                # 每5秒收集一次
                time.sleep(5)

    def get_stats(self):
        """获取处理统计信息"""
        stats = self.processing_stats.copy()
        if stats["processed_chunks"] > 0:
            stats["avg_processing_time"] = (
                stats["total_processing_time"] / stats["processed_chunks"]
            )
        else:
            stats["avg_processing_time"] = 0

        # 添加当前状态
        stats["current_state"] = self.state.value
        stats["state_duration"] = time.time() - self.last_state_change
        stats["queue_size"] = self.audio_queue.qsize()
        stats["queue_capacity"] = self.audio_queue.maxsize
        stats["history_seconds"] = self.current_history_seconds

        # 计算平均值
        if stats["memory_usage"]:
            stats["avg_memory_usage"] = sum(stats["memory_usage"]) / len(
                stats["memory_usage"]
            )
            stats["max_memory_usage"] = max(stats["memory_usage"])
        else:
            stats["avg_memory_usage"] = 0
            stats["max_memory_usage"] = 0

        if stats["queue_sizes"]:
            stats["avg_queue_size"] = sum(stats["queue_sizes"]) / len(
                stats["queue_sizes"]
            )
            stats["max_queue_size"] = max(stats["queue_sizes"])
        else:
            stats["avg_queue_size"] = 0
            stats["max_queue_size"] = 0

        return stats

    def save_state(self, filepath):
        """保存处理器状态到文件"""
        state = {
            "language": self.language,
            "last_transcript": self.last_transcript,
            "last_full_transcript": self.last_full_transcript,
            "current_history_seconds": self.current_history_seconds,
            "stats": self.get_stats(),
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            logger.info(f"状态已保存到 {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存状态失败: {str(e)}")
            return False

    def load_state(self, filepath):
        """从文件加载处理器状态"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                state = json.load(f)

            # 恢复状态
            if state["language"] == self.language:  # 只有语言匹配才恢复
                self.last_transcript = state.get("last_transcript", "")
                self.last_full_transcript = state.get("last_full_transcript", "")
                self.current_history_seconds = state.get(
                    "current_history_seconds", self.min_history_seconds
                )
                logger.info(f"状态已从 {filepath} 加载")
                return True
            else:
                logger.warning(
                    f"语言不匹配，无法加载状态: {state['language']} != {self.language}"
                )
                return False
        except Exception as e:
            logger.error(f"加载状态失败: {str(e)}")
            return False


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
