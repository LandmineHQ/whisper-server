#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
import time
import json
import traceback
import torch
import concurrent.futures
import re
from queue import Queue
from threading import Thread, Lock, Event
from enum import Enum
from audio.vad import VADProcessor  # 导入VAD处理器
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import uuid

logger = logging.getLogger("whisper-server")


class ProcessingState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    WAITING_MODEL = "waiting_model"


class UtteranceState(Enum):
    """语音段状态"""

    COLLECTING = "collecting"  # 正在收集音频
    PROCESSING = "processing"  # 正在处理转录
    COMPLETED = "completed"  # 转录完成
    FAILED = "failed"  # 转录失败


@dataclass
class Utterance:
    """语音段对象，表示一段连续的语音"""

    id: str  # 唯一标识符
    audio_chunks: List[np.ndarray]  # 音频数据块列表
    start_time: float  # 开始时间戳
    end_time: Optional[float] = None  # 结束时间戳
    duration: float = 0.0  # 音频持续时间(秒)
    state: UtteranceState = UtteranceState.COLLECTING  # 当前状态
    result: Dict[str, Any] = None  # 转录结果
    is_final: bool = False  # 是否为最终结果


class AudioProcessor:
    def __init__(
        self,
        sample_rate,
        language,
        whisper_model=None,
        max_audio_context_seconds=10.0,  # 最长10秒音频上下文
        max_utterance_time=30.0,  # 最长30秒语音段
        silence_threshold=1.0,  # 1秒静音判定为语音段结束
        use_vad=True,  # 启用VAD
        vad_aggressiveness=3,  # VAD灵敏度(0-3)
        interim_results=True,  # 是否提供中间结果
        interim_interval=2.0,  # 中间结果间隔(秒)
    ):
        """
        初始化基于语音段的实时音频处理器

        参数:
            sample_rate: 音频采样率
            language: 转录语言
            whisper_model: Whisper模型实例
            max_audio_context_seconds: 最大音频上下文时长(秒)
            max_utterance_time: 最大语音段时长(秒)
            silence_threshold: 静音阈值(秒)，超过此值视为语音段结束
            use_vad: 是否使用语音活动检测
            vad_aggressiveness: VAD灵敏度(0-3)
            interim_results: 是否提供中间结果
            interim_interval: 中间结果间隔(秒)
        """
        self.sample_rate = sample_rate
        self.language = language
        self.whisper_model = whisper_model
        self.max_audio_context_seconds = max_audio_context_seconds
        self.max_audio_samples = int(max_audio_context_seconds * sample_rate)
        self.max_utterance_time = max_utterance_time
        self.silence_threshold = silence_threshold
        self.silence_samples = int(silence_threshold * sample_rate)

        # 中间结果配置
        self.interim_results = interim_results
        self.interim_interval = interim_interval
        # 优化：添加中间结果记忆，避免重复
        self._last_interim_text = {}

        # 结果队列和转录历史
        self.result_queue = Queue()
        self.last_transcript = ""
        # 修改: 使用集合而非列表来防止重复
        self.transcription_history = []
        self.transcription_set = set()  # 用于快速检查重复

        # VAD配置
        self.use_vad = use_vad
        if use_vad:
            self.vad_processor = VADProcessor(
                sample_rate=sample_rate, aggressiveness=vad_aggressiveness
            )
            self.speech_detected = False
            self.silence_counter = 0
            # 修改: 降低VAD稳定性要求
            self.vad_stable_count = 0
            self.vad_stable_threshold = 0  # 立即响应语音检测

        # 当前语音段
        self.current_utterance = None
        # 语音段队列 - 等待处理的语音段
        self.utterance_queue = Queue()
        # 已完成的语音段 - 最近5个
        self.completed_utterances = []

        # 音频缓冲区 - 用于检测语音开始
        self.prebuffer = []
        self.prebuffer_duration = 0.0
        self.max_prebuffer_duration = 0.5  # 最大前置缓冲区时长(秒)

        # 同步和状态控制
        self.processing_lock = Lock()
        self.utterance_lock = Lock()
        self.state = ProcessingState.IDLE
        self.is_running = True
        self.stop_event = Event()

        # 统计数据
        self.stats = {
            "processed_utterances": 0,
            "total_processing_time": 0,
            "max_processing_time": 0,
            "timeouts": 0,
            "errors": 0,
            "vad_filtered": 0,
            "total_audio_chunks": 0,
            "utterances_started": 0,
            "utterances_completed": 0,
        }

        # 修改: 减少最小语音段间隔
        self.last_utterance_end_time = 0
        self.min_utterance_interval = 0.1  # 最小语音段间隔(秒)，从0.3秒减为0.1秒

        # 启动处理线程
        self.processor_thread = Thread(target=self._utterance_processor, daemon=True)
        self.processor_thread.start()

        # 如果启用中间结果，启动中间结果线程
        if self.interim_results:
            self.interim_thread = Thread(
                target=self._interim_results_processor, daemon=True
            )
            self.interim_thread.start()

        logger.info(
            f"实时AudioProcessor初始化: 采样率={sample_rate}, "
            f"语言={language}, 最大语音段={max_utterance_time}秒, "
            f"静音阈值={silence_threshold}秒, VAD={use_vad}, "
            f"中间结果={interim_results}"
        )

    def add_audio_chunk(self, audio_data):
        # 转换数据为浮点数组但不立即预处理
        original_float_data = (
            np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        )

        # 先在原始数据上检测语音能量水平
        original_energy = np.mean(np.abs(original_float_data))
        original_peak = np.max(np.abs(original_float_data))
        has_sufficient_energy = original_peak > 0.02  # 使用原始峰值判断

        # 再进行预处理用于VAD和转录
        processed_data = self._preprocess_audio(original_float_data)
        chunk_duration = len(processed_data) / self.sample_rate

        # 使用VAD和原始能量共同判断
        is_speech = self._detect_speech(processed_data, original_energy, original_peak)

        with self.utterance_lock:
            if is_speech:
                self._handle_speech_frame(processed_data, chunk_duration)
            else:
                self._handle_silence_frame(processed_data, chunk_duration)

    def _detect_speech(self, processed_audio, original_energy, original_peak):
        # VAD结果 (VAD可能在预处理后效果更好)
        contains_speech = False
        if self.use_vad:
            contains_speech = self.vad_processor.process_with_smoothing(processed_audio)

        # 使用原始音频特征判断能量
        is_low_energy = original_peak < 0.02  # 更高的阈值，基于原始峰值

        # 计算动态范围 (语音通常有较高的峰值/平均比)
        dynamic_ratio = original_peak / (original_energy + 1e-10)
        has_speech_dynamics = dynamic_ratio > 4.0

        # 综合判断
        return (
            (contains_speech and not is_low_energy)
            if self.use_vad
            else (not is_low_energy and has_speech_dynamics)
        )

    def _handle_speech_frame(self, audio_data, chunk_duration):
        """
        处理包含语音的音频帧

        参数:
            audio_data: 预处理后的音频数据
            chunk_duration: 音频帧时长(秒)
        """
        # 重置静音计数
        self.silence_counter = 0

        # 如果之前没有检测到语音，尝试开始新的语音段
        if not self.speech_detected:
            current_time = time.time()
            if (
                current_time - self.last_utterance_end_time
                > self.min_utterance_interval
            ):
                self._start_new_utterance()
                self.speech_detected = True
                logger.debug("VAD检测到语音开始，创建新语音段")
            else:
                logger.debug(
                    f"忽略快速连续的语音段，间隔过短: {current_time - self.last_utterance_end_time:.2f}秒"
                )

        # 将音频添加到当前语音段
        if self.current_utterance:
            self.current_utterance.audio_chunks.append(audio_data)
            self.current_utterance.duration += chunk_duration

            # 检查是否超过最大语音段时长
            if self.current_utterance.duration >= self.max_utterance_time:
                logger.debug(
                    f"语音段达到最大时长 {self.max_utterance_time}秒，强制结束"
                )
                self._end_current_utterance()

    def _handle_silence_frame(self, audio_data, chunk_duration):
        """
        处理静音音频帧

        参数:
            audio_data: 预处理后的音频数据
            chunk_duration: 音频帧时长(秒)
        """
        # 增加静音计数
        self.silence_counter += len(audio_data)

        # 如果当前有活跃的语音段
        if self.speech_detected and self.current_utterance:
            # 添加静音到当前语音段(作为尾部)
            self.current_utterance.audio_chunks.append(audio_data)
            self.current_utterance.duration += chunk_duration

            # 记录静音检测状态
            silence_ratio = self.silence_counter / self.silence_samples
            logger.debug(
                f"静音进度: {silence_ratio:.2f} ({self.silence_counter}/{self.silence_samples})"
            )

            # 检查是否达到静音阈值，结束当前语音段
            if self.silence_counter >= self.silence_samples:
                logger.debug(f"检测到 {self.silence_threshold}秒 静音，语音段结束")
                self._end_current_utterance()
                self.speech_detected = False
                self.silence_counter = 0
        else:
            # 没有活跃语音段，保留一些音频在前置缓冲区
            self._add_to_prebuffer(audio_data, chunk_duration)

    def get_latest_result(self):
        """获取最新的转录结果(非阻塞)"""
        if not self.result_queue.empty():
            result = self.result_queue.get()
            # 更新最后的转录结果
            if result.get("is_final", False):
                self.last_transcript = result["text"]
            return result
        return None

    def set_model(self, model):
        """设置Whisper模型实例"""
        self.whisper_model = model

    def close(self):
        """关闭处理器，停止所有线程"""
        self.is_running = False
        self.stop_event.set()

        # 结束当前语音段(如果有)
        with self.utterance_lock:
            if self.current_utterance:
                self._end_current_utterance()

        # 等待处理线程结束
        if self.processor_thread.is_alive():
            self.processor_thread.join(timeout=2.0)

        # 如果启用了中间结果，等待中间结果线程结束
        if (
            self.interim_results
            and hasattr(self, "interim_thread")
            and self.interim_thread.is_alive()
        ):
            self.interim_thread.join(timeout=2.0)

        logger.info("AudioProcessor已关闭")

    # 内部方法

    def _start_new_utterance(self):
        """开始一个新的语音段"""
        # 创建新的语音段
        utterance_id = str(uuid.uuid4())
        utterance = Utterance(
            id=utterance_id,
            audio_chunks=[],
            start_time=time.time(),
            state=UtteranceState.COLLECTING,
        )

        # 如果有预缓冲的音频，添加到语音段
        if self.prebuffer:
            utterance.audio_chunks.extend(self.prebuffer)
            utterance.duration = self.prebuffer_duration
            # 清空预缓冲区
            self.prebuffer = []
            self.prebuffer_duration = 0.0

        self.current_utterance = utterance
        self.stats["utterances_started"] += 1
        logger.debug(f"开始新语音段 {utterance_id}")

    def _end_current_utterance(self):
        """结束当前语音段并提交处理"""
        if not self.current_utterance:
            return

        # 更新语音段状态
        self.current_utterance.end_time = time.time()
        self.current_utterance.state = UtteranceState.PROCESSING

        # 更新最后语音段结束时间
        self.last_utterance_end_time = time.time()

        # 清除该语音段的中间结果记忆
        if self.current_utterance.id in self._last_interim_text:
            del self._last_interim_text[self.current_utterance.id]

        # 将语音段放入处理队列
        self.utterance_queue.put(self.current_utterance)
        logger.debug(
            f"语音段 {self.current_utterance.id} 结束，持续时间: {self.current_utterance.duration:.2f}秒"
        )

        # 重置当前语音段
        self.current_utterance = None

    def _add_to_prebuffer(self, audio_data, duration):
        """添加音频到预缓冲区，保持一定长度的上下文"""
        self.prebuffer.append(audio_data)
        self.prebuffer_duration += duration

        # 限制预缓冲区大小
        while (
            self.prebuffer_duration > self.max_prebuffer_duration
            and len(self.prebuffer) > 1
        ):
            removed = self.prebuffer.pop(0)
            self.prebuffer_duration -= len(removed) / self.sample_rate

    def _utterance_processor(self):
        """语音段处理线程 - 处理队列中的语音段"""
        logger.info("启动语音段处理线程")

        while self.is_running and not self.stop_event.is_set():
            try:
                # 从队列获取语音段，设置超时以便定期检查是否应该退出
                try:
                    utterance = self.utterance_queue.get(timeout=0.5)
                except:
                    continue

                # 处理语音段
                self._process_utterance(utterance)

            except Exception as e:
                logger.error(f"语音段处理错误: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(0.5)  # 简单的错误恢复

    def _interim_results_processor(self):
        """中间结果处理线程 - 为长语音段提供中间结果"""
        logger.info(f"启动中间结果处理线程，间隔: {self.interim_interval}秒")

        # 跟踪上次处理的语音段和位置
        last_utterance_id = None
        last_processed_samples = 0

        while self.is_running and not self.stop_event.is_set():
            try:
                # 检查当前是否有正在收集的语音段
                with self.utterance_lock:
                    current = self.current_utterance
                    if (
                        current
                        and current.state == UtteranceState.COLLECTING
                        and current.duration >= self.interim_interval * 1.5
                    ):
                        # 检测是否是新的语音段
                        if last_utterance_id != current.id:
                            last_utterance_id = current.id
                            last_processed_samples = 0  # 新语音段，重置位置

                        # 计算当前总样本数和新增样本数
                        total_samples = sum(
                            len(chunk) for chunk in current.audio_chunks
                        )
                        new_samples = total_samples - last_processed_samples

                        # 只有当有足够的新音频时才处理中间结果
                        if (
                            new_samples
                            >= self.interim_interval * 0.75 * self.sample_rate
                        ):
                            # 创建包含上下文的音频数据
                            context_seconds = 3.0  # 保留3秒上下文
                            context_samples = min(
                                int(context_seconds * self.sample_rate),
                                last_processed_samples,
                            )
                            start_pos = max(
                                0, total_samples - new_samples - context_samples
                            )

                            # 收集需要处理的音频块
                            chunks_to_process = []
                            samples_so_far = 0
                            for chunk in current.audio_chunks:
                                if samples_so_far + len(chunk) > start_pos:
                                    # 添加需要的部分
                                    start_in_chunk = max(0, start_pos - samples_so_far)
                                    chunks_to_process.append(chunk[start_in_chunk:])
                                elif samples_so_far >= start_pos:
                                    chunks_to_process.append(chunk)
                                samples_so_far += len(chunk)

                            utterance_id = current.id
                            duration = current.duration

                            # 更新已处理位置
                            last_processed_samples = total_samples
                        else:
                            chunks_to_process = None
                    else:
                        chunks_to_process = None
                        # 当前没有收集中的语音段，重置跟踪变量
                        if (
                            current is None
                            or current.state != UtteranceState.COLLECTING
                        ):
                            last_utterance_id = None
                            last_processed_samples = 0

                # 如果有足够新的音频，处理中间结果
                if chunks_to_process:
                    # 合并音频数据
                    audio_data = np.concatenate(chunks_to_process)

                    # 安全检查 - 限制最大长度
                    max_safe_duration = 10.0  # 降低到10秒
                    max_samples = int(max_safe_duration * self.sample_rate)
                    if len(audio_data) > max_samples:
                        logger.warning(
                            f"中间结果音频过长 ({len(audio_data)/self.sample_rate:.2f}秒)，截断处理"
                        )
                        audio_data = audio_data[-max_samples:]

                    # 处理中间结果
                    logger.debug(
                        f"处理语音段 {utterance_id} 的中间结果，总长: {duration:.2f}秒，处理音频长: {len(audio_data)/self.sample_rate:.2f}秒"
                    )
                    self._process_interim_result(audio_data, utterance_id)

                # 等待下一次中间结果间隔
                time.sleep(self.interim_interval)

            except Exception as e:
                logger.error(f"中间结果处理错误: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(1)  # 错误恢复

    def _process_interim_result(self, audio_data, utterance_id):
        """处理中间结果音频并生成转录"""
        if not self.whisper_model:
            return

        # 增强: 进一步验证音频质量
        if len(audio_data) == 0:
            logger.debug("跳过空的中间结果音频")
            return

        audio_stats = {
            "length": len(audio_data) / self.sample_rate,
            "mean": np.mean(np.abs(audio_data)),
            "peak": np.max(np.abs(audio_data)),
            "non_zero": np.count_nonzero(audio_data) / len(audio_data),
        }

        # 更严格的音频质量检查
        if audio_stats["mean"] < 0.001 or audio_stats["peak"] < 0.01:
            logger.debug(f"跳过低质量中间结果音频，能量过低: {audio_stats}")
            return

        # 检查是否包含无效值
        if not np.all(np.isfinite(audio_data)):
            logger.warning(f"中间结果音频包含无效值(NaN/Inf)，跳过处理")
            return

        start_time = time.time()  # 记录处理开始时间

        try:
            # 准备Whisper选项 - 中间结果使用更快的设置
            options = {
                "language": self.language,
                "initial_prompt": self._get_context_prompt(),
                "verbose": False,
                # 对中间结果使用更快的设置
                "beam_size": 1,  # 减少beam search宽度
                "best_of": 1,  # 不生成多个候选
                "temperature": 0,  # 使用贪婪解码
            }

            # 执行转录 - 较短的超时
            with torch.no_grad():
                result = self._execute_transcription_with_timeout(
                    audio_data, options, timeout=2.0
                )

                if result is None:
                    logger.warning(f"中间结果转录超时，跳过")
                    return

            # 处理转录结果
            transcribed_text = result["text"].strip()

            # 如果结果为空，不发送
            if not transcribed_text:
                logger.debug(f"中间结果转录为空，跳过")
                return

            # 优化：检查是否与上一个结果相同
            if (
                utterance_id in self._last_interim_text
                and self._last_interim_text[utterance_id] == transcribed_text
            ):
                logger.debug(f"跳过重复的中间结果: '{transcribed_text}'")
                return

            # 更新最后的中间结果
            self._last_interim_text[utterance_id] = transcribed_text

            # 计算处理时间
            processing_time = time.time() - start_time

            # 将中间结果放入队列，格式保持与最终结果一致
            self.result_queue.put(
                {
                    "text": transcribed_text,
                    "is_final": False,  # 这是中间结果
                    "language": result.get("language", self.language),
                    "utterance_id": utterance_id,
                    "interim": True,
                    "duration": len(audio_data) / self.sample_rate,  # 添加音频时长
                    "processing_time": processing_time,  # 添加处理时间
                }
            )

            logger.debug(
                f"生成中间结果: '{transcribed_text}'，处理时间: {processing_time:.2f}秒"
            )

        except torch.cuda.OutOfMemoryError:
            logger.error("GPU内存不足，尝试截断音频长度")
            # 将音频长度减半再尝试
            if len(audio_data) > 1000:
                self._process_interim_result(
                    audio_data[-len(audio_data) // 2 :], utterance_id
                )
        except Exception as e:
            logger.error(f"中间结果处理错误: {str(e)}")
            logger.error(traceback.format_exc())

    def _process_utterance(self, utterance):
        """处理单个语音段"""
        if not self.whisper_model:
            self.state = ProcessingState.WAITING_MODEL
            # 通知模型未加载
            self.result_queue.put(
                {
                    "text": "正在加载语音识别模型...",
                    "is_final": False,
                    "language": self.language,
                    "utterance_id": utterance.id,
                }
            )
            return

        # 更新处理状态
        self.state = ProcessingState.PROCESSING

        try:
            # 记录转录开始时间
            start_time = time.time()

            # 【增强】检查是否存在有效的音频块
            if not utterance.audio_chunks:
                logger.warning(f"语音段 {utterance.id} 没有音频数据，跳过处理")
                utterance.state = UtteranceState.FAILED
                self.state = ProcessingState.IDLE
                return

            # 【增强】过滤掉无效音频块 - 更严格检查
            valid_chunks = []
            for chunk in utterance.audio_chunks:
                # 检查是否有足够的非零值
                if len(chunk) > 0 and np.isfinite(chunk).all():
                    non_zero_ratio = np.count_nonzero(chunk) / len(chunk)
                    if non_zero_ratio >= 0.01:  # 至少1%的非零值
                        valid_chunks.append(chunk)
                    else:
                        logger.debug(
                            f"跳过几乎全零的音频块，非零比例: {non_zero_ratio:.4f}"
                        )
                else:
                    logger.debug(
                        f"跳过无效音频块，长度: {len(chunk) if len(chunk) > 0 else 0}"
                    )

            if not valid_chunks:
                logger.warning(
                    f"语音段 {utterance.id} 过滤后没有有效音频数据，跳过处理"
                )
                utterance.state = UtteranceState.FAILED
                self.state = ProcessingState.IDLE
                return

            utterance.audio_chunks = valid_chunks

            # 合并语音段的所有音频块
            audio_data = np.concatenate(utterance.audio_chunks)

            # 【增强】添加更详细的音频统计
            audio_stats = {
                "duration": len(audio_data) / self.sample_rate,
                "mean": np.mean(np.abs(audio_data)),
                "peak": np.max(np.abs(audio_data)),
                "non_zero_ratio": np.count_nonzero(audio_data)
                / max(1, len(audio_data)),
            }

            logger.debug(
                f"语音段 {utterance.id} 音频统计: 时长={audio_stats['duration']:.2f}秒, "
                f"平均能量={audio_stats['mean']:.6f}, 峰值={audio_stats['peak']:.6f}, "
                f"非零比例={audio_stats['non_zero_ratio']:.4f}"
            )

            # 【增强】更严格的音频质量检查
            min_audio_length = 0.3  # 增加到0.3秒
            min_samples = int(min_audio_length * self.sample_rate)

            if len(audio_data) < min_samples:
                logger.warning(
                    f"语音段 {utterance.id} 音频数据过短 ({len(audio_data)/self.sample_rate:.3f}秒)，低于最小要求({min_audio_length}秒)，跳过处理"
                )
                utterance.state = UtteranceState.FAILED
                self.state = ProcessingState.IDLE
                return

            # 【增强】检查音频能量和有效性
            if audio_stats["mean"] < 0.001 or audio_stats["peak"] < 0.01:
                logger.warning(
                    f"语音段 {utterance.id} 音频能量过低，跳过处理: 平均={audio_stats['mean']:.6f}, 峰值={audio_stats['peak']:.6f}"
                )
                utterance.state = UtteranceState.FAILED
                self.state = ProcessingState.IDLE
                return

            if audio_stats["non_zero_ratio"] < 0.01:  # 非零值太少
                logger.warning(
                    f"语音段 {utterance.id} 有效音频内容不足，非零比例: {audio_stats['non_zero_ratio']:.4f}，跳过处理"
                )
                utterance.state = UtteranceState.FAILED
                self.state = ProcessingState.IDLE
                return

            # 检查是否有无效值 (NaN, Inf)
            if not np.all(np.isfinite(audio_data)):
                logger.warning(f"语音段 {utterance.id} 包含无效值(NaN/Inf)，跳过处理")
                utterance.state = UtteranceState.FAILED
                self.state = ProcessingState.IDLE
                return

            # 【增强】确保音频数据足够长 - Whisper可能需要一定的最小长度
            min_whisper_duration = 0.5  # Whisper建议的最小音频长度
            min_whisper_samples = int(min_whisper_duration * self.sample_rate)

            if len(audio_data) < min_whisper_samples:
                # 对过短音频进行重复延长，确保满足最小长度要求
                repeats_needed = int(np.ceil(min_whisper_samples / len(audio_data)))
                logger.info(
                    f"语音段 {utterance.id} 音频过短，通过重复 {repeats_needed} 次满足最小长度要求"
                )
                audio_data = np.tile(audio_data, repeats_needed)[:min_whisper_samples]

            # 【增强】规范化音频数据 - 保险措施
            audio_data = np.clip(audio_data, -1.0, 1.0)  # 确保在 [-1, 1] 范围内

            # 准备Whisper选项
            options = {
                "language": self.language,
                "initial_prompt": self._get_context_prompt(),
                "verbose": False,
            }

            # 执行转录 - 添加超时保护
            with torch.no_grad():
                result = self._execute_transcription_with_timeout(audio_data, options)

                if result is None:
                    # 转录超时，记录并返回
                    self.stats["timeouts"] += 1
                    logger.warning(f"语音段 {utterance.id} 转录超时")
                    utterance.state = UtteranceState.FAILED
                    self.state = ProcessingState.IDLE
                    return

            # 处理转录结果
            transcribed_text = result["text"].strip()
            processing_time = time.time() - start_time

            # 如果结果为空，可能是静音或背景噪音
            if not transcribed_text:
                logger.debug(f"语音段 {utterance.id} 转录结果为空")
                utterance.state = UtteranceState.COMPLETED
                self.state = ProcessingState.IDLE
                return

            # 更新统计信息
            self.stats["processed_utterances"] += 1
            self.stats["total_processing_time"] += processing_time
            self.stats["max_processing_time"] = max(
                self.stats["max_processing_time"], processing_time
            )
            self.stats["utterances_completed"] += 1

            # 更新语音段状态和结果
            utterance.state = UtteranceState.COMPLETED
            utterance.result = {
                "text": transcribed_text,
                "language": result.get("language", self.language),
                "processing_time": processing_time,
            }

            # 将结果放入队列
            self.result_queue.put(
                {
                    "text": transcribed_text,
                    "is_final": True,  # 这是最终结果
                    "language": result.get("language", self.language),
                    "processing_time": processing_time,
                    "utterance_id": utterance.id,
                    "duration": utterance.duration,
                }
            )

            # 记录转录历史，最多保留最近5条，同时避免重复
            self._add_to_transcription_history(transcribed_text)

            # 保存已完成的语音段
            self._add_completed_utterance(utterance)

        except torch.cuda.OutOfMemoryError:
            # 优化：处理GPU内存不足的情况
            self.stats["errors"] += 1
            self.state = ProcessingState.ERROR
            utterance.state = UtteranceState.FAILED
            logger.error(f"语音段 {utterance.id} 转录处理GPU内存不足")

            # 尝试减少音频长度重试
            try:
                if utterance.duration > 5.0:  # 只对长音频尝试重试
                    logger.info(f"尝试用减半长度重新转录语音段 {utterance.id}")
                    # 创建缩短的语音段
                    half_duration = utterance.duration / 2
                    total_samples = sum(len(chunk) for chunk in utterance.audio_chunks)
                    keep_samples = int(half_duration * self.sample_rate)

                    # 保留后半部分（通常更重要）
                    new_chunks = []
                    collected = 0
                    for chunk in reversed(utterance.audio_chunks):
                        if collected + len(chunk) <= keep_samples:
                            new_chunks.insert(0, chunk)
                            collected += len(chunk)
                        else:
                            if keep_samples > collected:
                                new_chunks.insert(
                                    0, chunk[-(keep_samples - collected) :]
                                )
                            break

                    # 更新语音段
                    utterance.audio_chunks = new_chunks
                    utterance.duration = half_duration

                    # 重新处理
                    self._process_utterance(utterance)
                    return
            except Exception as retry_error:
                logger.error(f"重试转录失败: {str(retry_error)}")

        except Exception as e:
            self.stats["errors"] += 1
            self.state = ProcessingState.ERROR
            utterance.state = UtteranceState.FAILED
            logger.error(f"语音段 {utterance.id} 转录处理错误: {str(e)}")
            logger.error(traceback.format_exc())

        finally:
            # 重置处理状态
            self.state = ProcessingState.IDLE

    def _get_context_prompt(self):
        """获取上下文提示，基于之前的转录结果"""
        if not self.transcription_history:
            return None

        # 修改: 使用最近一条非重复转录作为上下文，避免循环反馈
        # 不再拼接多条历史，减少重复风险
        if len(self.transcription_history) > 0:
            return self.transcription_history[-1]
        return None

    def _add_to_transcription_history(self, text):
        """添加转录结果到历史记录，避免重复"""
        # 标准化文本 - 去除多余空格
        text = " ".join(text.split())

        # 如果文本已在集合中则不添加
        if text in self.transcription_set:
            return

        # 添加到历史和集合
        self.transcription_history.append(text)
        self.transcription_set.add(text)

        # 保持历史记录长度限制
        while len(self.transcription_history) > 5:
            removed = self.transcription_history.pop(0)
            # 检查是否需要从集合中移除（可能有重复）
            if removed not in self.transcription_history:
                self.transcription_set.remove(removed)

    def _add_completed_utterance(self, utterance):
        """添加已完成的语音段到历史记录"""
        self.completed_utterances.append(utterance)
        # 保留最近5个语音段
        if len(self.completed_utterances) > 5:
            self.completed_utterances.pop(0)

    def _execute_transcription_with_timeout(self, audio_data, options, timeout=5.0):
        """使用超时执行转录，防止长时间阻塞"""
        # 修复: 添加更严格的音频有效性检查
        if len(audio_data) == 0:
            logger.warning("尝试转录空音频数据，已跳过")
            return None

        # 更全面的音频有效性检查
        audio_energy = np.mean(np.abs(audio_data))
        audio_peak = np.max(np.abs(audio_data))
        non_zero_ratio = np.count_nonzero(audio_data) / len(audio_data)

        # 记录诊断信息
        logger.debug(
            f"转录音频特征: 长度={len(audio_data)/self.sample_rate:.3f}秒, 均值={audio_energy:.6f}, "
            f"峰值={audio_peak:.6f}, 非零比例={non_zero_ratio:.3f}"
        )

        # 验证音频数据的有效性
        if audio_peak < 0.005 or non_zero_ratio < 0.01:  # 如果峰值太小或非零值太少
            logger.warning(
                f"音频质量不足，峰值={audio_peak:.6f}, 非零比例={non_zero_ratio:.3f}，跳过转录"
            )
            return None

        if not np.isfinite(audio_data).all():  # 检查NaN和Inf
            logger.warning("音频数据包含无效值(NaN或inf)，跳过转录")
            return None

        # 规范化音频数据保险起见 - 确保没有极端值
        audio_data = np.clip(audio_data, -1.0, 1.0)  # 限制在[-1,1]范围内

        # 优化：根据音频长度动态调整超时
        audio_duration = len(audio_data) / self.sample_rate
        dynamic_timeout = min(max(timeout, audio_duration * 0.5), 10.0)  # 不超过10秒

        # 使用线程池执行带超时的转录
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self.whisper_model.transcribe, audio_data, **options
            )
            try:
                # 等待结果，使用动态超时
                return future.result(timeout=dynamic_timeout)
            except concurrent.futures.TimeoutError:
                logger.warning(f"转录超时 ({dynamic_timeout:.1f}秒)")
                return None

    def _preprocess_audio(self, audio_data):
        """音频预处理"""
        # 应用噪声门限 - 降低阈值使VAD更敏感
        noise_threshold = 0.002  # 从0.005降低到0.002
        audio_data = np.where(np.abs(audio_data) < noise_threshold, 0, audio_data)

        # 音量规范化
        if np.max(np.abs(audio_data)) > 0:
            normalized = audio_data / np.max(np.abs(audio_data)) * 0.95  # 提高音量
            return normalized

        return audio_data

    def get_stats(self):
        """获取处理统计信息"""
        stats = self.stats.copy()

        # 计算平均处理时间
        if stats["processed_utterances"] > 0:
            stats["avg_processing_time"] = (
                stats["total_processing_time"] / stats["processed_utterances"]
            )
        else:
            stats["avg_processing_time"] = 0

        # 计算VAD过滤率
        if stats["total_audio_chunks"] > 0:
            stats["vad_filter_rate"] = (
                stats["vad_filtered"] / stats["total_audio_chunks"]
            )
        else:
            stats["vad_filter_rate"] = 0

        # 添加当前状态
        stats["current_state"] = self.state.value
        stats["speech_detected"] = self.speech_detected
        stats["utterance_queue_size"] = self.utterance_queue.qsize()

        # 添加当前语音段信息
        if self.current_utterance:
            stats["current_utterance"] = {
                "id": self.current_utterance.id,
                "duration": self.current_utterance.duration,
                "chunks": len(self.current_utterance.audio_chunks),
            }

        return stats
