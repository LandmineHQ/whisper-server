#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import webrtcvad
import logging

logger = logging.getLogger("whisper-server")

class VADProcessor:
    """语音活动检测处理器"""
    
    def __init__(self, sample_rate=16000, aggressiveness=3, frame_duration=30):
        """
        初始化VAD处理器
        
        参数:
            sample_rate: 音频采样率
            aggressiveness: VAD灵敏度(0-3)
            frame_duration: VAD帧长度(毫秒)
        """
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_duration = frame_duration  # 毫秒
        self.frame_size = int(sample_rate * frame_duration / 1000)
        self.window = []  # 保存最近的VAD结果用于平滑
        self.window_size = 5
        
    def is_speech(self, audio_data):
        """
        检查音频片段是否包含语音
        
        参数:
            audio_data: 浮点音频数据 [-1, 1]
            
        返回:
            bool: 是否包含语音
        """
        # 将float32转回int16以供VAD使用
        pcm_data = (audio_data * 32768).astype(np.int16).tobytes()

        # 分帧检测语音
        frames = [
            pcm_data[i : i + self.frame_size * 2]
            for i in range(0, len(pcm_data), self.frame_size * 2)
        ]

        if not frames:
            return False

        speech_frames = sum(
            1
            for frame in frames
            if len(frame) == self.frame_size * 2
            and self.vad.is_speech(frame, self.sample_rate)
        )

        # 如果超过25%的帧包含语音，则认为这段音频含有语音
        speech_ratio = speech_frames / len(frames) if frames else 0
        return speech_ratio > 0.25
        
    def process_with_smoothing(self, audio_data):
        """使用窗口平滑处理音频片段的VAD结果"""
        contains_speech = self.is_speech(audio_data)
        
        # 更新VAD窗口用于平滑决策
        self.window.append(contains_speech)
        if len(self.window) > self.window_size:
            self.window.pop(0)
        
        # 平滑VAD决策，减少抖动
        speech_ratio = sum(self.window) / len(self.window)
        smooth_result = speech_ratio > 0.3  # 如果30%以上帧检测到语音，则认为有语音
        
        return smooth_result