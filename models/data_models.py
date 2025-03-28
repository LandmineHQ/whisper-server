#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


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
