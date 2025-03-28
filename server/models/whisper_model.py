#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import torch
import whisper
import numpy as np
import psutil
import logging

logger = logging.getLogger("whisper-server")

# 忽略警告
warnings.filterwarnings(
    "ignore", "You are using `torch.load` with `weights_only=False`"
)


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
