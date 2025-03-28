# 音频管理类
import pyaudio


class AudioManager:
    def __init__(self):
        self.pyaudio_instance = None
        self.audio_stream = None

    def init_audio(self, callback=None):
        """初始化音频采集"""
        try:
            p = pyaudio.PyAudio()
            self.pyaudio_instance = p

            # 获取默认输入设备
            default_input = p.get_default_input_device_info()["index"]
            if callback:
                callback(f"使用默认音频输入设备: 设备 #{default_input}")

            # 音频配置
            sample_rate = 16000
            channels = 1
            chunk_size = 1600  # 50ms at 16kHz
            audio_format = pyaudio.paInt16

            stream = p.open(
                format=audio_format,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=default_input,
                frames_per_buffer=chunk_size,
            )
            self.audio_stream = stream
            return True, stream
        except Exception as e:
            if callback:
                callback(f"音频设备初始化错误: {str(e)}")
            return False, None

    def cleanup_audio(self):
        """清理音频资源"""
        if self.audio_stream:
            if self.audio_stream.is_active():
                self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None

        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            self.pyaudio_instance = None
