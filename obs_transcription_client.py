# pyinstaller --onefile --windowed obs_transcription_client.py

import asyncio
import websockets
import json
import base64
import pyaudio
import obsws_python as obs
import time
import tkinter as tk
from tkinter import ttk


# GUI配置界面
class ConfigGUI:
    def __init__(self, root):
        self.root = root
        root.title("OBS转录客户端")
        root.geometry("400x300")

        # 设置风格
        style = ttk.Style()
        style.configure("TButton", padding=5, font=("Helvetica", 10))
        style.configure("TLabel", font=("Helvetica", 10))

        # 创建框架
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # OBS设置
        ttk.Label(main_frame, text="OBS设置").grid(
            column=0, row=0, sticky=tk.W, pady=(0, 5)
        )

        obs_frame = ttk.LabelFrame(main_frame, padding="5")
        obs_frame.grid(column=0, row=1, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(obs_frame, text="主机:").grid(column=0, row=0, sticky=tk.W)
        self.obs_host = ttk.Entry(obs_frame, width=15)
        self.obs_host.insert(0, "localhost")
        self.obs_host.grid(column=1, row=0, sticky=tk.W, padx=5)

        ttk.Label(obs_frame, text="端口:").grid(column=2, row=0, sticky=tk.W)
        self.obs_port = ttk.Entry(obs_frame, width=6)
        self.obs_port.insert(0, "4455")
        self.obs_port.grid(column=3, row=0, sticky=tk.W, padx=5)

        ttk.Label(obs_frame, text="密码:").grid(
            column=0, row=1, sticky=tk.W, pady=(5, 0)
        )
        self.obs_password = ttk.Entry(obs_frame, width=20, show="*")
        self.obs_password.grid(
            column=1, row=1, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=(5, 0)
        )

        ttk.Label(obs_frame, text="文本源:").grid(
            column=0, row=2, sticky=tk.W, pady=(5, 0)
        )
        self.text_source = ttk.Entry(obs_frame, width=20)
        self.text_source.insert(0, "转录文本")
        self.text_source.grid(
            column=1, row=2, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=(5, 0)
        )

        # 服务器设置
        ttk.Label(main_frame, text="转录服务器设置").grid(
            column=0, row=2, sticky=tk.W, pady=(0, 5)
        )

        server_frame = ttk.LabelFrame(main_frame, padding="5")
        server_frame.grid(column=0, row=3, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(server_frame, text="服务器地址:").grid(column=0, row=0, sticky=tk.W)
        self.server_url = ttk.Entry(server_frame, width=30)
        self.server_url.insert(0, "ws://localhost:8765/transcribe")
        self.server_url.grid(column=1, row=0, sticky=(tk.W, tk.E), padx=5)

        # 启动按钮
        self.start_button = ttk.Button(
            main_frame, text="启动转录", command=self.start_transcription
        )
        self.start_button.grid(column=0, row=4, sticky=tk.E, pady=10)

        # 状态标签
        self.status = ttk.Label(main_frame, text="就绪")
        self.status.grid(column=0, row=5, sticky=tk.W)

        # 设置网格权重
        main_frame.columnconfigure(0, weight=1)
        obs_frame.columnconfigure(3, weight=1)
        server_frame.columnconfigure(1, weight=1)

    def start_transcription(self):
        # 保存配置
        config = {
            "obs_host": self.obs_host.get(),
            "obs_port": int(self.obs_port.get()),
            "obs_password": self.obs_password.get(),
            "text_source": self.text_source.get(),
            "server_url": self.server_url.get(),
        }

        # 更新UI
        self.start_button.configure(state="disabled")
        self.status.configure(text="正在启动...")

        # 启动转录客户端
        self.root.after(100, lambda: self.run_client(config))

    def run_client(self, config):
        # 创建一个新线程运行异步代码
        import threading

        client_thread = threading.Thread(
            target=run_transcription_client, args=(config, self.update_status)
        )
        client_thread.daemon = True
        client_thread.start()

    def update_status(self, text):
        self.root.after(0, lambda: self.status.configure(text=text))


# 转录客户端功能实现
def run_transcription_client(config, status_callback):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(transcription_client(config, status_callback))
    except Exception as e:
        status_callback(f"错误: {str(e)}")
    finally:
        loop.close()


async def transcription_client(config, status_callback):
    """与转录服务器交互的客户端"""
    # 连接到OBS WebSocket
    obs_client = obs.obsws(
        config["obs_host"], config["obs_port"], config["obs_password"]
    )
    try:
        obs_client.connect()
        status_callback("已连接到OBS WebSocket")
    except Exception as e:
        status_callback(f"连接OBS失败: {str(e)}")
        return

    # 设置音频捕获
    p = pyaudio.PyAudio()

    # 列出可用的输入设备
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get("deviceCount")
    default_input = p.get_default_input_device_info()["index"]

    status_callback(f"使用默认音频输入设备: 设备 #{default_input}")

    # 音频配置
    sample_rate = 16000
    channels = 1
    chunk_size = 1024
    audio_format = pyaudio.paInt16

    stream = p.open(
        format=audio_format,
        channels=channels,
        rate=sample_rate,
        input=True,
        input_device_index=default_input,
        frames_per_buffer=chunk_size,
    )

    # 连接到转录服务器
    try:
        async with websockets.connect(config["server_url"]) as websocket:
            status_callback("已连接到转录服务器")

            # 发送初始化消息
            await websocket.send(
                json.dumps(
                    {
                        "type": "init",
                        "config": {
                            "language": "zh-CN",
                            "sample_rate": sample_rate,
                            "encoding": "LINEAR16",
                            "channels": channels,
                            "interim_results": True,
                        },
                    }
                )
            )

            # 等待初始化确认
            response = await websocket.recv()
            data = json.loads(response)
            if data.get("type") != "init_ack":
                status_callback(f"初始化失败: {data}")
                return

            status_callback(f"会话已初始化: {data.get('session_id')}")

            # 创建任务处理接收消息
            receive_task = asyncio.create_task(
                handle_messages(websocket, obs_client, config, status_callback)
            )

            # 主循环：捕获并发送音频
            status_callback("开始音频捕获...")
            try:
                while True:
                    audio_chunk = stream.read(chunk_size, exception_on_overflow=False)

                    # 编码并发送音频数据
                    audio_base64 = base64.b64encode(audio_chunk).decode("utf-8")
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "audio",
                                "data": audio_base64,
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                    )

                    # 短暂暂停，避免发送过多数据
                    await asyncio.sleep(0.01)
            except KeyboardInterrupt:
                status_callback("用户中断，正在关闭...")
            except Exception as e:
                status_callback(f"音频捕获错误: {str(e)}")
            finally:
                # 清理资源
                receive_task.cancel()
                stream.stop_stream()
                stream.close()
    except Exception as e:
        status_callback(f"WebSocket连接错误: {str(e)}")
    finally:
        p.terminate()
        obs_client.disconnect()


async def handle_messages(websocket, obs_client, config, status_callback):
    """处理来自转录服务器的消息"""
    try:
        while True:
            message = await websocket.recv()
            data = json.loads(message)

            if data.get("type") == "transcription":
                text = data.get("text", "")
                is_final = data.get("is_final", False)

                status_callback(f"{'最终' if is_final else '中间'} 转录: {text}")

                # 更新OBS文本源
                try:
                    obs_client.call(
                        obs.requests.SetInputSettings(
                            inputName=config["text_source"],
                            inputSettings={"text": text},
                        )
                    )
                except Exception as e:
                    status_callback(f"更新OBS失败: {str(e)}")

            elif data.get("type") == "error":
                status_callback(
                    f"服务器错误: [{data.get('code')}] {data.get('message')}"
                )
    except asyncio.CancelledError:
        # 任务被取消，正常退出
        pass
    except Exception as e:
        status_callback(f"处理消息时出错: {str(e)}")


def main():
    root = tk.Tk()
    gui = ConfigGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
