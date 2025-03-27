# pyinstaller --onefile --windowed obs_transcription_client.py

import asyncio
import websockets
import json
import pyaudio
import obsws_python as obs
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from datetime import datetime
import configparser
import os


# GUI配置界面
class ConfigGUI:
    def __init__(self, root):
        self.root = root
        root.title("OBS转录客户端")
        root.geometry("600x500")  # 增加窗口尺寸以容纳日志区域

        # 设置风格
        style = ttk.Style()
        style.configure("TButton", padding=5, font=("Helvetica", 10))
        style.configure("TLabel", font=("Helvetica", 10))

        # 创建主框架
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ===== 配置区域 =====
        config_frame = ttk.Frame(main_frame)
        config_frame.pack(fill=tk.X, expand=False, pady=(0, 10))

        # 右上角按钮区域
        buttons_frame = ttk.Frame(config_frame)
        buttons_frame.grid(column=0, row=0, sticky=tk.E)

        self.save_button = ttk.Button(
            buttons_frame, text="保存设置", command=self.save_config, width=12
        )
        self.save_button.pack(side=tk.LEFT, padx=(0, 5))

        self.default_button = ttk.Button(
            buttons_frame, text="恢复默认设置", command=self.restore_defaults, width=12
        )
        self.default_button.pack(side=tk.LEFT)

        # OBS设置
        ttk.Label(config_frame, text="OBS设置").grid(
            column=0, row=1, sticky=tk.W, pady=(0, 5)
        )

        obs_frame = ttk.LabelFrame(config_frame, padding="5")
        obs_frame.grid(column=0, row=2, sticky=(tk.W, tk.E), pady=(0, 10))

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
        ttk.Label(config_frame, text="转录服务器设置").grid(
            column=0, row=3, sticky=tk.W, pady=(0, 5)
        )

        server_frame = ttk.LabelFrame(config_frame, padding="5")
        server_frame.grid(column=0, row=4, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(server_frame, text="服务器地址:").grid(column=0, row=0, sticky=tk.W)
        self.server_url = ttk.Entry(server_frame, width=30)
        self.server_url.insert(0, "ws://localhost:8765")
        self.server_url.grid(column=1, row=0, sticky=(tk.W, tk.E), padx=5)

        # 启动按钮和状态
        control_frame = ttk.Frame(config_frame)
        control_frame.grid(column=0, row=5, sticky=(tk.W, tk.E), pady=5)
        control_frame.columnconfigure(0, weight=1)  # 确保框架能够正确扩展

        self.start_button = ttk.Button(
            control_frame, text="启动转录", command=self.start_transcription, width=12
        )
        self.start_button.pack(side=tk.RIGHT)

        self.status = ttk.Label(control_frame, text="就绪")
        self.status.pack(side=tk.LEFT)

        # 设置网格权重
        config_frame.columnconfigure(0, weight=1)
        obs_frame.columnconfigure(3, weight=1)
        server_frame.columnconfigure(1, weight=1)

        # ===== 日志区域 =====
        log_frame = ttk.LabelFrame(main_frame, text="运行日志", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)

        # 创建滚动文本区域用于日志显示
        self.log_area = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, height=15, font=("Consolas", 9)
        )
        self.log_area.pack(fill=tk.BOTH, expand=True)
        self.log_area.config(state=tk.DISABLED)  # 设为只读

        # 添加初始日志
        self.add_log("OBS转录客户端已启动")
        self.add_log("请配置连接参数后点击「启动转录」")

        # 尝试加载配置文件
        self.load_config()

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
        self.update_status("正在启动...", is_log=True)

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

    def update_status(self, text, is_log=True):
        """更新状态标签和日志区域"""
        self.root.after(0, lambda: self._update_ui(text, is_log))

    def _update_ui(self, text, is_log):
        # 更新状态标签
        if text == "ready":
            # 特殊情况：重置启动按钮
            self.start_button.configure(state="normal")
            self.status.configure(text="就绪")
        else:
            self.status.configure(text=text)

        # 如果需要，也添加到日志
        if is_log:
            self.add_log(text)

    def add_log(self, message):
        """添加消息到日志区域"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        self.log_area.config(state=tk.NORMAL)
        self.log_area.insert(tk.END, log_entry)
        self.log_area.see(tk.END)  # 自动滚动到底部
        self.log_area.config(state=tk.DISABLED)

    def save_config(self):
        """保存当前配置到config.ini文件"""
        config = configparser.ConfigParser()

        # OBS设置
        config["OBS"] = {
            "host": self.obs_host.get(),
            "port": self.obs_port.get(),
            "password": self.obs_password.get(),
            "text_source": self.text_source.get(),
        }

        # 服务器设置
        config["SERVER"] = {"url": self.server_url.get()}

        # 写入文件
        try:
            with open("config.ini", "w") as configfile:
                config.write(configfile)
            self.add_log("配置已保存到 config.ini")
            messagebox.showinfo("保存成功", "配置已成功保存到 config.ini")
        except Exception as e:
            self.add_log(f"保存配置失败: {str(e)}")
            messagebox.showerror("保存失败", f"无法保存配置: {str(e)}")

    def restore_defaults(self):
        """恢复默认设置"""
        if messagebox.askyesno("恢复默认", "确定要恢复所有设置到默认值吗？"):
            # OBS默认设置
            self.obs_host.delete(0, tk.END)
            self.obs_host.insert(0, "localhost")

            self.obs_port.delete(0, tk.END)
            self.obs_port.insert(0, "4455")

            self.obs_password.delete(0, tk.END)

            self.text_source.delete(0, tk.END)
            self.text_source.insert(0, "转录文本")

            # 服务器默认设置
            self.server_url.delete(0, tk.END)
            self.server_url.insert(0, "ws://localhost:8765")

            self.add_log("已恢复默认设置")

    def load_config(self):
        """从config.ini加载配置（如果存在）"""
        if not os.path.exists("config.ini"):
            self.add_log("未找到配置文件，使用默认配置")
            return

        config = configparser.ConfigParser()
        try:
            config.read("config.ini")

            # 加载OBS设置
            if "OBS" in config:
                if "host" in config["OBS"]:
                    self.obs_host.delete(0, tk.END)
                    self.obs_host.insert(0, config["OBS"]["host"])

                if "port" in config["OBS"]:
                    self.obs_port.delete(0, tk.END)
                    self.obs_port.insert(0, config["OBS"]["port"])

                if "password" in config["OBS"]:
                    self.obs_password.delete(0, tk.END)
                    self.obs_password.insert(0, config["OBS"]["password"])

                if "text_source" in config["OBS"]:
                    self.text_source.delete(0, tk.END)
                    self.text_source.insert(0, config["OBS"]["text_source"])

            # 加载服务器设置
            if "SERVER" in config and "url" in config["SERVER"]:
                self.server_url.delete(0, tk.END)
                self.server_url.insert(0, config["SERVER"]["url"])

            self.add_log("已加载配置文件")
        except Exception as e:
            self.add_log(f"加载配置失败: {str(e)}")


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
    # 连接到OBS WebSocket，使用原始导入方式
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

    try:
        # 列出可用的输入设备
        info = p.get_host_api_info_by_index(0)
        num_devices = info.get("deviceCount")
        default_input = p.get_default_input_device_info()["index"]

        status_callback(f"使用默认音频输入设备: 设备 #{default_input}")

        # 音频配置
        sample_rate = 16000
        channels = 1
        chunk_size = 1600  # 调整为更合适的大小(50ms at 16kHz)
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
            status_callback(f"正在连接到转录服务器: {config['server_url']}")
            async with websockets.connect(config["server_url"]) as websocket:
                status_callback("已连接到转录服务器")

                # 发送初始化消息
                init_message = {
                    "type": "init",
                    "config": {
                        "language": "zh",
                        "sample_rate": sample_rate,
                        "encoding": "LINEAR16",
                        "channels": channels,
                    },
                }
                status_callback(f"发送初始化消息: {json.dumps(init_message)}")

                await websocket.send(json.dumps(init_message))

                # 等待初始化确认
                response = await websocket.recv()
                data = json.loads(response)
                if data.get("type") != "init_ack":
                    status_callback(f"初始化失败: {data}")
                    return

                session_id = data.get("session_id", "unknown")
                status_callback(f"会话已初始化: {session_id}")

                # 创建任务处理接收消息
                receive_task = asyncio.create_task(
                    handle_messages(websocket, obs_client, config, status_callback)
                )

                # 主循环：捕获并发送音频
                status_callback("开始音频捕获...")

                # 跟踪发送的数据量
                total_bytes_sent = 0
                chunks_sent = 0
                start_time = time.time()

                try:
                    while True:
                        audio_chunk = stream.read(
                            chunk_size, exception_on_overflow=False
                        )

                        # 直接发送音频数据作为二进制帧
                        await websocket.send(audio_chunk)

                        # 更新统计信息
                        chunks_sent += 1
                        total_bytes_sent += len(audio_chunk)

                        # 每秒显示一次传输统计
                        elapsed = time.time() - start_time
                        if elapsed >= 5.0:  # 每5秒显示一次统计
                            kbps = (total_bytes_sent * 8 / 1000) / elapsed
                            status_callback(
                                f"音频传输: {chunks_sent}个块, {total_bytes_sent/1024:.1f}KB, {kbps:.1f}Kbps",
                                is_log=True,
                            )
                            # 重置计数器
                            start_time = time.time()
                            chunks_sent = 0
                            total_bytes_sent = 0

                        # 短暂暂停，避免发送过多数据
                        await asyncio.sleep(0.01)
                except KeyboardInterrupt:
                    status_callback("用户中断，正在关闭...")
                except Exception as e:
                    status_callback(f"音频捕获错误: {str(e)}")
                finally:
                    # 清理资源
                    receive_task.cancel()

                    # 发送结束会话请求
                    try:
                        end_message = {"type": "end"}
                        status_callback(f"发送结束会话请求: {json.dumps(end_message)}")
                        await websocket.send(json.dumps(end_message))

                        # 等待结束确认
                        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        status_callback(f"收到会话结束确认: {response}")
                    except Exception as e:
                        status_callback(f"结束会话错误: {str(e)}")
                    finally:
                        stream.stop_stream()
                        stream.close()
        except Exception as e:
            status_callback(f"WebSocket连接错误: {str(e)}")
    except Exception as e:
        status_callback(f"音频设备初始化错误: {str(e)}")
    finally:
        p.terminate()
        try:
            obs_client.disconnect()
            status_callback("已断开OBS连接")
        except:
            pass

        status_callback("转录服务已停止，可以重新启动")
        # 在UI线程中重新启用启动按钮
        asyncio.get_event_loop().call_soon_threadsafe(
            lambda: status_callback("ready", is_log=False)
        )


async def handle_messages(websocket, obs_client, config, status_callback):
    """处理来自转录服务器的消息"""
    try:
        while True:
            message = await websocket.recv()
            data = json.loads(message)

            if data.get("type") == "transcription":
                text = data.get("text", "")
                is_final = data.get("is_final", False)
                language = data.get("language", "")

                # 简化的日志输出，避免过多的中间结果
                if is_final or len(text) < 50:
                    status_callback(
                        f"{'最终' if is_final else '中间'} 转录 [{language}]: {text}"
                    )

                # 更新OBS文本源
                try:
                    obs_client.call(
                        obs.requests.SetInputSettings(  # 使用原始的引用方式
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
            elif data.get("type") == "end_ack":
                status_callback(f"会话已结束: {data.get('session_id')}")
            else:
                status_callback(f"收到未知消息类型: {data.get('type')}")
    except asyncio.CancelledError:
        # 任务被取消，正常退出
        status_callback("消息接收任务已取消")
    except Exception as e:
        status_callback(f"处理消息时出错: {str(e)}")


def main():
    root = tk.Tk()
    gui = ConfigGUI(root)

    # 添加窗口关闭事件处理
    def on_closing():
        gui.add_log("正在关闭应用程序...")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
