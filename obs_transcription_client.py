# pyinstaller --onefile --windowed .\main.py

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
import threading


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

        # 修改：将文本源从Entry改为Combobox
        self.text_source = ttk.Combobox(obs_frame, width=20, state="disabled")
        self.text_source.set("转录文本")  # 使用set而不是insert
        self.text_source.grid(
            column=1, row=2, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=(5, 0)
        )

        # 添加刷新按钮
        self.refresh_button = ttk.Button(
            obs_frame, text="刷新", command=self.update_text_sources, width=6
        )
        self.refresh_button.grid(column=3, row=2, sticky=tk.E, padx=5, pady=(5, 0))
        self.refresh_button.configure(state="disabled")  # 初始状态为禁用

        # 添加OBS连接按钮
        self.connect_obs_button = ttk.Button(
            obs_frame, text="连接OBS", command=self.toggle_obs_connection, width=12
        )
        self.connect_obs_button.grid(
            column=0, row=3, columnspan=4, sticky=tk.E, pady=(5, 0)
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

        # 添加服务器连接按钮
        self.connect_server_button = ttk.Button(
            server_frame,
            text="连接转录服务器",
            command=self.toggle_server_connection,
            width=16,
        )
        self.connect_server_button.grid(
            column=0, row=1, columnspan=2, sticky=tk.E, pady=(5, 0)
        )

        # 启动按钮和状态
        control_frame = ttk.Frame(config_frame)
        control_frame.grid(column=0, row=5, sticky=(tk.W, tk.E), pady=5)
        control_frame.columnconfigure(0, weight=1)  # 确保框架能够正确扩展

        self.start_button = ttk.Button(
            control_frame, text="启动转录", command=self.toggle_transcription, width=12
        )
        self.start_button.pack(side=tk.RIGHT)
        # 初始禁用转录按钮，直到OBS和服务器都已连接
        self.start_button.configure(state=tk.DISABLED)

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

        # ===== 状态栏区域 =====
        self.stats_frame = ttk.Frame(main_frame, padding="5")
        self.stats_frame.pack(fill=tk.X, expand=False, pady=(5, 0))

        # 左侧显示普通状态
        self.status = ttk.Label(self.stats_frame, text="就绪")
        self.status.pack(side=tk.LEFT)

        # 右侧显示传输统计
        self.stats_label = ttk.Label(self.stats_frame, text="")
        self.stats_label.pack(side=tk.RIGHT)

        # 添加初始日志
        self.add_log("OBS转录客户端已启动")
        self.add_log("请配置连接参数后分别连接OBS和转录服务器，然后启动转录")

        # 尝试加载配置文件
        self.load_config()

        # 连接状态和转录控制变量
        self.obs_connected = False
        self.server_connected = False
        self.server_url_valid = False
        self.obs_client = None
        self.transcription_active = False
        self.stop_event = threading.Event()
        self.client_thread = None
        self.audio_stream = None
        self.pyaudio_instance = None

        # 服务器WebSocket相关变量
        self.server_websocket = None
        self.server_websocket_loop = None
        self.server_thread = None

        # 新增：转录任务队列
        self.transcription_queue = asyncio.Queue(maxsize=1)
        self.transcription_task = None

    def update_stats(self, stats_text):
        """更新UI中的传输统计信息"""
        self.root.after(0, lambda: self.stats_label.configure(text=stats_text))

    def update_text_sources(self):
        """从OBS获取所有文本GDI+源并更新下拉框"""
        if not self.obs_client:
            return

        try:
            # 获取所有输入源
            list = self.obs_client.get_input_list()

            # 筛选文本GDI+源
            text_sources = []
            for input_item in list.inputs:
                input_kind = input_item.get("inputKind", "")
                input_name = input_item.get("inputName", "")

                # 检查是否为文本GDI+源 (text_gdiplus_v3)
                if input_kind == "text_gdiplus_v3":
                    text_sources.append(input_name)

            # 更新下拉框
            self.text_source["values"] = text_sources

            # 如果列表中有项目且当前未选择，则选择第一项
            if text_sources:
                current_value = self.text_source.get()
                if not current_value or current_value not in text_sources:
                    self.text_source.set(text_sources[0])
                self.add_log(
                    f"已更新文本源列表，共找到 {len(text_sources)} 个文本GDI+源"
                )
            else:
                self.add_log("未在OBS中找到任何文本GDI+源，请在OBS中添加一个文本源")
                messagebox.showinfo(
                    "未找到文本源",
                    "未在OBS中找到任何文本GDI+源，请在OBS中添加一个文本源后再试。",
                )

        except Exception as e:
            self.add_log(f"获取文本源列表时出错: {str(e)}")

    def check_transcription_button_state(self):
        """根据OBS和服务器的连接状态更新转录按钮状态"""
        if self.obs_connected and self.server_connected:
            self.start_button.configure(state=tk.NORMAL)
        else:
            self.start_button.configure(state=tk.DISABLED)
            # 如果正在转录但某个连接断开，停止转录
            if self.transcription_active:
                self.stop_transcription()
                self.add_log("连接断开，已停止转录")

    def toggle_obs_connection(self):
        """切换OBS连接状态"""
        if not self.obs_connected:
            self.connect_to_obs()
        else:
            self.disconnect_from_obs()

    def connect_to_obs(self):
        """连接到OBS WebSocket"""
        host = self.obs_host.get()
        try:
            port = int(self.obs_port.get())
        except ValueError:
            messagebox.showerror("错误", "OBS端口必须是数字")
            return

        password = self.obs_password.get()

        # 更新状态
        self.update_status("正在连接OBS...", is_log=True)
        self.connect_obs_button.configure(text="连接中...", state=tk.DISABLED)

        # 在后台线程中连接OBS，避免UI卡顿
        def connect_obs_thread():
            try:
                self.obs_client = obs.ReqClient(
                    host=host,
                    port=port,
                    password=password,
                )

                # 连接成功，不再需要验证特定文本源
                self.root.after(0, self._obs_connected_success)

            except Exception as e:
                self.root.after(0, lambda: self._obs_connected_failure(str(e)))

        threading.Thread(target=connect_obs_thread, daemon=True).start()

    def _obs_connected_success(self):
        """OBS连接成功后的UI更新"""
        self.obs_connected = True
        self.connect_obs_button.configure(text="断开OBS", state=tk.NORMAL)
        self.update_status("OBS已连接", is_log=True)

        # 启用文本源下拉框和刷新按钮
        self.text_source.configure(state="readonly")  # readonly允许选择但不允许直接编辑
        self.refresh_button.configure(state=tk.NORMAL)

        # 获取并更新文本源列表
        self.update_text_sources()

        self.check_transcription_button_state()

    def _obs_connected_failure(self, error_msg):
        """OBS连接失败后的UI更新"""
        self.obs_connected = False
        self.connect_obs_button.configure(text="连接OBS", state=tk.NORMAL)
        self.update_status(f"OBS连接失败: {error_msg}", is_log=True)
        self.check_transcription_button_state()

    def disconnect_from_obs(self):
        """断开OBS连接"""
        if self.obs_client:
            try:
                # 添加: 主动断开OBS WebSocket连接
                # 根据obsws_python库的实现，ReqClient内部维护了一个ws_client对象
                # 需要调用ws_client的disconnect方法
                if hasattr(self.obs_client, "ws_client") and self.obs_client.ws_client:
                    self.obs_client.ws_client.disconnect()
                elif hasattr(self.obs_client, "disconnect"):
                    self.obs_client.disconnect()

                self.update_status("已断开OBS WebSocket连接", is_log=True)
            except Exception as e:
                self.update_status(f"断开OBS连接时发生错误: {str(e)}", is_log=True)

            self.obs_client = None
            self.obs_connected = False
            self.connect_obs_button.configure(text="连接OBS", state=tk.NORMAL)

            # 禁用文本源下拉框和刷新按钮
            self.text_source.configure(state="disabled")
            self.refresh_button.configure(state="disabled")

            self.update_status("已断开OBS连接", is_log=True)
            self.check_transcription_button_state()

    def toggle_server_connection(self):
        """切换服务器连接状态"""
        if not self.server_connected:
            self.connect_to_server()
        else:
            self.disconnect_from_server()

    def connect_to_server(self):
        """连接到转录服务器"""
        server_url = self.server_url.get()
        self.update_status(f"正在连接转录服务器: {server_url}", is_log=True)
        self.connect_server_button.configure(text="连接中...", state=tk.DISABLED)

        # 重新创建转录任务队列，确保队列是干净的
        self.transcription_queue = asyncio.Queue(maxsize=1)

        # 在后台线程中连接服务器
        def connect_server_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # 创建WebSocket连接
                websocket = loop.run_until_complete(
                    websockets.connect(server_url, ping_timeout=5)
                )

                # 保存websocket连接和loop以便后续使用
                self.server_websocket = websocket
                self.server_websocket_loop = loop

                # 添加转录任务处理器
                self.transcription_task = loop.create_task(
                    self._process_transcription_tasks()
                )

                # 通知UI线程连接成功
                self.root.after(0, self._server_connected_success)

                # 保持event loop运行以处理websocket消息和转录任务
                loop.run_forever()

            except Exception as e:
                self.root.after(0, lambda: self._server_connected_failure(str(e)))
                if loop.is_running():
                    loop.stop()
                loop.close()

        self.server_thread = threading.Thread(target=connect_server_thread, daemon=True)
        self.server_thread.start()

    async def _process_transcription_tasks(self):
        """处理转录任务队列中的任务"""
        while True:
            # 等待转录任务
            task_args = await self.transcription_queue.get()

            try:
                # 调用转录函数处理任务
                await transcription_client(*task_args)
            except Exception as e:
                self.update_status(f"转录任务处理错误: {str(e)}", is_log=True)
            finally:
                # 标记任务完成
                self.transcription_queue.task_done()

    def _server_connected_success(self):
        """服务器验证成功后的UI更新"""
        self.server_connected = True
        self.connect_server_button.configure(text="断开服务器", state=tk.NORMAL)
        self.update_status("转录服务器已连接", is_log=True)
        self.check_transcription_button_state()

    def _server_connected_failure(self, error_msg):
        """服务器验证失败后的UI更新"""
        self.server_connected = False
        self.connect_server_button.configure(text="连接转录服务器", state=tk.NORMAL)
        self.update_status(f"服务器连接失败: {error_msg}", is_log=True)
        self.check_transcription_button_state()

    def disconnect_from_server(self):
        """断开服务器连接状态"""
        # 先检查是否有正在进行的转录，如果有则先停止
        if self.transcription_active:
            self.stop_event.set()  # 立即设置停止信号
            self.add_log("检测到活动的转录会话，正在停止...")
            # 直接重置转录状态，因为WebSocket即将关闭
            self.transcription_active = False
            self.start_button.configure(text="启动转录", state=tk.DISABLED)
            # 清理音频资源
            self.cleanup_audio()
            # 清除传输统计
            self.update_stats("")

        if self.transcription_task:
            # 取消转录任务处理器
            if self.server_websocket_loop and self.server_websocket_loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._cancel_transcription_task(), self.server_websocket_loop
                )
            self.transcription_task = None

        if self.server_websocket:
            try:
                # 创建一个新的event loop来关闭websocket连接
                async def close_websocket():
                    await self.server_websocket.close()

                if (
                    self.server_websocket_loop
                    and self.server_websocket_loop.is_running()
                ):
                    # 在当前loop中安排关闭任务
                    asyncio.run_coroutine_threadsafe(
                        close_websocket(), self.server_websocket_loop
                    )
                    # 停止事件循环
                    self.server_websocket_loop.stop()
                else:
                    # 如果loop不在运行，创建一个新的loop来关闭
                    temp_loop = asyncio.new_event_loop()
                    temp_loop.run_until_complete(close_websocket())
                    temp_loop.close()

                self.update_status("已关闭WebSocket连接", is_log=True)
            except Exception as e:
                self.update_status(f"关闭WebSocket连接时出错: {str(e)}", is_log=True)

            self.server_websocket = None
            self.server_websocket_loop = None
            # 确保清空任务队列的引用
            self.transcription_queue = None

        self.server_connected = False
        self.connect_server_button.configure(text="连接转录服务器", state=tk.NORMAL)
        self.update_status("已断开服务器连接", is_log=True)
        self.check_transcription_button_state()

    async def _cancel_transcription_task(self):
        """取消转录任务处理器"""
        if self.transcription_task and not self.transcription_task.done():
            self.transcription_task.cancel()
            try:
                await self.transcription_task
            except asyncio.CancelledError:
                pass

    def toggle_transcription(self):
        """切换转录状态：启动或停止"""
        if not self.transcription_active:
            # 当前未转录，启动转录
            self.start_transcription()
        else:
            # 当前正在转录，停止转录
            self.stop_transcription()

    def stop_transcription(self):
        """停止转录过程"""
        if self.transcription_active:
            self.update_status("正在停止转录...", is_log=True)
            # 设置停止信号
            self.stop_event.set()
            # 按钮文字暂时变为"正在停止..."
            self.start_button.configure(text="正在停止...", state="disabled")

            # 停止事件将被转录任务检测到

    def start_transcription(self):
        """启动转录过程"""
        # 确保已经连接到OBS和验证了服务器
        if not self.obs_connected or not self.server_connected:
            self.add_log("错误：请先连接OBS和转录服务器")
            return

        # 配置信息
        config = {
            "text_source": self.text_source.get(),
            "server_url": self.server_url.get(),
        }

        # 重置停止事件
        self.stop_event.clear()

        # 更新UI
        self.start_button.configure(text="停止转录")
        self.update_status("正在启动转录...", is_log=True)

        # 设置转录状态为活跃
        self.transcription_active = True

        # 初始化音频采集
        try:
            self.init_audio()

            # 将转录任务添加到队列
            if self.server_websocket_loop and self.server_websocket:
                asyncio.run_coroutine_threadsafe(
                    self.transcription_queue.put(
                        (
                            config,
                            self,
                            self.stop_event,
                            self.obs_client,
                            self.audio_stream,
                            self.server_websocket,
                        )
                    ),
                    self.server_websocket_loop,
                )
                self.add_log("已将转录任务添加到队列")
            else:
                self.update_status("错误：服务器连接无效", is_log=True)
                self.transcription_active = False
                self.start_button.configure(text="启动转录")
        except Exception as e:
            self.update_status(f"启动转录失败: {str(e)}", is_log=True)
            self.transcription_active = False
            self.start_button.configure(text="启动转录")

    def init_audio(self):
        """初始化音频采集"""
        try:
            p = pyaudio.PyAudio()
            self.pyaudio_instance = p

            # 获取默认输入设备
            default_input = p.get_default_input_device_info()["index"]
            self.add_log(f"使用默认音频输入设备: 设备 #{default_input}")

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
            return True
        except Exception as e:
            self.add_log(f"音频设备初始化错误: {str(e)}")
            raise e

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

    def update_status(self, text, is_log=True):
        """更新状态标签和日志区域"""
        self.root.after(0, lambda: self._update_ui(text, is_log))

    def _update_ui(self, text, is_log):
        # 更新状态标签
        if text == "ready":
            # 特殊情况：重置转录状态和按钮
            self.transcription_active = False
            self.start_button.configure(text="启动转录", state="normal")
            self.status.configure(text="就绪")
            # 清理音频资源
            self.cleanup_audio()
            # 清除传输统计
            self.update_stats("")
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

            # 修改：对于Combobox使用set
            self.text_source.set("转录文本")

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
                    # 修改：对于Combobox使用set而不是insert
                    self.text_source.set(config["OBS"]["text_source"])

            # 加载服务器设置
            if "SERVER" in config and "url" in config["SERVER"]:
                self.server_url.delete(0, tk.END)
                self.server_url.insert(0, config["SERVER"]["url"])

            self.add_log("已加载配置文件")
        except Exception as e:
            self.add_log(f"加载配置失败: {str(e)}")


# 转录客户端功能实现
async def transcription_client(
    config,
    status_callback,
    stop_event,
    obs_client,
    audio_stream,
    websocket,
):
    """与转录服务器交互的客户端"""
    try:
        status_callback.update_status("开始转录流程")

        # 检查WebSocket是否有效
        try:
            pong = await asyncio.wait_for(websocket.ping(), timeout=1)
            await pong  # 等待pong响应
            status_callback.update_status("使用现有转录服务器连接")
        except Exception as e:
            status_callback.update_status(f"WebSocket连接失效: {str(e)}")
            return

        # 音频配置
        sample_rate = 16000
        channels = 1
        chunk_size = 1600  # 50ms at 16kHz

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
        status_callback.update_status(f"发送初始化消息: {json.dumps(init_message)}")

        await websocket.send(json.dumps(init_message))

        # 等待初始化确认
        response = await websocket.recv()
        data = json.loads(response)
        if data.get("type") != "init_ack":
            status_callback.update_status(f"初始化失败: {data}")
            return

        session_id = data.get("session_id", "unknown")
        status_callback.update_status(f"会话已初始化: {session_id}")

        # 创建任务处理接收消息
        receive_task = asyncio.create_task(
            handle_messages(websocket, obs_client, config, status_callback)
        )

        # 主循环：捕获并发送音频
        status_callback.update_status("开始音频捕获...")

        # 跟踪发送的数据量
        total_bytes_sent = 0
        chunks_sent = 0
        start_time = time.time()

        try:
            while not stop_event.is_set():  # 检查停止信号
                audio_chunk = audio_stream.read(chunk_size, exception_on_overflow=False)

                # 直接发送音频数据作为二进制帧
                await websocket.send(audio_chunk)

                # 更新统计信息
                chunks_sent += 1
                total_bytes_sent += len(audio_chunk)

                # 更频繁地更新UI（每秒一次）但仍然每5秒重置计数器
                elapsed = time.time() - start_time
                if elapsed >= 1.0:
                    # 计算统计数据
                    kbps = (total_bytes_sent * 8 / 1000) / elapsed
                    stats_text = (
                        f"传输: {total_bytes_sent / 1024:.1f}KB, {kbps:.1f}Kbps"
                    )

                    # 更新UI状态栏而不是日志
                    status_callback.update_stats(stats_text)

                    # 只有当累计了5秒或更长时间才重置计数器
                    if elapsed >= 5.0:
                        start_time = time.time()
                        chunks_sent = 0
                        total_bytes_sent = 0

                # 短暂暂停，避免发送过多数据
                await asyncio.sleep(0.01)

            status_callback.update_status("检测到停止信号，正在关闭转录...")
        except KeyboardInterrupt:
            status_callback.update_status("用户中断，正在关闭...")
        except Exception as e:
            status_callback.update_status(f"音频捕获错误: {str(e)}")
        finally:
            # 清理资源
            receive_task.cancel()

            # 发送结束会话请求
            try:
                # 使用更可靠的方式检查WebSocket连接状态
                # 尝试发送结束会话请求，如果失败则捕获异常
                try:
                    end_message = {"type": "end"}
                    status_callback.update_status(
                        f"发送结束会话请求: {json.dumps(end_message)}"
                    )

                    # 尝试发送结束消息
                    await websocket.send(json.dumps(end_message))

                    # 等待结束确认，但使用try-except捕获可能的超时
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        status_callback.update_status(f"收到会话结束确认: {response}")
                    except asyncio.TimeoutError:
                        # 超时但不关闭连接
                        status_callback.update_status(
                            "等待会话结束确认超时，但保持连接"
                        )
                except (
                    websockets.exceptions.ConnectionClosed,
                    ConnectionResetError,
                    RuntimeError,  # 捕获更多可能的异常
                ) as e:
                    status_callback.update_status(
                        f"发送结束请求时出错: {str(e)}，但保持连接状态"
                    )
            except Exception as e:
                # 捕获所有其他异常，但不关闭连接
                status_callback.update_status(f"结束会话处理错误(保持连接): {str(e)}")

    except Exception as e:
        status_callback.update_status(f"转录客户端错误: {str(e)}")
    finally:
        # 转录服务已停止，可以重新启动
        status_callback.update_status("转录服务已停止，可以重新启动")
        # 在UI线程中重新启用启动按钮
        status_callback.update_status("ready", is_log=False)


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
                    status_callback.update_status(
                        f"{'最终' if is_final else '中间'} 转录 [{language}]: {text}"
                    )

                # 更新OBS文本源
                try:
                    obs_client.set_input_settings(
                        config["text_source"], {"text": text}, True
                    )
                except Exception as e:
                    status_callback.update_status(f"更新OBS失败: {str(e)}")

            elif data.get("type") == "error":
                status_callback.update_status(
                    f"服务器错误: [{data.get('code')}] {data.get('message')}"
                )
            elif data.get("type") == "end_ack":
                status_callback.update_status(f"会话已结束: {data.get('session_id')}")
            else:
                status_callback.update_status(f"收到未知消息类型: {data.get('type')}")
    except asyncio.CancelledError:
        # 任务被取消，正常退出
        status_callback.update_status("消息接收任务已取消")
    except Exception as e:
        status_callback.update_status(f"处理消息时出错: {str(e)}")


def main():
    root = tk.Tk()
    gui = ConfigGUI(root)

    # 添加窗口关闭事件处理
    def on_closing():
        # 如果连接了OBS或转录服务器，先断开
        if gui.obs_connected:
            gui.disconnect_from_obs()
        if gui.server_connected:
            gui.disconnect_from_server()

        # 如果转录正在运行，先停止它
        if gui.transcription_active:
            gui.stop_transcription()

        gui.add_log("正在关闭应用程序...")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
