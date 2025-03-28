# GUI配置界面
import asyncio
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

from core.audio_manager import AudioManager
from core.obs_manager import OBSManager
from core.server_connection import ServerConnection
from core.transcription_client import TranscriptionClient
from utils.config_manager import ConfigManager
from utils.logger import Logger


class ConfigGUI:
    def __init__(self, root):
        self.root = root
        root.title("OBS转录客户端")
        root.geometry("600x650")  # 增加窗口尺寸以容纳日志区域

        # 初始化管理器
        self.obs_manager = OBSManager()
        self.audio_manager = AudioManager()
        self.server_connection = ServerConnection()
        self.config_manager = ConfigManager()

        # 设置风格
        style = ttk.Style()
        style.configure("TButton", padding=5, font=("Helvetica", 10))
        style.configure("TLabel", font=("Helvetica", 10))

        # 创建主框架
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ===== 配置区域 =====
        config_frame = ttk.Frame(main_frame)
        config_frame.pack(fill=tk.X, expand=False, side=tk.TOP, pady=(0, 10))

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

        # 文本源下拉框
        self.text_source = ttk.Combobox(obs_frame, width=20, state="disabled")
        self.text_source.set("转录文本")
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

        # ===== 状态栏区域 =====
        self.stats_frame = ttk.Frame(main_frame, padding="5")
        self.stats_frame.pack(fill=tk.X, side=tk.BOTTOM)

        # 左侧显示普通状态
        self.status = ttk.Label(self.stats_frame, text="就绪")
        self.status.pack(side=tk.LEFT)

        # 右侧显示传输统计
        self.stats_label = ttk.Label(self.stats_frame, text="")
        self.stats_label.pack(side=tk.RIGHT)

        # ===== 日志区域 =====
        log_frame = ttk.LabelFrame(main_frame, text="运行日志", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)

        # 创建滚动文本区域用于日志显示
        self.log_area = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, height=15, font=("Consolas", 9)
        )
        self.log_area.pack(fill=tk.BOTH, expand=True)
        self.log_area.config(state=tk.DISABLED)  # 设为只读

        # 初始化日志管理器
        self.logger = Logger(self.log_area)

        # 添加初始日志
        self.logger.add_log("OBS转录客户端已启动")
        self.logger.add_log("请配置连接参数后分别连接OBS和转录服务器，然后启动转录")

        # 尝试加载配置文件
        self.load_config()

        # 转录控制变量
        self.transcription_active = False
        self.stop_event = threading.Event()

    def update_stats(self, stats_text):
        """更新UI中的传输统计信息"""
        self.root.after(0, lambda: self.stats_label.configure(text=stats_text))

    def update_text_sources(self):
        """从OBS获取所有文本GDI+源并更新下拉框"""
        text_sources, error = self.obs_manager.get_text_sources()

        if error:
            self.logger.add_log(f"获取文本源列表时出错: {error}")
            return

        # 更新下拉框
        self.text_source["values"] = text_sources

        # 如果列表中有项目且当前未选择，则选择第一项
        if text_sources:
            current_value = self.text_source.get()
            if not current_value or current_value not in text_sources:
                self.text_source.set(text_sources[0])
            self.logger.add_log(
                f"已更新文本源列表，共找到 {len(text_sources)} 个文本GDI+源"
            )
        else:
            self.logger.add_log("未在OBS中找到任何文本GDI+源，请在OBS中添加一个文本源")
            messagebox.showinfo(
                "未找到文本源",
                "未在OBS中找到任何文本GDI+源，请在OBS中添加一个文本源后再试。",
            )

    def check_transcription_button_state(self):
        """根据OBS和服务器的连接状态更新转录按钮状态"""
        if self.obs_manager.obs_connected and self.server_connection.server_connected:
            self.start_button.configure(state=tk.NORMAL)
        else:
            self.start_button.configure(state=tk.DISABLED)
            # 如果正在转录但某个连接断开，停止转录
            if self.transcription_active:
                self.stop_transcription()
                self.logger.add_log("连接断开，已停止转录")

    def toggle_obs_connection(self):
        """切换OBS连接状态"""
        if not self.obs_manager.obs_connected:
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

        # 连接成功后的回调
        def success_callback():
            self.root.after(0, self._obs_connected_success)

        # 连接失败后的回调
        def failure_callback(error_msg):
            self.root.after(0, lambda: self._obs_connected_failure(error_msg))

        # 开始连接
        self.obs_manager.connect_to_obs(
            host, port, password, success_callback, failure_callback
        )

    def _obs_connected_success(self):
        """OBS连接成功后的UI更新"""
        self.obs_manager.obs_connected = True
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
        self.obs_manager.obs_connected = False
        self.connect_obs_button.configure(text="连接OBS", state=tk.NORMAL)
        self.update_status(f"OBS连接失败: {error_msg}", is_log=True)
        self.check_transcription_button_state()

    def disconnect_from_obs(self):
        """断开OBS连接"""
        success, error = self.obs_manager.disconnect_from_obs()

        if not success:
            self.update_status(f"断开OBS连接时发生错误: {error}", is_log=True)

        # 更新UI
        self.connect_obs_button.configure(text="连接OBS", state=tk.NORMAL)

        # 禁用文本源下拉框和刷新按钮
        self.text_source.configure(state="disabled")
        self.refresh_button.configure(state="disabled")

        self.update_status("已断开OBS连接", is_log=True)
        self.check_transcription_button_state()

    def toggle_server_connection(self):
        """切换服务器连接状态"""
        if not self.server_connection.server_connected:
            self.connect_to_server()
        else:
            self.disconnect_from_server()

    def connect_to_server(self):
        """连接到转录服务器"""
        server_url = self.server_url.get()
        self.update_status(f"正在连接转录服务器: {server_url}", is_log=True)
        self.connect_server_button.configure(text="连接中...", state=tk.DISABLED)

        # 连接成功后的回调
        def success_callback():
            self.root.after(0, self._server_connected_success)

        # 连接失败后的回调
        def failure_callback(error_msg):
            self.root.after(0, lambda: self._server_connected_failure(error_msg))

        # 开始连接
        self.server_connection.connect_to_server(
            server_url, success_callback, failure_callback
        )

    def _server_connected_success(self):
        """服务器验证成功后的UI更新"""
        self.server_connection.server_connected = True
        self.connect_server_button.configure(text="断开服务器", state=tk.NORMAL)
        self.update_status("转录服务器已连接", is_log=True)
        self.check_transcription_button_state()

    def _server_connected_failure(self, error_msg):
        """服务器验证失败后的UI更新"""
        self.server_connection.server_connected = False
        self.connect_server_button.configure(text="连接转录服务器", state=tk.NORMAL)
        self.update_status(f"服务器连接失败: {error_msg}", is_log=True)
        self.check_transcription_button_state()

    def disconnect_from_server(self):
        """断开服务器连接状态"""
        # 先检查是否有正在进行的转录，如果有则先停止
        if self.transcription_active:
            self.stop_event.set()  # 立即设置停止信号
            self.logger.add_log("检测到活动的转录会话，正在停止...")
            # 直接重置转录状态，因为WebSocket即将关闭
            self.transcription_active = False
            self.start_button.configure(text="启动转录", state=tk.DISABLED)
            # 清理音频资源
            self.audio_manager.cleanup_audio()
            # 清除传输统计
            self.update_stats("")

        # 检查是否连接了转录服务器
        if not self.server_connection.server_connected:
            self.update_status("服务器未连接，无需断开", is_log=True)
            return

        # 断开服务器连接
        success, error = self.server_connection.disconnect_from_server()
        if not success:
            self.update_status(f"关闭WebSocket连接时出错: {error}", is_log=True)

        self.server_connection.server_connected = False
        self.connect_server_button.configure(text="连接转录服务器", state=tk.NORMAL)
        self.update_status("已断开服务器连接", is_log=True)
        self.check_transcription_button_state()

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
        if (
            not self.obs_manager.obs_connected
            or not self.server_connection.server_connected
        ):
            self.logger.add_log("错误：请先连接OBS和转录服务器")
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
            success, stream = self.audio_manager.init_audio(self.logger.add_log)
            if not success:
                raise Exception("音频初始化失败")

            # 将转录任务添加到队列
            success = self.server_connection.add_transcription_task(
                TranscriptionClient.run_transcription,
                config,
                self,
                self.stop_event,
                self.obs_manager.obs_client,
                stream,
                self.server_connection.server_websocket,
            )

            if success:
                self.logger.add_log("已将转录任务添加到队列")
            else:
                self.update_status("错误：服务器连接无效", is_log=True)
                self.transcription_active = False
                self.start_button.configure(text="启动转录")
        except Exception as e:
            self.update_status(f"启动转录失败: {str(e)}", is_log=True)
            self.transcription_active = False
            self.start_button.configure(text="启动转录")

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
            self.audio_manager.cleanup_audio()
            # 清除传输统计
            self.update_stats("")
        else:
            self.status.configure(text=text)

        # 如果需要，也添加到日志
        if is_log:
            self.logger.add_log(text)

    def add_log(self, message):
        """添加消息到日志区域（兼容性方法）"""
        self.logger.add_log(message)

    def save_config(self):
        """保存当前配置到config.ini文件"""
        config_dict = {
            "OBS": {
                "host": self.obs_host.get(),
                "port": self.obs_port.get(),
                "password": self.obs_password.get(),
                "text_source": self.text_source.get(),
            },
            "SERVER": {"url": self.server_url.get()},
        }

        # 写入文件
        success, error = self.config_manager.save_config(config_dict)
        if success:
            self.logger.add_log("配置已保存到 config.ini")
            messagebox.showinfo("保存成功", "配置已成功保存到 config.ini")
        else:
            self.logger.add_log(f"保存配置失败: {error}")
            messagebox.showerror("保存失败", f"无法保存配置: {error}")

    def restore_defaults(self):
        """恢复默认设置"""
        if messagebox.askyesno("恢复默认", "确定要恢复所有设置到默认值吗？"):
            # OBS默认设置
            self.obs_host.delete(0, tk.END)
            self.obs_host.insert(0, "localhost")

            self.obs_port.delete(0, tk.END)
            self.obs_port.insert(0, "4455")

            self.obs_password.delete(0, tk.END)

            # 文本源
            self.text_source.set("转录文本")

            # 服务器默认设置
            self.server_url.delete(0, tk.END)
            self.server_url.insert(0, "ws://localhost:8765")

            self.logger.add_log("已恢复默认设置")

    def load_config(self):
        """从config.ini加载配置（如果存在）"""
        config, error = self.config_manager.load_config()
        if error:
            self.logger.add_log(f"配置加载失败或未找到: {error}")
            return

        try:
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
                    self.text_source.set(config["OBS"]["text_source"])

            # 加载服务器设置
            if "SERVER" in config and "url" in config["SERVER"]:
                self.server_url.delete(0, tk.END)
                self.server_url.insert(0, config["SERVER"]["url"])

            self.logger.add_log("已加载配置文件")
        except Exception as e:
            self.logger.add_log(f"加载配置失败: {str(e)}")
