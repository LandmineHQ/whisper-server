# 服务器连接管理类
import asyncio
import websockets
import threading


class ServerConnection:
    def __init__(self):
        self.server_websocket = None
        self.server_websocket_loop = None
        self.server_thread = None
        self.server_connected = False
        self.transcription_queue = None
        self.transcription_task = None

    def connect_to_server(self, server_url, success_callback, failure_callback):
        """连接到转录服务器"""
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
                success_callback()

                # 保持event loop运行以处理websocket消息和转录任务
                loop.run_forever()

            except Exception as e:
                failure_callback(str(e))
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
                await task_args[0](*task_args[1:])
            except Exception as e:
                print(f"转录任务处理错误: {str(e)}")
            finally:
                # 标记任务完成
                self.transcription_queue.task_done()

    def add_transcription_task(self, task_func, *args):
        """添加转录任务到队列"""
        if self.server_websocket_loop and self.server_websocket:
            asyncio.run_coroutine_threadsafe(
                self.transcription_queue.put((task_func, *args)),
                self.server_websocket_loop,
            )
            return True
        return False

    def disconnect_from_server(self):
        """断开服务器连接状态"""
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

                return True, ""
            except Exception as e:
                return False, str(e)
            finally:
                self.server_websocket = None
                self.server_websocket_loop = None
