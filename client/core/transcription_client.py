# 转录客户端核心功能
import asyncio
import json
import time
import websockets


class TranscriptionClient:
    @staticmethod
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
                    status_callback.update_status(
                        f"会话已结束: {data.get('session_id')}"
                    )
                else:
                    status_callback.update_status(
                        f"收到未知消息类型: {data.get('type')}"
                    )
        except asyncio.CancelledError:
            # 任务被取消，正常退出
            status_callback.update_status("消息接收任务已取消")
        except Exception as e:
            status_callback.update_status(f"处理消息时出错: {str(e)}")

    @staticmethod
    async def run_transcription(
        config,
        status_callback,
        stop_event,
        obs_client,
        audio_stream,
        websocket,
    ):
        """与转录服务器交互的客户端主函数"""
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
                TranscriptionClient.handle_messages(
                    websocket, obs_client, config, status_callback
                )
            )

            # 主循环：捕获并发送音频
            status_callback.update_status("开始音频捕获...")

            # 跟踪发送的数据量
            total_bytes_sent = 0
            chunks_sent = 0
            start_time = time.time()

            try:
                while not stop_event.is_set():  # 检查停止信号
                    audio_chunk = audio_stream.read(
                        chunk_size, exception_on_overflow=False
                    )

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
                            response = await asyncio.wait_for(
                                websocket.recv(), timeout=2.0
                            )
                            status_callback.update_status(
                                f"收到会话结束确认: {response}"
                            )
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
                    status_callback.update_status(
                        f"结束会话处理错误(保持连接): {str(e)}"
                    )

        except Exception as e:
            status_callback.update_status(f"转录客户端错误: {str(e)}")
        finally:
            # 转录服务已停止，可以重新启动
            status_callback.update_status("转录服务已停止，可以重新启动")
            # 在UI线程中重新启用启动按钮
            status_callback.update_status("ready", is_log=False)
