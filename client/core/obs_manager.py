# OBS连接管理类
import obsws_python as obs
import threading


class OBSManager:
    def __init__(self):
        self.obs_client = None
        self.obs_connected = False

    def connect_to_obs(self, host, port, password, success_callback, failure_callback):
        """连接到OBS WebSocket"""

        # 在后台线程中连接OBS，避免UI卡顿
        def connect_obs_thread():
            try:
                self.obs_client = obs.ReqClient(
                    host=host,
                    port=port,
                    password=password,
                )
                # 连接成功
                success_callback()
            except Exception as e:
                failure_callback(str(e))

        thread = threading.Thread(target=connect_obs_thread, daemon=True)
        thread.start()

    def disconnect_from_obs(self):
        """断开OBS连接"""
        if self.obs_client:
            try:
                # 主动断开OBS WebSocket连接
                if hasattr(self.obs_client, "ws_client") and self.obs_client.ws_client:
                    self.obs_client.ws_client.disconnect()
                elif hasattr(self.obs_client, "disconnect"):
                    self.obs_client.disconnect()
                return True, ""
            except Exception as e:
                return False, str(e)
            finally:
                self.obs_client = None
                self.obs_connected = False

        return True, ""

    def get_text_sources(self):
        """从OBS获取所有文本GDI+源"""
        if not self.obs_client:
            return [], "未连接到OBS"

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

            return text_sources, ""
        except Exception as e:
            return [], str(e)

    def update_text_source(self, source_name, text):
        """更新OBS文本源内容"""
        if not self.obs_client:
            return False, "未连接到OBS"

        try:
            self.obs_client.set_input_settings(source_name, {"text": text}, True)
            return True, ""
        except Exception as e:
            return False, str(e)
