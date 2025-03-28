#!/usr/bin/env python
# pyinstaller --onefile --windowed .\main.py


def check_dependencies():
    """检查必要的依赖是否已安装"""
    missing_deps = []

    try:
        import tkinter
    except ImportError:
        missing_deps.append("tkinter (GUI库)")

    try:
        import pyaudio
    except ImportError:
        missing_deps.append("pyaudio (音频处理库)")

    try:
        import obsws_python
    except ImportError:
        missing_deps.append("obsws_python (OBS WebSocket库)")

    try:
        import websockets
    except ImportError:
        missing_deps.append("websockets (WebSocket客户端库)")

    if missing_deps:
        print("错误: 缺少以下必要的依赖:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\n安装指南:")
        print("  pip install pyaudio obsws_python websockets")
        if "tkinter" in str(missing_deps):
            print("\n要安装tkinter，请参考系统特定的指南:")
            print("  Ubuntu/Debian: sudo apt-get install python3-tk")
            print("  CentOS/RHEL/Fedora: sudo dnf install python3-tkinter")
            print("  Arch Linux: sudo pacman -S tk")
        return False
    return True


def main():
    # 先检查依赖
    if not check_dependencies():
        return

    import tkinter as tk
    from gui.config_gui import ConfigGUI

    root = tk.Tk()
    gui = ConfigGUI(root)

    # 添加窗口关闭事件处理
    def on_closing():
        # 如果连接了OBS，先断开
        if gui.obs_manager.obs_connected:
            gui.disconnect_from_obs()

        # 如果连接了服务器，先断开
        if gui.server_connection.server_connected:
            gui.disconnect_from_server()

        # 如果转录正在运行，先停止它
        if gui.transcription_active:
            gui.stop_transcription()

        gui.logger.add_log("正在关闭应用程序...")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
