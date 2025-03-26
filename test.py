import requests
import os


def test_whisper_transcription():
    """
    测试Whisper服务器的转写功能
    """
    # 服务器URL
    url = "http://localhost:9001/transcribe"

    # 音频文件路径
    audio_file_path = "audio.mp3"

    # 检查文件是否存在
    if not os.path.exists(audio_file_path):
        print(f"错误: 文件 {audio_file_path} 不存在")
        return

    try:
        # 准备文件
        files = {
            "audio": (
                os.path.basename(audio_file_path),
                open(audio_file_path, "rb"),
                "audio/mpeg",
            )
        }

        # 发送请求
        print("正在发送音频文件到服务器进行转写...")
        response = requests.post(url, files=files)

        # 检查响应
        if response.status_code == 200:
            result = response.json()
            print("转写成功!")
            print("转写文本:", result.get("text", ""))

            # 如果需要，可以打印更多详细信息
            if "segments" in result:
                print("\n分段信息:")
                for i, segment in enumerate(result["segments"]):
                    print(f"段落 {i+1}: {segment.get('text', '')}")
                    print(
                        f"  开始时间: {segment.get('start', 0):.2f}秒, 结束时间: {segment.get('end', 0):.2f}秒"
                    )
        else:
            print(f"错误: 服务器返回状态码 {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        # 确保文件被关闭
        if "files" in locals() and "audio" in files:
            files["audio"][1].close()


if __name__ == "__main__":
    test_whisper_transcription()
