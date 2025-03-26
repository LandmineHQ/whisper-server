from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import tempfile
import os

app = Flask(__name__)
CORS(app)

# 限制服务器资源使用
os.environ["OMP_NUM_THREADS"] = "4"  # 限制为4个CPU核心
os.environ["MKL_NUM_THREADS"] = "4"  # Intel MKL库也使用4个核心

# 加载模型 - 选择与您下载的模型相匹配
model = whisper.load_model("large-v3-turbo")


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]

    # 保存临时文件
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "input.wav")
    audio_file.save(temp_path)

    # 转写
    result = model.transcribe(temp_path)

    # 清理
    os.remove(temp_path)
    os.rmdir(temp_dir)

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9001)
