# Whisper-Server: 实时语音转文字转录系统

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenAI Whisper](https://img.shields.io/badge/AI-OpenAI%20Whisper-brightgreen)

一个由OpenAI的Whisper模型驱动的实时语音转录系统，具有低延迟音频处理的WebSocket服务器和OBS Studio集成客户端。

## 🌟 功能特点

- **实时转录** 使用OpenAI的Whisper large-v3-turbo模型
- **基于WebSocket的通信** 实现高效的二进制音频流传输
- **多语言支持** 提供全球语言无障碍访问
- **OBS Studio集成** 用于直播字幕/字幕显示
- **低延迟处理** 针对实时应用场景优化
- **用户友好的GUI客户端** 提供配置管理功能

## 🏗️ 系统架构

系统由两个主要组件组成：

1. **Whisper WebSocket服务器**:
   - 通过WebSocket接收实时音频流
   - 使用OpenAI的Whisper模型处理音频
   - 实时返回转录结果
   - 管理多个并发会话

2. **OBS转录客户端**:
   - 从默认输入设备捕获音频
   - 将音频流传输到转录服务器
   - 使用转录结果更新OBS中的文本源
   - 提供配置和监控的用户界面

``` txt
┌─────────────┐     WebSocket     ┌──────────────┐
│   OBS       │      音频流        │              │
│ 转录客户端   │ ────────────────> │   Whisper    │
│             │                   │   服务器      │
│             │   文本结果         │              |
│             │ <──────────────── │              │
└─────────────┘                   └──────────────┘
       │                                 │
       │                                 │
       ▼                                 ▼
┌─────────────┐                   ┌──────────────┐
│  OBS Studio │                   │   OpenAI     │
│  文本源     │                   │   Whisper    │
└─────────────┘                   │   模型       │
                                  └──────────────┘
```

## 📋 系统需求

### 服务器需求

- Python 3.8+
- torch
- whisper
- websockets
- numpy

### 客户端需求

- Python 3.8+
- obsws-python
- websockets
- pyaudio
- tkinter (大多数Python发行版中已包含)

## 🔧 安装

### 服务器设置

1. 克隆仓库:

   ```bash
   git clone https://github.com/LandmineHQ/whisper-server.git
   cd whisper-server
   ```

2. 安装依赖项:

   ```bash
   pip install torch numpy websockets openai-whisper
   ```

3. 运行服务器:

   ```bash
   python whisper_server.py
   ```

   默认情况下，服务器运行在 `0.0.0.0:8765`。你可以通过设置环境变量来修改:

   ```bash
   HOST=127.0.0.1 PORT=8000 python whisper_server.py
   ```

### 客户端设置

1. 安装额外依赖项:

   ```bash
   pip install obsws-python pyaudio
   ```

2. 运行OBS集成客户端:

   ```bash
   python obs_transcription_client.py
   ```

## 🎮 使用指南

### 服务器配置

服务器使用默认设置，配置极简。主要设置可通过环境变量修改:

- `HOST`: 服务器绑定地址 (默认: 0.0.0.0)
- `PORT`: 服务器端口 (默认: 8765)

默认情况下，服务器加载"tiny"级别的Whisper模型，以实现更快的处理速度和更低的资源需求。你可以在代码中修改模型大小(如"base", "small", "medium", "large", "large-v3-turbo")。

### OBS客户端设置

1. **在OBS Studio中启用WebSocket服务器**:
   - 进入工具 → obs-websocket设置
   - 启用WebSocket服务器
   - 如需要，设置密码
   - 记下端口号 (默认: 4455)

2. **在OBS中创建文本源**:
   - 在场景中添加新的"文本 (GDI+)"源
   - 命名它 (例如, "转录文本")
   - 根据需要配置字体、大小、颜色等

3. **配置客户端**:
   - 启动OBS转录客户端
   - 输入OBS连接详情 (主机、端口、密码)
   - 点击"连接OBS"并从下拉菜单中选择你的文本源
   - 输入WebSocket服务器URL (例如, ws://localhost:8765)
   - 点击"连接转录服务器"
   - 两个连接都建立后，点击"开始转录"

4. **保存配置**:
   - 点击"保存设置"将配置保存以供将来使用

## 📡 WebSocket API参考

### 连接端点

``` txt
ws://[服务器地址]:[端口]
```

默认端口: `8765`

### 消息格式

#### 控制消息 (客户端到服务器)

**初始化请求**:

```json
{
  "type": "init",
  "config": {
    "language": "zh",       // 语言代码 (如 "zh", "en", "ja")
    "sample_rate": 16000,   // 音频采样率 (Hz)
    "encoding": "LINEAR16", // 音频编码格式
    "channels": 1           // 音频通道数
  }
}
```

**结束会话请求**:

```json
{
  "type": "end"
}
```

#### 音频数据

音频通过WebSocket二进制帧发送:

- PCM 16位整数，小端字节序
- 采样率与初始化配置匹配
- 通道数与初始化配置匹配

#### 服务器响应

**初始化确认**:

```json
{
  "type": "init_ack",
  "session_id": "uuid字符串",
  "status": "ready"
}
```

**转录结果**:

```json
{
  "type": "transcription",
  "text": "转录的文本内容",
  "is_final": true,
  "language": "zh"
}
```

**结束会话确认**:

```json
{
  "type": "end_ack",
  "session_id": "uuid字符串"
}
```

**错误消息**:

```json
{
  "type": "error",
  "code": "错误代码",
  "message": "错误描述"
}
```

## 🔍 性能考量

- 服务器使用每会话一线程模型，采用非阻塞I/O
- 音频以约50-100毫秒的块进行处理
- 系统维护有限的音频历史记录，用于上下文感知转录
- 转录结果以最小延迟实时提供
- OBS客户端包含音频吞吐量的统计监控功能

## 🛡️ 错误处理

- 自动清理不活跃会话(5分钟超时)
- 通过WebSocket消息提供健壮的错误报告
- 优雅处理连接中断
- 客户端具备自动重连能力

## 📝 最佳实践

1. **音频质量**:
   - 使用16kHz或更高采样率获得最佳识别效果
   - 确保麦克风输入质量高，减少背景噪音

2. **网络优化**:
   - 在网络条件较差的情况下考虑增加音频缓冲区大小
   - 监控WebSocket连接状态

3. **资源管理**:
   - 使用完毕后显式结束会话
   - 不使用时断开连接，需要时重新连接

4. **隐私考虑**:
   - 告知用户音频数据将被发送到服务器处理
   - 考虑实现音量检测功能，只传输有声音的音频

## 🤝 贡献

欢迎贡献！请随时提交拉取请求。

## 📄 许可证

该项目采用MIT许可证 - 详情请参阅LICENSE文件。

## 🙏 致谢

- [OpenAI Whisper](https://github.com/openai/whisper) 提供语音识别模型
- [OBS Studio](https://obsproject.com/) 提供流媒体软件
- [obsws-python](https://github.com/aiovideostudio/obsws-python) 提供OBS WebSocket集成

---

详细API文档，请参阅 [server.md](docs/server.md)。
