# Whisper WebSocket 实时语音转录 API 文档

## 目录

1. [接口概述](#接口概述)
2. [连接信息](#连接信息)
3. [消息格式](#消息格式)
4. [通信流程](#通信流程)
5. [错误处理](#错误处理)
6. [示例代码](#示例代码)
7. [最佳实践](#最佳实践)

## 接口概述

该WebSocket API提供实时语音转录服务，使用OpenAI的Whisper large-v3-turbo模型进行语音识别。通过WebSocket传输音频数据，服务端实时返回转录结果。

**主要特点**：

- 支持多种语言
- 低延迟实时转录
- 优化的二进制音频传输（无Base64编码）
- 基于会话的通信模型

## 连接信息

**WebSocket端点**：`ws://[服务器地址]:[端口]`

默认端口：`8765`

## 消息格式

通信分为两种类型的消息：

1. **控制消息**：使用JSON文本格式
2. **音频数据**：使用WebSocket二进制帧直接传输

### 控制消息（客户端发送）

#### 1. 初始化请求

```json
{
  "type": "init",
  "config": {
    "language": "zh",       // 语言代码（如"zh", "en", "ja"等）
    "sample_rate": 16000,   // 音频采样率（赫兹）
    "encoding": "LINEAR16", // 音频编码格式
    "channels": 1           // 音频通道数
  }
}
```

| 参数 | 类型 | 必须 | 描述 |
|------|------|------|------|
| language | String | 否 | 音频语言, 默认"zh" |
| sample_rate | Integer | 否 | 音频采样率, 默认16000 |
| encoding | String | 否 | 音频编码格式, 目前仅支持"LINEAR16" |
| channels | Integer | 否 | 通道数, 默认1 |

#### 2. 结束会话请求

```json
{
  "type": "end"
}
```

### 音频数据（客户端发送）

音频数据直接通过WebSocket的二进制帧发送，格式要求：

- PCM 16位整数，小端字节序
- 采样率与初始化时指定的sample_rate一致
- 单声道/双声道根据初始化时指定的channels决定

建议：每次发送约20-100ms的音频数据（大约640-3200字节，对于16kHz采样率的单声道音频）

### 服务端响应（JSON文本格式）

#### 1. 初始化确认

```json
{
  "type": "init_ack",
  "session_id": "uuid-string",
  "status": "ready"
}
```

#### 2. 转录结果

```json
{
  "type": "transcription",
  "text": "这是转录的文本内容",
  "is_final": true,
  "language": "zh"
}
```

| 参数 | 类型 | 描述 |
|------|------|------|
| text | String | 转录的文本内容 |
| is_final | Boolean | 是否为最终结果 |
| language | String | 检测到的语言代码 |

#### 3. 结束会话确认

```json
{
  "type": "end_ack",
  "session_id": "uuid-string"
}
```

#### 4. 错误消息

```json
{
  "type": "error",
  "code": "error_code",
  "message": "错误描述"
}
```

常见错误代码:

- `invalid_request`: 无效的请求或会话未初始化
- `json_parse_error`: JSON解析错误
- `audio_processing_error`: 音频处理错误
- `server_error`: 服务端内部错误

## 通信流程

1. **建立连接**：客户端与服务端建立WebSocket连接
2. **初始化会话**：客户端发送init消息，服务端返回session_id
3. **传输音频**：客户端以二进制帧形式发送音频数据
4. **接收转录**：服务端处理音频并返回转录结果
5. **结束会话**：客户端发送end消息，服务端确认并关闭会话

```
客户端                                     服务端
  |                                         |
  |       WebSocket连接建立                 |
  |---------------------------------------->|
  |                                         |
  |       发送初始化请求 (JSON)             |
  |---------------------------------------->|
  |                                         |
  |       接收初始化确认 (JSON)             |
  |<----------------------------------------|
  |                                         |
  |       发送音频数据 (二进制)             |
  |---------------------------------------->|
  |                                         |
  |       接收转录结果 (JSON)               |
  |<----------------------------------------|
  |                                         |
  |       发送更多音频 (二进制)             |
  |---------------------------------------->|
  |                                         |
  |       接收更多转录结果 (JSON)           |
  |<----------------------------------------|
  |                                         |
  |       发送结束会话请求 (JSON)           |
  |---------------------------------------->|
  |                                         |
  |       接收结束会话确认 (JSON)           |
  |<----------------------------------------|
  |                                         |
  |       WebSocket连接关闭                 |
  |<--------------------------------------->|
```

## 错误处理

- 会话自动超时：如果5分钟内没有活动，会话将被自动关闭
- 连接错误：当WebSocket连接意外断开，客户端应实现重连逻辑
- 服务错误：根据返回的错误代码和消息进行相应处理

## 示例代码

### Python 客户端示例

```python
import asyncio
import websockets
import json
import wave
import time

async def transcribe_microphone_audio(server_url="ws://localhost:8765"):
    """从麦克风捕获音频并实时转录"""
    
    # 导入必要的库
    import pyaudio
    
    # PyAudio配置
    CHUNK = 1600  # 每个缓冲区的帧数 (50ms at 16kHz)
    FORMAT = pyaudio.paInt16  # 16位PCM
    CHANNELS = 1  # 单声道
    RATE = 16000  # 采样率
    
    p = pyaudio.PyAudio()
    
    try:
        # 打开麦克风流
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("* 正在连接到转录服务...")
        async with websockets.connect(server_url) as websocket:
            # 初始化会话
            await websocket.send(json.dumps({
                "type": "init",
                "config": {
                    "language": "zh",
                    "sample_rate": RATE,
                    "channels": CHANNELS,
                    "encoding": "LINEAR16"
                }
            }))
            
            # 接收初始化确认
            response = await websocket.recv()
            resp_data = json.loads(response)
            if resp_data.get("type") != "init_ack":
                print(f"初始化失败: {response}")
                return
            
            session_id = resp_data.get("session_id")
            print(f"* 会话已初始化 (ID: {session_id})")
            print("* 开始录音，按Ctrl+C停止...")
            
            # 设置进行并发接收和发送
            stop_flag = False
            
            # 接收器协程
            async def receiver():
                try:
                    while not stop_flag:
                        try:
                            # 非阻塞接收，超时1秒
                            response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                            result = json.loads(response)
                            
                            if result.get("type") == "transcription":
                                print(f"\r转录: {result.get('text')}")
                            elif result.get("type") == "error":
                                print(f"\r错误: {result.get('message')}")
                        except asyncio.TimeoutError:
                            # 超时，继续循环
                            pass
                        except Exception as e:
                            print(f"接收错误: {str(e)}")
                            break
                except asyncio.CancelledError:
                    pass
            
            # 发送器协程
            async def sender():
                try:
                    while not stop_flag:
                        # 从麦克风读取数据
                        audio_data = stream.read(CHUNK, exception_on_overflow=False)
                        # 直接发送二进制数据
                        await websocket.send(audio_data)
                        await asyncio.sleep(0.01)  # 稍微让出CPU
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"发送错误: {str(e)}")
            
            # 启动并发任务
            receiver_task = asyncio.create_task(receiver())
            sender_task = asyncio.create_task(sender())
            
            try:
                # 等待用户中断
                while True:
                    await asyncio.sleep(0.1)
            except KeyboardInterrupt:
                print("\n* 停止录音...")
            finally:
                # 标记停止并取消任务
                stop_flag = True
                receiver_task.cancel()
                sender_task.cancel()
                
                # 结束会话
                await websocket.send(json.dumps({"type": "end"}))
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    print(f"会话结束: {response}")
                except:
                    print("无法接收会话结束确认")
    
    finally:
        # 清理资源
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()

# 运行示例
if __name__ == "__main__":
    asyncio.run(transcribe_microphone_audio())
```

### JavaScript 客户端示例

```javascript
class WhisperTranscriptionClient {
  constructor(serverUrl = 'ws://localhost:8765') {
    this.serverUrl = serverUrl;
    this.websocket = null;
    this.isRecording = false;
    this.mediaStream = null;
    this.audioContext = null;
    this.processor = null;
    this.onTranscriptionCallback = null;
    this.onErrorCallback = null;
  }

  async connect(language = 'zh') {
    return new Promise((resolve, reject) => {
      try {
        this.websocket = new WebSocket(this.serverUrl);
        
        this.websocket.onopen = () => {
          console.log('WebSocket连接已建立');
          
          // 发送初始化请求
          const initMessage = {
            type: 'init',
            config: {
              language: language,
              sample_rate: 16000,
              channels: 1,
              encoding: 'LINEAR16'
            }
          };
          
          this.websocket.send(JSON.stringify(initMessage));
        };
        
        this.websocket.onmessage = (event) => {
          const message = JSON.parse(event.data);
          
          if (message.type === 'init_ack') {
            console.log(`会话已初始化, ID: ${message.session_id}`);
            resolve(message.session_id);
          } else if (message.type === 'transcription') {
            if (this.onTranscriptionCallback) {
              this.onTranscriptionCallback(message.text, message.is_final);
            }
          } else if (message.type === 'error') {
            console.error(`服务器错误: ${message.code} - ${message.message}`);
            if (this.onErrorCallback) {
              this.onErrorCallback(message.code, message.message);
            }
          }
        };
        
        this.websocket.onerror = (error) => {
          console.error('WebSocket错误:', error);
          reject(error);
        };
        
        this.websocket.onclose = (event) => {
          console.log(`WebSocket关闭: 代码=${event.code}, 原因=${event.reason}`);
          this.stopRecording();
        };
        
      } catch (error) {
        reject(error);
      }
    });
  }

  async startRecording() {
    if (this.isRecording) return;
    
    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.audioContext = new AudioContext();
      
      // 创建音频处理节点
      const source = this.audioContext.createMediaStreamSource(this.mediaStream);
      this.processor = this.audioContext.createScriptProcessor(1024, 1, 1);
      
      // 重采样至16kHz (如果需要)
      const sampleRate = this.audioContext.sampleRate;
      const resamplingRatio = 16000 / sampleRate;
      
      this.processor.onaudioprocess = (e) => {
        if (!this.isRecording) return;
        
        const inputData = e.inputBuffer.getChannelData(0);
        
        // 创建16位整数PCM数据
        const pcmData = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          // 将Float32音频数据[-1.0, 1.0]转换为Int16[-32768, 32767]
          pcmData[i] = Math.min(Math.max(inputData[i] * 32767, -32768), 32767);
        }
        
        // 将Int16Array发送为二进制数据
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
          this.websocket.send(pcmData.buffer);
        }
      };
      
      // 连接节点
      source.connect(this.processor);
      this.processor.connect(this.audioContext.destination);
      
      this.isRecording = true;
      console.log('开始录音和转录');
      
    } catch (error) {
      console.error('启动录音失败:', error);
      throw error;
    }
  }

  stopRecording() {
    if (!this.isRecording) return;
    
    this.isRecording = false;
    
    // 断开处理节点
    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }
    
    // 关闭音频上下文
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
    
    // 关闭麦克风流
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }
    
    // 发送结束会话请求
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify({ type: 'end' }));
    }
    
    console.log('停止录音和转录');
  }

  disconnect() {
    this.stopRecording();
    
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
  }

  onTranscription(callback) {
    this.onTranscriptionCallback = callback;
  }

  onError(callback) {
    this.onErrorCallback = callback;
  }
}

// 使用示例
async function startWhisperTranscription() {
  const client = new WhisperTranscriptionClient();
  
  client.onTranscription((text, isFinal) => {
    const resultElement = document.getElementById('transcription-result');
    resultElement.textContent = text;
    if (isFinal) {
      resultElement.style.fontWeight = 'bold';
    } else {
      resultElement.style.fontWeight = 'normal';
    }
  });
  
  client.onError((code, message) => {
    console.error(`转录错误 [${code}]: ${message}`);
    document.getElementById('status').textContent = `错误: ${message}`;
  });
  
  try {
    await client.connect();
    document.getElementById('status').textContent = '已连接';
    
    const startButton = document.getElementById('start-button');
    const stopButton = document.getElementById('stop-button');
    
    startButton.addEventListener('click', async () => {
      await client.startRecording();
      document.getElementById('status').textContent = '正在录音...';
    });
    
    stopButton.addEventListener('click', () => {
      client.stopRecording();
      document.getElementById('status').textContent = '已停止';
    });
    
    window.addEventListener('beforeunload', () => {
      client.disconnect();
    });
    
  } catch (error) {
    console.error('连接失败:', error);
    document.getElementById('status').textContent = '连接失败';
  }
}
```

## 最佳实践

1. **音频分块**：
   - 每个音频块控制在20-100ms长度（16kHz采样率约640-3200字节）
   - 过大块会增加延迟，过小块会增加网络开销

2. **错误处理**：
   - 实现断线重连机制，包括指数退避策略
   - 保存会话状态以便在重连时恢复

3. **音频质量**：
   - 使用16kHz或更高采样率获得最佳识别效果
   - 确保麦克风输入质量高，减少背景噪音

4. **网络优化**：
   - 在较差网络环境下考虑增加音频缓冲区大小
   - 监控和记录WebSocket连接状态

5. **资源管理**：
   - 使用完毕后显式调用`end`消息关闭会话
   - 长时间不使用时断开连接，需要时重连

6. **隐私考虑**：
   - 告知用户音频数据会被发送到服务器进行处理
   - 考虑在客户端实现音量检测，只传输有声音的片段

7. **兼容性**：
   - 在WebSocket不支持的环境下提供备选方案（如HTTP轮询）
   - 考虑在低功耗设备上减少处理和传输频率
