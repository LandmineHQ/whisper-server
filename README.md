# Whisper-Server: Real-time Speech-to-Text Transcription System

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenAI Whisper](https://img.shields.io/badge/AI-OpenAI%20Whisper-brightgreen)

A real-time speech transcription system powered by OpenAI's Whisper model, featuring a WebSocket server for low-latency audio processing and an OBS Studio integration client.

## 🌟 Features

- **Real-time transcription** using OpenAI's Whisper large-v3-turbo model
- **WebSocket-based communication** for efficient binary audio streaming
- **Multi-language support** for global accessibility
- **OBS Studio integration** for live streaming captions/subtitles
- **Low latency processing** optimized for real-time applications
- **User-friendly GUI client** with configuration management

## 🏗️ System Architecture

The system consists of two main components:

1. **Whisper WebSocket Server**:
   - Accepts real-time audio streams via WebSocket
   - Processes audio using OpenAI's Whisper model
   - Returns transcription results in real-time
   - Manages multiple concurrent sessions

2. **OBS Transcription Client**:
   - Captures audio from default input device
   - Streams audio to the transcription server
   - Updates text sources in OBS with transcription results
   - Provides user interface for configuration and monitoring

``` txt
┌─────────────┐     WebSocket     ┌──────────────┐
│   OBS       │   Audio Stream    │              │
│ Transcription│ ────────────────>│   Whisper    │
│   Client    │                   │   Server     │
│             │   Text Results    │              │
│             │ <────────────────│              │
└─────────────┘                   └──────────────┘
       │                                 │
       │                                 │
       ▼                                 ▼
┌─────────────┐                   ┌──────────────┐
│  OBS Studio │                   │   OpenAI     │
│  Text Source│                   │   Whisper    │
└─────────────┘                   │   Model      │
                                  └──────────────┘
```

## 📋 Requirements

### Server Requirements

- Python 3.8+
- torch
- whisper
- websockets
- numpy

### Client Requirements

- Python 3.8+
- obsws-python
- websockets
- pyaudio
- tkinter (included in most Python distributions)

## 🔧 Installation

### Server Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/LandmineHQ/whisper-server.git
   cd whisper-server
   ```

2. Install dependencies:

   ```bash
   pip install torch numpy websockets openai-whisper
   ```

3. Run the server:

   ```bash
   python whisper_server.py
   ```

   By default, the server runs on `0.0.0.0:8765`. You can modify this by setting environment variables:

   ```bash
   HOST=127.0.0.1 PORT=8000 python whisper_server.py
   ```

### Client Setup

1. Install additional dependencies:

   ```bash
   pip install obsws-python pyaudio
   ```

2. Run the OBS integration client:

   ```bash
   python obs_transcription_client.py
   ```

## 🎮 Usage Guide

### Server Configuration

The server requires minimal configuration as it uses default settings. Key settings can be modified via environment variables:

- `HOST`: Server bind address (default: 0.0.0.0)
- `PORT`: Server port (default: 8765)

By default, the server loads the "tiny" Whisper model for faster processing and lower resource requirements. You can modify the model size in the code (e.g., "base", "small", "medium", "large", "large-v3-turbo").

### OBS Client Setup

1. **Enable WebSocket Server in OBS Studio**:
   - Go to Tools → obs-websocket Settings
   - Enable the WebSocket server
   - Set a password if needed
   - Note the port (default: 4455)

2. **Create a Text Source in OBS**:
   - Add a new "Text (GDI+)" source to your scene
   - Name it (e.g., "Transcription Text")
   - Configure font, size, color, etc. as desired

3. **Configure the Client**:
   - Launch the OBS Transcription Client
   - Enter OBS connection details (host, port, password)
   - Click "Connect OBS" and select your text source from the dropdown
   - Enter the WebSocket server URL (e.g., ws://localhost:8765)
   - Click "Connect Transcription Server"
   - Once both connections are established, click "Start Transcription"

4. **Save Configuration**:
   - Click "Save Settings" to store your configuration for future use

## 📡 WebSocket API Reference

### Connection Endpoint

``` txt
ws://[server-address]:[port]
```

Default port: `8765`

### Message Format

#### Control Messages (Client to Server)

**Initialization Request**:

```json
{
  "type": "init",
  "config": {
    "language": "zh",       // Language code (e.g., "zh", "en", "ja")
    "sample_rate": 16000,   // Audio sample rate (Hz)
    "encoding": "LINEAR16", // Audio encoding format
    "channels": 1           // Audio channels
  }
}
```

**End Session Request**:

```json
{
  "type": "end"
}
```

#### Audio Data

Audio is sent as binary WebSocket frames:

- PCM 16-bit integer, little-endian
- Sample rate matching the initialization config
- Channels matching the initialization config

#### Server Responses

**Initialization Acknowledgment**:

```json
{
  "type": "init_ack",
  "session_id": "uuid-string",
  "status": "ready"
}
```

**Transcription Result**:

```json
{
  "type": "transcription",
  "text": "Transcribed text content",
  "is_final": true,
  "language": "zh"
}
```

**End Session Acknowledgment**:

```json
{
  "type": "end_ack",
  "session_id": "uuid-string"
}
```

**Error Message**:

```json
{
  "type": "error",
  "code": "error_code",
  "message": "Error description"
}
```

## 🔍 Performance Considerations

- The server uses a thread-per-session model with non-blocking I/O
- Audio is processed in chunks of approximately 50-100ms
- The system maintains a limited audio history for context-aware transcription
- Transcription results are provided in real-time with minimal latency
- The OBS client includes statistics monitoring for audio throughput

## 🛡️ Error Handling

- Automatic session cleanup for inactive sessions (5-minute timeout)
- Robust error reporting via WebSocket messages
- Graceful handling of connection drops
- Automatic reconnection capabilities in the client

## 📝 Best Practices

1. **Audio Quality**:
   - Use 16kHz or higher sample rate for best recognition results
   - Ensure good microphone input quality with minimal background noise

2. **Network Optimization**:
   - Consider increasing audio buffer size in poor network conditions
   - Monitor WebSocket connection status

3. **Resource Management**:
   - Explicitly end sessions when finished
   - Disconnect when not in use, reconnect when needed

4. **Privacy Considerations**:
   - Inform users that audio data is being processed by a server
   - Consider implementing volume detection to only transmit audio with sound

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the speech recognition model
- [OBS Studio](https://obsproject.com/) for the streaming software
- [obsws-python](https://github.com/aiovideostudio/obsws-python) for OBS WebSocket integration

---

For detailed API documentation, see [server.md](docs/server.md).
