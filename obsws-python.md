# obsws-python 用法示例文档

## 目录

1. [简介](#简介)
2. [安装要求](#安装要求)
3. [安装方法](#安装方法)
4. [基础连接配置](#基础连接配置)
5. [请求(Requests)示例](#请求requests示例)
6. [事件(Events)处理示例](#事件events处理示例)
7. [属性访问](#属性访问)
8. [错误处理](#错误处理)
9. [日志记录](#日志记录)

## 简介

obsws-python 是一个为 OBS Studio WebSocket v5.0 接口开发的 Python SDK，它简化了与 OBS Studio 进行交互的过程。通过这个库，您可以控制 OBS 的场景切换、音频设置、录制状态等众多功能。

> 注意：当前版本未实现官方文档中的所有端点。

## 安装要求

* [OBS Studio](https://obsproject.com/)
* [OBS Websocket v5 插件](https://github.com/obsproject/obs-websocket/releases/tag/5.0.0)
  * 在 OBS Studio 28 及以上版本中已默认包含此插件
  * 对于早期版本需手动安装
* Python 3.9 或更高版本

## 安装方法

```bash
pip install obsws-python
```

## 基础连接配置

obsws-python 提供了两种配置连接参数的方式：

### 方式一：使用关键字参数

```python
import obsws_python as obs

# 通过关键字参数传递连接信息
client = obs.ReqClient(
    host='localhost',  
    port=4455,  
    password='mystrongpass',  
    timeout=3
)
```

### 方式二：使用配置文件

在用户主目录创建 `config.toml` 文件：

```toml
[connection]
host = "localhost"
port = 4455
password = "mystrongpass"
```

然后直接初始化客户端：

```python
import obsws_python as obs

# 会自动从配置文件加载连接信息
client = obs.ReqClient()
```

> 参数优先级：关键字参数 > 配置文件 > 默认值

## 请求(Requests)示例

请求方法命名采用蛇形命名法(snake_case)，对应 OBS WebSocket API 中的方法。

### 获取 OBS 版本信息

```python
import obsws_python as obs

client = obs.ReqClient()

# 获取版本信息
response = client.get_version()
print(f"OBS 版本: {response.obs_version}")
print(f"WebSocket 版本: {response.obs_web_socket_version}")
```

### 场景操作

```python
# 获取当前场景
current_scene = client.get_current_program_scene()
print(f"当前场景: {current_scene.current_program_scene_name}")

# 切换到指定场景
client.set_current_program_scene("直播中")

# 获取场景列表
scenes = client.get_scene_list()
for scene in scenes.scenes:
    print(f"场景名称: {scene['sceneName']}")
```

### 音频控制

```python
# 静音/取消静音麦克风
client.toggle_input_mute('Mic/Aux')

# 设置音量 (0.0-1.0)
client.set_input_volume('Mic/Aux', 0.5)

# 获取音量
volume = client.get_input_volume('Mic/Aux')
print(f"麦克风音量: {volume.input_volume}")
```

### 录制控制

```python
# 开始录制
client.start_record()

# 停止录制
client.stop_record()

# 获取录制状态
status = client.get_record_status()
print(f"录制状态: {'正在录制' if status.output_active else '未录制'}")
```

### 直接使用 send 方法（处理原始数据）

```python
# 使用raw=True获取原始JSON响应
response = client.send("GetVersion", raw=True)
print(f"原始响应数据: {response}")
```

## 事件(Events)处理示例

使用 `EventClient` 处理 OBS 发出的事件。回调函数命名规则为 "on_" + 事件名称的蛇形命名法形式。

```python
import obsws_python as obs

# 创建事件客户端
event_client = obs.EventClient()

# 场景切换事件处理
def on_current_program_scene_changed(data):
    print(f"场景已切换至: {data.scene_name}")

# 录制状态变化事件处理
def on_record_state_changed(data):
    if data.output_active:
        print(f"录制已开始，文件保存至: {data.output_path}")
    else:
        print("录制已停止")

# 音频静音状态变化事件处理
def on_input_mute_state_changed(data):
    status = "已静音" if data.input_muted else "已取消静音"
    print(f"输入源 '{data.input_name}' {status}")

# 注册回调函数
event_client.callback.register(on_current_program_scene_changed)
event_client.callback.register(on_record_state_changed)
event_client.callback.register(on_input_mute_state_changed)

# 批量注册多个回调函数
event_client.callback.register([on_current_program_scene_changed, on_input_mute_state_changed])

# 查看已注册的回调函数
registered_callbacks = event_client.callback.get()
print(f"已注册的回调函数: {registered_callbacks}")

# 取消注册回调函数
event_client.callback.deregister(on_input_mute_state_changed)

# 保持程序运行以接收事件
import time
try:
    print("开始监听OBS事件，按Ctrl+C退出...")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("停止监听OBS事件")
```

## 属性访问

对于请求响应和事件数据，您可以使用 `attrs()` 方法查看可用属性。

```python
# 查看请求响应的可用属性
response = client.get_version()
print(f"可用属性: {response.attrs()}")

# 在事件回调中查看可用属性
def on_scene_created(data):
    print(f"场景创建事件属性: {data.attrs()}")
    print(f"场景名称: {data.scene_name}")
```

## 错误处理

obsws-python 提供了几种异常类型用于错误处理：

```python
import obsws_python as obs

try:
    client = obs.ReqClient(timeout=1)  # 设置较短的超时时间
    
    # 可能引发 OBSSDKRequestError 的操作
    client.set_current_program_scene("不存在的场景")
    
except obs.error.OBSSDKTimeoutError:
    print("连接超时，请检查OBS是否运行以及WebSocket服务是否启用")
    
except obs.error.OBSSDKRequestError as e:
    print(f"请求错误 - 请求名称: {e.req_name}, 错误代码: {e.code}")
    if e.code == 601:
        print("没有找到指定的资源")
    elif e.code == 604:
        print("请求的操作无效")
    
except obs.error.OBSSDKError as e:
    # 捕获所有其他OBS SDK错误
    print(f"OBS SDK错误: {e}")
```

## 日志记录

开启调试日志查看原始消息交换：

```python
import obsws_python as obs
import logging

# 设置日志级别为DEBUG以查看所有通信内容
logging.basicConfig(level=logging.DEBUG)

client = obs.ReqClient()
response = client.get_version()
```

## 完整示例

结合请求和事件的综合示例：

```python
import obsws_python as obs
import time
import logging

# 设置日志级别
logging.basicConfig(level=logging.INFO)

# 创建请求客户端和事件客户端
req_client = obs.ReqClient(host='localhost', port=4455)
event_client = obs.EventClient(host='localhost', port=4455)

# 定义事件处理函数
def on_scene_changed(data):
    print(f"场景已切换到: {data.scene_name}")
    
def on_streaming_state_changed(data):
    if data.output_active:
        print("直播已开始")
    else:
        print("直播已结束")

# 注册事件处理函数
event_client.callback.register([
    on_scene_changed,
    on_streaming_state_changed
])

# 获取版本信息
version = req_client.get_version()
print(f"OBS版本: {version.obs_version}")
print(f"WebSocket版本: {version.obs_web_socket_version}")

# 获取场景列表
scenes = req_client.get_scene_list()
print("可用场景:")
for scene in scenes.scenes:
    print(f"- {scene['sceneName']}")

# 设置场景
current_scene = scenes.scenes[0]['sceneName']
print(f"切换到场景: {current_scene}")
req_client.set_current_program_scene(current_scene)

# 保持程序运行以接收事件
try:
    print("监听OBS事件中，按Ctrl+C退出...")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("程序已退出")
```

通过本文档的示例，您应该能够开始使用 obsws-python 库与 OBS Studio 进行交互。更多详细信息，请参考[官方文档](https://github.com/obsproject/obs-websocket/blob/master/docs/generated/protocol.md#obs-websocket-501-protocol)。
