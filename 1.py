import asyncio
import json
import threading
import time
import nls
import pyaudio
from fastapi import FastAPI, HTTPException, WebSocket, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
import os
from openai import OpenAI
from uuid import uuid4

# 实时语音转文本配置
URL = "wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1"
TOKEN = "f42a185d8bae4622a9aae50128857d85"  # 替换为你的token
APPKEY = "JNbplHCzbWk4PVCH"  # 替换为你的appkey

# 配置麦克风参数
CHUNK = 640  # 每次读取的音频块大小
FORMAT = pyaudio.paInt16  # 音频格式
CHANNELS = 1  # 声道数
RATE = 16000  # 采样率

# 大模型推理服务配置
client = OpenAI(
    api_key = "9ec5a9f8-d11c-4571-8394-bd196b209fe4",
    base_url = "https://ark.cn-beijing.volces.com/api/v3",
)

# FastAPI应用
app = FastAPI()

# 允许跨域请求
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

# 定义大模型推理服务的请求和响应格式
class InferenceRequest(BaseModel):
    text: str

class InferenceResponse(BaseModel):
    result: str

# 定义激活请求和响应格式
class ActivationRequest(BaseModel):
    key: str

class ActivationResponse(BaseModel):
    duration: int
    expire_at: str

# 实时语音转文本类
class SpeechToText:
    def __init__(self, tid, websocket: WebSocket = None):
        self.__th = threading.Thread(target=self.__test_run)
        self.__id = tid
        self.__running = False
        self.result = ""
        self.latest_result = ""  # 用于保存最新的结果
        self.websocket = websocket

    def start(self):
        self.__running = True
        self.__th.start()

    def stop(self):
        self.__running = False
        self.__th.join()

    def test_on_sentence_end(self, message, *args):
        print("test_on_sentence_end:{}".format(message))
        try:
            message_dict = json.loads(message)
            result = message_dict["payload"]["result"]
            self.latest_result = result  # 保存最新的结果
            print('存储的内容为：', self.latest_result)
            # 在非主线程中创建一个新的事件循环
            if self.websocket:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.send_result(result))
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    async def send_result(self, result):
        if self.websocket:
            await self.websocket.send_text(result)

    def test_on_result_chg(self, message, *args):
        print("test_on_chg:{}".format(message))

    def __test_run(self):
        print("thread:{} start..".format(self.__id))
        sr = nls.NlsSpeechTranscriber(
            url=URL,
            token=TOKEN,
            appkey=APPKEY,
            on_sentence_end=self.test_on_sentence_end,
            on_result_changed=self.test_on_result_chg,
            callback_args=[self.__id]
        )

        print("{}: session start".format(self.__id))
        try:
            sr.start(aformat="pcm", enable_intermediate_result=True, enable_punctuation_prediction=True, enable_inverse_text_normalization=True)

            # 打开麦克风流
            audio = pyaudio.PyAudio()
            stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

            while self.__running:
                data = stream.read(CHUNK)
                sr.send_audio(data)
                time.sleep(0.01)

            # 发送最后的音频数据并等待处理完成
            sr.stop()
            time.sleep(1)
        except Exception as e:
            print(f"Error in speech transcription: {e}")
            self.__running = False

        print("{}: sr stopped".format(self.__id))

        # 关闭麦克风流
        stream.stop_stream()
        stream.close()
        audio.terminate()

# 保存每个会话的对话上下文
sessions = {}

# 调用大模型推理服务
async def large_model_inference(session_id, text, websocket: WebSocket):
    if session_id not in sessions:
        sessions[session_id] = []

    # 将新的用户消息添加到上下文
    sessions[session_id].append({"role": "user", "content": text})

    # 构建完整的对话上下文
    messages = [{"role": "system", "content": "你是面试智能助手，请把主要精力放在文本中的问题上，有限回答文本后面的问题"}] + sessions[session_id]

    response = client.chat.completions.create(
        model="ep-20240725171057-kfzcp",
        messages=messages,
        stream=True
    )

    async def send_message():
        for chunk in response:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            print(f"Received chunk: {content}")
            await websocket.send_text(f"INFERENCE:{content}")
            await asyncio.sleep(0)  # Yield control to the event loop

            # 将模型的回复添加到上下文
            sessions[session_id].append({"role": "assistant", "content": content})

    await send_message()

# 定义根路径的处理函数
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/前端加key.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# 激活key的端点
@app.post("/activate_key/", response_model=ActivationResponse)
async def activate_key(request: ActivationRequest):
    key = request.key
    try:
        with open("keys_durations.json", "r") as file:
            data = json.load(file)

        if (key in data):
            entry = data[key]
            duration = entry["duration"]
            expire_at = entry["expire_at"]

            if expire_at == "":
                expire_at = (datetime.now() + timedelta(seconds=duration)).isoformat()
                entry["expire_at"] = expire_at
                with open("keys_durations.json", "w") as file:
                    json.dump(data, file, indent=4)
            else:
                expire_at_dt = datetime.fromisoformat(expire_at)
                if datetime.now() > expire_at_dt:
                    raise HTTPException(status_code=403, detail="Key has expired.")

            return {"duration": duration, "expire_at": expire_at}
        else:
            raise HTTPException(status_code=404, detail="Key not found.")
    except Exception as e:
        print(f"Error activating key: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# WebSocket端点
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid4())  # 为每个连接生成唯一的会话 ID
    stt = SpeechToText(session_id, websocket)
    stt.start()
    try:
        while True:
            data = await websocket.receive_text()
            if data == "stop":
                stt.stop()
                break
            elif data:
                print(f"Received data for inference: {data}")
                asyncio.create_task(large_model_inference(session_id, data, websocket))
    except Exception as e:
        print(f"WebSocket错误: {e}")
        await websocket.send_text(f"WebSocket错误: {e}")
    finally:
        await websocket.close()

# 定义处理文本的API端点
@app.post("/process_text/")
async def process_text(request: InferenceRequest):
    try:
        text = request.text
        print(f"Received text: {text}")

        if not text:
            raise HTTPException(status_code=400, detail="Text is empty")

        return {"text": text}
    except Exception as e:
        print(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 启动API服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
