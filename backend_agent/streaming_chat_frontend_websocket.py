from fastapi import FastAPI, WebSocket
import requests
import json

app = FastAPI()


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    while True:
        message = await websocket.receive_text()

        # 调用 vLLM 流式接口
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "qwen-muice-dpo",
                "messages": [{"role": "user", "content": message}],
                "stream": True,
                "temperature": 0.7,
                "max_tokens": 512
            },
            stream=True
        )

        # 发送流式响应
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8').replace('data: ', ''))
                if 'choices' in data and len(data['choices']) > 0:
                    delta = data['choices'][0].get('delta', {})
                    if 'content' in delta:
                        await websocket.send_text(delta['content'])

        await websocket.send_text("[DONE]")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8081)