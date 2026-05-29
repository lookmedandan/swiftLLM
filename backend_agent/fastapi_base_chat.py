from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    history: list = []


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # 调用 vLLM 服务
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "qwen-muice-dpo",
            "messages": [{"role": "user", "content": request.message}],
            "temperature": 0.7,
            "max_tokens": 512
        }
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="模型调用失败")

    result = response.json()
    return ChatResponse(response=result["choices"][0]["message"]["content"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)