from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# 允许前端跨域访问（CORS）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    history: list = []


class ChatResponse(BaseModel):
    response: str


# ========== 模型加载（启动时加载一次）==========
print("正在加载模型，请稍候...")

# 模型路径 - 根据你的实际情况修改
MODEL_PATH = "/Users/mac/Public/Ai_agent/sft-dpo-train/qwen1.5b_muice_dpo_final"

# Mac M 系列芯片用 mps，Intel Mac 用 cpu
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"使用设备: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if DEVICE == "mps" else torch.float32,
    device_map=DEVICE if DEVICE == "mps" else None,
    trust_remote_code=True
)
model.eval()

print("模型加载完成！")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # 构建对话格式（Qwen 的 ChatML 格式）
        messages = [{"role": "user", "content": request.message}]

        # 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 编码输入
        model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

        # 生成回复
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )

        # 解码输出（去掉输入部分，只保留生成的内容）
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return ChatResponse(response=response_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型推理错误: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)