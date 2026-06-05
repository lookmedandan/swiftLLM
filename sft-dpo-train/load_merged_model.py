from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载合并后的模型
model_path = "./qwen1.5b_muice_merged"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 或 torch.bfloat16
    device_map="auto"  # 自动分配 GPU/CPU
)

tokenizer = AutoTokenizer.from_pretrained(model_path)


# 单轮对话
def chat(message, max_new_tokens=512):
    messages = [{"role": "user", "content": message}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取 assistant 回复
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    return response


# 测试
print(chat("你好，请介绍一下自己"))