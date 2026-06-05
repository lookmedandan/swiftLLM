from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # vLLM 不需要真实 API key
)

response = client.chat.completions.create(
    model="qwen-muice",
    messages=[
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "你好"}
    ],
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)