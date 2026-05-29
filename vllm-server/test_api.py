import requests

response = requests.post('http://localhost:8000/v1/chat/completions', json={
    "model": "qwen-muice-dpo",
    "messages": [{"role": "user", "content": "你好"}],
    "temperature": 0.7,
    "max_tokens": 512
})

print(response.json()['choices'][0]['message']['content'])