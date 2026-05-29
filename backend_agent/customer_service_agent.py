import requests
import json


class CustomerServiceAgent:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.system_prompt = """你是一个专业的客服助手。请根据用户的问题提供准确、友好的回答。
        如果问题超出你的知识范围，请礼貌地告知用户你会转接人工客服。"""

    def chat(self, user_message, history=None):
        messages = [{"role": "system", "content": self.system_prompt}]

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_message})

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": "qwen-muice-dpo",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 512
            }
        )

        return response.json()["choices"][0]["message"]["content"]

    def handle_complaint(self, complaint):
        """处理投诉"""
        prompt = f"用户投诉：{complaint}\n请给出安抚和解决方案："
        return self.chat(prompt)

    def handle_inquiry(self, question):
        """处理咨询"""
        prompt = f"用户咨询：{question}\n请详细解答："
        return self.chat(prompt)


# 使用示例
agent = CustomerServiceAgent()
response = agent.chat("我的订单什么时候发货？")
print(response)