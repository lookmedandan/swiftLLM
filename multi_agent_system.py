import requests
from typing import List, Dict


class MultiAgentSystem:
    def __init__(self, vllm_url="http://localhost:8000"):
        self.vllm_url = vllm_url
        self.agents = {}

    def register_agent(self, name, system_prompt, skills=None):
        """注册 Agent"""
        self.agents[name] = {
            "system_prompt": system_prompt,
            "skills": skills or []
        }

    def call_agent(self, agent_name, task, context=None):
        """调用特定 Agent"""
        agent = self.agents.get(agent_name)
        if not agent:
            return f"Agent {agent_name} 不存在"

        messages = [{"role": "system", "content": agent["system_prompt"]}]

        if context:
            messages.append({"role": "user", "content": f"上下文：{context}"})

        messages.append({"role": "user", "content": task})

        response = requests.post(
            f"{self.vllm_url}/v1/chat/completions",
            json={
                "model": "qwen-muice-dpo",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 512
            }
        )

        return response.json()["choices"][0]["message"]["content"]

    def route_task(self, user_input):
        """任务路由"""
        # 分析用户意图，路由到合适的 Agent
        routing_prompt = f"""分析用户意图，选择最合适的 Agent：
可选 Agent：{list(self.agents.keys())}
用户输入：{user_input}
请只返回 Agent 名称："""

        response = requests.post(
            f"{self.vllm_url}/v1/chat/completions",
            json={
                "model": "qwen-muice-dpo",
                "messages": [{"role": "user", "content": routing_prompt}],
                "temperature": 0.1,
                "max_tokens": 50
            }
        )

        agent_name = response.json()["choices"][0]["message"]["content"].strip()
        return self.call_agent(agent_name, user_input)


# 使用示例
system = MultiAgentSystem()

# 注册多个 Agent
system.register_agent(
    "sales",
    "你是销售专家，擅长产品推荐和销售技巧。",
    ["产品推荐", "价格谈判", "促销活动"]
)

system.register_agent(
    "support",
    "你是技术支持，擅长解决产品使用问题。",
    ["故障排查", "使用指导", "技术咨询"]
)

system.register_agent(
    "hr",
    "你是 HR 助手，擅长人事相关事务。",
    ["招聘咨询", "员工福利", "培训安排"]
)

# 路由任务
response = system.route_task("我想了解一下你们的产品")
print(response)