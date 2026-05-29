from ..backend_agent.multi_agent_system import MultiAgentSystem

system = MultiAgentSystem()

# 注册 Agent
system.register_agent("sales", "你是销售专家...")
system.register_agent("support", "你是技术支持...")

# 自动路由任务
response = system.route_task("我想了解一下你们的产品")
print(response)