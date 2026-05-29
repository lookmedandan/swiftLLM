from ..backend_agent.customer_service_agent import CustomerServiceAgent

agent = CustomerServiceAgent()
response = agent.chat("我的订单什么时候发货？")
print(response)

# 处理投诉
response = agent.handle_complaint("产品质量太差了！")
print(response)