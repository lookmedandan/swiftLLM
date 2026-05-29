from ..backend_agent.kbqa_agent import RAGAgent

rag = RAGAgent()

# 添加知识文档
rag.add_document("我们公司的工作时间是周一至周五 9:00-18:00")
rag.add_document("退换货政策：7天无理由退换")

# 提问
response = rag.answer("你们什么时候上班？")
print(response)