import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class RAGAgent:
    def __init__(self, vllm_url="http://localhost:8000"):
        self.vllm_url = vllm_url
        self.knowledge_base = []
        self.embeddings = []

    def add_document(self, text):
        """添加文档到知识库"""
        self.knowledge_base.append(text)
        # 这里应该调用 embedding 模型获取向量
        # 简化示例，实际需要用 embedding API

    def search(self, query, top_k=3):
        """检索相关知识"""
        # 简化实现，实际应该用向量数据库
        scores = []
        for doc in self.knowledge_base:
            # 计算相似度（简化）
            score = len(set(query) & set(doc)) / len(set(query) | set(doc))
            scores.append(score)

        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.knowledge_base[i] for i in top_indices]

    def answer(self, question):
        """基于知识库回答问题"""
        # 检索相关知识
        contexts = self.search(question)
        context_text = "\n".join(contexts)

        # 构建 prompt
        prompt = f"""基于以下知识回答问题：

{context_text}

问题：{question}
回答："""

        # 调用模型
        response = requests.post(
            f"{self.vllm_url}/v1/chat/completions",
            json={
                "model": "qwen-muice-dpo",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 512
            }
        )

        return response.json()["choices"][0]["message"]["content"]


# 使用示例
rag = RAGAgent()
rag.add_document("我们公司的工作时间是周一至周五 9:00-18:00")
rag.add_document("退换货政策：7天无理由退换")
response = rag.answer("你们什么时候上班？")
print(response)