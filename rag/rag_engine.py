"""
本地 RAG 引擎 - 支持混合检索 + 重排序
"""
import os
import re
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# 路径配置
HERE = Path(os.path.dirname(os.path.abspath(__file__)))
EMBEDDING_MODEL = str(HERE.parent.parent / 'model' / 'bge-large-zh-v1.5')
RERANKER_MODEL = str(HERE.parent.parent / 'model' / 'bge-reranker-base')
PERSIST_DIR = str(HERE / 'rag_data')

# 如果本地模型不存在，回退到 HuggingFace
if not os.path.exists(EMBEDDING_MODEL):
    EMBEDDING_MODEL = 'BAAI/bge-large-zh-v1.5'
    print(f"本地 Embedding 模型不存在，将使用: {EMBEDDING_MODEL}")

if not os.path.exists(RERANKER_MODEL):
    RERANKER_MODEL = 'BAAI/bge-reranker-base'
    print(f"本地重排序模型不存在，将使用: {RERANKER_MODEL}")

# 重排序模型常量
CROSS_ENCODER_MODEL = RERANKER_MODEL

import jieba
from sentence_transformers import SentenceTransformer


class Document:
    def __init__(self, content: str, source: str = "", metadata: Dict = None):
        self.content = content
        self.source = source
        self.metadata = metadata or {}
        self.embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "source": self.source,
            "metadata": self.metadata
        }


def clean_source(filename: str) -> str:
    """清理文件名中的冗余后缀"""
    cleaned = re.sub(r'\s*\([^)]*z-library[^)]*\)', '', filename)
    cleaned = re.sub(r'\s*\([^)]*1lib[^)]*\)', '', cleaned)
    cleaned = re.sub(r'\s*\([^)]*z-lib[^)]*\)', '', cleaned)
    cleaned = re.sub(r'\.pdf\s+\([^)]+\)\.pdf', '.pdf', cleaned)
    return cleaned.strip()


class HybridRAG:
    """混合检索 RAG：向量检索 + 关键词检索 + 重排序"""

    def __init__(self,
                 embedding_model: str = EMBEDDING_MODEL,
                 persist_dir: str = PERSIST_DIR):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)

        self.embedder = self._load_embedder(embedding_model)
        self.reranker = self._load_reranker()
        self.documents: List[Document] = []
        self.embeddings: List[np.ndarray] = []
        self.inverted_index: Dict[str, List[int]] = {}
        self.doc_keywords: List[set] = []

        self._load()
        print(f"RAG 初始化完成，知识库共 {len(self.documents)} 条文档")

    def _load_embedder(self, model_name: str):
        print(f"正在加载 Embedding 模型: {model_name}...")
        embedder = SentenceTransformer(model_name)
        print("Embedding 模型加载完成！")
        return embedder

    def _load_reranker(self):
        """加载交叉编码器重排序模型"""
        try:
            from sentence_transformers import CrossEncoder
            print(f"正在加载重排序模型: {CROSS_ENCODER_MODEL}...")
            reranker = CrossEncoder(CROSS_ENCODER_MODEL)
            print("重排序模型加载完成！")
            return reranker
        except Exception as e:
            print(f"⚠ 重排序模型加载失败: {e}，将使用向量分数排序")
            return None

    def _load(self):
        """从磁盘加载知识库（JSON 格式）"""
        docs_file = self.persist_dir / "documents.json"
        embs_file = self.persist_dir / "embeddings.json"

        if docs_file.exists() and embs_file.exists():
            try:
                with open(docs_file, 'r', encoding='utf-8') as f:
                    docs_data = json.load(f)

                with open(embs_file, 'r') as f:
                    embs_data = json.load(f)

                self.documents = [Document(**item) for item in docs_data]
                self.embeddings = [np.array(emb) for emb in embs_data]

                # 重建关键词索引
                self._build_keyword_index()

                print(f"从磁盘加载了 {len(self.documents)} 条文档")
            except Exception as e:
                print(f"加载失败: {e}，将创建新知识库")

    def _save(self):
        """保存知识库到磁盘（JSON 格式）"""
        try:
            docs_data = [doc.to_dict() for doc in self.documents]
            with open(self.persist_dir / "documents.json", 'w', encoding='utf-8') as f:
                json.dump(docs_data, f, ensure_ascii=False, indent=2)

            embs_data = [emb.tolist() for emb in self.embeddings]
            with open(self.persist_dir / "embeddings.json", 'w') as f:
                json.dump(embs_data, f)

            print(f"知识库已保存，共 {len(self.documents)} 条文档")
        except Exception as e:
            print(f"保存失败: {e}")

    def _extract_keywords(self, text: str) -> set:
        """提取中文关键词"""
        words = jieba.lcut(text)
        keywords = {w.strip() for w in words
                   if len(w.strip()) > 1
                   and not w.strip().isdigit()
                   and not re.match(r'^[^\w]+$', w.strip())}
        return keywords

    def _build_keyword_index(self):
        """构建倒排索引"""
        self.inverted_index = {}
        self.doc_keywords = []

        for doc_id, doc in enumerate(self.documents):
            keywords = self._extract_keywords(doc.content)
            self.doc_keywords.append(keywords)

            for kw in keywords:
                if kw not in self.inverted_index:
                    self.inverted_index[kw] = []
                self.inverted_index[kw].append(doc_id)

    def add_documents(self, texts: List[str], sources: List[str] = None,
                      metadata_list: List[Dict] = None, auto_save: bool = True):
        if not texts:
            return

        sources = sources or [""] * len(texts)
        metadata_list = metadata_list or [{}] * len(texts)

        print(f"正在向量化 {len(texts)} 条文档...")
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)

        for text, source, meta, emb in zip(texts, sources, metadata_list, embeddings):
            doc = Document(content=text, source=source, metadata=meta)
            self.documents.append(doc)
            self.embeddings.append(emb)

        # 重建关键词索引
        self._build_keyword_index()

        print(f"已添加 {len(texts)} 条文档，知识库共 {len(self.documents)} 条")

        if auto_save:
            self._save()

    def add_text_chunks(self, text: str, chunk_size: int = 300,
                        chunk_overlap: int = 30, source: str = ""):
        source = clean_source(source)
        chunks = self._split_text(text, chunk_size, chunk_overlap)
        if not chunks:
            return

        self.add_documents(
            texts=chunks,
            sources=[source] * len(chunks),
            metadata_list=[{"chunk_index": i, "total_chunks": len(chunks)}
                           for i in range(len(chunks))],
            auto_save=False
        )
        self._save()

    def _split_text(self, text: str, chunk_size: int = 300, overlap: int = 30) -> List[str]:
        """按语义段落分块，优先在标题、空行处切分"""
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

        chunks = []
        current = []
        current_len = 0

        for para in paragraphs:
            if len(para) < 20 and not any(c in para for c in "。，；"):
                if current:
                    chunks.append("\n".join(current))
                    current = []
                    current_len = 0
                chunks.append(para)
                continue

            if current_len + len(para) < chunk_size:
                current.append(para)
                current_len += len(para)
            else:
                if current:
                    chunks.append("\n".join(current))
                current = current[-1:] if overlap > 0 else []
                current_len = sum(len(p) for p in current)
                current.append(para)
                current_len += len(para)

        if current:
            chunks.append("\n".join(current))

        return chunks

    # ========== 混合检索核心 ==========

    def _vector_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """向量检索"""
        if not self.documents:
            return []

        query_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(self.embeddings, query_emb)
        top_indices = np.argsort(scores)[-top_k*2:][::-1]

        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def _keyword_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """关键词检索（BM25 风格）"""
        if not self.inverted_index:
            return []

        query_keywords = self._extract_keywords(query)
        if not query_keywords:
            return []

        doc_hits: Dict[int, int] = {}
        for kw in query_keywords:
            if kw in self.inverted_index:
                for doc_id in self.inverted_index[kw]:
                    doc_hits[doc_id] = doc_hits.get(doc_id, 0) + 1

        results = []
        avg_doc_len = np.mean([len(kws) for kws in self.doc_keywords]) if self.doc_keywords else 1

        for doc_id, hits in doc_hits.items():
            if doc_id >= len(self.doc_keywords):
                continue

            doc_len = len(self.doc_keywords[doc_id])
            score = hits * (1 + np.log1p(avg_doc_len / max(doc_len, 1)))
            results.append((doc_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _reciprocal_rank_fusion(self,
                                vector_results: List[Tuple[int, float]],
                                keyword_results: List[Tuple[int, float]],
                                k: float = 60.0) -> List[Tuple[int, float]]:
        """RRF 融合：倒数排序融合"""
        scores: Dict[int, float] = {}

        for rank, (doc_id, _) in enumerate(vector_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        for rank, (doc_id, _) in enumerate(keyword_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return fused

    def _rerank(self, query: str, candidates: List[Tuple[int, float]], top_k: int = 5) -> List[Dict]:
        """交叉编码器重排序"""
        if not candidates:
            return []

        if self.reranker is None or len(candidates) <= 1:
            results = []
            for doc_id, score in candidates[:top_k]:
                if doc_id < len(self.documents):
                    doc = self.documents[doc_id]
                    results.append({
                        "content": doc.content,
                        "source": doc.source,
                        "score": round(score, 4),
                        "metadata": doc.metadata,
                        "type": "local",
                        "rank_method": "vector+keyword"
                    })
            return results

        # 准备重排序输入
        pairs = []
        doc_ids = []
        for doc_id, _ in candidates:
            if doc_id < len(self.documents):
                pairs.append([query, self.documents[doc_id].content[:512]])
                doc_ids.append(doc_id)

        # 重排序打分
        rerank_scores = self.reranker.predict(pairs)

        scored_results = list(zip(doc_ids, rerank_scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in scored_results[:top_k]:
            doc = self.documents[doc_id]
            results.append({
                "content": doc.content,
                "source": doc.source,
                "score": round(float(score), 4),
                "metadata": doc.metadata,
                "type": "local",
                "rank_method": "reranker"
            })

        return results

    def search(self,
               query: str,
               top_k: int = 5,
               score_threshold: float = 0.3,
               use_hybrid: bool = True) -> List[Dict]:
        """
        混合检索入口
        """
        if not self.documents or not query or not query.strip():
            return []

        query = str(query).strip()

        # 1. 向量检索
        vector_results = self._vector_search(query, top_k=top_k)
        print(f"  向量检索: {len(vector_results)} 条")

        if not use_hybrid:
            return self._rerank(query, vector_results, top_k=top_k)

        # 2. 关键词检索
        keyword_results = self._keyword_search(query, top_k=top_k)
        print(f"  关键词检索: {len(keyword_results)} 条")

        # 3. RRF 融合
        fused = self._reciprocal_rank_fusion(vector_results, keyword_results)
        print(f"  融合后: {len(fused)} 条")

        # 4. 重排序
        results = self._rerank(query, fused, top_k=top_k)

        # 过滤低分
        results = [r for r in results if r["score"] >= score_threshold]

        return results

    def delete_by_source(self, source: str):
        original = len(self.documents)
        keep = [i for i, doc in enumerate(self.documents) if doc.source != source]

        self.documents = [self.documents[i] for i in keep]
        self.embeddings = [self.embeddings[i] for i in keep]

        self._build_keyword_index()

        deleted = original - len(self.documents)
        if deleted > 0:
            self._save()
            print(f"已删除来源 '{source}' 的 {deleted} 条文档")
        return deleted

    def clear(self):
        """清空知识库"""
        self.documents = []
        self.embeddings = []
        self.inverted_index = {}
        self.doc_keywords = []

        for f in ["documents.json", "embeddings.json", "documents.pkl", "embeddings.pkl"]:
            file_path = self.persist_dir / f
            if file_path.exists():
                file_path.unlink()

        print("知识库已清空")

    def get_stats(self) -> Dict:
        sources = {}
        for doc in self.documents:
            sources[doc.source] = sources.get(doc.source, 0) + 1

        return {
            "total_documents": len(self.documents),
            "sources": sources,
            "persist_dir": str(self.persist_dir),
            "has_reranker": self.reranker is not None
        }


# 全局单例
rag = HybridRAG()