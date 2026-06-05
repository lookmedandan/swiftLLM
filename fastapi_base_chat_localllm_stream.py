"""
FastAPI 后端 - 本地模型 + RAG + 流式输出 + 多对话管理 + 持久化日志
"""
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch
import json
import os
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from rag.rag_engine import rag
from rag.document_loader import DocumentLoader, load_folder

# ========== 路径配置 ==========
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = BASE_DIR.parent / "frontend"
CONVERSATIONS_DIR = BASE_DIR / "conversations"
CONVERSATIONS_DIR.mkdir(exist_ok=True)

# 模型路径（相对 backend 目录）
MODEL_PATH = str(BASE_DIR.parent / "sft-dpo-train" / "qwen1.5b_muice_dpo_final")

# ========== FastAPI 应用 ==========
app = FastAPI(title="AI Agent with RAG", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 托管前端静态文件 ==========
if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
    print(f"✓ 前端目录已挂载: {FRONTEND_DIR}")
else:
    print(f"⚠ 前端目录不存在: {FRONTEND_DIR}")

@app.get("/")
async def root():
    return RedirectResponse(url="/frontend/rag_markdown.html")

# ========== 对话管理器 ==========
class ConversationManager:
    def __init__(self):
        self.conversations: Dict[str, Dict] = {}
        self._load_all()

    def _load_all(self):
        for conv_file in CONVERSATIONS_DIR.glob("*.json"):
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    conv = json.load(f)
                    self.conversations[conv['id']] = conv
            except Exception as e:
                print(f"加载对话失败 {conv_file}: {e}")
        print(f"已加载 {len(self.conversations)} 个历史对话")

    def create(self, title: str = "新对话") -> str:
        conv_id = f"conv_{uuid.uuid4().hex[:8]}"
        conversation = {
            "id": conv_id,
            "title": title,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": [],
            "search_logs": []
        }
        self.conversations[conv_id] = conversation
        self._save(conv_id)
        return conv_id

    def get(self, conv_id: str) -> Optional[Dict]:
        return self.conversations.get(conv_id)

    def add_message(self, conv_id: str, role: str, content: str, sources: List[Dict] = None, log_id: str = None):
        conv = self.conversations.get(conv_id)
        if not conv:
            return
        message = {
            "id": f"msg_{len(conv['messages'])}",
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "sources": sources or [],
            "log_id": log_id
        }
        conv['messages'].append(message)
        conv['updated_at'] = datetime.now().isoformat()
        self._save(conv_id)

    def add_search_log(self, conv_id: str, log_entry: Dict) -> str:
        conv = self.conversations.get(conv_id)
        if not conv:
            return ""
        log_id = f"log_{len(conv['search_logs'])}_{uuid.uuid4().hex[:4]}"
        log_entry['id'] = log_id
        log_entry['timestamp'] = datetime.now().isoformat()
        conv['search_logs'].append(log_entry)
        conv['updated_at'] = datetime.now().isoformat()
        self._save(conv_id)
        return log_id

    def get_search_log(self, conv_id: str, log_id: str) -> Optional[Dict]:
        conv = self.conversations.get(conv_id)
        if not conv:
            return None
        for log in conv['search_logs']:
            if log['id'] == log_id:
                return log
        return None

    def list_conversations(self) -> List[Dict]:
        return [
            {
                "id": c['id'],
                "title": c['title'],
                "created_at": c['created_at'],
                "updated_at": c['updated_at'],
                "message_count": len(c['messages'])
            }
            for c in sorted(self.conversations.values(), key=lambda x: x['updated_at'], reverse=True)
        ]

    def delete(self, conv_id: str):
        if conv_id in self.conversations:
            del self.conversations[conv_id]
            file_path = CONVERSATIONS_DIR / f"{conv_id}.json"
            if file_path.exists():
                file_path.unlink()

    def update_title(self, conv_id: str, title: str):
        conv = self.conversations.get(conv_id)
        if conv:
            conv['title'] = title
            conv['updated_at'] = datetime.now().isoformat()
            self._save(conv_id)

    def _save(self, conv_id: str):
        conv = self.conversations.get(conv_id)
        if conv:
            file_path = CONVERSATIONS_DIR / f"{conv_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conv, f, ensure_ascii=False, indent=2)

conv_manager = ConversationManager()

# ========== 模型加载 ==========
print("=" * 50)
print("正在初始化 AI Agent...")
print("=" * 50)

print(f"模型路径: {MODEL_PATH}")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"使用设备: {DEVICE}")

print("加载 LLM 模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if DEVICE == "mps" else torch.float32,
    device_map=DEVICE if DEVICE == "mps" else None,
    trust_remote_code=True
)
model.eval()
print("✓ LLM 模型加载完成")

print("初始化 RAG 知识库...")
print(f"✓ 知识库就绪，共 {len(rag.documents)} 条文档")
print("=" * 50)
print("系统初始化完成！")
print("=" * 50)

# ========== 数据模型 ==========
class ChatRequest(BaseModel):
    message: str
    use_rag: bool = True
    top_k: int = 3

class CreateConversationRequest(BaseModel):
    title: Optional[str] = "新对话"

class UpdateTitleRequest(BaseModel):
    title: str

class DocumentRequest(BaseModel):
    documents: List[str]
    sources: Optional[List[str]] = None

# ========== 对话管理接口 ==========

@app.post("/conversations")
async def create_conversation(request: CreateConversationRequest = None):
    title = request.title if request else "新对话"
    conv_id = conv_manager.create(title)
    return {"id": conv_id, "title": title}

@app.get("/conversations")
async def list_conversations():
    return {"conversations": conv_manager.list_conversations()}

@app.get("/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    conv = conv_manager.get(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="对话不存在")
    return conv

@app.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    conv_manager.delete(conv_id)
    return {"message": "对话已删除"}

@app.put("/conversations/{conv_id}/title")
async def update_title(conv_id: str, request: UpdateTitleRequest):
    conv = conv_manager.get(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="对话不存在")
    conv_manager.update_title(conv_id, request.title)
    return {"message": "标题已更新"}

@app.get("/conversations/{conv_id}/logs")
async def get_conversation_logs(conv_id: str):
    conv = conv_manager.get(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="对话不存在")
    return {"logs": conv['search_logs']}

@app.get("/conversations/{conv_id}/log/{log_id}")
async def get_conversation_log_detail(conv_id: str, log_id: str):
    log = conv_manager.get_search_log(conv_id, log_id)
    if not log:
        raise HTTPException(status_code=404, detail="日志不存在")
    return log

# ========== RAG 管理接口 ==========

@app.get("/rag/stats")
async def get_stats():
    return rag.get_stats()

@app.post("/rag/documents")
async def add_documents(request: DocumentRequest):
    sources = request.sources or [""] * len(request.documents)
    rag.add_documents(texts=request.documents, sources=sources)
    return {
        "message": f"成功添加 {len(request.documents)} 条文档",
        "total": len(rag.documents),
        "stats": rag.get_stats()
    }

@app.post("/rag/upload")
async def upload_file(file: UploadFile = File(...), chunk_size: int = 300):
    temp_path = f"./temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    loader = DocumentLoader()
    ext = os.path.splitext(file.filename)[1].lower()

    try:
        if ext == '.txt':
            docs = loader.load_txt(temp_path)
        elif ext == '.md':
            docs = loader.load_md(temp_path)
        elif ext == '.pdf':
            docs = loader.load_pdf(temp_path)
        elif ext == '.docx':
            docs = loader.load_docx(temp_path)
        else:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

        for doc in docs:
            rag.add_text_chunks(text=doc["content"], chunk_size=chunk_size, source=file.filename)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            "message": f"文件 '{file.filename}' 上传成功",
            "total": len(rag.documents),
            "stats": rag.get_stats()
        }
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/search")
async def search_documents(query: str, top_k: int = 3):
    results = rag.search(query, top_k=top_k)
    return {"query": query, "results": results, "total_docs": len(rag.documents)}

@app.delete("/rag/documents")
async def clear_documents():
    rag.clear()
    return {"message": "知识库已清空"}

@app.post("/rag/load-directory")
async def load_directory(path: str = "./knowledge_base"):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"目录不存在: {path}")
    count = load_folder(path, rag_instance=rag)
    return {
        "message": f"从 '{path}' 加载了 {count} 条文档",
        "stats": rag.get_stats()
    }

# ========== 聊天接口 ==========

@app.get("/chat/stream")
async def chat_stream(
    message: str,
    conv_id: Optional[str] = None,
    use_rag: bool = True,
    top_k: int = 3
):
    if not conv_id:
        conv_id = conv_manager.create("新对话")

    conv = conv_manager.get(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="对话不存在")

    print(f"\n[流式请求] conv={conv_id} | {message[:50]}...")

    conv_manager.add_message(conv_id, "user", message)

    contexts = []
    log_id = None

    if use_rag:
        start_time = time.time()
        vector_results = rag._vector_search(message, top_k=top_k)
        keyword_results = rag._keyword_search(message, top_k=top_k)
        fused = rag._reciprocal_rank_fusion(vector_results, keyword_results)
        contexts = rag._rerank(message, fused, top_k=top_k)
        duration = (time.time() - start_time) * 1000

        log_entry = {
            "query": message,
            "vector_results": [
                {"doc_id": doc_id, "score": round(score, 4),
                 "content": rag.documents[doc_id].content[:100] + "..."}
                for doc_id, score in vector_results[:5]
            ],
            "keyword_results": [
                {"doc_id": doc_id, "score": round(score, 4),
                 "content": rag.documents[doc_id].content[:100] + "..."}
                for doc_id, score in keyword_results[:5]
            ],
            "fused_results": [
                {"doc_id": doc_id, "rrf_score": round(score, 4)}
                for doc_id, score in fused[:5]
            ],
            "final_results": contexts,
            "duration_ms": round(duration, 2)
        }
        log_id = conv_manager.add_search_log(conv_id, log_entry)
        print(f"  检索耗时: {duration:.2f}ms, log_id={log_id}")

    prompt = build_prompt(message, contexts)

    def generate():
        full_response = ""

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = dict(
            input_ids=inputs.input_ids,
            streamer=streamer,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        yield f"data: {json.dumps({'conv_id': conv_id, 'log_id': log_id}, ensure_ascii=False)}\n\n"

        if contexts:
            yield f"data: {json.dumps({'sources': contexts, 'log_id': log_id}, ensure_ascii=False)}\n\n"

        for new_text in streamer:
            if new_text:
                full_response += new_text
                data = json.dumps({"token": new_text}, ensure_ascii=False)
                yield f"data: {data}\n\n"

        conv_manager.add_message(conv_id, "assistant", full_response, contexts, log_id)

        yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
        thread.join()
        print("  生成完成")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/chat")
async def chat(request: ChatRequest):
    contexts = []
    if request.use_rag:
        contexts = rag.search(request.message, top_k=request.top_k)

    prompt = build_prompt(request.message, contexts)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return {"response": response, "sources": contexts}

# ========== 辅助函数 ==========

def build_prompt(message: str, contexts: List[Dict]) -> str:
    if contexts:
        context_text = "\n\n".join([
            f"[参考 {i+1}] {ctx['content'][:400]}"
            for i, ctx in enumerate(contexts)
        ])

        return f"""你是一个专业的智能客服助手。请根据以下参考资料回答用户问题。
如果参考资料不足以回答，请基于你的知识回答，但要明确说明。

参考资料：
{context_text}

用户问题：{message}

请用中文简洁回答："""
    else:
        return message

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)