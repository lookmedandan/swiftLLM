"""
api.py - RAG 后端 API
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag.rag_engine import rag

app = FastAPI()

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    results: list
    total: int

@app.post("/search", response_model=QueryResponse)
def search(req: QueryRequest):
    results = rag.search(req.query, top_k=req.top_k, score_threshold=0.5)
    return QueryResponse(results=results, total=len(results))

@app.get("/stats")
def stats():
    return rag.get_stats()

@app.post("/load")
def load_documents():
    from rag.document_loader import load_folder
    loaded = load_folder("./documents/pdfs/地理与旅游", rag)
    return {"loaded": len(loaded), "total": len(rag.documents)}