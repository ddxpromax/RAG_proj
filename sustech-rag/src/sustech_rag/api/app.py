from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from sustech_rag.generation.rag import RAGService
from sustech_rag.retrieval.service import RetrievalService

app = FastAPI(title="SUSTech Campus RAG", version="0.1.0")
retrieval_service = RetrievalService()
rag_service = RAGService(retrieval_service)


class ChatRequest(BaseModel):
    question: str
    mode: str = "bm25"
    use_llm: bool = True


class RetrieveRequest(BaseModel):
    question: str
    mode: str = "bm25"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "sustech-rag"}


@app.post("/retrieve")
def retrieve(req: RetrieveRequest) -> dict:
    result = retrieval_service.retrieve(req.question, req.mode)
    return {
        "hits": [hit.model_dump(mode="json") for hit in result["hits"]],
        "trace": result["trace"],
    }


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    return rag_service.answer(req.question, mode=req.mode, use_llm=req.use_llm).model_dump(mode="json")


@app.post("/chat/no-rag")
def chat_no_rag(req: ChatRequest) -> dict:
    return rag_service.answer(req.question, mode="no_rag", use_llm=req.use_llm).model_dump(mode="json")
