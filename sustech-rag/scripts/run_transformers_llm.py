from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from sustech_rag.common.config import PROJECT_ROOT, configure_model_cache, load_yaml


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float = 0.1
    max_tokens: int = 800


def important_terms(text: str) -> list[str]:
    stop = {
        "南方科技大学",
        "南科大",
        "什么",
        "哪些",
        "有关",
        "相关",
        "根据",
        "官方",
        "资料",
        "怎么",
        "是否",
        "多少",
        "要点",
        "内容",
        "南方",
        "科技",
        "大学",
        "提供",
        "服务",
        "开放",
        "时间",
    }
    terms = re.findall(r"[A-Za-z]{2,}\d*|20[0-3]\d|[\u4e00-\u9fff]{2,}", text)
    return [term for term in dict.fromkeys(terms) if term not in stop][:8]


def best_snippet(question: str, body: str, limit: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", body).strip()
    if not cleaned:
        return ""
    terms = important_terms(question)
    sentences = [s.strip() for s in re.split(r"(?<=[。！？!?；;])\s*", cleaned) if len(s.strip()) >= 12]
    if not sentences:
        return cleaned[:limit] + ("..." if len(cleaned) > limit else "")
    scored = []
    for index, sentence in enumerate(sentences[:40]):
        score = sum(2 for term in terms if term in sentence)
        score += sum(1 for key in ["开放", "时间", "招生", "章程", "专业", "学分", "申请", "办理"] if key in question and key in sentence)
        score -= index * 0.02
        scored.append((score, sentence))
    selected = [sentence for _, sentence in sorted(scored, key=lambda item: item[0], reverse=True)[:2]]
    snippet = " ".join(selected).strip() or cleaned
    return snippet[:limit] + ("..." if len(snippet) > limit else "")


def has_transformers_weights(model_path: str) -> bool:
    path = Path(model_path)
    if not path.exists():
        return False
    return any((path / name).exists() for name in ["model.safetensors", "pytorch_model.bin"]) or bool(
        list(path.glob("*.safetensors"))
    )


def evidence_generate(messages: list[dict], max_tokens: int = 800) -> str:
    user_content = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
    question_match = re.search(r"问题[:：]\s*(.+?)(?:\n|$)", user_content)
    question = question_match.group(1).strip() if question_match else "用户问题"
    blocks = re.findall(r"\[(\d+)\]\n标题[:：](.*?)\n.*?内容[:：](.*?)(?=\n\n\[\d+\]|\Z)", user_content, flags=re.S)
    if not blocks:
        return "当前没有提供可用证据，无法根据官方资料回答。"
    lines = [f"针对“{question}”，根据提供的官方资料可以回答如下："]
    for source_id, title, body in blocks[:3]:
        snippet = best_snippet(question, body)
        lines.append(f"- {snippet} [{source_id}]")
    lines.append("以上结论均来自检索证据；如证据不足，应以官方原文为准。")
    return "\n".join(lines)[: max_tokens * 3]


def create_extractive_app() -> FastAPI:
    app = FastAPI(title="Local Evidence Generator OpenAI-compatible Service")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "backend": "extractive"}

    @app.post("/v1/chat/completions")
    def chat(req: ChatCompletionRequest) -> dict:
        text = evidence_generate([m.model_dump() for m in req.messages], max_tokens=req.max_tokens)
        return {
            "id": "local-evidence-chat",
            "object": "chat.completion",
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        }

    return app


def create_transformers_app(model_path: str) -> FastAPI:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    app = FastAPI(title="Local Transformers OpenAI-compatible LLM")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "backend": "transformers", "model_path": model_path}

    @app.post("/v1/chat/completions")
    def chat(req: ChatCompletionRequest) -> dict:
        messages = [m.model_dump() for m in req.messages]
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                do_sample=req.temperature > 0,
                temperature=max(req.temperature, 1e-5),
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        return {
            "id": "local-transformers-chat",
            "object": "chat.completion",
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--backend", choices=["auto", "transformers", "extractive"], default=None)
    args = parser.parse_args()
    configure_model_cache()
    model_config = load_yaml(PROJECT_ROOT / "configs" / "models.yaml")["generator"]
    model_path = args.model_path or model_config["local_path"]
    backend = args.backend or model_config.get("backend", "auto")
    if backend == "extractive" or (backend == "auto" and not has_transformers_weights(model_path)):
        app = create_extractive_app()
    else:
        app = create_transformers_app(model_path)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
