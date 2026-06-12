from __future__ import annotations

import httpx

from sustech_rag.common.config import PROJECT_ROOT, load_yaml


class LLMClient:
    def __init__(self, models_config: str = str(PROJECT_ROOT / "configs" / "models.yaml")) -> None:
        config = load_yaml(models_config)["generator"]
        self.api_base = config["api_base"].rstrip("/")
        self.api_key = config.get("api_key", "EMPTY")
        self.model = config["served_model_name"]
        self.temperature = float(config.get("temperature", 0.1))
        self.max_tokens = int(config.get("max_tokens", 800))

    def chat(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=120, trust_env=False) as client:
            response = client.post(f"{self.api_base}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"]
