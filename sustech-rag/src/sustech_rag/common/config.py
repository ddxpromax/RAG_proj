from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATHS = PROJECT_ROOT / "configs" / "paths.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def expand_templates(value: Any, mapping: dict[str, Any]) -> Any:
    if isinstance(value, str):
        out = value
        for key, replacement in mapping.items():
            if isinstance(replacement, (str, int, float)):
                out = out.replace("${" + key + "}", str(replacement))
        return os.path.expandvars(out)
    if isinstance(value, dict):
        local = {**mapping, **{k: v for k, v in value.items() if isinstance(v, str)}}
        return {k: expand_templates(v, local) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_templates(v, mapping) for v in value]
    return value


def load_paths(path: str | Path = DEFAULT_PATHS) -> dict[str, Any]:
    raw = load_yaml(path)
    root_mapping = {
        "project_root": raw.get("project_root", str(PROJECT_ROOT)),
        "storage_root": raw.get("storage_root", "/root/autodl-fs/sustech-rag"),
    }
    return expand_templates(raw, root_mapping)


def ensure_dirs(paths: dict[str, Any] | None = None) -> None:
    paths = paths or load_paths()
    candidates: list[str] = []
    for section in ("data", "models", "indexes"):
        value = paths.get(section, {})
        if isinstance(value, dict):
            candidates.extend(str(v) for v in value.values())
    for key in ("logs", "cache"):
        if paths.get(key):
            candidates.append(str(paths[key]))
    for item in candidates:
        Path(item).mkdir(parents=True, exist_ok=True)


def configure_model_cache(paths: dict[str, Any] | None = None) -> None:
    paths = paths or load_paths()
    models = paths.get("models", {})
    if models.get("hf_home"):
        os.environ.setdefault("HF_HOME", str(models["hf_home"]))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(models["hf_home"]))
    if models.get("modelscope_cache"):
        os.environ.setdefault("MODELSCOPE_CACHE", str(models["modelscope_cache"]))

