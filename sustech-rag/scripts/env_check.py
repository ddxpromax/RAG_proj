from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sustech_rag.common.config import configure_model_cache, ensure_dirs, load_paths


def main() -> None:
    paths = load_paths()
    ensure_dirs(paths)
    configure_model_cache(paths)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Project root: {paths['project_root']}")
    print(f"Storage root: {paths['storage_root']}")
    for key in ("HF_HOME", "MODELSCOPE_CACHE", "TRANSFORMERS_CACHE"):
        print(f"{key}: {os.environ.get(key, '')}")
    for mod in ["httpx", "bs4", "fastapi", "gradio", "qdrant_client", "rank_bm25", "jieba", "fitz"]:
        print(f"{mod}: {'ok' if importlib.util.find_spec(mod) else 'missing'}")
    print(f"docker: {shutil.which('docker') or 'missing'}")


if __name__ == "__main__":
    main()

