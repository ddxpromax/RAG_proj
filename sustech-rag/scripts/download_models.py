from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sustech_rag.common.config import PROJECT_ROOT, configure_model_cache, ensure_dirs, load_yaml


def snapshot(model_id: str, target: Path, provider: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and any(target.iterdir()):
        print(f"Already exists: {target}")
        return target
    if provider == "modelscope":
        from modelscope import snapshot_download

        downloaded = Path(snapshot_download(model_id))
    else:
        from huggingface_hub import snapshot_download

        downloaded = Path(snapshot_download(repo_id=model_id))
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(downloaded, target)
    print(f"Downloaded {model_id} -> {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=["embedding", "reranker", "generator", "all"],
        default="embedding",
    )
    parser.add_argument("--provider", choices=["modelscope", "huggingface"], default=None)
    args = parser.parse_args()

    ensure_dirs()
    configure_model_cache()
    config = load_yaml(PROJECT_ROOT / "configs" / "models.yaml")
    targets = ["embedding", "reranker", "generator"] if args.target == "all" else [args.target]
    for name in targets:
        item = config[name]
        provider = args.provider or item.get("provider", "modelscope")
        snapshot(item["model_id"], Path(item["local_path"]), provider)


if __name__ == "__main__":
    main()

