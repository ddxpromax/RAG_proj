from __future__ import annotations

import argparse

from sustech_rag.common.config import configure_model_cache, ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser(prog="sustech-rag")
    parser.add_argument("command", choices=["init-dirs", "env"])
    args = parser.parse_args()
    if args.command == "init-dirs":
        ensure_dirs()
        configure_model_cache()
        print("Initialized SUSTech RAG directories and model cache environment.")
    elif args.command == "env":
        configure_model_cache()
        print("Model cache environment configured.")


if __name__ == "__main__":
    main()

