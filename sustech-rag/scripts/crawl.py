from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sustech_rag.ingestion.crawler import crawl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sources.yaml")
    parser.add_argument("--source", action="append", help="Limit to one source_id; can be repeated.")
    args = parser.parse_args()
    asyncio.run(crawl(args.config, limit_sources=args.source))


if __name__ == "__main__":
    main()

