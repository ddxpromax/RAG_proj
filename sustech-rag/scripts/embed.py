from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sustech_rag.indexing.qdrant_index import build_qdrant_placeholder

build_qdrant_placeholder()

