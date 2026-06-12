from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sustech_rag.indexing.bm25 import build_bm25

build_bm25()

