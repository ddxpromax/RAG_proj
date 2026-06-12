from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sustech_rag.common.config import configure_model_cache, ensure_dirs

ensure_dirs()
configure_model_cache()
print("Directories initialized.")

