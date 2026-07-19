import sys
from pathlib import Path

import os


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
os.environ.setdefault("EMBEDDING_PROVIDER", "hash")
