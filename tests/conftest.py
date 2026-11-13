import os
import sys
import tempfile
from pathlib import Path

os.environ["NONO_DATA_DIR"] = tempfile.mkdtemp(prefix="nono-rag-test-")
os.environ["EMBEDDING_BACKEND"] = "hash"
os.environ["RAG_API_KEY"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
