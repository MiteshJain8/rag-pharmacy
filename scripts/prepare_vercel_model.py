"""Bundle the embedding model so a cold function need not download it."""

from pathlib import Path

from fastembed import TextEmbedding

root = Path(__file__).resolve().parents[1]
TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=str(root / "model_cache"))
