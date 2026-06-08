"""Task 4 - load markdown, chunk, embed, and persist a local index."""

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
INDEX_PATH = INDEX_DIR / "local_chunks.json"

# Recursive character chunking is resilient for mixed legal/news markdown where
# headings are inconsistent after DOCX conversion.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Local hashed token embeddings keep the classroom harness deterministic without
# requiring model downloads; this can be swapped for BAAI/bge-m3 in production.
EMBEDDING_MODEL = "local-hashed-token-embedding-v1"
EMBEDDING_DIM = 256

VECTOR_STORE = "local-json"  # "local-json" | "weaviate" | "chromadb" | "faiss"


def load_documents() -> list[dict]:
    """Read markdown files from data/standardized."""
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            continue
        rel_path = md_file.relative_to(STANDARDIZED_DIR)
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "source_path": str(rel_path).replace("\\", "/"),
                    "type": rel_path.parts[0] if len(rel_path.parts) > 1 else "unknown",
                },
            }
        )
    return documents


def _split_long_text(text: str) -> list[str]:
    parts = re.split(r"(\. |\n|; |, )", text)
    chunks, current = [], ""
    for i in range(0, len(parts), 2):
        piece = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
        if len(current) + len(piece) <= CHUNK_SIZE:
            current += piece
        else:
            if current.strip():
                chunks.append(current.strip())
            current = piece
            while len(current) > CHUNK_SIZE:
                chunks.append(current[:CHUNK_SIZE].strip())
                current = current[CHUNK_SIZE - CHUNK_OVERLAP :]
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _chunk_text(text: str) -> list[str]:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    chunks, current = [], ""
    for block in blocks:
        if len(block) > CHUNK_SIZE:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_long_text(block))
            continue

        candidate = f"{current}\n\n{block}".strip() if current else block
        if len(candidate) <= CHUNK_SIZE:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            overlap = current[-CHUNK_OVERLAP:] if current else ""
            current = f"{overlap}\n{block}".strip()
            if len(current) > CHUNK_SIZE:
                chunks.extend(_split_long_text(current))
                current = ""
    if current:
        chunks.append(current.strip())
    return [chunk[:CHUNK_SIZE] for chunk in chunks if chunk.strip()]


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk documents with metadata preserved."""
    chunks = []
    for doc_index, doc in enumerate(documents):
        for chunk_index, chunk_text in enumerate(_chunk_text(doc.get("content", ""))):
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "doc_index": doc_index,
                        "chunk_index": chunk_index,
                    },
                }
            )
    return chunks


def tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)
    tokens = []
    for token in raw_tokens:
        normalized = unicodedata.normalize("NFD", token)
        folded = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
        tokens.append(token)
        if folded != token:
            tokens.append(folded)
    return tokens


def text_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    vector = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Attach deterministic local embeddings to chunks."""
    embedded = []
    for chunk in chunks:
        item = {**chunk}
        item["embedding"] = text_embedding(chunk.get("content", ""))
        embedded.append(item)
    return embedded


def index_to_vectorstore(chunks: list[dict]):
    """Persist chunks to the configured local vector store."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    return INDEX_PATH


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
