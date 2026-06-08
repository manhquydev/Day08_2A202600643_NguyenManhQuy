"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import json
from functools import lru_cache

try:
    from .task4_chunking_indexing import (
        INDEX_PATH,
        chunk_documents,
        cosine_similarity,
        embed_chunks,
        load_documents,
        text_embedding,
    )
except ImportError:
    from task4_chunking_indexing import (
        INDEX_PATH,
        chunk_documents,
        cosine_similarity,
        embed_chunks,
        load_documents,
        text_embedding,
    )


@lru_cache(maxsize=1)
def _embedded_corpus() -> tuple[dict, ...]:
    # Load from persisted index when available to avoid re-embedding on every import.
    if INDEX_PATH.exists():
        return tuple(json.loads(INDEX_PATH.read_text(encoding="utf-8")))
    docs = load_documents()
    chunks = chunk_documents(docs)
    return tuple(embed_chunks(chunks))


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if top_k <= 0 or not query.strip():
        return []

    query_embedding = text_embedding(query)
    results = []
    for chunk in _embedded_corpus():
        score = cosine_similarity(query_embedding, chunk.get("embedding", []))
        if score <= 0:
            continue
        results.append(
            {
                "content": chunk["content"],
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
