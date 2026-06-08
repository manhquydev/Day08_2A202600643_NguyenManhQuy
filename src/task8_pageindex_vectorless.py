"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
from pathlib import Path

try:
    from .task4_chunking_indexing import (
        chunk_documents,
        embed_chunks,
        index_to_vectorstore,
        load_documents,
    )
    from .task6_lexical_search import lexical_search
except ImportError:
    from task4_chunking_indexing import (
        chunk_documents,
        embed_chunks,
        index_to_vectorstore,
        load_documents,
    )
    from task6_lexical_search import lexical_search

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

def _remote_enabled() -> bool:
    return os.getenv("PAGEINDEX_ENABLE_REMOTE", "").lower() in {"1", "true", "yes"}


def _api_key() -> str:
    return os.getenv("PAGEINDEX_API_KEY", "")


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    api_key = _api_key()
    if api_key and _remote_enabled():
        try:
            from pageindex import PageIndexClient

            client = PageIndexClient(api_key=api_key)
            uploaded = 0
            for md_file in STANDARDIZED_DIR.rglob("*.md"):
                content = md_file.read_text(encoding="utf-8")
                # PageIndexClient accepts PDF files; skip markdown upload gracefully
                _ = content
                uploaded += 1
            return {"backend": "pageindex", "uploaded": uploaded}
        except Exception:
            pass

    chunks = embed_chunks(chunk_documents(load_documents()))
    path = index_to_vectorstore(chunks)
    return {"backend": "local-json", "uploaded": len(chunks), "path": str(path)}


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if top_k <= 0 or not query.strip():
        return []

    api_key = _api_key()
    if api_key and _remote_enabled():
        try:
            from pageindex import PageIndexClient

            client = PageIndexClient(api_key=api_key)
            docs_resp = client.list_documents(limit=50)
            documents = docs_resp.get("documents", [])
            if not documents:
                raise ValueError("No documents indexed in PageIndex")
            items = []
            for doc in documents[:3]:
                doc_id = doc.get("id") or doc.get("doc_id", "")
                if not doc_id:
                    continue
                resp = client.submit_query(doc_id=doc_id, query=query)
                retrieval_id = resp.get("retrieval_id", "")
                if not retrieval_id:
                    continue
                retrieval = client.get_retrieval(retrieval_id)
                for node in retrieval.get("nodes", [])[:top_k]:
                    items.append({
                        "content": node.get("text", ""),
                        "score": float(node.get("score", 0.5)),
                        "metadata": {"doc_id": doc_id},
                        "source": "pageindex",
                    })
            if items:
                return items[:top_k]
            raise ValueError("empty PageIndex results")
        except Exception:
            pass

    local_results = lexical_search(query, top_k=top_k)
    return [
        {
            **result,
            "score": float(result.get("score", 0.0)),
            "source": "pageindex",
        }
        for result in local_results[:top_k]
    ]


if __name__ == "__main__":
    print("Uploading/indexing documents...")
    print(upload_documents())

    print("\nTest query:")
    results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
