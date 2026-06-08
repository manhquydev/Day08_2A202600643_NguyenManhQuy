"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import math
import unicodedata
from collections import Counter
from functools import lru_cache

try:
    from .task4_chunking_indexing import chunk_documents, load_documents, tokenize
except ImportError:
    from task4_chunking_indexing import chunk_documents, load_documents, tokenize

CORPUS: list[dict] = []


def _normalize_token(token: str) -> str:
    decomposed = unicodedata.normalize("NFD", token.lower())
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _tokens(text: str) -> list[str]:
    return [_normalize_token(token) for token in tokenize(text)]


class LocalBM25:
    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.tokenized_corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.avgdl = (
            sum(len(doc) for doc in tokenized_corpus) / len(tokenized_corpus)
            if tokenized_corpus
            else 0.0
        )
        document_frequency: Counter[str] = Counter()
        for doc in tokenized_corpus:
            document_frequency.update(set(doc))
        doc_count = len(tokenized_corpus)
        self.idf = {
            term: max(0.0, math.log((doc_count - freq + 0.5) / (freq + 0.5) + 1.0))
            for term, freq in document_frequency.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = []
        for doc in self.tokenized_corpus:
            frequencies = Counter(doc)
            doc_len = len(doc) or 1
            score = 0.0
            for term in query_tokens:
                tf = frequencies.get(term, 0)
                if not tf:
                    continue
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1))
                score += self.idf.get(term, 0.0) * (tf * (self.k1 + 1)) / denominator
            scores.append(score)
        return scores


@lru_cache(maxsize=1)
def _corpus() -> tuple[dict, ...]:
    chunks = chunk_documents(load_documents())
    return tuple(chunks)


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [_tokens(doc.get("content", "")) for doc in corpus]
    return LocalBM25(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if top_k <= 0 or not query.strip():
        return []

    corpus = list(_corpus())
    if not corpus:
        return []

    bm25 = build_bm25_index(corpus)
    scores = bm25.get_scores(_tokens(query))
    ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

    results = []
    for idx in ranked_indices:
        if scores[idx] <= 0:
            continue
        results.append(
            {
                "content": corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": corpus[idx].get("metadata", {}),
            }
        )
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
