"""Task 7 - rerank candidates with Jina when configured, otherwise locally."""

import os
import unicodedata
from collections import Counter

import requests

try:
    from .task4_chunking_indexing import cosine_similarity, text_embedding, tokenize
except ImportError:
    from task4_chunking_indexing import cosine_similarity, text_embedding, tokenize


def _jina_remote_enabled() -> bool:
    return os.getenv("JINA_ENABLE_REMOTE", "").lower() in {"1", "true", "yes"}


def _normalize_tokens(text: str) -> set[str]:
    tokens = set()
    for token in tokenize(text):
        folded = unicodedata.normalize("NFD", token.lower())
        tokens.add("".join(ch for ch in folded if unicodedata.category(ch) != "Mn"))
    return tokens


def _lexical_relevance(query: str, document: str) -> float:
    query_tokens = _normalize_tokens(query)
    document_tokens = _normalize_tokens(document)
    if not query_tokens or not document_tokens:
        return 0.0
    overlap = len(query_tokens & document_tokens) / len(query_tokens)
    density = sum(Counter(_normalize_tokens(document))[t] for t in query_tokens & document_tokens)
    return overlap + min(density / 20.0, 0.5)


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """Rerank candidates with optional Jina API and local fallback."""
    if top_k <= 0 or not candidates:
        return []

    api_key = os.getenv("JINA_API_KEY", "")
    if api_key and _jina_remote_enabled():
        try:
            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": [c.get("content", "") for c in candidates],
                    "top_n": top_k,
                },
                timeout=30,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            return [
                {
                    **candidates[item["index"]],
                    "score": float(item.get("relevance_score", 0.0)),
                    "rerank_method": "jina-cross-encoder",
                }
                for item in results
            ]
        except Exception:
            pass

    reranked = []
    max_base = max((float(c.get("score", 0.0)) for c in candidates), default=1.0) or 1.0
    for candidate in candidates:
        base = float(candidate.get("score", 0.0)) / max_base
        relevance = _lexical_relevance(query, candidate.get("content", ""))
        item = {**candidate}
        item["score"] = 0.65 * relevance + 0.35 * base
        item["rerank_method"] = "local-token-overlap"
        reranked.append(item)
    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Select relevant but less redundant candidates with MMR."""
    selected: list[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < top_k:
        best_idx, best_score = remaining[0], float("-inf")
        for idx in remaining:
            embedding = candidates[idx].get("embedding", [])
            relevance = cosine_similarity(query_embedding, embedding)
            diversity_penalty = max(
                (
                    cosine_similarity(embedding, candidates[sel_idx].get("embedding", []))
                    for sel_idx in selected
                ),
                default=0.0,
            )
            score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if score > best_score:
                best_idx, best_score = idx, score
        item = {**candidates[best_idx], "score": float(best_score), "rerank_method": "mmr"}
        candidates[best_idx] = item
        selected.append(best_idx)
        remaining.remove(best_idx)
    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """Fuse ranked lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = item.get("content", "")
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    results = []
    max_score = max(scores.values(), default=1.0) or 1.0
    for content, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]:
        item = {
            **content_map[content],
            "score": float(score / max_score),
            "rerank_method": "rrf",
        }
        results.append(item)
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """Unified reranking interface."""
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        query_embedding = text_embedding(query)
        mmr_candidates = []
        for candidate in candidates:
            item = {**candidate}
            item.setdefault("embedding", text_embedding(item.get("content", "")))
            mmr_candidates.append(item)
        return rerank_mmr(query_embedding, mmr_candidates, top_k)
    elif method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
