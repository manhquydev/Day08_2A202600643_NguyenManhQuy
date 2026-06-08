"""RAG runner adapters used by evaluation configs."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from src.task10_generation import generate_with_citation


@dataclass(frozen=True)
class EvalConfig:
    name: str
    description: str
    use_reranking: bool
    top_k: int = 5
    score_threshold: float = 0.3


CONFIGS = [
    EvalConfig("hybrid_rerank", "Hybrid semantic + BM25, RRF merge, reranking on", True),
    EvalConfig("hybrid_no_rerank", "Hybrid semantic + BM25, RRF merge, reranking off", False),
]


def run_rag(question: str, config: EvalConfig, rag_pipeline: Any | None = None) -> dict:
    if rag_pipeline is None:
        return _invoke_configurable(generate_with_citation, question, config)
    if hasattr(rag_pipeline, "generate_with_citation"):
        return _invoke_configurable(rag_pipeline.generate_with_citation, question, config)
    if hasattr(rag_pipeline, "answer"):
        return _invoke_configurable(rag_pipeline.answer, question, config)
    if callable(rag_pipeline):
        return _invoke_configurable(rag_pipeline, question, config)
    raise TypeError("rag_pipeline must be callable or expose answer/generate_with_citation")


def _invoke_configurable(method: Any, question: str, config: EvalConfig) -> dict:
    kwargs = {
        "top_k": config.top_k,
        "use_reranking": config.use_reranking,
        "score_threshold": config.score_threshold,
    }
    signature = inspect.signature(method)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if accepts_kwargs:
        return method(question, **kwargs)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    if "use_reranking" not in supported:
        raise TypeError("RAG evaluation method must accept use_reranking for A/B comparison")
    return method(question, **supported)
