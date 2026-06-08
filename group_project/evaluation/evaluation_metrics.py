"""Deterministic metric helpers for offline RAG evaluation."""

from __future__ import annotations

import re
import unicodedata
from statistics import mean

REQUIRED_KEYS = {"question", "expected_answer", "expected_context"}
METRIC_NAMES = [
    "faithfulness",
    "answer_relevance",
    "context_recall",
    "context_precision",
]

STOPWORDS = {
    "la", "va", "cua", "co", "the", "duoc", "trong", "theo", "ve", "cho",
    "cac", "nhung", "mot", "nguoi", "nay", "do", "khi", "voi", "tu", "bi",
    "da", "de", "hoac", "khong", "phai", "tai", "nam", "dieu", "khoan",
}


def validate_golden_dataset(dataset: list[dict]) -> None:
    if len(dataset) < 15:
        raise ValueError(f"golden_dataset.json needs >=15 cases, found {len(dataset)}")
    for index, item in enumerate(dataset, start=1):
        missing = REQUIRED_KEYS - set(item)
        if missing:
            raise ValueError(f"Case {index} missing keys: {sorted(missing)}")
        empty = [key for key in REQUIRED_KEYS if not str(item.get(key, "")).strip()]
        if empty:
            raise ValueError(f"Case {index} has empty fields: {empty}")


def clip(score: float) -> float:
    return max(0.0, min(1.0, round(float(score), 4)))


def source_label(source: dict, index: int) -> str:
    metadata = source.get("metadata", {})
    return str(
        metadata.get("source")
        or metadata.get("source_path")
        or metadata.get("path")
        or source.get("source")
        or f"source-{index}"
    )


def local_metrics(item: dict, answer: str, contexts: list[str], sources: list[dict]) -> dict:
    context_text = "\n".join(contexts)
    expected = item["expected_answer"]
    expected_context = item["expected_context"]
    gold = f"{item['question']} {expected} {expected_context}"
    relevant_contexts = [
        ctx for ctx in contexts if _coverage(ctx, gold) >= 0.08 or _coverage(ctx, expected) >= 0.15
    ]
    source_text = " ".join(source_label(src, i) for i, src in enumerate(sources, start=1))
    context_recall = 0.75 * _coverage(context_text, expected) + 0.25 * max(
        _coverage(context_text, expected_context),
        _coverage(source_text, expected_context),
    )
    return {
        "faithfulness": clip(_coverage(context_text, answer)),
        "answer_relevance": clip(0.8 * _coverage(answer, expected) + 0.2 * _coverage(answer, item["question"])),
        "context_recall": clip(context_recall),
        "context_precision": clip(len(relevant_contexts) / len(contexts) if contexts else 0.0),
    }


def summarize_cases(cases: list[dict]) -> dict:
    if not cases:
        raise ValueError("No evaluation cases produced")
    return {metric: clip(mean(case["metrics"][metric] for case in cases)) for metric in METRIC_NAMES} | {
        "average": clip(mean(case["average"] for case in cases)),
        "questions_run": len(cases),
    }


def _coverage(text: str, reference: str) -> float:
    reference_tokens = _tokens(reference)
    if not reference_tokens:
        return 0.0
    return len(_tokens(text) & reference_tokens) / len(reference_tokens)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\w]+", _fold(text), flags=re.UNICODE)
        if len(token) > 1 and token not in STOPWORDS
    }


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
