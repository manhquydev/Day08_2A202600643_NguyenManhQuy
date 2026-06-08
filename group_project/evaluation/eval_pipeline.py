"""Executable RAG evaluation pipeline for the group submission."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .evaluation_metrics import (
        METRIC_NAMES,
        REQUIRED_KEYS,
        clip,
        local_metrics,
        source_label,
        summarize_cases,
        validate_golden_dataset,
    )
    from .evaluation_rag_runner import CONFIGS, EvalConfig, run_rag as _run_rag
    from .evaluation_report import export_results_to_markdown
except ImportError:
    from evaluation_metrics import (
        METRIC_NAMES,
        REQUIRED_KEYS,
        clip,
        local_metrics,
        source_label,
        summarize_cases,
        validate_golden_dataset,
    )
    from evaluation_rag_runner import CONFIGS, EvalConfig, run_rag as _run_rag
    from evaluation_report import export_results_to_markdown

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def load_golden_dataset() -> list[dict]:
    """Load golden dataset from JSON file."""
    return json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))


def _evaluate_local_cases(
    rag_pipeline: Any | None,
    golden_dataset: list[dict],
    config: EvalConfig,
) -> list[dict]:
    cases = []
    for item in golden_dataset:
        result = _run_rag(item["question"], config, rag_pipeline)
        sources = result.get("sources", []) or []
        contexts = [str(source.get("content", "")) for source in sources if source.get("content")]
        answer = str(result.get("answer", "")).strip()
        metrics = local_metrics(item, answer, contexts, sources)
        cases.append(
            {
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "expected_context": item["expected_context"],
                "answer": answer,
                "contexts": contexts,
                "sources": [source_label(source, i) for i, source in enumerate(sources, start=1)],
                "metrics": metrics,
                "average": clip(mean(metrics.values())),
            }
        )
    return cases


def _try_apply_deepeval(cases: list[dict]) -> tuple[str, list[str]]:
    if os.getenv("DEEPEVAL_ENABLE_REMOTE", "").lower() not in {"1", "true", "yes"}:
        return "local_heuristic_fallback", ["DEEPEVAL_ENABLE_REMOTE is not enabled; used deterministic local fallback."]
    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            FaithfulnessMetric,
        )
        from deepeval.test_case import LLMTestCase
    except Exception as exc:
        return "local_heuristic_fallback", [f"DeepEval import failed: {exc}. Used local fallback."]

    metric_factories = {
        "faithfulness": FaithfulnessMetric,
        "answer_relevance": AnswerRelevancyMetric,
        "context_recall": ContextualRecallMetric,
        "context_precision": ContextualPrecisionMetric,
    }
    warnings = []
    failures = 0
    for case in cases:
        test_case = LLMTestCase(
            input=case["question"],
            actual_output=case["answer"],
            expected_output=case["expected_answer"],
            retrieval_context=case["contexts"],
        )
        for metric_name, factory in metric_factories.items():
            try:
                metric = factory(threshold=0.7)
                metric.measure(test_case)
                case["metrics"][metric_name] = clip(metric.score or 0.0)
            except Exception as exc:
                failures += 1
                warnings.append(f"DeepEval {metric_name} failed for one case: {exc}")
        case["average"] = clip(mean(case["metrics"].values()))
    backend = "deepeval" if failures == 0 else "mixed_deepeval_local_fallback"
    return backend, warnings


def evaluate_with_deepeval(
    rag_pipeline: Any | None,
    golden_dataset: list[dict],
    config: EvalConfig | None = None,
) -> dict:
    """Selected evaluator: DeepEval, with explicit offline fallback."""
    validate_golden_dataset(golden_dataset)
    active_config = config or CONFIGS[0]
    cases = _evaluate_local_cases(rag_pipeline, golden_dataset, active_config)
    backend, warnings = _try_apply_deepeval(cases)
    return {
        "framework": "DeepEval",
        "metric_backend": backend,
        "config": active_config.__dict__,
        "summary": summarize_cases(cases),
        "cases": cases,
        "warnings": warnings,
    }


def compare_configs(rag_pipeline: Any | None, golden_dataset: list[dict]) -> dict:
    """Compare A/B configs on the full dataset."""
    return {
        config.name: evaluate_with_deepeval(rag_pipeline, golden_dataset, config=config)
        for config in CONFIGS
    }


def export_results(results: dict, comparison: dict) -> None:
    """Export evaluation results to results.md."""
    export_results_to_markdown(results, comparison, CONFIGS, METRIC_NAMES, RESULTS_PATH)


def main() -> int:
    dataset = load_golden_dataset()
    validate_golden_dataset(dataset)
    comparison = compare_configs(None, dataset)
    primary = comparison[CONFIGS[0].name]
    export_results(primary, comparison)
    print(
        "Evaluation complete: "
        f"{primary['summary']['questions_run']} questions x {len(CONFIGS)} configs; "
        f"report={RESULTS_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
