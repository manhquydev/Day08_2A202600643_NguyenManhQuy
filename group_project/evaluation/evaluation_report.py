"""Markdown export for RAG evaluation results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def export_results_to_markdown(
    results: dict,
    comparison: dict,
    configs: list,
    metric_names: list[str],
    results_path: Path,
) -> None:
    config_a, config_b = configs[0].name, configs[1].name
    summary_a = comparison[config_a]["summary"]
    summary_b = comparison[config_b]["summary"]
    warnings = sorted({w for result in comparison.values() for w in result["warnings"]})
    all_cases = [
        {**case, "config": name}
        for name, result in comparison.items()
        for case in result["cases"]
    ]
    bottom_cases = sorted(all_cases, key=lambda case: case["average"])[:3]

    lines = [
        "# RAG Evaluation Results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Framework sử dụng",
        "",
        f"- Primary framework implemented: {results['framework']}",
        f"- Framework/backend used for reported scores: {results['metric_backend']}",
        "- Command: `python group_project/evaluation/eval_pipeline.py`",
        f"- Golden dataset: {summary_a['questions_run']} questions",
        "",
        "## A/B Configs",
        "",
        "| Config | Description | top_k | score_threshold |",
        "|---|---|---:|---:|",
    ]
    for config in configs:
        lines.append(
            f"| `{config.name}` | {config.description} | {config.top_k} | {config.score_threshold:.2f} |"
        )

    lines.extend([
        "",
        "## Overall Scores",
        "",
        "| Metric | Config A hybrid_rerank | Config B hybrid_no_rerank | Delta A-B |",
        "|---|---:|---:|---:|",
    ])
    for metric in metric_names + ["average"]:
        delta = summary_a[metric] - summary_b[metric]
        lines.append(f"| {metric} | {_fmt(summary_a[metric])} | {_fmt(summary_b[metric])} | {_fmt(delta)} |")

    winner = config_a if summary_a["average"] >= summary_b["average"] else config_b
    lines.extend([
        "",
        "## A/B Comparison Analysis",
        "",
        f"Config tốt hơn theo average: `{winner}`.",
        "Config A bật reranking sau RRF merge; Config B giữ cùng hybrid retrieval nhưng tắt reranking để đo tác động riêng của bước rerank.",
        "",
        "## Worst Performers (Bottom 3)",
        "",
        "| # | Config | Question | Faithfulness | Relevance | Recall | Precision | Failure Stage | Root Cause |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ])
    for index, case in enumerate(bottom_cases, start=1):
        stage, cause = _failure_stage(case["metrics"], metric_names)
        question = case["question"].replace("|", " ")[:90]
        metrics = case["metrics"]
        lines.append(
            f"| {index} | `{case['config']}` | {question} | {_fmt(metrics['faithfulness'])} | "
            f"{_fmt(metrics['answer_relevance'])} | {_fmt(metrics['context_recall'])} | "
            f"{_fmt(metrics['context_precision'])} | {stage} | {cause} |"
        )

    lines.extend([
        "",
        "## Recommendations",
        "",
        "### Cải tiến 1",
        "Action: bổ sung metadata điều/khoản, heading và source path khi chunk corpus.",
        "Expected impact: tăng context recall cho câu hỏi pháp luật chi tiết.",
        "",
        "### Cải tiến 2",
        "Action: tinh chỉnh reranker local bằng feature runtime như query-token overlap, source type và citation density.",
        "Expected impact: tăng context precision cho dataset nhỏ.",
        "",
        "### Cải tiến 3",
        "Action: chạy DeepEval remote khi có API key để có LLM-as-judge score chính thức.",
        "Expected impact: giảm phụ thuộc heuristic offline.",
    ])
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(["", "## Unresolved Questions", "", "- None."])
    results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _failure_stage(metrics: dict, metric_names: list[str]) -> tuple[str, str]:
    weakest = min(metric_names, key=lambda metric: metrics[metric])
    reasons = {
        "faithfulness": ("Generation grounding", "Answer tokens weakly supported by retrieved context"),
        "answer_relevance": ("Answer relevance", "Answer misses expected answer terms"),
        "context_recall": ("Retrieval recall", "Retrieved context misses expected evidence"),
        "context_precision": ("Retrieval precision", "Many retrieved chunks are off-target"),
    }
    return reasons[weakest]


def _fmt(score: float) -> str:
    return f"{score:.3f}"
