# RAG Evaluation Results

Generated: 2026-06-08 19:48:15

## Framework sử dụng

- Primary framework implemented: DeepEval
- Framework/backend used for reported scores: local_heuristic_fallback
- Command: `python group_project/evaluation/eval_pipeline.py`
- Golden dataset: 19 questions

## A/B Configs

| Config | Description | top_k | score_threshold |
|---|---|---:|---:|
| `hybrid_rerank` | Hybrid semantic + BM25, RRF merge, reranking on | 5 | 0.30 |
| `hybrid_no_rerank` | Hybrid semantic + BM25, RRF merge, reranking off | 5 | 0.30 |

## Overall Scores

| Metric | Config A hybrid_rerank | Config B hybrid_no_rerank | Delta A-B |
|---|---:|---:|---:|
| faithfulness | 0.885 | 0.875 | 0.010 |
| answer_relevance | 0.714 | 0.701 | 0.013 |
| context_recall | 0.858 | 0.830 | 0.028 |
| context_precision | 0.990 | 0.958 | 0.032 |
| average | 0.862 | 0.841 | 0.021 |

## A/B Comparison Analysis

Config tốt hơn theo average: `hybrid_rerank`.
Config A bật reranking sau RRF merge; Config B giữ cùng hybrid retrieval nhưng tắt reranking để đo tác động riêng của bước rerank.

## Worst Performers (Bottom 3)

| # | Config | Question | Faithfulness | Relevance | Recall | Precision | Failure Stage | Root Cause |
|---|---|---|---:|---:|---:|---:|---|---|
| 1 | `hybrid_no_rerank` | Heroin được ghi trong danh mục chất ma túy như thế nào? | 0.826 | 0.429 | 0.407 | 1.000 | Retrieval recall | Retrieved context misses expected evidence |
| 2 | `hybrid_no_rerank` | Sơn Ngọc Minh được bài báo giới thiệu là ai trước khi bị bắt? | 0.812 | 0.648 | 0.834 | 0.600 | Retrieval precision | Many retrieved chunks are off-target |
| 3 | `hybrid_rerank` | Heroin được ghi trong danh mục chất ma túy như thế nào? | 0.779 | 0.486 | 0.651 | 1.000 | Answer relevance | Answer misses expected answer terms |

## Recommendations

### Cải tiến 1
Action: bổ sung metadata điều/khoản, heading và source path khi chunk corpus.
Expected impact: tăng context recall cho câu hỏi pháp luật chi tiết.

### Cải tiến 2
Action: tinh chỉnh reranker local bằng feature runtime như query-token overlap, source type và citation density.
Expected impact: tăng context precision cho dataset nhỏ.

### Cải tiến 3
Action: chạy DeepEval remote khi có API key để có LLM-as-judge score chính thức.
Expected impact: giảm phụ thuộc heuristic offline.

## Warnings

- DEEPEVAL_ENABLE_REMOTE is not enabled; used deterministic local fallback.

## Unresolved Questions

- None.
