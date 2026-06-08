from group_project.evaluation import eval_pipeline, evaluation_rag_runner


def test_golden_dataset_has_required_shape():
    dataset = eval_pipeline.load_golden_dataset()
    eval_pipeline.validate_golden_dataset(dataset)
    assert len(dataset) >= 15
    assert all(eval_pipeline.REQUIRED_KEYS <= set(item) for item in dataset)


def test_local_metrics_are_populated_from_rag_output():
    item = {
        "question": "MDMA thuộc nhóm nào?",
        "expected_answer": "MDMA nằm trong Danh mục I nhóm IB các chất hướng thần.",
        "expected_context": "quy-dinh-cac-danh-muc-chat-cam-va-ten-chat.md Danh mục I IB",
    }
    answer = "MDMA nằm trong Danh mục I nhóm IB các chất hướng thần."
    contexts = ["Danh mục I IB. MDMA là chất hướng thần."]
    metrics = eval_pipeline.local_metrics(item, answer, contexts, [{"metadata": {"source": "doc.md"}}])
    assert set(metrics) == set(eval_pipeline.METRIC_NAMES)
    assert all(0.0 <= score <= 1.0 for score in metrics.values())
    assert metrics["answer_relevance"] > 0
    assert metrics["context_recall"] > 0


def test_compare_configs_uses_real_reranking_toggle(monkeypatch):
    seen = []

    def fake_run_rag(question, config, rag_pipeline=None):
        seen.append(config.use_reranking)
        suffix = "rerank" if config.use_reranking else "no rerank"
        return {
            "answer": f"answer {suffix}",
            "sources": [
                {
                    "content": f"{question} answer {suffix}",
                    "metadata": {"source": f"{suffix}.md"},
                }
            ],
        }

    monkeypatch.setattr(eval_pipeline, "_run_rag", fake_run_rag)
    dataset = eval_pipeline.load_golden_dataset()
    comparison = eval_pipeline.compare_configs(None, dataset)

    assert set(comparison) == {"hybrid_rerank", "hybrid_no_rerank"}
    assert True in seen and False in seen
    assert comparison["hybrid_rerank"]["summary"]["questions_run"] == len(dataset)
    assert comparison["hybrid_no_rerank"]["summary"]["questions_run"] == len(dataset)


def test_run_rag_passes_config_to_generation(monkeypatch):
    seen = []

    def fake_generate(question, top_k, use_reranking, score_threshold):
        seen.append(
            {
                "question": question,
                "top_k": top_k,
                "use_reranking": use_reranking,
                "score_threshold": score_threshold,
            }
        )
        return {"answer": "ok", "sources": []}

    monkeypatch.setattr(evaluation_rag_runner, "generate_with_citation", fake_generate)
    for config in eval_pipeline.CONFIGS:
        evaluation_rag_runner.run_rag("test question", config)

    assert [item["use_reranking"] for item in seen] == [True, False]
    assert all(item["top_k"] == 5 for item in seen)
    assert all(item["score_threshold"] == 0.3 for item in seen)


def test_export_results_writes_no_placeholders(monkeypatch, tmp_path):
    def fake_run_rag(question, config, rag_pipeline=None):
        return {
            "answer": "MDMA nằm trong Danh mục I nhóm IB các chất hướng thần.",
            "sources": [
                {
                    "content": "MDMA là chất hướng thần trong Danh mục I IB.",
                    "metadata": {"source": "quy-dinh-cac-danh-muc-chat-cam-va-ten-chat.md"},
                }
            ],
        }

    output_path = tmp_path / "results.md"
    monkeypatch.setattr(eval_pipeline, "_run_rag", fake_run_rag)
    monkeypatch.setattr(eval_pipeline, "RESULTS_PATH", output_path)
    dataset = eval_pipeline.load_golden_dataset()
    comparison = eval_pipeline.compare_configs(None, dataset)
    eval_pipeline.export_results(comparison["hybrid_rerank"], comparison)

    content = output_path.read_text(encoding="utf-8")
    assert "Framework/backend used for reported scores" in content
    assert "hybrid_no_rerank" in content
    assert "|  |" not in content
    assert "placeholder" not in content.lower()
