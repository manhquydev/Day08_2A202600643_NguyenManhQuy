"""Task 10 - generate evidence-bound answers with citations."""

import os
import re

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:
    from task9_retrieval_pipeline import retrieve

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Place high-score chunks at prompt edges to reduce lost-in-middle risk."""
    if len(chunks) <= 2:
        return list(chunks)

    head = [chunks[i] for i in range(0, len(chunks), 2)]
    tail = [chunks[i] for i in range(1, len(chunks), 2)]
    return head + list(reversed(tail))

def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks with source labels for citation."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source") or metadata.get("url") or f"Source {i}"
        doc_type = metadata.get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type} | "
            f"Score: {float(chunk.get('score', 0.0)):.3f}]\n{chunk.get('content', '')}"
        )
    return "\n\n---\n\n".join(context_parts)


def _citation_for_chunk(chunk: dict) -> str:
    metadata = chunk.get("metadata", {})
    source = metadata.get("source") or metadata.get("url") or "Nguon khong ro"
    year_match = re.search(r"(20\d{2}|19\d{2})", " ".join(str(v) for v in metadata.values()))
    year = year_match.group(1) if year_match else "n.d."
    return f"[{source}, {year}]"


def _fallback_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "I cannot verify this information"

    statements = []
    for chunk in chunks[:3]:
        content = re.sub(r"\s+", " ", chunk.get("content", "")).strip()
        if not content:
            continue
        sentence_match = re.search(r"(.{80,260}?[.!?])\s", content + " ")
        excerpt = sentence_match.group(1) if sentence_match else content[:240]
        statements.append(f"- {excerpt.strip()} {_citation_for_chunk(chunk)}")

    if not statements:
        return "I cannot verify this information"
    return (
        f"Cau hoi: {query}\n"
        "Thong tin co the xac minh tu context hien co:\n"
        + "\n".join(statements)
    )


def _openai_remote_enabled() -> bool:
    return os.getenv("OPENAI_ENABLE_REMOTE", "").lower() in {"1", "true", "yes"}

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Run retrieval, context ordering, and cited generation."""
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and not api_key.startswith("sk-xxx") and _openai_remote_enabled():
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content or ""
        except Exception:
            answer = _fallback_answer(query, reordered)
    else:
        answer = _fallback_answer(query, reordered)

    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": reordered[0].get("source", "none") if reordered else "none",
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
