"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import re
import zipfile
from html import unescape
from pathlib import Path

try:
    from markitdown import MarkItDown
except ImportError:  # MarkItDown is optional in the local harness.
    MarkItDown = None

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _safe_slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip())
    return re.sub(r"-+", "-", value).strip("-").lower() or "document"


def _docx_to_text(filepath: Path) -> str:
    with zipfile.ZipFile(filepath) as archive:
        xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<[^>]+>", "", xml)
    return re.sub(r"\n{3,}", "\n\n", unescape(xml)).strip()


def _html_to_text(filepath: Path) -> str:
    content = filepath.read_text(encoding="utf-8", errors="ignore")
    content = re.sub(r"<(script|style).*?</\1>", " ", content, flags=re.I | re.S)
    content = re.sub(r"</(p|h1|h2|h3|li)>", "\n", content, flags=re.I)
    content = re.sub(r"<[^>]+>", " ", content)
    return re.sub(r"\s+\n", "\n", unescape(content)).strip()


def _convert_file_text(filepath: Path) -> str:
    if MarkItDown is not None and filepath.suffix.lower() in {".pdf", ".docx", ".doc"}:
        try:
            return MarkItDown().convert(str(filepath)).text_content.strip()
        except Exception:
            pass

    suffix = filepath.suffix.lower()
    if suffix == ".docx":
        return _docx_to_text(filepath)
    if suffix in {".md", ".txt"}:
        return filepath.read_text(encoding="utf-8", errors="ignore").strip()
    if suffix == ".html":
        return _html_to_text(filepath)
    if suffix == ".json":
        data = json.loads(filepath.read_text(encoding="utf-8"))
        return data.get("content_markdown") or data.get("content") or ""
    raise ValueError(f"Unsupported file type: {filepath}")


def _write_markdown(output_path: Path, source: Path, title: str, body: str, extra: dict | None = None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source": source.name,
        "source_path": str(source.relative_to(LANDING_DIR)),
        **(extra or {}),
    }
    header = [f"# {title}", ""]
    for key, value in metadata.items():
        header.append(f"**{key}:** {value}")
    header.extend(["", "---", "", body.strip()])
    output_path.write_text("\n".join(header), encoding="utf-8")


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            text = _convert_file_text(filepath)
            if len(text) < 50:
                raise ValueError(f"Converted legal file too short: {filepath}")
            output_path = output_dir / f"{_safe_slug(filepath.stem)}.md"
            _write_markdown(
                output_path,
                filepath,
                filepath.stem.replace("-", " ").title(),
                text,
                {"type": "legal"},
            )
            print(f"  Saved: {output_path}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.iterdir():
        suffix = filepath.suffix.lower()
        if suffix in {".json", ".html", ".md", ".txt"}:
            print(f"Converting: {filepath.name}")
            output_path = output_dir / f"{_safe_slug(filepath.stem)}.md"
            if suffix != ".json":
                body = _convert_file_text(filepath)
                _write_markdown(
                    output_path,
                    filepath,
                    filepath.stem.replace("-", " ").title(),
                    body,
                    {"type": "news"},
                )
                print(f"  Saved: {output_path}")
                continue

            data = json.loads(filepath.read_text(encoding="utf-8"))
            body = data.get("content_markdown") or data.get("content") or ""
            if len(body) < 50:
                raise ValueError(f"Converted news file too short: {filepath}")
            _write_markdown(
                output_path,
                filepath,
                data.get("title", filepath.stem),
                body,
                {
                    "type": "news",
                    "url": data.get("url", ""),
                    "date_crawled": data.get("date_crawled", ""),
                },
            )
            print(f"  Saved: {output_path}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\nDone! Output tai:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
