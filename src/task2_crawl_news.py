"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
import re
from html import unescape
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://thanhnien.vn/cong-an-tphcm-bat-ca-si-long-nhat-va-son-ngoc-minh-lien-quan-den-ma-tuy-185260520123807384.htm",
    "https://thanhnien.vn/ca-si-son-ngoc-minh-vua-bi-bat-vi-lien-quan-den-ma-tuy-la-ai-18526052012481811.htm",
    "https://thanhnien.vn/miu-le-va-loi-xin-loi-muon-mang-cua-loat-sao-viet-vuong-vao-ma-tuy-18526051513021689.htm",
    "https://thanhnien.vn/ca-si-long-nhat-bi-bat-showbiz-viet-lien-tiep-chan-dong-vi-ma-tuy-18526052013032001.htm",
    "https://thanhnien.vn/hanh-trinh-nghe-thuat-day-thi-phi-cua-long-nhat-truoc-khi-bi-bat-18526052017185611.htm",
]


def _strip_tags(value: str) -> str:
    value = re.sub(r"<(script|style).*?</\1>", " ", value, flags=re.I | re.S)
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", unescape(value)).strip()


def _metadata(html: str, key: str) -> str:
    patterns = [
        rf'<meta[^>]+property=["\']{re.escape(key)}["\'][^>]+content=["\'](.*?)["\']',
        rf'<meta[^>]+name=["\']{re.escape(key)}["\'][^>]+content=["\'](.*?)["\']',
    ]
    for pattern in patterns:
        match = re.search(pattern, html, flags=re.I | re.S)
        if match:
            return _strip_tags(match.group(1))
    return ""


def _extract_article(url: str, html: str) -> dict:
    title = _metadata(html, "og:title")
    if not title:
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
        title = _strip_tags(title_match.group(1)) if title_match else urlparse(url).path

    description = _metadata(html, "description")
    paragraphs = []
    for raw in re.findall(r"<p\b[^>]*>(.*?)</p>", html, flags=re.I | re.S):
        text = _strip_tags(raw)
        if len(text) > 40 and not text.lower().startswith(("image", "ảnh:")):
            paragraphs.append(text)

    content = "\n\n".join(dict.fromkeys(paragraphs))
    if description and description not in content:
        content = f"{description}\n\n{content}".strip()

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(timespec="seconds"),
        "content_markdown": f"# {title}\n\n{content}",
    }


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    headers = {"User-Agent": "Mozilla/5.0 compatible RAG classroom crawler"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    article = _extract_article(url, response.text)
    if len(article["content_markdown"]) < 500:
        raise ValueError(f"Article content too short after extraction: {url}")
    return article


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("Hay dien ARTICLE_URLS truoc khi chay!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
