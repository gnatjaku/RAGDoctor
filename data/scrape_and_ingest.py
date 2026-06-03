#!/usr/bin/env python3
"""
Crawler Medonet category pages, scrape linked articles, run a dry-run analysis,
and optionally ingest them into RAGDoctor with auto-sized chunking.

Usage examples:
    python data/scrape_and_ingest.py --dry-run
    python data/scrape_and_ingest.py --url https://www.medonet.pl/choroby-od-a-do-z,kategoria,195.html --ingest
    python data/scrape_and_ingest.py --ingest --chunk-size 12000 --chunk-overlap 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import unicodedata
from collections import deque
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError


DEFAULT_URL = "https://www.medonet.pl/choroby-od-a-do-z,kategoria,195.html"
DEFAULT_API = "http://localhost:8000"
DEFAULT_MAX_EMBEDDING_CHUNK_SIZE = 6000
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
}
CONTENT_TAGS = ("p", "h2", "h3", "h4", "li", "blockquote")
DROP_TAGS = ("script", "style", "figure", "aside", "nav", "form", "noscript", "svg")


@dataclass(slots=True)
class ScrapedDocument:
    url: str
    title: str
    text: str
    char_count: int


@dataclass(slots=True)
class MongoLookupConfig:
    uri: str
    db_name: str
    collection_name: str


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    clean = parsed._replace(fragment="", query="")
    return urlunparse(clean)


def fetch_soup(url: str, session: requests.Session) -> BeautifulSoup:
    response = session.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.content, "lxml", from_encoding="utf-8")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def resolve_links(page_url: str, soup: BeautifulSoup) -> Iterable[str]:
    for tag in soup.select("a[href]"):
        href = tag.get("href", "").strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        yield normalize_url(urljoin(page_url, href))


def looks_like_article(url: str, seed_host: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()

    if parsed.netloc and parsed.netloc != seed_host:
        return False
    if not path.endswith(".html"):
        return False
    if "kategoria" in path or "/tagi/" in path or "/szukaj/" in path or "/autorzy" in path:
        return False
    return True


def discover_article_links(
    seed_url: str,
    session: requests.Session,
    *,
    max_depth: int,
    max_pages: int,
) -> list[str]:
    seed_url = normalize_url(seed_url)
    seed_host = urlparse(seed_url).netloc
    queue = deque([(seed_url, 0)])
    visited_pages: set[str] = set()
    discovered_articles: set[str] = set()

    while queue and len(visited_pages) < max_pages:
        current_url, depth = queue.popleft()
        if current_url in visited_pages:
            continue

        visited_pages.add(current_url)
        print(f"[crawl] depth={depth} page={current_url}")

        try:
            soup = fetch_soup(current_url, session)
        except Exception as exc:
            print(f"[crawl] skip {current_url} ({exc})", file=sys.stderr)
            continue

        for candidate in resolve_links(current_url, soup):
            parsed = urlparse(candidate)
            if parsed.netloc != seed_host:
                continue

            if looks_like_article(candidate, seed_host):
                discovered_articles.add(candidate)

            if depth < max_depth and candidate not in visited_pages:
                queue.append((candidate, depth + 1))

    return sorted(discovered_articles)


def extract_title(soup: BeautifulSoup) -> str:
    for selector in (
        "meta[property='og:title']",
        "h1.ods-article-header__title",
        "h1.article-title",
        "h1.title",
        "[class*='article'] h1",
        "main h1",
        "h1",
        "title",
    ):
        tag = soup.select_one(selector)
        if not tag:
            continue
        value = tag.get("content") or tag.get_text(" ", strip=True)
        if value:
            return normalize_text(value)
    return ""


def extract_text_parts(soup: BeautifulSoup) -> list[str]:
    parts: list[str] = []

    for block in soup.find_all("div", class_="ods-a-body-text"):
        for tag in block(DROP_TAGS):
            tag.decompose()
        text = block.get_text(" ", strip=True)
        if text:
            parts.append(text)
    if parts:
        return parts

    for selector in (
        "article.ods-article-body",
        "div.article-body",
        "div.article__body",
        "div.article-content",
        "article",
        "main",
    ):
        container = soup.select_one(selector)
        if not container:
            continue
        for tag in container(DROP_TAGS):
            tag.decompose()
        parts = [
            element.get_text(" ", strip=True)
            for element in container.find_all(CONTENT_TAGS)
            if element.get_text(" ", strip=True)
        ]
        if parts:
            return parts

    body = soup.find("body")
    if body:
        for tag in body(("script", "style", "nav", "footer", "header", "noscript", "svg")):
            tag.decompose()
        text = body.get_text(" ", strip=True)
        if text:
            return [text]

    return []


def scrape_article(url: str, session: requests.Session) -> ScrapedDocument | None:
    soup = fetch_soup(url, session)
    title = extract_title(soup)
    text = normalize_text(" ".join(extract_text_parts(soup)))

    if len(text) < 200:
        return None

    return ScrapedDocument(
        url=url,
        title=title or url,
        text=text,
        char_count=len(text),
    )


def round_up(value: int, step: int) -> int:
    return int(math.ceil(value / step) * step)


def choose_chunk_size(docs: list[ScrapedDocument], buffer_chars: int) -> int:
    max_chars = max(doc.char_count for doc in docs)
    return max(800, round_up(max_chars + buffer_chars, 128))


def choose_chunk_overlap(chunk_size: int, explicit_overlap: int | None) -> int:
    if explicit_overlap is not None:
        return min(explicit_overlap, chunk_size - 1)
    if chunk_size <= 2048:
        return min(120, chunk_size - 1)
    return min(200, max(0, chunk_size // 20), chunk_size - 1)


def estimate_chunks(char_count: int, chunk_size: int, chunk_overlap: int) -> int:
    if char_count <= chunk_size:
        return 1
    effective_step = max(1, chunk_size - chunk_overlap)
    return 1 + math.ceil((char_count - chunk_size) / effective_step)


def source_id_for(url: str) -> str:
    return "medonet-" + hashlib.md5(url.encode("utf-8")).hexdigest()[:12]


def source_name_for(title: str, url: str) -> str:
    base = re.sub(r"[^\w\-.]", "_", title).strip("._")
    if not base:
        base = source_id_for(url)
    return base[:120] + ".txt"


def build_mongo_lookup_config() -> MongoLookupConfig | None:
    load_dotenv()

    uri = (
        os.getenv("MONGODB_URI")
        or os.getenv("MONGO_URI")
        or os.getenv("MONGODB_URL")
    )
    db_name = os.getenv("MONGODB_DB") or os.getenv("MONGO_DB_NAME")
    collection_name = os.getenv("MONGODB_COLLECTION") or os.getenv("MONGO_COLLECTION_NAME")

    if not uri or not db_name or not collection_name:
        return None

    return MongoLookupConfig(
        uri=uri.strip().strip('"').strip("'"),
        db_name=db_name,
        collection_name=collection_name,
    )


def source_exists(source_id: str, lookup_config: MongoLookupConfig) -> bool:
    client = MongoClient(
        lookup_config.uri,
        serverSelectionTimeoutMS=3000,
        connectTimeoutMS=3000,
        socketTimeoutMS=3000,
    )
    try:
        collection = client[lookup_config.db_name][lookup_config.collection_name]
        return collection.count_documents({"source_id": source_id}, limit=1) > 0
    except PyMongoError as exc:
        raise RuntimeError(f"Mongo lookup failed for skip-already-ingested: {exc}") from exc
    finally:
        client.close()


def dry_run_report(
    docs: list[ScrapedDocument],
    *,
    chunk_size: int,
    chunk_overlap: int,
    max_embedding_chunk_size: int,
) -> None:
    total_chars = sum(doc.char_count for doc in docs)
    max_chars = max(doc.char_count for doc in docs)
    min_chars = min(doc.char_count for doc in docs)

    print("\n[dry-run] Summary")
    print(f"[dry-run] articles: {len(docs)}")
    print(f"[dry-run] total chars: {total_chars}")
    print(f"[dry-run] min chars/article: {min_chars}")
    print(f"[dry-run] max chars/article: {max_chars}")
    print(f"[dry-run] recommended chunk_size: {chunk_size}")
    print(f"[dry-run] recommended chunk_overlap: {chunk_overlap}")
    print(f"[dry-run] embedding-safe chunk cap: {max_embedding_chunk_size}")

    multi_chunk = 0
    for index, doc in enumerate(sorted(docs, key=lambda item: item.char_count, reverse=True), start=1):
        effective_chunk_size = min(chunk_size, max_embedding_chunk_size)
        effective_chunk_overlap = min(chunk_overlap, choose_chunk_overlap(effective_chunk_size, None))
        chunks = estimate_chunks(doc.char_count, effective_chunk_size, effective_chunk_overlap)
        if chunks > 1:
            multi_chunk += 1
        print(
            f"[dry-run] {index:03d}. chars={doc.char_count:5d} "
            f"chunks={chunks:2d} title={doc.title[:90]} url={doc.url}"
        )

    print(f"[dry-run] articles fitting in one chunk: {len(docs) - multi_chunk}/{len(docs)}")


def ingest_documents(
    docs: list[ScrapedDocument],
    *,
    api_base: str,
    chunk_size: int,
    chunk_overlap: int,
    max_embedding_chunk_size: int,
    skip_already_ingested: bool,
) -> list[dict[str, str | int]]:
    ingest_url = f"{api_base.rstrip('/')}/ingest"
    failures: list[dict[str, str | int]] = []
    lookup_config = build_mongo_lookup_config() if skip_already_ingested else None

    if skip_already_ingested and lookup_config is None:
        raise RuntimeError(
            "skip-already-ingested requires MONGODB_URI/MONGODB_DB/MONGODB_COLLECTION "
            "(or compatible MONGO_* variables) in the environment"
        )

    for index, doc in enumerate(docs, start=1):
        print(f"[ingest] {index}/{len(docs)} {doc.url}")
        doc_source_id = source_id_for(doc.url)
        if lookup_config and source_exists(doc_source_id, lookup_config):
            print(f"[ingest] skipped existing source_id={doc_source_id}")
            continue

        attempt_sizes: list[int] = []
        current_size = min(chunk_size, max_embedding_chunk_size)

        while True:
            if current_size not in attempt_sizes:
                attempt_sizes.append(current_size)
            if current_size <= 1000:
                break
            current_size = max(1000, round_up(current_size // 2, 250))

        last_error = ""
        for attempt_size in attempt_sizes:
            attempt_overlap = min(chunk_overlap, choose_chunk_overlap(attempt_size, None))
            payload = {
                "source_id": doc_source_id,
                "source_name": source_name_for(doc.title, doc.url),
                "text": doc.text,
                "metadata": {
                    "url": doc.url,
                    "title": doc.title,
                    "source_type": "web_article",
                    "site": urlparse(doc.url).netloc,
                    "char_count": doc.char_count,
                },
                "chunk_size": attempt_size,
                "chunk_overlap": attempt_overlap,
            }
            response = requests.post(
                ingest_url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json; charset=utf-8"},
                timeout=600,
            )
            if response.ok:
                print(f"[ingest] result={response.json()}")
                break

            last_error = response.text.strip()
            print(
                f"[ingest] failed chunk_size={attempt_size} chunk_overlap={attempt_overlap} "
                f"status={response.status_code} body={last_error}"
            )
        else:
            failures.append(
                {
                    "url": doc.url,
                    "title": doc.title,
                    "char_count": doc.char_count,
                    "error": last_error or "unknown ingest error",
                }
            )

    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Follow links from a Medonet page, scrape article content, and optionally ingest it."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Seed URL to crawl.")
    parser.add_argument("--max-depth", type=int, default=0, help="How many additional link levels to follow beyond the seed page.")
    parser.add_argument("--max-pages", type=int, default=200, help="Maximum number of pages to fetch while discovering links.")
    parser.add_argument("--chunk-buffer", type=int, default=64, help="Extra characters added above the longest article for auto chunk sizing.")
    parser.add_argument("--chunk-size", default="auto", help="'auto' or an integer chunk size in characters.")
    parser.add_argument("--chunk-overlap", default="auto", help="'auto' or an integer overlap in characters.")
    parser.add_argument(
        "--max-embedding-chunk-size",
        type=int,
        default=DEFAULT_MAX_EMBEDDING_CHUNK_SIZE,
        help=f"Upper bound for chunk size sent to the embedding backend (default: {DEFAULT_MAX_EMBEDDING_CHUNK_SIZE}).",
    )
    parser.add_argument("--api", default=DEFAULT_API, help=f"RAGDoctor API base URL (default: {DEFAULT_API}).")
    parser.add_argument(
        "--skip-already-ingested",
        action="store_true",
        help="Skip documents whose source_id already exists in MongoDB.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only crawl and report sizes. This is the default behavior.")
    parser.add_argument("--ingest", action="store_true", help="Ingest after the dry-run analysis finishes.")
    parser.add_argument("--min-chars", type=int, default=500, help="Minimum extracted text length to treat a page as an article.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    session = requests.Session()

    print(f"[start] seed={args.url}")
    article_links = discover_article_links(
        seed_url=args.url,
        session=session,
        max_depth=args.max_depth,
        max_pages=args.max_pages,
    )

    if not article_links:
        print("[error] No article links discovered.", file=sys.stderr)
        return 1

    print(f"[crawl] discovered article links: {len(article_links)}")
    docs: list[ScrapedDocument] = []

    for index, url in enumerate(article_links, start=1):
        print(f"[scrape] {index}/{len(article_links)} {url}")
        try:
            doc = scrape_article(url, session)
        except Exception as exc:
            print(f"[scrape] skip {url} ({exc})", file=sys.stderr)
            continue

        if not doc or doc.char_count < args.min_chars:
            print(f"[scrape] ignored {url} (content too short)")
            continue

        docs.append(doc)
        print(f"[scrape] title={doc.title}")
        print(f"[scrape] chars={doc.char_count}")

    if not docs:
        print("[error] No articles with usable content were scraped.", file=sys.stderr)
        return 1

    if args.chunk_size == "auto":
        chunk_size = choose_chunk_size(docs, args.chunk_buffer)
    else:
        chunk_size = int(args.chunk_size)

    explicit_overlap = None if args.chunk_overlap == "auto" else int(args.chunk_overlap)
    chunk_overlap = choose_chunk_overlap(chunk_size, explicit_overlap)

    dry_run_report(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_embedding_chunk_size=args.max_embedding_chunk_size,
    )

    if not args.ingest:
        print("[dry-run] no ingest performed")
        return 0

    failures = ingest_documents(
        docs,
        api_base=args.api,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_embedding_chunk_size=args.max_embedding_chunk_size,
        skip_already_ingested=args.skip_already_ingested,
    )
    if failures:
        print(f"[ingest] completed with failures: {len(failures)}", file=sys.stderr)
        for failure in failures:
            print(
                f"[ingest] failed url={failure['url']} chars={failure['char_count']} error={failure['error']}",
                file=sys.stderr,
            )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
