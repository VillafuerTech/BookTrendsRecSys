"""Scrape Goodreads user rating histories for collected reviewers."""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Set

if TYPE_CHECKING:  # pragma: no cover
    import requests
    from bs4 import BeautifulSoup
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

requests = None
BeautifulSoup = None
HTTPAdapter = None
Retry = None

BASE_URL = "https://www.goodreads.com"
LOGGER = logging.getLogger(__name__)


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)


def require_dependencies() -> None:
    global requests, BeautifulSoup, HTTPAdapter, Retry
    if requests is not None and BeautifulSoup is not None and HTTPAdapter is not None and Retry is not None:
        return
    try:
        import requests as _requests
        from bs4 import BeautifulSoup as _BeautifulSoup
        from requests.adapters import HTTPAdapter as _HTTPAdapter
        from urllib3.util.retry import Retry as _Retry
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise RuntimeError(
            "Missing optional dependency. Install requirements with 'pip install -r requirements.txt'."
        ) from exc
    requests = _requests
    BeautifulSoup = _BeautifulSoup
    HTTPAdapter = _HTTPAdapter
    Retry = _Retry


def build_session(user_agent: str) -> "requests.Session":
    require_dependencies()
    session = requests.Session()
    retry = Retry(
        total=5,
        read=5,
        connect=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": user_agent})
    return session


def iter_review_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.glob("goodreads_reviews_*.json") if p.is_file())


def load_user_ids(review_dir: Path) -> Set[int]:
    user_ids: Set[int] = set()
    for path in iter_review_files(review_dir):
        try:
            with path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            for review in data.get("reviews", []):
                uid = review.get("user_id")
                if isinstance(uid, int):
                    user_ids.add(uid)
        except json.JSONDecodeError as exc:
            LOGGER.warning("Failed to parse %s: %s", path, exc)
    return user_ids


def fetch_page(
    session: "requests.Session",
    url: str,
    min_delay: float,
    max_delay: float,
) -> Optional[str]:
    require_dependencies()
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        LOGGER.warning("Failed to fetch %s: %s", url, exc)
        return None
    finally:
        time.sleep(random.uniform(min_delay, max_delay))


def parse_rating(row) -> Optional[float]:
    rating_span = row.select_one(".staticStars:not(.greyText)")
    if rating_span and rating_span.has_attr("title"):
        try:
            return float(rating_span["title"].split(" ")[0])
        except (ValueError, IndexError):
            return None
    rating_span = row.select_one("span[aria-label*='out of 5']")
    if rating_span and rating_span.has_attr("aria-label"):
        try:
            return float(rating_span["aria-label"].split(" ")[0])
        except (ValueError, IndexError):
            return None
    return None


def parse_rating_text(row) -> Optional[str]:
    rating_span = row.select_one(".staticStars:not(.greyText)")
    if rating_span and rating_span.has_attr("title"):
        return rating_span["title"]
    rating_span = row.select_one("span[aria-label*='out of 5']")
    if rating_span and rating_span.has_attr("aria-label"):
        return rating_span["aria-label"]
    return None


def parse_book_entry(row) -> Optional[Dict[str, object]]:
    title_anchor = row.select_one("a.bookTitle") or row.select_one("a[href*='/book/show/']")
    if not title_anchor:
        return None
    title = title_anchor.get_text(strip=True)
    book_url = title_anchor.get("href") or ""
    if book_url and not book_url.startswith("http"):
        book_url = f"{BASE_URL}{book_url}"
    book_id = None
    if book_url:
        slug = book_url.rstrip("/").split("/")[-1]
        leading = slug.split("-")[0]
        if leading.isdigit():
            book_id = int(leading)

    author = None
    author_cell = row.select_one("a.authorName") or row.select_one("td.field.author")
    if author_cell:
        author = author_cell.get_text(strip=True)

    rating_value = parse_rating(row)
    rating_text = parse_rating_text(row)

    date_read_cell = row.select_one("td.field.date_read .value") or row.select_one(".date_read_value")
    date_read = date_read_cell.get_text(strip=True) if date_read_cell else None

    date_added_cell = row.select_one("td.field.date_added .value") or row.select_one(".date_added_value")
    date_added = date_added_cell.get_text(strip=True) if date_added_cell else None

    shelves = []
    shelves_cell = row.select_one("td.field.shelves") or row.select_one(".shelves")
    if shelves_cell:
        shelves = [a.get_text(strip=True) for a in shelves_cell.select("a") if a.get_text(strip=True)]

    review_anchor = row.select_one("a[href*='/review/show/']")
    review_url = review_anchor.get("href") if review_anchor else None
    if review_url and not review_url.startswith("http"):
        review_url = f"{BASE_URL}{review_url}"

    return {
        "book_id": book_id,
        "book_title": title,
        "book_url": book_url,
        "author": author,
        "user_rating": rating_value,
        "user_rating_text": rating_text,
        "date_read": date_read or "not set",
        "date_added": date_added,
        "shelves": shelves,
        "review_url": review_url,
    }


def parse_user_page(html: str) -> List[Dict[str, object]]:
    require_dependencies()
    soup = BeautifulSoup(html, "lxml")
    rows = soup.select("tr.bookalike")
    if not rows:
        rows = soup.select("table#books tr")
    entries: List[Dict[str, object]] = []
    for row in rows:
        entry = parse_book_entry(row)
        if entry:
            entries.append(entry)
    return entries


def crawl_user(
    session: requests.Session,
    user_id: int,
    max_pages: int,
    min_delay: float,
    max_delay: float,
) -> Dict[str, object]:
    items: List[Dict[str, object]] = []
    seen = set()
    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}/review/list/{user_id}?page={page}&per_page=100&shelf=read"
        LOGGER.info("Fetching user %s page %s", user_id, page)
        html = fetch_page(session, url, min_delay, max_delay)
        if html is None:
            break
        page_items = parse_user_page(html)
        before = len(items)
        for item in page_items:
            key = (item.get("book_id"), item.get("date_added"))
            if key in seen:
                continue
            seen.add(key)
            items.append(item)
        LOGGER.info(
            "User %s page %s yielded %d new ratings", user_id, page, len(items) - before
        )
        if not page_items:
            LOGGER.info("No ratings on page %s for user %s; stopping", page, user_id)
            break
    return {"source_user_id": user_id, "count": len(items), "items": items}


def save_user_history(data: Dict[str, object], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    with out_file.open("r", encoding="utf-8") as fp:
        json.load(fp)
    LOGGER.info("Saved %s with %d items", out_file, data.get("count", 0))


def main(args: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Scrape Goodreads user rating histories")
    parser.add_argument("--in", dest="input_dir", type=Path, default=Path("data/raw/reviews"))
    parser.add_argument("--out", dest="output_dir", type=Path, default=Path("data/raw/users"))
    parser.add_argument("--min-delay", type=float, default=1.0)
    parser.add_argument("--max-delay", type=float, default=3.0)
    parser.add_argument("--max-pages", type=int, default=300)
    parser.add_argument("--user-agent", type=str, default="Mozilla/5.0 (compatible; BookTrendsBot/1.0)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parsed = parser.parse_args(args=args)

    setup_logging(Path("logs/collect.log"))

    user_ids = load_user_ids(parsed.input_dir)
    if not user_ids:
        LOGGER.info("No users found in %s", parsed.input_dir)
        print(
            f"SUMMARY: users=0 processed=0 skipped=0 output_dir={parsed.output_dir}")
        return 0

    session = build_session(parsed.user_agent)

    total = len(user_ids)
    processed = 0
    skipped = 0
    for user_id in sorted(user_ids):
        out_file = parsed.output_dir / f"goodreads_ratings_{user_id}.json"
        if parsed.resume and out_file.exists() and not parsed.overwrite:
            LOGGER.info("Skipping %s (already exists)", out_file)
            skipped += 1
            continue
        try:
            data = crawl_user(
                session=session,
                user_id=user_id,
                max_pages=parsed.max_pages,
                min_delay=parsed.min_delay,
                max_delay=parsed.max_delay,
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("Error while crawling user %s: %s", user_id, exc)
            continue
        try:
            save_user_history(data, out_file)
            processed += 1
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("Failed to save history for user %s: %s", user_id, exc)

    LOGGER.info(
        "Completed user crawl. Total users: %d, processed: %d, skipped: %d",
        total,
        processed,
        skipped,
    )
    print(
        f"SUMMARY: users={total} processed={processed} skipped={skipped} output_dir={parsed.output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
