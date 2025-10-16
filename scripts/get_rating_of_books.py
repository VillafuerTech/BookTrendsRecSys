"""Scrape Goodreads reviews for a list of seed books."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    import requests
    from bs4 import BeautifulSoup
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

requests = None
BeautifulSoup = None
HTTPAdapter = None
Retry = None


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

BASE_URL = "https://www.goodreads.com"
LOGGER = logging.getLogger(__name__)


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)


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


def read_seed_books(seed_path: Path) -> List[Dict[str, str]]:
    seeds: List[Dict[str, str]] = []
    with seed_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if "book_id" not in reader.fieldnames:
            raise ValueError("Seed file must contain a 'book_id' column")
        for row in reader:
            if row.get("book_id"):
                seeds.append(row)
    return seeds


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
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)


def _extract_user_id(href: str) -> Optional[int]:
    if not href:
        return None
    parts = href.rstrip("/").split("/")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
        leading = part.split("-")[0]
        if leading.isdigit():
            return int(leading)
    return None


def parse_reviews(html: str) -> List[Dict[str, Optional[object]]]:
    require_dependencies()
    soup = BeautifulSoup(html, "lxml")
    rows = soup.select("tr.bookalike.review")
    if not rows:
        rows = soup.select("table#books tr")
    reviews: List[Dict[str, Optional[object]]] = []
    for row in rows:
        user_anchor = row.select_one("a.user") or row.select_one("a[href*='/user/show/']")
        username = user_anchor.get_text(strip=True) if user_anchor else None
        user_id = (
            _extract_user_id(user_anchor["href"]) if user_anchor and user_anchor.has_attr("href") else None
        )

        rating_span = row.select_one(".ShelfStatus span[aria-label*='out of 5']")
        rating_value: Optional[float] = None
        if rating_span and rating_span.has_attr("aria-label"):
            label = rating_span["aria-label"]
            try:
                rating_value = float(label.split(" ")[0])
            except (ValueError, IndexError):
                rating_value = None
        if rating_value is None:
            alt_rating = row.select_one("span.staticStars:not(.greyText)")
            if alt_rating and alt_rating.has_attr("title"):
                try:
                    rating_value = float(alt_rating["title"].split(" ")[0])
                except (ValueError, IndexError):
                    rating_value = None

        review_text = row.select_one(".reviewText")
        review_body = review_text.get_text(strip=True) if review_text else None

        date_span = row.select_one(".date_read_value") or row.select_one(".date_updated")
        review_date = date_span.get_text(strip=True) if date_span else None

        likes_span = row.select_one(".likesCount")
        comments_span = row.select_one(".commentsCount")

        def _parse_int(text: Optional[str]) -> int:
            if not text:
                return 0
            digits = "".join(ch for ch in text if ch.isdigit())
            return int(digits) if digits else 0

        reviews.append(
            {
                "user": username,
                "user_id": user_id,
                "user_rating": rating_value,
                "review_date": review_date,
                "review": review_body,
                "likes": _parse_int(likes_span.get_text(strip=True) if likes_span else None),
                "comments": _parse_int(comments_span.get_text(strip=True) if comments_span else None),
            }
        )
    return reviews


def parse_book_metadata(soup):
    title = soup.select_one("h1#bookTitle") or soup.select_one("h1[data-testid='bookTitle']")
    avg_rating = soup.select_one("span[itemprop='ratingValue']") or soup.select_one(
        "div.RatingStatistics__rating"
    )
    title_text = title.get_text(strip=True) if title else ""
    try:
        avg_rating_value = float(avg_rating.get_text(strip=True)) if avg_rating else None
    except ValueError:
        avg_rating_value = None
    return {"book_title": title_text, "average_rating": avg_rating_value}


def crawl_book(
    session: "requests.Session",
    book_id: str,
    max_pages: int,
    min_delay: float,
    max_delay: float,
) -> Dict[str, object]:
    reviews: List[Dict[str, object]] = []
    book_title: str = ""
    average_rating: Optional[float] = None
    seen_reviews = set()
    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}/book/show/{book_id}?page={page}"
        LOGGER.info("Fetching book %s page %s", book_id, page)
        html = fetch_page(session, url, min_delay, max_delay)
        if html is None:
            break
        require_dependencies()
        soup = BeautifulSoup(html, "lxml")
        if not book_title:
            meta = parse_book_metadata(soup)
            book_title = meta.get("book_title", "") or ""
            average_rating = meta.get("average_rating")
        page_reviews = parse_reviews(html)
        before_count = len(reviews)
        for review in page_reviews:
            key = (review.get("user_id"), review.get("review_date"))
            if key in seen_reviews:
                continue
            seen_reviews.add(key)
            reviews.append(review)
        LOGGER.info(
            "Book %s page %s yielded %d new reviews", book_id, page, len(reviews) - before_count
        )
        if not page_reviews:
            LOGGER.info("No reviews found on page %s for book %s; stopping", page, book_id)
            break
    data: Dict[str, object] = {
        "book_id": int(book_id) if str(book_id).isdigit() else book_id,
        "book_title": book_title,
        "average_rating": average_rating,
        "reviews": reviews,
        "source_url": f"{BASE_URL}/book/show/{book_id}",
        "fetched_at": datetime.utcnow().isoformat(),
    }
    return data


def validate_output(path: Path) -> None:
    with path.open("r", encoding="utf-8") as fp:
        json.load(fp)


def save_book_reviews(data: Dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    validate_output(out_path)
    LOGGER.info(
        "Saved %d reviews for book %s to %s",
        len(data.get("reviews", [])),
        data.get("book_id"),
        out_path,
    )


def main(args: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Scrape Goodreads reviews for seed books")
    parser.add_argument("--seeds", type=Path, default=Path("src/ingest/assets/seed_books.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/raw/reviews"))
    parser.add_argument("--min-delay", type=float, default=1.0)
    parser.add_argument("--max-delay", type=float, default=3.0)
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--user-agent", type=str, default="Mozilla/5.0 (compatible; BookTrendsBot/1.0)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parsed = parser.parse_args(args=args)

    setup_logging(Path("logs/collect.log"))

    try:
        seeds = read_seed_books(parsed.seeds)
    except FileNotFoundError:
        LOGGER.error("Seed file %s not found", parsed.seeds)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error("Failed to read seed books: %s", exc)
        return 1

    session = build_session(parsed.user_agent)
    total = len(seeds)
    processed = 0
    skipped = 0

    for seed in seeds:
        book_id = seed.get("book_id")
        if not book_id:
            continue
        out_file = parsed.out / f"goodreads_reviews_{book_id}.json"
        if parsed.resume and out_file.exists() and not parsed.overwrite:
            LOGGER.info("Skipping %s (already exists)", out_file)
            skipped += 1
            continue
        try:
            book_data = crawl_book(
                session=session,
                book_id=str(book_id),
                max_pages=parsed.max_pages,
                min_delay=parsed.min_delay,
                max_delay=parsed.max_delay,
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("Error while crawling book %s: %s", book_id, exc)
            continue
        try:
            save_book_reviews(book_data, out_file)
            processed += 1
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("Failed to save data for book %s: %s", book_id, exc)

    LOGGER.info(
        "Completed book crawl. Total seeds: %d, processed: %d, skipped: %d",
        total,
        processed,
        skipped,
    )
    print(
        f"SUMMARY: seeds={total} processed={processed} skipped={skipped} output_dir={parsed.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
