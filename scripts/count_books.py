"""Aggregate Goodreads user JSON files into book occurrence counts."""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

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


def iter_user_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.glob("goodreads_ratings_*.json") if p.is_file())


def extract_book_slug(item: Dict[str, object]) -> Optional[str]:
    book_url = item.get("book_url")
    if isinstance(book_url, str) and book_url:
        slug = book_url.rstrip("/").split("/")[-1]
        return slug or None
    book_id = item.get("book_id")
    if isinstance(book_id, int):
        return str(book_id)
    if isinstance(book_id, str) and book_id:
        return book_id
    return None


def count_books(directory: Path) -> Counter:
    counter: Counter = Counter()
    for path in iter_user_files(directory):
        try:
            with path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
        except json.JSONDecodeError as exc:
            LOGGER.warning("Failed to read %s: %s", path, exc)
            continue
        for item in data.get("items", []):
            if not isinstance(item, dict):
                continue
            slug = extract_book_slug(item)
            if slug:
                counter[slug] += 1
    return counter


def write_counts(counter: Counter, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["book_slug", "count"])
        for slug, count in counter.most_common():
            writer.writerow([slug, count])
    # Validation read-back
    with output.open("r", encoding="utf-8") as csvfile:
        list(csv.reader(csvfile))
    LOGGER.info("Wrote %d book counts to %s", len(counter), output)


def main(args: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Count Goodreads books from user JSON files")
    parser.add_argument("--in", dest="input_dir", type=Path, default=Path("data/raw/users"))
    parser.add_argument("--out", dest="output", type=Path, default=Path("data/interim/book_counts.csv"))
    parsed = parser.parse_args(args=args)

    setup_logging(Path("logs/collect.log"))

    counter = count_books(parsed.input_dir)
    if not counter:
        LOGGER.info("No book data found in %s", parsed.input_dir)
    write_counts(counter, parsed.output)

    LOGGER.info("Completed counting books")
    print(
        f"SUMMARY: unique_books={len(counter)} output={parsed.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
