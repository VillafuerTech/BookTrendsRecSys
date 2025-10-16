"""Build a training dataset from Goodreads user rating JSON files."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

pd = None

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


def parse_book_id(item: dict) -> Optional[int]:
    book_id = item.get("book_id")
    if isinstance(book_id, int):
        return book_id
    if isinstance(book_id, str) and book_id.isdigit():
        return int(book_id)
    book_url = item.get("book_url")
    if isinstance(book_url, str) and book_url:
        slug = book_url.rstrip("/").split("/")[-1]
        leading = slug.split("-")[0]
        if leading.isdigit():
            return int(leading)
    return None


def parse_rating(item: dict) -> Optional[float]:
    rating = item.get("user_rating")
    if isinstance(rating, (int, float)):
        return float(rating)
    if isinstance(rating, str):
        try:
            return float(rating)
        except ValueError:
            return None
    return None


def build_rows(directory: Path) -> List[dict]:
    rows: List[dict] = []
    for path in iter_user_files(directory):
        try:
            with path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
        except json.JSONDecodeError as exc:
            LOGGER.warning("Failed to parse %s: %s", path, exc)
            continue
        user_id = data.get("source_user_id")
        if not isinstance(user_id, int):
            continue
        for item in data.get("items", []):
            if not isinstance(item, dict):
                continue
            book_id = parse_book_id(item)
            rating = parse_rating(item)
            if book_id is None or rating is None:
                continue
            if not (1 <= rating <= 5):
                continue
            rows.append({"userid": user_id, "bookid": book_id, "rating": rating})
    rows.sort(key=lambda r: (r["userid"], r["bookid"]))
    return rows


def write_dataset(rows: List[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    global pd
    if pd is None:
        try:
            import pandas as _pd
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "Missing optional dependency. Install requirements with 'pip install -r requirements.txt'."
            ) from exc
        pd = _pd
    df = pd.DataFrame(rows, columns=["userid", "bookid", "rating"])
    df.to_csv(output, index=False)
    df_read = pd.read_csv(output)
    if list(df_read.columns) != ["userid", "bookid", "rating"]:
        raise ValueError("Output dataset has unexpected columns")
    LOGGER.info("Wrote dataset with %d rows to %s", len(df), output)


def main(args: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build Goodreads training dataset")
    parser.add_argument("--in", dest="input_dir", type=Path, default=Path("data/raw/users"))
    parser.add_argument(
        "--out",
        dest="output",
        type=Path,
        default=Path("data/processed/goodreads_dataset.csv"),
    )
    parsed = parser.parse_args(args=args)

    setup_logging(Path("logs/collect.log"))

    rows = build_rows(parsed.input_dir)
    write_dataset(rows, parsed.output)

    LOGGER.info("Completed dataset build")
    print(
        f"SUMMARY: rows={len(rows)} output={parsed.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
