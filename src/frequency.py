

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Dict, List, Optional

import requests

NGRAM_URL = "https://books.google.com/ngrams/json"

PHRASES: Dict[str, Dict[str, List[str]]] = {
    "half_past": {
        "ru": [
            "половина третьего", "половина седьмого", "половина десятого",
            "половина двенадцатого", "половина пятого",
        ],
        "en": [
            "half past two", "half past six", "half past nine",
            "half past eleven", "half past four",
        ],
    },
    "quarter_past": {
        "ru": [
            "четверть шестого", "четверть девятого", "четверть одиннадцатого",
            "четверть второго",
        ],
        "en": [
            "quarter past five", "quarter past eight", "quarter past ten",
            "quarter past one",
        ],
    },
    "n_minutes": {
        "ru": [
            "двадцать минут восьмого", "пять минут девятого",
            "десять минут одиннадцатого", "двадцать пять минут второго",
            "двадцать минут третьего", "десять минут седьмого",
            "пять минут первого", "двадцать пять минут десятого",
        ],
        "en": [
            "twenty past seven", "five past eight", "ten past ten",
            "twenty five past one", "twenty past two", "ten past six",
            "five past twelve", "twenty five past nine",
        ],
    },
}

CORPUS = {"ru": "ru-2019", "en": "en-2019"}


def lookup(phrase: str, lang: str, pause: float = 1.5) -> Optional[float]:
    """Mean relative frequency of one phrase over 1980-2019, or None."""
    try:
        response = requests.get(
            NGRAM_URL,
            params={
                "content": phrase,
                "year_start": 1980,
                "year_end": 2019,
                "corpus": CORPUS[lang],
                "smoothing": 3,
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=45,
        )
        time.sleep(pause)  
        if response.status_code != 200:
            return None
        payload = response.json()
        if not payload:
            return 0.0  
        series = payload[0].get("timeseries") or []
        return statistics.mean(series) if series else 0.0
    except Exception: 
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/frequency.json")
    parser.add_argument("--pause", type=float, default=1.5)
    args = parser.parse_args()

    results: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for category, by_lang in PHRASES.items():
        results[category] = {}
        for lang, phrases in by_lang.items():
            results[category][lang] = {}
            for phrase in phrases:
                value = lookup(phrase, lang, args.pause)
                results[category][lang][phrase] = value
                shown = "failed" if value is None else f"{value:.3e}"
                print(f"  {lang}  {phrase:<32} {shown}", flush=True)

    print("\nmean frequency per category (1980-2019, Google Books):")
    print(f'{"category":<15}{"Russian":>14}{"English":>14}')
    print("-" * 43)
    summary = {}
    for category in PHRASES:
        row = {}
        for lang in ("ru", "en"):
            values = [v for v in results[category][lang].values() if v is not None]
            row[lang] = statistics.mean(values) if values else None
        summary[category] = row
        ru = "n/a" if row["ru"] is None else f'{row["ru"]:.3e}'
        en = "n/a" if row["en"] is None else f'{row["en"]:.3e}'
        print(f"{category:<15}{ru:>14}{en:>14}")

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"per_phrase": results, "per_category": summary},
                  handle, ensure_ascii=False, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
