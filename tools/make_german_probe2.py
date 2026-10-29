from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SOURCE = os.path.join(ROOT, "data", "tasks_time_v2.json")
TARGET = os.path.join(ROOT, "data", "tasks_time_de2.json")

ITEMS = [
    ("00:30", "half_past", "into_hour",
     "halb eins nachts", "half past twelve at night"),
    ("01:30", "half_past", "into_hour",
     "halb zwei nachts", "half past one at night"),
    ("02:30", "half_past", "into_hour",
     "halb drei morgens", "half past two in the morning"),
    ("03:30", "half_past", "into_hour",
     "halb vier morgens", "half past three in the morning"),
    ("05:30", "half_past", "into_hour",
     "halb sechs morgens", "half past five in the morning"),
    ("07:30", "half_past", "into_hour",
     "halb acht morgens", "half past seven in the morning"),
    ("08:30", "half_past", "into_hour",
     "halb neun morgens", "half past eight in the morning"),
    ("09:30", "half_past", "into_hour",
     "halb zehn vormittags", "half past nine in the morning"),
    ("10:30", "half_past", "into_hour",
     "halb elf vormittags", "half past ten in the morning"),
    ("12:30", "half_past", "into_hour",
     "halb eins mittags", "half past twelve in the afternoon"),
    ("13:30", "half_past", "into_hour",
     "halb zwei nachmittags", "half past one in the afternoon"),
    ("15:30", "half_past", "into_hour",
     "halb vier nachmittags", "half past three in the afternoon"),
    ("17:30", "half_past", "into_hour",
     "halb sechs abends", "half past five in the evening"),
    ("19:30", "half_past", "into_hour",
     "halb acht abends", "half past seven in the evening"),
    ("22:30", "half_past", "into_hour",
     "halb elf abends", "half past ten in the evening"),

    ("01:15", "quarter_into", "into_hour_regional",
     "viertel zwei nachts", "quarter past one at night"),
    ("03:15", "quarter_into", "into_hour_regional",
     "viertel vier morgens", "quarter past three in the morning"),
    ("06:15", "quarter_into", "into_hour_regional",
     "viertel sieben morgens", "quarter past six in the morning"),
    ("09:15", "quarter_into", "into_hour_regional",
     "viertel zehn vormittags", "quarter past nine in the morning"),
    ("14:15", "quarter_into", "into_hour_regional",
     "viertel drei nachmittags", "quarter past two in the afternoon"),
    ("19:15", "quarter_into", "into_hour_regional",
     "viertel acht abends", "quarter past seven in the evening"),

    ("01:45", "quarter_into", "into_hour_regional",
     "dreiviertel zwei nachts", "quarter to two at night"),
    ("04:45", "quarter_into", "into_hour_regional",
     "dreiviertel fünf morgens", "quarter to five in the morning"),
    ("07:45", "quarter_into", "into_hour_regional",
     "dreiviertel acht morgens", "quarter to eight in the morning"),
    ("10:45", "quarter_into", "into_hour_regional",
     "dreiviertel elf vormittags", "quarter to eleven in the morning"),
    ("15:45", "quarter_into", "into_hour_regional",
     "dreiviertel vier nachmittags", "quarter to four in the afternoon"),
    ("20:45", "quarter_into", "into_hour_regional",
     "dreiviertel neun abends", "quarter to nine in the evening"),

    ("01:10", "n_minutes", "from_hour",
     "zehn nach eins nachts", "ten past one at night"),
    ("03:20", "n_minutes", "from_hour",
     "zwanzig nach drei morgens", "twenty past three in the morning"),
    ("06:05", "n_minutes", "from_hour",
     "fünf nach sechs morgens", "five past six in the morning"),
    ("07:25", "n_minutes", "from_hour",
     "fünfundzwanzig nach sieben morgens", "twenty five past seven in the morning"),
    ("09:10", "n_minutes", "from_hour",
     "zehn nach neun vormittags", "ten past nine in the morning"),
    ("11:20", "n_minutes", "from_hour",
     "zwanzig nach elf vormittags", "twenty past eleven in the morning"),
    ("13:05", "n_minutes", "from_hour",
     "fünf nach eins nachmittags", "five past one in the afternoon"),
    ("15:25", "n_minutes", "from_hour",
     "fünfundzwanzig nach drei nachmittags", "twenty five past three in the afternoon"),
    ("16:10", "n_minutes", "from_hour",
     "zehn nach vier nachmittags", "ten past four in the afternoon"),
    ("18:20", "n_minutes", "from_hour",
     "zwanzig nach sechs abends", "twenty past six in the evening"),
    ("21:05", "n_minutes", "from_hour",
     "fünf nach neun abends", "five past nine in the evening"),
    ("23:10", "n_minutes", "from_hour",
     "zehn nach elf abends", "ten past eleven at night"),

    ("02:45", "literal", "digital", "02:45", "02:45"),
    ("04:20", "literal", "digital", "04:20", "04:20"),
    ("13:35", "literal", "digital", "13:35", "13:35"),
    ("17:05", "literal", "digital", "17:05", "17:05"),
    ("21:50", "literal", "digital", "21:50", "21:50"),

    ("03:40", "subtractive", "subtractive",
     "zwanzig vor vier morgens", "twenty to four in the morning"),
    ("06:45", "subtractive", "subtractive",
     "viertel vor sieben morgens", "quarter to seven in the morning"),
    ("13:50", "subtractive", "subtractive",
     "zehn vor zwei nachmittags", "ten to two in the afternoon"),
    ("21:40", "subtractive", "subtractive",
     "zwanzig vor zehn abends", "twenty to ten in the evening"),
]

LABEL_EN = "alarm"
LABEL_DE = "Wecker"

FEWSHOT_TIMES = ("09:05", "19:10", "08:25")

NOTE = (
    "Powered German probe, 48 items. Same cells and same carrier as "
    "data/tasks_time_de.json, with the into_hour cell grown from 5 items to 15 "
    "and the regional cell from 2 to 12, because five items could not separate "
    "any two models. Kept as a separate file because the first run happened "
    "before the scale hypothesis existed, and merging would test that "
    "hypothesis partly on the data that suggested it; see F57 for the "
    "pre-registration. No clock time here repeats one from the first probe and "
    "none is a few-shot demonstration. German counts INTO the coming hour with "
    "a closed set of three items (halb, viertel, dreiviertel) and has no "
    "productive counting-into form, so from_hour items can only be N nach X. "
    "The twelve regional items are flagged de_regional: viertel N and "
    "dreiviertel N are used in the east and south, their gold label is "
    "contested among native speakers, and they must be excluded from any "
    "primary comparison unless deliberately included. viertel is lowercase "
    "throughout on purpose so the minimal pair differs by one word and not "
    "also by capitalisation. WORDING NOT YET CHECKED BY A NATIVE SPEAKER — see "
    "tools/make_german_probe2.py."
)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    with open(SOURCE, "r", encoding="utf-8") as handle:
        meta = json.load(handle)["meta"]

    times = [row[0] for row in ITEMS]
    duplicates = {t for t in times if times.count(t) > 1}
    if duplicates:
        raise SystemExit(f"duplicate clock times: {sorted(duplicates)}")
    leaked = set(times) & set(FEWSHOT_TIMES)
    if leaked:
        raise SystemExit(f"clock time is also a few-shot demonstration: {leaked}")

    with open(os.path.join(ROOT, "data", "tasks_time_de.json"), "r",
              encoding="utf-8") as handle:
        first = {t["checks"]["en"][0]["time"] for t in json.load(handle)["tasks"]}
    overlap = set(times) & first
    if overlap:
        raise SystemExit(f"clock time already used by the first German probe: "
                         f"{sorted(overlap)}")

    tasks = []
    for index, (time, difficulty, direction, german, english) in enumerate(ITEMS, 1):
        regional = direction.endswith("_regional")
        tasks.append({
            "id": f"h{index:02d}",
            "difficulty": difficulty,
            "de_direction": direction,
            "de_regional": regional,
            "n_steps": 1,
            "trap": f"{german} = {time}",
            "instruction": {
                "en": f"Set an alarm for {english}.",
                "de": f"Stelle den Wecker auf {german}.",
            },
            "checks": {
                "en": [{"kind": "alarm_exists", "time": time}],
                "de": [{"kind": "alarm_exists", "time": time}],
            },
            "gold": {
                "en": [
                    {"tool": "set_alarm", "args": {"time": time, "label": LABEL_EN}},
                    {"tool": "done", "args": {}},
                ],
                "de": [
                    {"tool": "set_alarm", "args": {"time": time, "label": LABEL_DE}},
                    {"tool": "done", "args": {}},
                ],
            },
        })

    out = {
        "meta": {
            "today": meta["today"],
            "weekday": meta["weekday"],
            "version": "time-de-2.0",
            "note": NOTE,
        },
        "tasks": tasks,
    }

    with open(TARGET, "w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"wrote {TARGET} with {len(tasks)} tasks")
    counts: dict = {}
    for task in tasks:
        counts[task["de_direction"]] = counts.get(task["de_direction"], 0) + 1
    for direction, n in sorted(counts.items()):
        print(f"  de_direction {direction:<20} {n}")
    print(f"  flagged de_regional: {sum(t['de_regional'] for t in tasks)} "
          f"(contested gold, exclude by default)")
    print()
    print("The German half, for a native speaker to read straight through:")
    for task in tasks:
        print(f"  {task['id']}  {task['checks']['de'][0]['time']}  "
              f"{task['de_direction']:<20} {task['instruction']['de']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
