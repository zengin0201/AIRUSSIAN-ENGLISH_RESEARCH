from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SOURCE = os.path.join(ROOT, "data", "tasks_time_v2.json")
TARGET = os.path.join(ROOT, "data", "tasks_time_de_halb.json")

LABEL_DE = "Wecker"
LABEL_EN = "alarm"
FEWSHOT_TIMES = ("09:05", "19:10", "08:25")

ITEMS = [
    ("06:30", "into_fixed", "halb sieben morgens", "half past six in the morning"),
    ("08:30", "into_fixed", "halb neun morgens", "half past eight in the morning"),
    ("10:30", "into_fixed", "halb elf morgens", "half past ten in the morning"),
    ("14:30", "into_fixed", "halb drei nachmittags", "half past two in the afternoon"),
    ("16:30", "into_fixed", "halb fünf nachmittags", "half past four in the afternoon"),
    ("18:30", "into_fixed", "halb sieben abends", "half past six in the evening"),
    ("20:30", "into_fixed", "halb neun abends", "half past eight in the evening"),
    ("21:30", "into_fixed", "halb zehn abends", "half past nine in the evening"),

    ("06:20", "into_composed", "zehn vor halb sieben morgens",
     "twenty past six in the morning"),
    ("08:20", "into_composed", "zehn vor halb neun morgens",
     "twenty past eight in the morning"),
    ("10:25", "into_composed", "fünf vor halb elf morgens",
     "twenty five past ten in the morning"),
    ("14:20", "into_composed", "zehn vor halb drei nachmittags",
     "twenty past two in the afternoon"),
    ("16:25", "into_composed", "fünf vor halb fünf nachmittags",
     "twenty five past four in the afternoon"),
    ("18:20", "into_composed", "zehn vor halb sieben abends",
     "twenty past six in the evening"),
    ("20:25", "into_composed", "fünf vor halb neun abends",
     "twenty five past eight in the evening"),
    ("21:20", "into_composed", "zehn vor halb zehn abends",
     "twenty past nine in the evening"),

    ("07:35", "into_composed", "fünf nach halb acht morgens",
     "twenty five to eight in the morning"),
    ("09:40", "into_composed", "zehn nach halb zehn morgens",
     "twenty to ten in the morning"),
    ("15:35", "into_composed", "fünf nach halb vier nachmittags",
     "twenty five to four in the afternoon"),
    ("17:40", "into_composed", "zehn nach halb sechs nachmittags",
     "twenty to six in the afternoon"),
    ("19:35", "into_composed", "fünf nach halb acht abends",
     "twenty five to eight in the evening"),
    ("20:40", "into_composed", "zehn nach halb neun abends",
     "twenty to nine in the evening"),

    ("06:20", "from_hour", "zwanzig nach sechs morgens",
     "twenty past six in the morning"),
    ("08:20", "from_hour", "zwanzig nach acht morgens",
     "twenty past eight in the morning"),
    ("10:25", "from_hour", "fünfundzwanzig nach zehn morgens",
     "twenty five past ten in the morning"),
    ("14:20", "from_hour", "zwanzig nach zwei nachmittags",
     "twenty past two in the afternoon"),
    ("16:25", "from_hour", "fünfundzwanzig nach vier nachmittags",
     "twenty five past four in the afternoon"),
    ("18:20", "from_hour", "zwanzig nach sechs abends",
     "twenty past six in the evening"),
    ("20:25", "from_hour", "fünfundzwanzig nach acht abends",
     "twenty five past eight in the evening"),
    ("21:20", "from_hour", "zwanzig nach neun abends",
     "twenty past nine in the evening"),

    ("07:12", "digital", "07:12", "07:12"),
    ("11:48", "digital", "11:48", "11:48"),
    ("15:03", "digital", "15:03", "15:03"),
    ("19:27", "digital", "19:27", "19:27"),
    ("22:36", "digital", "22:36", "22:36"),
]

CARRIER_DE = "Stelle den Wecker auf {}."
CARRIER_EN = "Set an alarm for {}."

NOTE = (
    "The productivity contrast inside German. The earlier German probe varied "
    "productivity across two languages; this varies it inside one. halb acht "
    "(07:30) is a stored counting-into phrase; zehn vor halb acht (07:20) is "
    "counting into the hour assembled from that phrase, with the surface number "
    "not the answer and the named hour not yet arrived, which is the profile "
    "that breaks in Russian; zwanzig nach sieben (07:20) counts from the "
    "completed hour. The from_hour and into_composed cells ask for the SAME "
    "eight clock times with BYTE-IDENTICAL English, since English has only one "
    "rendering of 06:20, so a difference between them cannot be the task or the "
    "time and can only be the German construction. This file exists because the "
    "paper claimed German has no productive counting-into form, which is false; "
    "see F70. Daypart words avoid the 02:00-05:00 window that an earlier review "
    "found mislabelled as morgens, and vormittags is not used at all. The "
    "post-positioned daypart is kept despite being an English calque, because "
    "every earlier German probe uses it and changing it here would make the "
    "files incomparable. PREDICTIONS REGISTERED IN NOTES F71 BEFORE THE RUN. "
    "WORDING NOT CHECKED BY A NATIVE SPEAKER."
)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    with open(SOURCE, "r", encoding="utf-8") as handle:
        meta = json.load(handle)["meta"]

    leaked = {t for t, *_ in ITEMS} & set(FEWSHOT_TIMES)
    if leaked:
        raise SystemExit(f"clock time is also a few-shot demonstration: {leaked}")

    composed = sorted(t for t, c, *_ in ITEMS if c == "into_composed")
    from_hour = sorted(t for t, c, *_ in ITEMS if c == "from_hour")
    overlap = sorted(set(composed) & set(from_hour))
    if len(overlap) != len(from_hour):
        raise SystemExit(
            f"from_hour times must all appear in into_composed; "
            f"matched {len(overlap)} of {len(from_hour)}")
    english = {}
    for time, cell, _de, en in ITEMS:
        if cell in ("into_composed", "from_hour"):
            if time in english and english[time] != en:
                raise SystemExit(
                    f"{time}: the minimal pair needs identical English, got "
                    f"{english[time]!r} and {en!r}")
            english[time] = en

    tasks = []
    for index, (time, cell, german, eng) in enumerate(ITEMS, 1):
        tasks.append({
            "id": f"v{index:02d}",
            "difficulty": cell,
            "de_cell": cell,
            "n_steps": 1,
            "trap": f"{german} = {time}",
            "instruction": {
                "en": CARRIER_EN.format(eng),
                "de": CARRIER_DE.format(german),
            },
            "checks": {
                "en": [{"kind": "alarm_exists", "time": time}],
                "de": [{"kind": "alarm_exists", "time": time}],
            },
            "gold": {
                "en": [{"tool": "set_alarm", "args": {"time": time, "label": LABEL_EN}},
                       {"tool": "done", "args": {}}],
                "de": [{"tool": "set_alarm", "args": {"time": time, "label": LABEL_DE}},
                       {"tool": "done", "args": {}}],
            },
        })

    out = {
        "meta": {
            "today": meta["today"],
            "weekday": meta["weekday"],
            "version": "time-de-halb-1.0",
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
        counts[task["de_cell"]] = counts.get(task["de_cell"], 0) + 1
    for cell, n in sorted(counts.items()):
        print(f"  {cell:<16} {n}")
    print(f"\n  minimal pair verified: {len(from_hour)} clock times appear in both")
    print("  into_composed and from_hour, with identical English in each.")
    print()
    print("The German half, for a native speaker to read straight through:")
    for task in tasks:
        print(f"  {task['id']}  {task['checks']['de'][0]['time']}  "
              f"{task['de_cell']:<14} {task['instruction']['de']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
