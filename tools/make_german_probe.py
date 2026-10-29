from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SOURCE = os.path.join(ROOT, "data", "tasks_time_v2.json")
TARGET = os.path.join(ROOT, "data", "tasks_time_de.json")

GERMAN = {
    "g01": ("Stelle den Wecker auf halb drei nachmittags.", "into_hour"),
    "g02": ("Stelle den Wecker auf halb sieben morgens.", "into_hour"),
    "g03": ("Stelle den Wecker auf halb zehn abends.", "into_hour"),
    "g04": ("Stelle den Wecker auf halb zwölf vormittags.", "into_hour"),
    "g05": ("Stelle den Wecker auf halb fünf morgens.", "into_hour"),

    "g06": ("Stelle den Wecker auf viertel nach fünf morgens.", "from_hour"),
    "g07": ("Stelle den Wecker auf viertel neun morgens.", "into_hour_regional"),
    "g08": ("Stelle den Wecker auf viertel nach zehn abends.", "from_hour"),
    "g09": ("Stelle den Wecker auf viertel zwei nachmittags.", "into_hour_regional"),

    "g10": ("Stelle den Wecker auf zwanzig nach sieben morgens.", "from_hour"),
    "g11": ("Stelle den Wecker auf fünf nach acht abends.", "from_hour"),
    "g12": ("Stelle den Wecker auf zehn nach zehn morgens.", "from_hour"),
    "g13": ("Stelle den Wecker auf fünfundzwanzig nach eins nachmittags.", "from_hour"),
    "g14": ("Stelle den Wecker auf zwanzig nach zwei nachmittags.", "from_hour"),
    "g15": ("Stelle den Wecker auf zehn nach sechs morgens.", "from_hour"),
    "g16": ("Stelle den Wecker auf fünf nach zwölf mittags.", "from_hour"),
    "g17": ("Stelle den Wecker auf fünfundzwanzig nach neun abends.", "from_hour"),

    "g18": ("Stelle den Wecker auf 07:20.", "digital"),
    "g19": ("Stelle den Wecker auf 14:30.", "digital"),
    "g20": ("Stelle den Wecker auf 21:15.", "digital"),
    "g21": ("Stelle den Wecker auf 05:45.", "digital"),

    "g22": ("Stelle den Wecker auf zwanzig vor neun morgens.", "subtractive"),
    "g23": ("Stelle den Wecker auf viertel vor sechs morgens.", "subtractive"),
    "g24": ("Stelle den Wecker auf zehn vor sieben abends.", "subtractive"),
}

LABEL_DE = "Wecker"

NOTE = (
    "German mirror of the 24-item probe. The English half is byte-identical to "
    "data/tasks_time_v2.json, so English acts as the same within-item ceiling "
    "in every language file and they can be pooled. German is the falsification "
    "test, not another replication. German counts INTO the coming hour as "
    "Russian does, but only with a closed set of three lexical items (halb, "
    "viertel, dreiviertel); there is no productive German form for arbitrary "
    "minutes, so 07:20 can only be zwanzig nach sieben. Direction of counting "
    "is held constant and productivity is varied. The memorisation reading "
    "predicts German into_hour is read CORRECTLY, because three frequent "
    "phrases are memorisable where a rule is not, and therefore that German "
    "looks like English throughout. The rival reading, that counting into the "
    "coming hour is simply hard, predicts German into_hour fails like Russian. "
    "German is also high-resource in every corpus, which removes the resource "
    "explanation without needing to measure anything. Two items (g07, g09) are "
    "flagged de_regional: viertel neun and viertel zwei are used in the east "
    "and south and may not parse for a northern speaker, so their gold label is "
    "contested among natives and they must be excluded from any primary "
    "comparison unless deliberately included. viertel is lowercase throughout "
    "on purpose, so that viertel nach fuenf and viertel neun differ by one word "
    "and not also by capitalisation. WORDING NOT YET CHECKED BY A NATIVE "
    "SPEAKER — see tools/make_german_probe.py."
)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    with open(SOURCE, "r", encoding="utf-8") as handle:
        source = json.load(handle)

    missing = {t["id"] for t in source["tasks"]} - set(GERMAN)
    if missing:
        raise SystemExit(f"no German wording for: {sorted(missing)}")

    tasks = []
    for task in source["tasks"]:
        wording, direction = GERMAN[task["id"]]
        time = task["checks"]["en"][0]["time"]

        tasks.append({
            "id": task["id"],
            "difficulty": task["difficulty"],
            "de_direction": direction,
            "de_regional": direction.endswith("_regional"),
            "n_steps": 1,
            "trap": f"{wording} = {time}",
            "instruction": {
                "en": task["instruction"]["en"],
                "de": wording,
            },
            "checks": {
                "en": [dict(c) for c in task["checks"]["en"]],
                "de": [dict(c) for c in task["checks"]["en"]],
            },
            "gold": {
                "en": [dict(a) for a in task["gold"]["en"]],
                "de": [
                    {"tool": "set_alarm", "args": {"time": time, "label": LABEL_DE}},
                    {"tool": "done", "args": {}},
                ],
            },
        })

    out = {
        "meta": {
            "today": source["meta"]["today"],
            "weekday": source["meta"]["weekday"],
            "version": "time-de-1.0",
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
    regional = [t["id"] for t in tasks if t["de_regional"]]
    print(f"  flagged de_regional: {', '.join(regional)} "
          f"(contested gold, exclude by default)")
    print()
    print("The German half, for a native speaker to read straight through:")
    for task in tasks:
        print(f"  {task['id']}  {task['checks']['en'][0]['time']}  "
              f"{task['de_direction']:<20} {task['instruction']['de']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
