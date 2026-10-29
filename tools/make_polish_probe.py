from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SOURCE = os.path.join(ROOT, "data", "tasks_time_v2.json")
TARGET = os.path.join(ROOT, "data", "tasks_time_pl.json")

POLISH = {
    "g01": ("Ustaw budzik na wpół do trzeciej po południu.", "into_hour"),
    "g02": ("Ustaw budzik na wpół do siódmej rano.", "into_hour"),
    "g03": ("Ustaw budzik na wpół do dziesiątej wieczorem.", "into_hour"),
    "g04": ("Ustaw budzik na wpół do dwunastej przed południem.", "into_hour"),
    "g05": ("Ustaw budzik na wpół do piątej rano.", "into_hour"),

    "g06": ("Ustaw budzik na kwadrans po piątej rano.", "from_hour"),
    "g07": ("Ustaw budzik na kwadrans po ósmej rano.", "from_hour"),
    "g08": ("Ustaw budzik na kwadrans po dziesiątej wieczorem.", "from_hour"),
    "g09": ("Ustaw budzik na kwadrans po pierwszej po południu.", "from_hour"),

    "g10": ("Ustaw budzik na dwadzieścia po siódmej rano.", "from_hour"),
    "g11": ("Ustaw budzik na pięć po ósmej wieczorem.", "from_hour"),
    "g12": ("Ustaw budzik na dziesięć po dziesiątej rano.", "from_hour"),
    "g13": ("Ustaw budzik na dwadzieścia pięć po pierwszej po południu.", "from_hour"),
    "g14": ("Ustaw budzik na dwadzieścia po drugiej po południu.", "from_hour"),
    "g15": ("Ustaw budzik na dziesięć po szóstej rano.", "from_hour"),
    "g16": ("Ustaw budzik na pięć po dwunastej w południe.", "from_hour"),
    "g17": ("Ustaw budzik na dwadzieścia pięć po dziewiątej wieczorem.", "from_hour"),

    "g18": ("Ustaw budzik na 07:20.", "digital"),
    "g19": ("Ustaw budzik na 14:30.", "digital"),
    "g20": ("Ustaw budzik na 21:15.", "digital"),
    "g21": ("Ustaw budzik na 05:45.", "digital"),

    "g22": ("Ustaw budzik za dwadzieścia dziewiąta rano.", "subtractive"),
    "g23": ("Ustaw budzik za kwadrans szósta rano.", "subtractive"),
    "g24": ("Ustaw budzik za dziesięć siódma wieczorem.", "subtractive"),
}

LABEL_PL = "budzik"

NOTE = (
    "Polish mirror of the 24-item probe. The English half is byte-identical to "
    "data/tasks_time_v2.json, so English acts as the same within-item ceiling "
    "in both files and the two can be pooled. The Polish half is the crossed "
    "design: Polish counts INTO the coming hour for «wpół do X» (07:30 = wpół "
    "do ósmej) and FROM the completed hour for «kwadrans po X» and «N po X» "
    "(07:20 = dwadzieścia po siódmej). Russian counts into the hour for all "
    "three. So `pl_direction` splits the same clock times by construction "
    "inside one language. Prediction from the memorisation reading: Polish "
    "`into_hour` behaves like Russian and Polish `from_hour` behaves like "
    "English. The rival reading, that models are simply worse at Slavic or at "
    "lower-resource languages, predicts both Polish cells fail. WORDING NOT "
    "YET CHECKED BY A NATIVE SPEAKER — see tools/make_polish_probe.py."
)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    with open(SOURCE, "r", encoding="utf-8") as handle:
        source = json.load(handle)

    missing = {t["id"] for t in source["tasks"]} - set(POLISH)
    if missing:
        raise SystemExit(f"no Polish wording for: {sorted(missing)}")

    tasks = []
    for task in source["tasks"]:
        wording, direction = POLISH[task["id"]]
        time = task["checks"]["en"][0]["time"]

        tasks.append({
            "id": task["id"],
            "difficulty": task["difficulty"],
            "pl_direction": direction,
            "n_steps": 1,
            "trap": f"«{wording}» = {time}",
            "instruction": {
                "en": task["instruction"]["en"],
                "pl": wording,
            },
            "checks": {
                "en": [dict(c) for c in task["checks"]["en"]],
                "pl": [dict(c) for c in task["checks"]["en"]],
            },
            "gold": {
                "en": [dict(a) for a in task["gold"]["en"]],
                "pl": [
                    {"tool": "set_alarm", "args": {"time": time, "label": LABEL_PL}},
                    {"tool": "done", "args": {}},
                ],
            },
        })

    out = {
        "meta": {
            "today": source["meta"]["today"],
            "weekday": source["meta"]["weekday"],
            "version": "time-pl-1.0",
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
        counts[task["pl_direction"]] = counts.get(task["pl_direction"], 0) + 1
    for direction, n in sorted(counts.items()):
        print(f"  pl_direction {direction:<12} {n}")
    print()
    print("The Polish half, for a native speaker to read straight through:")
    for task in tasks:
        print(f"  {task['id']}  {task['checks']['en'][0]['time']}  "
              f"{task['pl_direction']:<12} {task['instruction']['pl']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
