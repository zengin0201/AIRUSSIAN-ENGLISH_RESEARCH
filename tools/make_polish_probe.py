"""
Generate data/tasks_time_pl.json from data/tasks_time_v2.json.

    python tools/make_polish_probe.py

Why Polish, and why it is not just "a third language".

Russian bundles three constructions that all count into the coming hour:
«половина восьмого» 07:30, «четверть восьмого» 07:15, «двадцать минут
восьмого» 07:20. All three name the hour being entered. All three fail.

Polish splits that bundle down the middle:

    07:30  wpół do ósmej          half TO the eighth   -> counts INTO the hour
    07:15  kwadrans po siódmej    quarter past seven   -> counts FROM the hour
    07:20  dwadzieścia po siódmej twenty past seven    -> counts FROM the hour

So Polish carries the Russian construction and the English construction at the
same time, in the same language, from the same speaker, in the same probe.

That turns a replication into a crossed design, and the two live explanations
of our Russian result make opposite predictions on it:

  (a) "models are worse at Slavic, or at lower-resource languages"
      -> all of Polish fails, including «dwadzieścia po siódmej»

  (b) "models memorise instances of the counting-into construction and never
      learn the rule"  (our reading)
      -> only «wpół do» fails; «po» is answered like English

Nothing else we could run separates those two as cheaply.

It also makes the probe partly self-checking, which matters because neither
author is a native speaker. If our Polish is simply bad, the models fail across
the board and we learn nothing — but we know we learned nothing, because the
`literal` control drops too. A bad translation cannot produce a clean split
between two constructions inside the same language.

**The wording still needs a native speaker before publication.** These sentences
were written from reference grammar, not by a Pole. The forms used are the
textbook ones — `wpół do` + genitive, `kwadrans po` / `N po` + locative, `za N`
+ nominative — but "textbook correct" and "what a person would say" are not the
same claim, and only the second one is the claim this probe needs.

One known wart, recorded rather than hidden: «za dwadzieścia dziewiąta» is the
ordinary way to say 08:40, but «za dwadzieścia» on its own can also mean "in
twenty minutes". The nominative «dziewiąta» is what forces the clock reading.
Russian «без двадцати девять» has no such ambiguity. `subtractive` is a control
category in both languages, so this does not touch the main comparison, but a
Polish `subtractive` score below the Russian one should be read with it in mind.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SOURCE = os.path.join(ROOT, "data", "tasks_time_v2.json")
TARGET = os.path.join(ROOT, "data", "tasks_time_pl.json")

# The Polish wording, written out per task instead of derived by substitution,
# so that a Polish speaker can check the whole probe by reading one table.
#
# `direction` is the thing the experiment turns on: whether the Polish phrase
# names the hour being counted INTO (as Russian does throughout) or the hour
# already COMPLETED (as English does throughout).
POLISH = {
    # half_past — Polish counts INTO the coming hour, exactly as Russian does.
    "g01": ("Ustaw budzik na wpół do trzeciej po południu.", "into_hour"),
    "g02": ("Ustaw budzik na wpół do siódmej rano.", "into_hour"),
    "g03": ("Ustaw budzik na wpół do dziesiątej wieczorem.", "into_hour"),
    "g04": ("Ustaw budzik na wpół do dwunastej przed południem.", "into_hour"),
    "g05": ("Ustaw budzik na wpół do piątej rano.", "into_hour"),

    # quarter_past — Polish counts FROM the completed hour, as English does.
    "g06": ("Ustaw budzik na kwadrans po piątej rano.", "from_hour"),
    "g07": ("Ustaw budzik na kwadrans po ósmej rano.", "from_hour"),
    "g08": ("Ustaw budzik na kwadrans po dziesiątej wieczorem.", "from_hour"),
    "g09": ("Ustaw budzik na kwadrans po pierwszej po południu.", "from_hour"),

    # n_minutes — also FROM the completed hour. This is the cell where Russian
    # is 0 out of 32. If Polish scores here, language is not the variable.
    "g10": ("Ustaw budzik na dwadzieścia po siódmej rano.", "from_hour"),
    "g11": ("Ustaw budzik na pięć po ósmej wieczorem.", "from_hour"),
    "g12": ("Ustaw budzik na dziesięć po dziesiątej rano.", "from_hour"),
    "g13": ("Ustaw budzik na dwadzieścia pięć po pierwszej po południu.", "from_hour"),
    "g14": ("Ustaw budzik na dwadzieścia po drugiej po południu.", "from_hour"),
    "g15": ("Ustaw budzik na dziesięć po szóstej rano.", "from_hour"),
    "g16": ("Ustaw budzik na pięć po dwunastej w południe.", "from_hour"),
    "g17": ("Ustaw budzik na dwadzieścia pięć po dziewiątej wieczorem.", "from_hour"),

    # literal — the probe's health check. Must be near perfect or the Polish
    # carrier sentence is broken and nothing else in the file means anything.
    "g18": ("Ustaw budzik na 07:20.", "digital"),
    "g19": ("Ustaw budzik na 14:30.", "digital"),
    "g20": ("Ustaw budzik na 21:15.", "digital"),
    "g21": ("Ustaw budzik na 05:45.", "digital"),

    # subtractive — subtractive in all three languages. See the wart above.
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
