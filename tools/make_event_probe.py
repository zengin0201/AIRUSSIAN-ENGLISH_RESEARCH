

from __future__ import annotations

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TODAY = "2026-09-07"
TITLE = "meeting"

# How the Russian time phrase attaches to "событие ... под названием".
# Key: category. Value: (preposition, accusative -> prepositional fixes).
RU_FRAME = {
    "half_past": ("в ", {"половину": "половине"}),
    "quarter_past": ("в ", {}),
    "n_minutes": ("в ", {}),
    "literal": ("в ", {}),
    "subtractive": ("", {}),          # «без двадцати девять» takes no preposition
}


def ru_phrase(instruction: str, category: str) -> str:
    """Pull the time phrase out of the alarm sentence and re-case it."""
    text = instruction.strip().rstrip(".")
    text = re.sub(r"^Поставь будильник\s*", "", text)
    text = re.sub(r"^на\s+", "", text)          # «на половину третьего» -> «половину третьего»
    preposition, fixes = RU_FRAME[category]
    for accusative, prepositional in fixes.items():
        text = text.replace(accusative, prepositional)
    return preposition + text


def en_phrase(instruction: str) -> str:
    text = instruction.strip().rstrip(".")
    text = re.sub(r"^Set an alarm for\s*", "", text)
    return "at " + text


def main() -> None:
    source = os.path.join(ROOT, "data", "tasks_time_v2.json")
    with open(source, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    tasks = []
    for task in data["tasks"]:
        category = task["difficulty"]
        time_value = task["checks"]["en"][0]["time"]

        en = (f'Add an event today {en_phrase(task["instruction"]["en"])} '
              f'called "{TITLE}".')
        ru = (f'Добавь событие на сегодня '
              f'{ru_phrase(task["instruction"]["ru"], category)} '
              f'под названием "{TITLE}".')

        check = {"kind": "event_exists", "date": TODAY, "time": time_value,
                 "title_contains": TITLE}
        gold = [
            {"tool": "add_event",
             "args": {"date": TODAY, "time": time_value, "title": TITLE}},
            {"tool": "done", "args": {}},
        ]

        tasks.append({
            "id": task["id"].replace("g", "e"),
            "difficulty": category,
            "n_steps": 1,
            "trap": task.get("trap", ""),
            "instruction": {"en": en, "ru": ru},
            "checks": {"en": [dict(check)], "ru": [dict(check)]},
            "gold": {"en": [dict(a) for a in gold], "ru": [dict(a) for a in gold]},
        })

    out = {
        "meta": {
            "today": TODAY,
            "weekday": data["meta"]["weekday"],
            "version": "event-1.0",
            "note": (
                "Second domain for the time probe. The same 24 spoken time "
                "expressions as data/tasks_time_v2.json, asked as calendar "
                "events instead of alarms, so the effect can be separated from "
                "the one tool it was first measured on. The event title is the "
                "same English identifier in both languages, so the only thing "
                "that differs between the two versions is the time. Generated "
                "by tools/make_event_probe.py; edit the generator, not this "
                "file."
            ),
        },
        "tasks": tasks,
    }

    target = os.path.join(ROOT, "data", "tasks_event.json")
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=2)
    print(f"written {target}: {len(tasks)} tasks")
    print()
    print("Read these aloud before running anything:")
    for task in tasks[:3] + tasks[9:11] + tasks[-3:]:
        print(f'  {task["id"]} [{task["difficulty"]}] -> {task["checks"]["en"][0]["time"]}')
        print(f'      EN  {task["instruction"]["en"]}')
        print(f'      RU  {task["instruction"]["ru"]}')


if __name__ == "__main__":
    main()
