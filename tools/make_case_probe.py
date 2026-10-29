from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SOURCE = os.path.join(ROOT, "data", "tasks_time_v2.json")
TARGET = os.path.join(ROOT, "data", "tasks_time_case.json")

LABEL_RU = "будильник"
LABEL_EN = "alarm"
TITLE_RU = "встреча"
TITLE_EN = "meeting"

FRAMES = {
    "at":   ("accusative", "set_alarm"),
    "from": ("genitive", "add_event"),
    "by":   ("dative", "set_alarm"),
}

ITEMS = [
    ("07:20", "n_minutes",
     "двадцать минут восьмого утра",
     "двадцати минут восьмого утра",
     "двадцати минутам восьмого утра",
     "twenty past seven in the morning"),
    ("10:10", "n_minutes",
     "десять минут одиннадцатого утра",
     "десяти минут одиннадцатого утра",
     "десяти минутам одиннадцатого утра",
     "ten past ten in the morning"),
    ("13:25", "n_minutes",
     "двадцать пять минут второго дня",
     "двадцати пяти минут второго дня",
     "двадцати пяти минутам второго дня",
     "twenty five past one in the afternoon"),
    ("20:05", "n_minutes",
     "пять минут девятого вечера",
     "пяти минут девятого вечера",
     "пяти минутам девятого вечера",
     "five past eight in the evening"),
    ("06:10", "n_minutes",
     "десять минут седьмого утра",
     "десяти минут седьмого утра",
     "десяти минутам седьмого утра",
     "ten past six in the morning"),

    ("14:30", "half_past",
     "половину третьего дня",
     "половины третьего дня",
     "половине третьего дня",
     "half past two in the afternoon"),
    ("06:30", "half_past",
     "половину седьмого утра",
     "половины седьмого утра",
     "половине седьмого утра",
     "half past six in the morning"),
    ("21:30", "half_past",
     "половину десятого вечера",
     "половины десятого вечера",
     "половине десятого вечера",
     "half past nine in the evening"),
    ("04:30", "half_past",
     "половину пятого утра",
     "половины пятого утра",
     "половине пятого утра",
     "half past four in the morning"),
    ("11:30", "half_past",
     "половину двенадцатого утра",
     "половины двенадцатого утра",
     "половине двенадцатого утра",
     "half past eleven in the morning"),

    ("07:45", "literal", "07:45", "07:45", "07:45", "07:45"),
    ("12:15", "literal", "12:15", "12:15", "12:15", "12:15"),
    ("16:40", "literal", "16:40", "16:40", "16:40", "16:40"),
    ("22:05", "literal", "22:05", "22:05", "22:05", "22:05"),
    ("03:50", "literal", "03:50", "03:50", "03:50", "03:50"),
]

MARGINAL = {("by", "n_minutes")}

CARRIER = {
    ("at", "ru"): "Поставь будильник на {}.",
    ("at", "en"): "Set an alarm for {}.",
    ("from", "ru"): "Добавь в календарь встречу с {}.",
    ("from", "en"): "Add a meeting to the calendar from {}.",
    ("by", "ru"): "Разбуди меня к {}.",
    ("by", "en"): "Wake me by {}.",
}

NOTE = (
    "Case-variation probe, 45 items, answering Richard Sproat's point that the "
    "main probe tests one surface shape of an inflecting paradigm. Three carrier "
    "frames put the same construction in three cases: на + accusative (the "
    "existing frame, carried over unchanged), с + genitive, к + dative. The "
    "prepositional form he named, двадцати минутах восьмого, is grammatical but "
    "does not arise in an imperative instruction to an assistant, and that is "
    "reported rather than faked with an invented sentence. Case is confounded "
    "with frame by construction, and the English half carries the same three "
    "frames as a control: English has no case, so if English is flat across the "
    "frames and Russian drops in the oblique two, the frame is not the "
    "difficulty. The from-frame uses add_event because an alarm does not start "
    "at a time; within any one frame both languages call the same tool at the "
    "same time. Cells: n_minutes (productive, 0 of 8 in the main probe), "
    "half_past (frequent fixed, 45 percent), literal (health check, digits do "
    "not inflect and must be flat). PREDICTION REGISTERED IN NOTES F60 BEFORE "
    "THE RUN. The dative of the productive form is marginal and flagged per "
    "item; see tools/make_case_probe.py."
)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    with open(SOURCE, "r", encoding="utf-8") as handle:
        meta = json.load(handle)["meta"]
    date = meta["today"]

    tasks = []
    index = 0
    for time, category, acc, gen, dat, english in ITEMS:
        russian_by_frame = {"at": acc, "from": gen, "by": dat}
        for frame in ("at", "from", "by"):
            index += 1
            tool = FRAMES[frame][1]
            case = FRAMES[frame][0]
            ru = CARRIER[(frame, "ru")].format(russian_by_frame[frame])
            en = CARRIER[(frame, "en")].format(english)

            if tool == "set_alarm":
                check = {"kind": "alarm_exists", "time": time}
                gold_ru = {"tool": "set_alarm",
                           "args": {"time": time, "label": LABEL_RU}}
                gold_en = {"tool": "set_alarm",
                           "args": {"time": time, "label": LABEL_EN}}
            else:
                check = {"kind": "event_exists", "date": date,
                         "time": time, "title_contains": ""}
                gold_ru = {"tool": "add_event",
                           "args": {"date": date, "time": time, "title": TITLE_RU}}
                gold_en = {"tool": "add_event",
                           "args": {"date": date, "time": time, "title": TITLE_EN}}

            tasks.append({
                "id": f"c{index:02d}",
                "difficulty": category,
                "case_frame": frame,
                "case": case,
                "marginal_wording": (frame, category) in MARGINAL,
                "n_steps": 1,
                "trap": f"{russian_by_frame[frame]} = {time}",
                "instruction": {"en": en, "ru": ru},
                "checks": {"en": [dict(check)], "ru": [dict(check)]},
                "gold": {
                    "en": [gold_en, {"tool": "done", "args": {}}],
                    "ru": [gold_ru, {"tool": "done", "args": {}}],
                },
            })

    out = {
        "meta": {
            "today": meta["today"],
            "weekday": meta["weekday"],
            "version": "time-case-1.0",
            "note": NOTE,
        },
        "tasks": tasks,
    }

    with open(TARGET, "w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"wrote {TARGET} with {len(tasks)} tasks")
    print()
    print(f"    {'cell':<12}" + "".join(f"{f:>10}" for f in ("at/acc", "from/gen", "by/dat")))
    for category in ("n_minutes", "half_past", "literal"):
        counts = [sum(1 for t in tasks
                      if t["difficulty"] == category and t["case_frame"] == f)
                  for f in ("at", "from", "by")]
        print(f"    {category:<12}" + "".join(f"{c:>10}" for c in counts))
    marginal = [t["id"] for t in tasks if t["marginal_wording"]]
    print(f"\n  flagged marginal_wording: {', '.join(marginal)}")
    print()
    print("The Russian half, for a native speaker to read straight through:")
    for task in tasks:
        mark = " <- marginal" if task["marginal_wording"] else ""
        print(f"  {task['id']}  {task['checks']['ru'][0]['time']}  "
              f"{task['case']:<11} {task['instruction']['ru']}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
