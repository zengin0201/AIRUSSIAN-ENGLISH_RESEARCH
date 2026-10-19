"""
Generate data/tasks_time_case.json — the case-variation probe.

    python tools/make_case_probe.py

Richard Sproat, in correspondence, 14 October 2026:

    "Двадцать минут восьмого will not always show up in that form, if for
    example the time expression is found in a context that requires a
    different case form (e.g. двадцати минутах восьмого). This adds further
    noise to the task."

He is right that the existing probe tests one surface shape. It asks for every
alarm with «на ...», which takes the accusative throughout — «на половину
третьего», where половину is accusative and половина is the citation form. For
«двадцать минут» and «четверть» the accusative is syncretic with the
nominative, so the whole probe effectively sits in one slot of a paradigm a
speaker produces in several.

This file varies the carrier frame so the same construction appears in three
cases.

    frame          preposition  case          Russian                   English
    F1 at          на           accusative    на двадцать минут восьмого   for ...
    F2 from        с            genitive      с двадцати минут восьмого    from ...
    F3 by          к            dative        к двадцати минутам восьмого  by ...

F1 is the existing frame, carried over unchanged so the new file is anchored to
the old result rather than floating free.

**On the prepositional, which is the form Sproat actually named.** «двадцати
минутах восьмого» is grammatical, but it does not arise in an imperative
instruction to an assistant: «в двадцати минутах» reads as a distance, not a
clock time, and the other prepositional contexts are discursive rather than
task-giving. Writing one anyway would test a sentence no user would type. The
three frames here are the oblique forms this register actually produces, and
reporting that the prepositional does not occur in it is part of the answer to
his question rather than an evasion of it.

**The confound, and the control for it.** Case is varied by changing the
preposition, so case is confounded with the frame: "block the calendar from X"
is a different sentence from "set an alarm for X" quite apart from morphology.
The English half carries the same three frames — for / from / by — and English
has no case. If English is flat across the three and Russian drops in the
oblique two, the frame is not the difficulty. That is the same within-item
ceiling the rest of the project uses, applied to a new axis.

**What each frame checks.** F1 and F3 are a single alarm. F2 is a calendar
event, because "from X" asks for something that starts at a time and an alarm
does not start. The time checked is identical in all three, and the English
half of each item asks for the same tool at the same time, so the comparison
that matters — Russian against English inside one frame — never crosses tools.

**Prediction, written before the run** (also in NOTES, F60):
the oblique frames are worse than F1 for the productive form, because the
oblique forms are rarer still, and the account the German probe left us with is
that rarity plus insufficient capacity is what breaks these readings. If F2 and
F3 match F1, case adds nothing and Sproat's worry is answered in the negative,
which is equally reportable.

**Wording not checked by a native speaker of Russian other than the author.**
The author is a native speaker, which the Polish and German probes could not
say. «к двадцати минутам восьмого» is the marginal one: «к половине восьмого»
is entirely ordinary, the productive form in the dative much less so. It is
flagged per item.
"""

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

# (frame, case, tool) -> how the sentence is built in each language.
FRAMES = {
    "at":   ("accusative", "set_alarm"),
    "from": ("genitive", "add_event"),
    "by":   ("dative", "set_alarm"),
}

# Each entry is (time, category, Russian forms by frame, English form).
# The Russian is written out per frame rather than inflected by code, so that
# the whole probe can be audited by reading one table.
#
#   accusative   на X          genitive     с X          dative   к X
ITEMS = [
    # ---- productive: N минут X-ого. The cell that is 0 of 8 everywhere. ----
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

    # ---- половина X: the frequent fixed phrase, 45% in the main probe. ----
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

    # ---- literal: the health check. Digits do not inflect, so this cell
    #      must be flat across all three frames in both languages. If it is
    #      not, the frame itself is broken and nothing else can be read. ----
    ("07:45", "literal", "07:45", "07:45", "07:45", "07:45"),
    ("12:15", "literal", "12:15", "12:15", "12:15", "12:15"),
    ("16:40", "literal", "16:40", "16:40", "16:40", "16:40"),
    ("22:05", "literal", "22:05", "22:05", "22:05", "22:05"),
    ("03:50", "literal", "03:50", "03:50", "03:50", "03:50"),
]

# «к двадцати минутам восьмого» is grammatical but much less ordinary than
# «к половине восьмого». Flagged so a reader is not asked to take it on trust.
MARGINAL = {("by", "n_minutes")}

CARRIER = {
    ("at", "ru"): "Поставь будильник на {}.",
    ("at", "en"): "Set an alarm for {}.",
    ("from", "ru"): "Добавь в календарь встречу с {}.",
    ("from", "en"): "Add a meeting to the calendar from {}.",
    # "Set an alarm for by X" is not English. «Разбуди меня к половине
    # восьмого» is entirely ordinary Russian and "wake me by X" is ordinary
    # English, so the carrier is natural in both and whatever marginality
    # remains sits on the time expression, which is where we want it.
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
                # title_contains is required by the verifier. An empty
                # string matches any title, which makes the check purely
                # date-and-time — exactly parallel to alarm_exists, which
                # checks only the time. Asking for a specific title would
                # make the Russian arm also get the word for "meeting"
                # right, which is a second thing to fail at and has
                # nothing to do with reading the clock.
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
