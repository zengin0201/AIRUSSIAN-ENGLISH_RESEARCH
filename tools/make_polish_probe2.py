from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SOURCE = os.path.join(ROOT, "data", "tasks_time_pl.json")
TARGET = os.path.join(ROOT, "data", "tasks_time_pl2.json")

FIXES = {
    "g22": ("Ustaw budzik za dwadzieścia dziewiąta rano.",
            "Ustaw budzik na za dwadzieścia dziewiąta rano."),
    "g23": ("Ustaw budzik za kwadrans szósta rano.",
            "Ustaw budzik na za kwadrans szósta rano."),
    "g24": ("Ustaw budzik za dziesięć siódma wieczorem.",
            "Ustaw budzik na za dziesięć siódma wieczorem."),
}

NOTE_SUFFIX = (
    " CORRECTED SUBTRACTIVE CARRIER: the three subtractive items originally "
    "read «Ustaw budzik za ...», omitting the «na» that governs the target "
    "time in every other item, so they parsed as setting a timer for a "
    "duration. Found independently by the assistant (F67) and by Gemini on a "
    "blind sheet with no intended times (F68), which marked all three wrong. "
    "Only that preposition changed. Register objections to «kwadrans» and to "
    "«dwadzieścia pięć po» were left alone: they did not predict failure and a "
    "native speaker rules on them. data/tasks_time_pl.json is kept unchanged "
    "because the pl_*.jsonl results were produced against it."
)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    with open(SOURCE, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    seen = set()
    for task in data["tasks"]:
        fix = FIXES.get(task["id"])
        if not fix:
            continue
        old, new = fix
        if task["instruction"]["pl"] != old:
            raise SystemExit(
                f"{task['id']}: expected {old!r}, found "
                f"{task['instruction']['pl']!r} — source file has moved")
        task["instruction"]["pl"] = new
        task["trap"] = f"{new} = {task['checks']['pl'][0]['time']}"
        seen.add(task["id"])

    missing = set(FIXES) - seen
    if missing:
        raise SystemExit(f"items not found in source: {sorted(missing)}")

    data["meta"]["version"] = "time-pl-2.0"
    data["meta"]["note"] = data["meta"]["note"] + NOTE_SUFFIX

    with open(TARGET, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"wrote {TARGET} with {len(data['tasks'])} tasks")
    print(f"  changed {len(seen)} items, all in the subtractive cell:")
    for task_id in sorted(seen):
        old, new = FIXES[task_id]
        print(f"    {task_id}")
        print(f"      was: {old}")
        print(f"      now: {new}")
    print()
    print("  everything else is byte-identical to data/tasks_time_pl.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
