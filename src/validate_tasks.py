

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

from .environment import DATE_RE, TIME_RE, TOOL_NAMES

# check kinds the environment can actually evaluate
CHECK_KINDS = {
    "alarm_exists",
    "alarm_absent",
    "event_exists",
    "list_contains",
    "list_not_contains",
    "list_size",
    "music_playing",
    "music_stopped",
    "note_contains",
}

# Fields that carry machine values. These MUST be identical across languages:
# a clock does not change when the request changes language.
STRUCTURAL_FIELDS = ("kind", "time", "date", "list_name", "size")

# Fields that carry human content. These are EXPECTED to differ.
CONTENT_FIELDS = ("label_contains", "title_contains", "item", "text", "artist")


def _structural_signature(check: Dict[str, Any]) -> Tuple:
    return tuple(str(check.get(f, "")) for f in STRUCTURAL_FIELDS)


def second_language(tasks: List[Dict[str, Any]]) -> str:
    """
    Which language is being compared against English in this file.

    A task file holds exactly two wordings of the same task. One of them is
    always English, because English is the within-item ceiling the whole design
    depends on. The other is read from the file rather than hardcoded, so that
    adding Polish does not mean writing Polish into a slot labelled Russian.
    That kind of mislabelling is how a measurement quietly stops meaning what
    its column heading says.
    """
    codes = set()
    for task in tasks:
        codes |= set(task.get("instruction", {}))
    codes.discard("en")
    if len(codes) != 1:
        raise ValueError(
            f"a task file must hold English plus exactly one other language, "
            f"found {sorted(codes) or ['nothing']} besides English"
        )
    return codes.pop()


def check_task(task: Dict[str, Any], l2: str = "ru") -> Tuple[List[str], List[str]]:
    """Return (errors, warnings) for one task. `l2` is the non-English half."""
    errors: List[str] = []
    warnings: List[str] = []
    tid = task.get("id", "<no id>")

    for field in ("id", "instruction", "checks", "gold"):
        if field not in task:
            errors.append(f"{tid}: missing field '{field}'")
    if errors:
        return errors, warnings

    for lang in ("en", l2):
        for field in ("instruction", "checks", "gold"):
            if lang not in task[field]:
                errors.append(f"{tid}: {field} has no '{lang}' version")
    if errors:
        return errors, warnings

    checks_en: List[Dict[str, Any]] = task["checks"]["en"]
    checks_ru: List[Dict[str, Any]] = task["checks"][l2]
    gold_en: List[Dict[str, Any]] = task["gold"]["en"]
    gold_ru: List[Dict[str, Any]] = task["gold"][l2]
    other = l2.upper()

    if len(checks_en) != len(checks_ru):
        errors.append(
            f"{tid}: {len(checks_en)} English checks but {len(checks_ru)} {other} "
            f"ones. The two languages are not being asked for the same thing."
        )
    else:
        for i, (ce, cr) in enumerate(zip(checks_en, checks_ru)):
            if _structural_signature(ce) != _structural_signature(cr):
                errors.append(
                    f"{tid}: check {i} differs structurally between languages: "
                    f"EN {_structural_signature(ce)} vs {other} {_structural_signature(cr)}"
                )
            for field in CONTENT_FIELDS:
                if (field in ce) != (field in cr):
                    errors.append(
                        f"{tid}: check {i} has '{field}' in only one language"
                    )

    for lang, checks in (("en", checks_en), (l2, checks_ru)):
        for i, check in enumerate(checks):
            kind = check.get("kind")
            if kind not in CHECK_KINDS:
                errors.append(f"{tid}: {lang} check {i} has unknown kind '{kind}'")
            if "time" in check and not TIME_RE.match(str(check["time"])):
                errors.append(
                    f"{tid}: {lang} check {i} time '{check['time']}' is not HH:MM"
                )
            if "date" in check and not DATE_RE.match(str(check["date"])):
                errors.append(
                    f"{tid}: {lang} check {i} date '{check['date']}' is not YYYY-MM-DD"
                )

    # ---- gold traces: same plan, different words -------------------------
    tools_en = [a.get("tool") for a in gold_en]
    tools_ru = [a.get("tool") for a in gold_ru]
    if tools_en != tools_ru:
        errors.append(
            f"{tid}: gold traces use different tool sequences.\n"
            f"        EN {tools_en}\n        RU {tools_ru}\n"
            f"        The Russian agent would be solving a different problem."
        )

    for lang, gold in (("en", gold_en), (l2, gold_ru)):
        for i, action in enumerate(gold):
            tool = action.get("tool")
            if tool not in TOOL_NAMES:
                errors.append(f"{tid}: {lang} gold step {i} calls unknown tool '{tool}'")
            args = action.get("args", {})
            if "time" in args and not TIME_RE.match(str(args["time"])):
                errors.append(f"{tid}: {lang} gold step {i} bad time '{args['time']}'")
            if "date" in args and not DATE_RE.match(str(args["date"])):
                errors.append(f"{tid}: {lang} gold step {i} bad date '{args['date']}'")
        if gold and gold[-1].get("tool") != "done":
            errors.append(f"{tid}: {lang} gold trace does not end with done")

    declared = task.get("n_steps")
    actual = len([a for a in gold_en if a.get("tool") != "done"])
    if declared is not None and declared != actual:
        warnings.append(
            f"{tid}: n_steps says {declared} but the gold trace has {actual} "
            f"real actions"
        )

    # ---- things only a human can settle ----------------------------------
    text_en = task["instruction"]["en"]
    text_ru = task["instruction"][l2]
    words_en = len(text_en.split())
    words_ru = len(text_ru.split())
    if words_en and abs(words_ru - words_en) / words_en > 0.5:
        warnings.append(
            f"{tid}: wording length differs a lot ({words_en} EN words vs "
            f"{words_ru} {other} words). Read both aloud and make sure one is not "
            f"spelling out something the other leaves implicit."
        )
    checks_content = any(
        field in check
        for check in checks_en + checks_ru
        for field in CONTENT_FIELDS
    )
    if checks_content:
        for lang, text in (("en", text_en), (l2, text_ru)):
            quotes = text.count("«") + text.count('"') + text.count("'")
            if quotes == 0:
                warnings.append(
                    f"{tid}: the {lang} wording quotes nothing, but its checks "
                    f"test user content, so the translate arms have no span to "
                    f"protect"
                )

    return errors, warnings


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a task file")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--quiet", action="store_true", help="Errors only.")
    args = parser.parse_args(argv)

    with open(args.tasks, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    tasks = data["tasks"]
    all_errors: List[str] = []
    all_warnings: List[str] = []

    ids = Counter(t.get("id") for t in tasks)
    for tid, count in ids.items():
        if count > 1:
            all_errors.append(f"duplicate task id '{tid}' appears {count} times")

    l2 = second_language(tasks)

    for task in tasks:
        errors, warnings = check_task(task, l2)
        all_errors.extend(errors)
        all_warnings.extend(warnings)

    print(f"{args.tasks}: {len(tasks)} tasks, comparing en against {l2}")
    difficulties = Counter(t.get("difficulty", "unknown") for t in tasks)
    print("  difficulty mix: " + ", ".join(f"{k}={v}" for k, v in sorted(difficulties.items())))
    steps = [len([a for a in t["gold"]["en"] if a.get("tool") != "done"]) for t in tasks]
    if steps:
        print(f"  gold actions per task: min={min(steps)} max={max(steps)} "
              f"mean={sum(steps) / len(steps):.1f}")

    if all_warnings and not args.quiet:
        print(f"\n{len(all_warnings)} warning(s) — judgement calls, not blockers:")
        for warning in all_warnings:
            print(f"  ! {warning}")

    if all_errors:
        print(f"\n{len(all_errors)} ERROR(S) — fix these before running a model:")
        for error in all_errors:
            print(f"  x {error}")
        return 1

    print("\nOK: every task is structurally parallel across languages.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
