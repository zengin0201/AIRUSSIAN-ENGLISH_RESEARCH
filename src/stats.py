from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .metrics import fisher_exact, wilson_ci

ALPHA = 0.01


def format_p(p: float, alpha: float = ALPHA) -> str:
    if p < alpha:
        return f"p < {('%.2f' % alpha).lstrip('0')}"
    if p >= 1:
        return "p = 1.0"
    return f"p = {('%.2g' % p).lstrip('0')}"

FAMILY = ("half_past", "quarter_past", "n_minutes")
FAMILY_ALL = FAMILY + ("into_hour",)

CATEGORY_LABEL = {
    "literal": "literal (control)",
    "half_past": "«половина X»",
    "quarter_past": "«четверть X»",
    "n_minutes": "«N минут X-ого»",
    "subtractive": "«без N» (control)",
    "into_hour": "counting into hour",
    "arithmetic": "arithmetic",
}

MAIN_RUNS = [
    ("Qwen2.5-3B", "results/v2_qwen3b.jsonl"),
    ("Phi-3.5-mini", "results/v2_phi.jsonl"),
    ("qwen3.6-27b", "results/v2_qwen27b.jsonl"),
    ("gpt-oss-120b", "results/v2_gptoss120b.jsonl"),
]
COT_RUNS = [
    ("Qwen2.5-3B", "results/v2_qwen3b.jsonl", "results/cot_qwen3b.jsonl"),
    ("Phi-3.5-mini", "results/v2_phi.jsonl", "results/cot_phi.jsonl"),
    ("gpt-oss-120b", "results/v2_gptoss120b.jsonl", "results/cot_gptoss120b.jsonl"),
]
EVENT_RUNS = [
    ("Qwen2.5-3B", "results/v2_qwen3b.jsonl", "results/event_qwen3b.jsonl"),
    ("Phi-3.5-mini", "results/v2_phi.jsonl", "results/event_phi.jsonl"),
    ("gpt-oss-120b", "results/v2_gptoss120b.jsonl", "results/event_gptoss120b.jsonl"),
]
FEWSHOT_RUNS = [
    ("Qwen2.5-3B", "results/v2_qwen3b.jsonl", "results/fs_qwen3b.jsonl"),
    ("Phi-3.5-mini", "results/v2_phi.jsonl", "results/fs_phi.jsonl"),
    ("qwen3.8-27b", "results/fs_qwen38.jsonl", "results/fs_qwen38.jsonl"),
    ("gpt-oss-120b", "results/v2_gptoss120b.jsonl", "results/fs_gptoss120b.jsonl"),
]

STABILITY_RUNS = [
    ("gpt-oss-120b", "results/stab_gptoss120b.jsonl"),
    ("Qwen2.5-3B", "results/stab_qwen3b.jsonl"),
]

TASKS_MAIN = "data/tasks_time_v2.json"

FIRST_PROBE_RUNS = [
    ("Qwen2.5-3B", "results/time_probe.jsonl"),
    ("Phi-3.5-mini", "results/time_phi.jsonl"),
    ("qwen3.6-27b", "results/time_qwen27b.jsonl"),
    ("gpt-oss-120b", "results/time_gptoss120b.jsonl"),
]
RUSSIAN_ARMS = ("ru", "ru_sys_ru", "ru_translate_safe")

TASKS_PL = "data/tasks_time_pl.json"
POLISH_RUNS = [
    ("Qwen2.5-3B", "results/pl_qwen3b.jsonl"),
    ("Phi-3.5-mini", "results/pl_phi.jsonl"),
    ("qwen3.6-27b", "results/pl_qwen27b.jsonl"),
    ("gpt-oss-120b", "results/pl_gptoss120b.jsonl"),
]


def load(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"  !! {path} not found, skipped")
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("record_type") != "meta":
                rows.append(row)
    return rows


def load_tasks(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        blob = json.load(handle)
    tasks = blob["tasks"] if isinstance(blob, dict) else blob
    return {t["id"]: t for t in tasks}


def tally(
    rows: Iterable[Dict[str, Any]],
    arm: str,
    categories: Optional[Sequence[str]] = None,
    ids: Optional[Sequence[str]] = None,
) -> Tuple[int, int]:
    ok = total = 0
    wanted_ids = set(ids) if ids else None
    wanted_cat = set(categories) if categories else None
    for row in rows:
        if row["arm"] != arm:
            continue
        if wanted_cat and row.get("difficulty") not in wanted_cat:
            continue
        if wanted_ids and row["task_id"] not in wanted_ids:
            continue
        if row.get("error"):
            continue
        total += 1
        ok += bool(row["success"])
    return ok, total


def crashes(rows: Iterable[Dict[str, Any]], arm: Optional[str] = None) -> int:
    return sum(1 for row in rows
               if row.get("error") and (arm is None or row.get("arm") == arm))


def minutes(value: str) -> int:
    hours, mins = value.split(":")
    return int(hours) * 60 + int(mins)


class Report:

    def __init__(self) -> None:
        self.p_values: List[float] = []
        self.bounds: List[float] = []

    def head(self, title: str) -> None:
        print()
        print("=" * 74)
        print(title)
        print("=" * 74)

    def compare(self, label: str, en: Tuple[int, int], other: Tuple[int, int],
                lang: str = "ru") -> float:
        en_ok, en_n = en
        other_ok, other_n = other
        p = fisher_exact(en_ok, en_n - en_ok, other_ok, other_n - other_ok)
        self.p_values.append(p)
        en_pct = 100 * en_ok / en_n if en_n else 0.0
        other_pct = 100 * other_ok / other_n if other_n else 0.0
        print(f"  {label:<26} en {en_ok:>3}/{en_n:<3} ({en_pct:5.1f}%)   "
              f"{lang} {other_ok:>3}/{other_n:<3} ({other_pct:5.1f}%)   "
              f"{format_p(p):<9} [exact {p:.2e}]")
        en_ci = wilson_ci(en_ok, en_n) if en_n else (0.0, 0.0)
        other_ci = wilson_ci(other_ok, other_n) if other_n else (0.0, 0.0)
        self.bounds += [*en_ci, *other_ci]
        print(f"  {'':<26} 99% CI    en [{en_ci[0]:.1f}, {en_ci[1]:.1f}]"
              f"{'':<6}{lang} [{other_ci[0]:.1f}, {other_ci[1]:.1f}]")
        return p


def section_gradient(report: Report) -> None:
    report.head("Table 1 — success by category, four models, greedy (en · ru)")
    order = ["literal", "half_past", "quarter_past", "n_minutes", "subtractive"]
    header = f"  {'model':<16}" + "".join(f"{CATEGORY_LABEL[c].split(' ')[0]:>18}" for c in order)
    print(header)
    pooled: Dict[str, Dict[str, List[int]]] = {
        c: {"en": [0, 0], "ru": [0, 0]} for c in order
    }
    for name, path in MAIN_RUNS:
        rows = load(path)
        if not rows:
            continue
        cells = []
        for cat in order:
            en = tally(rows, "en", [cat])
            ru = tally(rows, "ru", [cat])
            for arm, got in (("en", en), ("ru", ru)):
                pooled[cat][arm][0] += got[0]
                pooled[cat][arm][1] += got[1]
            cells.append(f"{en[0]}/{en[1]} · {ru[0]}/{ru[1]}")
        print(f"  {name:<16}" + "".join(f"{c:>18}" for c in cells))

    report.head("Table 2 — the gradient, pooled over the four models")
    for cat in ["half_past", "quarter_past", "n_minutes"]:
        report.compare(CATEGORY_LABEL[cat],
                       tuple(pooled[cat]["en"]), tuple(pooled[cat]["ru"]))
    fam_en = [sum(pooled[c]["en"][i] for c in FAMILY) for i in (0, 1)]
    fam_ru = [sum(pooled[c]["ru"][i] for c in FAMILY) for i in (0, 1)]
    report.compare("whole family", tuple(fam_en), tuple(fam_ru))
    print()
    for cat in ("literal", "subtractive"):
        report.compare(CATEGORY_LABEL[cat],
                       tuple(pooled[cat]["en"]), tuple(pooled[cat]["ru"]))
    print("  (the two controls must not separate; that is their whole job)")


def section_error(report: Report) -> None:
    report.head("The error is one hour — (written - correct) in minutes")
    tasks = load_tasks(TASKS_MAIN)
    for arm in ("en", "ru"):
        counts: Counter = Counter()
        total = missing = 0
        for _, path in MAIN_RUNS:
            for row in load(path):
                if row.get("difficulty") not in FAMILY_ALL or row["arm"] != arm:
                    continue
                total += 1
                alarms = row["final_state"]["alarms"]
                if not alarms:
                    missing += 1
                    continue
                want = tasks[row["task_id"]]["checks"]["en"][0]["time"]
                counts[minutes(alarms[0]["time"]) - minutes(want)] += 1
        print(f"  {arm}: n = {total}"
              + (f", no alarm written at all = {missing}" if missing else ""))
        for delta, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            tag = "  <- exactly one hour late" if delta == 60 else ""
            name = "correct" if delta == 0 else f"{delta:+d} min"
            print(f"      {name:>12} : {n:>3}{tag}")
        print()


def section_robustness(report: Report) -> None:
    report.head("Robustness — dropping the six «дня» items that are surface-ambiguous")
    tasks = load_tasks(TASKS_MAIN)
    family_ids, unambiguous, ambiguous = [], [], []
    for task_id, task in tasks.items():
        if task.get("difficulty") not in FAMILY_ALL:
            continue
        family_ids.append(task_id)
        blob = json.dumps(task, ensure_ascii=False)
        (unambiguous if "дня" not in blob else ambiguous).append(task_id)
    print(f"  family items {len(family_ids)}, "
          f"ambiguous «дня» {len(ambiguous)} -> {sorted(ambiguous)}")
    print(f"  «утра»/«вечера» only: {len(unambiguous)} -> {sorted(unambiguous)}")
    print()
    for label, ids in (("all items", family_ids), ("unambiguous only", unambiguous)):
        en = [0, 0]
        ru = [0, 0]
        for _, path in MAIN_RUNS:
            rows = load(path)
            for arm, acc in (("en", en), ("ru", ru)):
                ok, n = tally(rows, arm, FAMILY_ALL, ids)
                acc[0] += ok
                acc[1] += n
        report.compare(label, tuple(en), tuple(ru))


def section_cot(report: Report) -> None:
    report.head("Chain of thought — does being told to reason repair it?")
    print(f"  {'model':<16}{'ru «N минут» base':>20}{'+ CoT':>10}"
          f"{'ru all base':>14}{'+ CoT':>10}{'en all base':>14}{'+ CoT':>10}")
    cot_ok = cot_n = 0
    for name, base_path, cot_path in COT_RUNS:
        base, cot = load(base_path), load(cot_path)
        if not base or not cot:
            continue
        b_nm = tally(base, "ru", ["n_minutes"])
        c_nm = tally(cot, "ru_cot", ["n_minutes"])
        cot_ok += c_nm[0]
        cot_n += c_nm[1]
        cells = [b_nm, c_nm, tally(base, "ru"), tally(cot, "ru_cot"),
                 tally(base, "en"), tally(cot, "en_cot")]
        widths = [20, 10, 14, 10, 14, 10]
        print(f"  {name:<16}" + "".join(
            f"{f'{ok}/{n}':>{w}}" for (ok, n), w in zip(cells, widths)))
    print()
    print(f"  Russian productive form with chain of thought, pooled: "
          f"{cot_ok}/{cot_n}")
    print("  Qwen2.5-3B carries a caveat: it emits no reasoning at all on English")
    print("  input and reasons in Mandarin on Russian input, so its two CoT arms")
    print("  are not doing the same thing. Its 0/8 does not depend on that.")


def section_fewshot(report: Report) -> None:
    report.head("Few-shot — three worked examples, none at a probe time")
    print(f"  {'model':<16}{'ru «N минут» base':>20}{'+ few-shot':>13}"
          f"{'ru all base':>14}{'+ few-shot':>13}{'en all base':>14}{'+ few-shot':>13}")

    base_ok = base_n = fs_ok = fs_n = 0
    any_run = False
    for name, base_path, fs_path in FEWSHOT_RUNS:
        base, fs = load(base_path), load(fs_path)
        if not base or not fs:
            continue
        any_run = True
        b_nm = tally(base, "ru", ["n_minutes"])
        f_nm = tally(fs, "ru_fewshot", ["n_minutes"])
        base_ok += b_nm[0]
        base_n += b_nm[1]
        fs_ok += f_nm[0]
        fs_n += f_nm[1]
        cells = [b_nm, f_nm, tally(base, "ru"), tally(fs, "ru_fewshot"),
                 tally(base, "en"), tally(fs, "en_fewshot")]
        widths = [20, 13, 14, 13, 14, 13]
        print(f"  {name:<16}" + "".join(
            f"{f'{ok}/{n}':>{w}}" for (ok, n), w in zip(cells, widths)))

    if not any_run:
        print("  no few-shot runs yet. Commands are in NOTES.md, session 15.")
        return

    print()
    print(f"  Russian productive form: {base_ok}/{base_n} zero-shot "
          f"-> {fs_ok}/{fs_n} with three examples")
    if base_n and fs_n:
        p = fisher_exact(fs_ok, fs_n - fs_ok, base_ok, base_n - base_ok)
        report.p_values.append(p)
        print(f"  few-shot against zero-shot, same items: {format_p(p)} "
              f"[exact {p:.2e}]")


def section_domain(report: Report) -> None:
    report.head("Second domain — the same expressions asked as calendar events")
    print(f"  {'model':<16}{'alarm en':>12}{'event en':>12}{'alarm ru':>12}{'event ru':>12}")
    pooled = {k: [0, 0] for k in ("alarm_en", "event_en", "alarm_ru", "event_ru")}
    for name, alarm_path, event_path in EVENT_RUNS:
        alarm, event = load(alarm_path), load(event_path)
        if not alarm or not event:
            continue
        cells = []
        for key, rows, arm in (("alarm_en", alarm, "en"), ("event_en", event, "en"),
                               ("alarm_ru", alarm, "ru"), ("event_ru", event, "ru")):
            ok, n = tally(rows, arm)
            cells.append(f"{ok}/{n}")
            fam_ok, fam_n = tally(rows, arm, FAMILY_ALL)
            pooled[key][0] += fam_ok
            pooled[key][1] += fam_n
        print(f"  {name:<16}" + "".join(f"{c:>12}" for c in cells))
    print()
    print("  counting-into-the-hour family, pooled over the three models:")
    report.compare("set_alarm", tuple(pooled["alarm_en"]), tuple(pooled["alarm_ru"]))
    report.compare("add_event", tuple(pooled["event_en"]), tuple(pooled["event_ru"]))


def section_stability(report: Report) -> None:
    report.head("Stability — the whole probe repeated three times at temperature 0.7")
    for name, path in STABILITY_RUNS:
        rows = load(path)
        if not rows:
            continue
        by_arm: Dict[str, List[float]] = defaultdict(list)
        counts: Dict[Tuple[str, int], List[int]] = defaultdict(lambda: [0, 0])
        for row in rows:
            key = (row["arm"], row["repeat"])
            counts[key][0] += bool(row["success"])
            counts[key][1] += 1
        for (arm, _), (ok, n) in sorted(counts.items()):
            by_arm[arm].append(100 * ok / n)
        for arm, values in sorted(by_arm.items()):
            spread = statistics.stdev(values) if len(values) > 1 else 0.0
            shown = ", ".join(f"{v:.0f}%" for v in values)
            print(f"  {name:<14} {arm:<3} {shown:<22} sd = {spread:.1f} points")
        print()


def section_first_probe(report: Report) -> None:
    report.head("The first probe (18 items) — the numbers the notebook quotes")
    per_model = []
    for name, path in FIRST_PROBE_RUNS:
        rows = load(path)
        if not rows:
            continue
        per_model.append((name, rows))
        print(f"  {name:<16} into_hour  en {'%d/%d' % tally(rows, 'en', ['into_hour']):>6}"
              f"   ru {'%d/%d' % tally(rows, 'ru', ['into_hour']):>6}")
    print()

    def pool(models, arms):
        ok = n = 0
        for _, rows in models:
            for arm in arms:
                a, b = tally(rows, arm, ["into_hour"])
                ok += a
                n += b
        return ok, n

    qwen = per_model[:1]
    small = per_model[:2]
    report.compare("Qwen only, ru", pool(qwen, ["en"]), pool(qwen, ["ru"]))
    report.compare("Qwen, 3 Russian arms", pool(qwen, ["en"]), pool(qwen, RUSSIAN_ARMS))
    report.compare("two small models", pool(small, ["en"]), pool(small, ["ru"]))
    report.compare("two small, all ru arms", pool(small, ["en"]), pool(small, RUSSIAN_ARMS))
    report.compare("all four models", pool(per_model, ["en"]), pool(per_model, ["ru"]))


def section_polish(report: Report) -> None:
    report.head("Polish — the same language carrying both constructions")
    if not os.path.exists(TASKS_PL):
        print(f"  !! {TASKS_PL} not found — run tools/make_polish_probe.py")
        return

    tasks = load_tasks(TASKS_PL)
    direction = {tid: t.get("pl_direction") for tid, t in tasks.items()}
    ids_by_direction: Dict[str, List[str]] = defaultdict(list)
    for tid, kind in direction.items():
        ids_by_direction[kind].append(tid)

    order = ["digital", "into_hour", "from_hour", "subtractive"]
    gloss = {
        "digital": "«na 07:20» control",
        "into_hour": "«wpół do X» counts INTO",
        "from_hour": "«N po X» counts FROM",
        "subtractive": "«za N X» control",
    }

    pooled: Dict[str, Dict[str, List[int]]] = {
        k: {"en": [0, 0], "pl": [0, 0]} for k in order
    }
    any_run = False
    print(f"  {'model':<16}" + "".join(f"{k:>22}" for k in order))
    for name, path in POLISH_RUNS:
        rows = load(path)
        if not rows:
            continue
        any_run = True
        cells = []
        for kind in order:
            ids = ids_by_direction[kind]
            en = tally(rows, "en", None, ids)
            pl = tally(rows, "pl", None, ids)
            for arm, got in (("en", en), ("pl", pl)):
                pooled[kind][arm][0] += got[0]
                pooled[kind][arm][1] += got[1]
            cells.append(f"{en[0]}/{en[1]} · {pl[0]}/{pl[1]}")
        print(f"  {name:<16}" + "".join(f"{c:>22}" for c in cells))

    if not any_run:
        print("  no Polish runs yet. Commands are in NOTES.md, session 12.")
        return

    print("  (each cell is English · Polish)")
    print()
    print("  Polish against its own English half:")
    for kind in order:
        report.compare(gloss[kind], tuple(pooled[kind]["en"]),
                       tuple(pooled[kind]["pl"]), lang="pl")

    print()
    print("  Inside Polish alone, «po» against «wpół do». This is the contrast the")
    print("  probe was designed around and it turned out to be the wrong one: at")
    print("  scale both cells are at ceiling. The informative comparison is Polish")
    print("  against Russian on the same clock times, in the next section.")
    for label, subset in (("pooled over all models", POLISH_RUNS),
                          ("small models only, as reported in F34", POLISH_RUNS[:2])):
        into = [0, 0]
        frm = [0, 0]
        for _, path in subset:
            rows = load(path)
            if not rows:
                continue
            for acc, kind in ((into, "into_hour"), (frm, "from_hour")):
                ok, n = tally(rows, "pl", None, ids_by_direction[kind])
                acc[0] += ok
                acc[1] += n
        if not into[1]:
            continue
        p = fisher_exact(frm[0], frm[1] - frm[0], into[0], into[1] - into[0])
        report.p_values.append(p)
        print(f"    {label:<38} «po» {frm[0]}/{frm[1]}  «wpół do» {into[0]}/{into[1]}"
              f"   {format_p(p)}")

    literal = pooled["digital"]["pl"]
    if literal[1] and literal[0] / literal[1] < 0.9:
        print()
        print(f"  !! Polish digital times are only {literal[0]}/{literal[1]}."
              f" The wording is suspect; do not read the rows above as a"
              f" result about the construction.")


MATCHED_FORMS = [
    ("half (:30)", "half_past", "same: both count INTO"),
    ("quarter (:15)", "quarter_past", "crossed: PL counts FROM"),
    ("N minutes", "n_minutes", "crossed: PL counts FROM"),
]
PL_RU_PAIRS = [
    ("Qwen2.5-3B", "results/pl_qwen3b.jsonl", "results/v2_qwen3b.jsonl"),
    ("Phi-3.5-mini", "results/pl_phi.jsonl", "results/v2_phi.jsonl"),
    ("qwen3.6-27b", "results/pl_qwen27b.jsonl", "results/v2_qwen27b.jsonl"),
    ("gpt-oss-120b", "results/pl_gptoss120b.jsonl", "results/v2_gptoss120b.jsonl"),
]


def section_polish_vs_russian(report: Report) -> None:
    report.head("Polish against Russian — same times, construction matched or crossed")

    for label, models in (("all four models", PL_RU_PAIRS),
                          ("large models only (27B, 120B)", PL_RU_PAIRS[2:])):
        available = [(n, a, b) for n, a, b in models
                     if os.path.exists(a) and os.path.exists(b)]
        if not available:
            print(f"  {label}: runs missing, skipped")
            continue
        print(f"  {label}:")
        print(f"    {'clock form':<15}{'Polish':<10}{'Russian':<10}{'relation'}")
        buckets: Dict[str, List[int]] = {"same": [0, 0, 0, 0], "crossed": [0, 0, 0, 0]}
        for name, cat, relation in MATCHED_FORMS:
            pl_ok = pl_n = ru_ok = ru_n = 0
            for _, pl_path, ru_path in available:
                a, b = tally(load(pl_path), "pl", [cat])
                pl_ok += a
                pl_n += b
                a, b = tally(load(ru_path), "ru", [cat])
                ru_ok += a
                ru_n += b
            key = "same" if relation.startswith("same") else "crossed"
            for i, v in enumerate((pl_ok, pl_n, ru_ok, ru_n)):
                buckets[key][i] += v
            print(f"    {name:<15}{f'{pl_ok}/{pl_n}':<10}{f'{ru_ok}/{ru_n}':<10}{relation}")

        for key, caption in (("same", "both languages count INTO the hour"),
                             ("crossed", "Polish counts FROM, Russian counts INTO")):
            pl_ok, pl_n, ru_ok, ru_n = buckets[key]
            if not pl_n:
                continue
            p = fisher_exact(pl_ok, pl_n - pl_ok, ru_ok, ru_n - ru_ok)
            report.p_values.append(p)
            print(f"    {caption:<42} pl {pl_ok:>2}/{pl_n:<3} ru {ru_ok:>2}/{ru_n:<3}"
                  f"  {format_p(p)} [exact {p:.2e}]")
        print()

    print("  Corpus share is the same quantity in both halves of this comparison,")
    print("  so it cannot produce an effect that appears on one clock form and")
    print("  vanishes on another. The variable is the construction.")


GERMAN_RUNS = [
    ("Qwen2.5-3B", "3B", "results/de2_qwen3b.jsonl"),
    ("Phi-3.5-mini", "3.8B", "results/de2_phi.jsonl"),
    ("qwen3.8-27b", "27B", "results/de2_qwen38.jsonl"),
    ("gpt-oss-120b", "120B", "results/de2_gptoss120b.jsonl"),
]

GERMAN_CELLS = [
    ("literal", "digital", "health check"),
    ("n_minutes", "counting-from", "N nach X, the only option German has"),
    ("half_past", "counting-into", "halb X, the closed set"),
    ("quarter_into", "counting-into (regional)", "viertel X, east and south"),
    ("subtractive", "subtractive", "control, fails in English too"),
]


def section_german(report: Report) -> None:
    report.head("German — a closed counting-into set, across four model sizes")

    available = [(n, s, p) for n, s, p in GERMAN_RUNS if os.path.exists(p)]
    if not available:
        print("  runs missing, skipped")
        return

    print(f"    {'cell':<26}" + "".join(f"{s:>9}" for _, s, _ in available))
    totals: Dict[str, List[int]] = {}
    for difficulty, caption, _note in GERMAN_CELLS:
        cells = []
        ok_all = n_all = 0
        for _, _, path in available:
            ok, n = tally(load(path), "de", [difficulty])
            cells.append(f"{ok}/{n}")
            ok_all += ok
            n_all += n
        totals[caption] = [ok_all, n_all]
        print(f"    {caption:<26}" + "".join(f"{c:>9}" for c in cells))

    print()
    into_ok, into_n = totals["counting-into"]
    from_ok, from_n = totals["counting-from"]
    p = fisher_exact(into_ok, into_n - into_ok, from_ok, from_n - from_ok)
    report.p_values.append(p)
    print(f"    counting-into {into_ok}/{into_n} vs counting-from {from_ok}/{from_n}"
          f"   {format_p(p)} [exact {p:.2e}]")

    small = available[:2]
    large = available[2:]
    if len(small) == 2 and len(large) == 2:
        def cell(runs, difficulty):
            ok = n = 0
            for _, _, path in runs:
                a, b = tally(load(path), "de", [difficulty])
                ok += a
                n += b
            return ok, n

        s_ok, s_n = cell(small, "half_past")
        l_ok, l_n = cell(large, "half_past")
        p = fisher_exact(l_ok, l_n - l_ok, s_ok, s_n - s_ok)
        report.p_values.append(p)
        print(f"    counting-into, two large {l_ok}/{l_n} vs two small {s_ok}/{s_n}"
              f"   {format_p(p)} [exact {p:.2e}]")

    print()
    print("    within German, counting-into against counting-from, per model:")
    for name, size, path in available:
        rows = load(path)
        i_ok, i_n = tally(rows, "de", ["half_past"])
        f_ok, f_n = tally(rows, "de", ["n_minutes"])
        p = fisher_exact(i_ok, i_n - i_ok, f_ok, f_n - f_ok)
        report.p_values.append(p)
        print(f"      {name:<14} {size:>5}  into {i_ok:>2}/{i_n:<3} from {f_ok:>2}/{f_n:<3}"
              f"  {format_p(p)} [exact {p:.2e}]")

    print()
    print("    German is high-resource in every corpus, so a data-volume account")
    print("    has no room here. The counting-from cell is intact at every size")
    print("    while the counting-into cell collapses below 120B.")
    print()
    print("    Six of the twelve regional items render dreiviertel N in English")
    print("    as 'quarter to X', which is the subtractive control. The English")
    print("    half of that cell is therefore depressed on the small models by a")
    print("    known effect and is not a usable within-item ceiling there. The")
    print("    German-against-German comparisons above are unaffected.")


TASKS_CASE = "data/tasks_time_case.json"
CASE_RUNS = [
    ("Qwen2.5-3B", "results/case_qwen3b.jsonl"),
    ("Phi-3.5-mini", "results/case_phi.jsonl"),
    ("qwen3.8-27b", "results/case_qwen38.jsonl"),
    ("gpt-oss-120b", "results/case_gptoss120b.jsonl"),
]
CASE_FRAMES = [("at", "accusative"), ("from", "genitive"), ("by", "dative")]


def _case_ids(frame: str, category: str) -> List[str]:
    with open(TASKS_CASE, "r", encoding="utf-8") as handle:
        tasks = json.load(handle)["tasks"]
    return [t["id"] for t in tasks
            if t["case_frame"] == frame and t["difficulty"] == category]


def section_case(report: Report) -> None:
    report.head("Case variation — the same construction in three frames")

    available = [(n, p) for n, p in CASE_RUNS if os.path.exists(p)]
    if not available:
        print("  runs missing, skipped")
        return
    print(f"  models: {', '.join(n for n, _ in available)}")
    print()

    print(f"    {'cell':<24}" + "".join(f"{c:>22}" for _, c in CASE_FRAMES))
    totals: Dict[str, Dict[str, List[int]]] = {}
    for category in ("literal", "half_past", "n_minutes"):
        cells = []
        totals[category] = {}
        for frame, case in CASE_FRAMES:
            ids = _case_ids(frame, category)
            ru_ok = ru_n = en_ok = en_n = 0
            for _, path in available:
                rows = load(path)
                a, b = tally(rows, "ru", ids=ids)
                ru_ok += a
                ru_n += b
                a, b = tally(rows, "en", ids=ids)
                en_ok += a
                en_n += b
            totals[category][case] = [ru_ok, ru_n, en_ok, en_n]
            cells.append(f"ru {ru_ok}/{ru_n}  en {en_ok}/{en_n}")
        print(f"    {category:<24}" + "".join(f"{c:>22}" for c in cells))

    print()
    lit = totals["literal"]
    rates = [lit[c][0] / lit[c][1] if lit[c][1] else 0.0 for _, c in CASE_FRAMES]
    spread = max(rates) - min(rates)
    print(f"    digital across the three frames: "
          + ", ".join(f"{r*100:.0f}%" for r in rates)
          + f"  (spread {spread*100:.0f} points)")
    if spread > 0.15:
        print("    that spread is too large to read anything below it — the frame")
        print("    itself is doing work. Check the episodes before interpreting.")
    else:
        print("    flat enough to read the cells below.")
    lost = sum(crashes(load(path)) for _, path in available)
    if lost:
        print(f"    {lost} episodes crashed and are excluded from every count")
        print("    above; they are infrastructure failures, not model behaviour.")

    print()
    for category in ("half_past", "n_minutes"):
        acc = totals[category]["accusative"]
        obl = [0, 0]
        for case in ("genitive", "dative"):
            obl[0] += totals[category][case][0]
            obl[1] += totals[category][case][1]
        p = fisher_exact(acc[0], acc[1] - acc[0], obl[0], obl[1] - obl[0])
        report.p_values.append(p)
        print(f"    {category:<12} Russian accusative {acc[0]}/{acc[1]}"
              f" vs oblique {obl[0]}/{obl[1]}   {format_p(p)} [exact {p:.2e}]")

    for case in ("genitive", "dative"):
        acc = totals["half_past"]["accusative"]
        obl = totals["half_past"][case]
        p = fisher_exact(acc[0], acc[1] - acc[0], obl[0], obl[1] - obl[0])
        report.p_values.append(p)
        print(f"    half_past    Russian accusative {acc[0]}/{acc[1]}"
              f" vs {case} {obl[0]}/{obl[1]}   {format_p(p)} [exact {p:.2e}]")

    print()
    for category in ("half_past", "n_minutes"):
        acc = totals[category]["accusative"]
        obl = [0, 0]
        for case in ("genitive", "dative"):
            obl[0] += totals[category][case][2]
            obl[1] += totals[category][case][3]
        p = fisher_exact(acc[2], acc[3] - acc[2], obl[0], obl[1] - obl[0])
        report.p_values.append(p)
        print(f"    {category:<12} English accusative-frame {acc[2]}/{acc[3]}"
              f" vs oblique {obl[0]}/{obl[1]}   {format_p(p)} [exact {p:.2e}]")

    print()
    print("    The dative productive cell carries marginal_wording in the data:")
    print("    к двадцати минутам восьмого is grammatical but rare even for a")
    print("    native speaker. If it scores below the accusative productive")
    print("    cell, the wording is a live alternative explanation.")


TASKS_HALB = "data/tasks_time_de_halb.json"
HALB_RUNS = [
    ("Qwen2.5-3B", "3B", "results/halb_qwen3b.jsonl"),
    ("Phi-3.5-mini", "3.8B", "results/halb_phi.jsonl"),
    ("qwen3.8-27b", "27B", "results/halb_qwen38.jsonl"),
    ("gpt-oss-120b", "120B", "results/halb_gptoss120b.jsonl"),
]


def _halb_kind(task: Dict[str, Any]) -> str:
    german = task["instruction"]["de"]
    if task["de_cell"] != "into_composed":
        return task["de_cell"]
    if " vor halb" not in german:
        return "composed_nach"
    return "composed_vor"


def _halb_tasks() -> Dict[str, Dict[str, Any]]:
    with open(TASKS_HALB, "r", encoding="utf-8") as handle:
        return {t["id"]: t for t in json.load(handle)["tasks"]}


def section_halb(report: Report) -> None:
    report.head("German assembled counting-into — zehn vor halb against N nach")

    available = [(n, s, p) for n, s, p in HALB_RUNS if os.path.exists(p)]
    if not available:
        print("  runs missing, skipped")
        return
    tasks = _halb_tasks()

    cells = ["digital", "from_hour", "into_fixed", "composed_vor", "composed_nach"]
    print(f"    {'cell':<16}" + "".join(f"{s:>10}" for _, s, _ in available))
    for cell in cells:
        row = []
        for _, _, path in available:
            rows = [r for r in load(path)
                    if r.get("arm") == "de" and not r.get("error")
                    and _halb_kind(tasks[r["task_id"]]) == cell]
            row.append(f"{sum(bool(r['success']) for r in rows)}/{len(rows)}")
        print(f"    {cell:<16}" + "".join(f"{c:>10}" for c in row))

    print()
    print("    English control, composed_nach only:")
    row = []
    for _, s, path in available:
        rows = [r for r in load(path)
                if r.get("arm") == "en" and not r.get("error")
                and _halb_kind(tasks[r["task_id"]]) == "composed_nach"]
        row.append(f"{s} {sum(bool(r['success']) for r in rows)}/{len(rows)}")
    print("      " + "   ".join(row))
    print("      Rendered 'twenty five to eight' — the English subtractive form,")
    print("      which fails in English too. That cell is excluded below.")

    print()
    print("    minimal pair, same clock times, identical English:")
    for name, size, path in available:
        rows = [r for r in load(path) if r.get("arm") == "de" and not r.get("error")]
        comp = [r for r in rows if _halb_kind(tasks[r["task_id"]]) == "composed_vor"]
        times = {tasks[r["task_id"]]["checks"]["de"][0]["time"] for r in comp}
        frm = [r for r in rows
               if _halb_kind(tasks[r["task_id"]]) == "from_hour"
               and tasks[r["task_id"]]["checks"]["de"][0]["time"] in times]
        a_ok, a_n = sum(bool(r["success"]) for r in comp), len(comp)
        b_ok, b_n = sum(bool(r["success"]) for r in frm), len(frm)
        if not a_n or not b_n:
            continue
        p = fisher_exact(a_ok, a_n - a_ok, b_ok, b_n - b_ok)
        report.p_values.append(p)
        print(f"      {name:<14} {size:>5}  vor-halb {a_ok:>2}/{a_n:<3}"
              f" nach {b_ok:>2}/{b_n:<3}  {format_p(p)} [exact {p:.2e}]")

    print()
    print("    The assembled form is below the counting-from one on every model")
    print("    that is not at ceiling, and the gap clears .01 on Phi-3.5-mini,")
    print("    where German counting-from is 6 of 8 and the assembled form is 0")
    print("    of 8 with English at 7 of 8 in both cells. At 120B both are at")
    print("    ceiling and there is nothing to see.")


BARE_PAIRS = [
    ("gpt-oss-120b", "results/bare_gptoss120b.jsonl", "results/v2_gptoss120b.jsonl"),
    ("qwen3.8-27b", "results/bare_qwen38.jsonl", "results/v2_qwen38.jsonl"),
]
BARE_TASKS = "data/tasks_time_v2.json"


def section_bare(report: Report) -> None:
    report.head("The bare condition — same sentences, no tools, no JSON")

    with open(BARE_TASKS, "r", encoding="utf-8") as handle:
        tasks = {t["id"]: t for t in json.load(handle)["tasks"]}

    available = [(n, b, a) for n, b, a in BARE_PAIRS
                 if os.path.exists(b) and os.path.exists(a)]
    if not available:
        print("  runs missing, skipped")
        return

    for name, bare_path, agent_path in available:
        with open(bare_path, "r", encoding="utf-8") as handle:
            bare = [json.loads(line) for line in handle if line.strip()]
        bare = [r for r in bare if r.get("record_type") == "bare"
                and not r.get("error")]
        agent = [r for r in load(agent_path) if not r.get("error")]

        print(f"  {name}")
        print(f"    {'cell':<14}{'bare ru':>9}{'agent ru':>10}"
              f"{'bare en':>9}{'agent en':>10}")
        for cat in ("literal", "half_past", "quarter_past",
                    "n_minutes", "subtractive"):
            row = []
            for arm in ("ru", "en"):
                b = [r for r in bare
                     if r["arm"] == arm and r["difficulty"] == cat]
                a = [r for r in agent
                     if r.get("arm") == arm
                     and tasks[r["task_id"]]["difficulty"] == cat]
                row.append(f"{sum(bool(r['success']) for r in b)}/{len(b)}")
                row.append(f"{sum(bool(r['success']) for r in a)}/{len(a)}")
            print(f"    {cat:<14}{row[0]:>9}{row[1]:>10}{row[2]:>9}{row[3]:>10}")

        for arm in ("ru", "en"):
            b = [r for r in bare if r["arm"] == arm]
            a = [r for r in agent if r.get("arm") == arm]
            b_ok, a_ok = (sum(bool(r["success"]) for r in b),
                          sum(bool(r["success"]) for r in a))
            p = fisher_exact(b_ok, len(b) - b_ok, a_ok, len(a) - a_ok)
            report.p_values.append(p)
            print(f"      {arm}: bare {b_ok}/{len(b)} vs agent {a_ok}/{len(a)}"
                  f"   {format_p(p)} [exact {p:.2e}]")
        print()

    print("    Removing the agent machinery does not repair the construction.")
    print("    The failure is in resolving the phrase; the agent frame is what")
    print("    makes the wrong value invisible once it is resolved.")


def section_frequency(report: Report) -> None:
    report.head("Corpus frequency against accuracy")
    path = "results/frequency.json"
    if not os.path.exists(path):
        print(f"  !! {path} not found — run `python -m src.frequency` first")
        return
    with open(path, "r", encoding="utf-8") as handle:
        blob = json.load(handle)
    per_category = blob["per_category"]

    accuracy: Dict[str, Dict[str, List[int]]] = {
        c: {"en": [0, 0], "ru": [0, 0]} for c in FAMILY
    }
    for _, run_path in MAIN_RUNS:
        rows = load(run_path)
        for cat in FAMILY:
            for arm in ("en", "ru"):
                ok, n = tally(rows, arm, [cat])
                accuracy[cat][arm][0] += ok
                accuracy[cat][arm][1] += n

    print(f"  {'category':<18}{'freq ru':>12}{'freq en':>12}{'acc ru':>9}{'acc en':>9}")
    for cat in FAMILY:
        freq = per_category[cat]
        ru_ok, ru_n = accuracy[cat]["ru"]
        en_ok, en_n = accuracy[cat]["en"]
        print(f"  {CATEGORY_LABEL[cat]:<18}{freq['ru']:>12.2e}{freq['en']:>12.2e}"
              f"{100 * ru_ok / ru_n:>8.0f}%{100 * en_ok / en_n:>8.0f}%")

    ru_drop = per_category[FAMILY[0]]["ru"] / per_category[FAMILY[-1]]["ru"]
    en_drop = per_category[FAMILY[0]]["en"] / per_category[FAMILY[-1]]["en"]
    print()
    print(f"  commonest to rarest: Russian falls {ru_drop:.1f}x, "
          f"English falls {en_drop:.1f}x")
    print("  English accuracy does not move. That dissociation is the result.")


_NUM = r"(\.[0-9]+|[0-9]+(?:\.[0-9]+)?(?:[eE]-?[0-9]+)?)"
P_EQUALS = re.compile(r"p\s*=\s*" + _NUM, re.IGNORECASE)
P_LESS = re.compile(r"p\s*<\s*" + _NUM, re.IGNORECASE)
CI_BRACKET = re.compile(r"\[\s*(\d{1,3}(?:\.\d)?)\s*[,–—-]\s*"
                        r"(\d{1,3}(?:\.\d)?)\s*\]")


def check_document(path: str, computed: List[float], alpha: float = ALPHA,
                   bounds: Optional[List[float]] = None) -> int:
    if not os.path.exists(path):
        print(f"  !! {path} not found")
        return 0
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    smallest = min(computed) if computed else 1.0
    bad = 0
    for line_no, line in enumerate(text.splitlines(), 1):
        for raw in P_EQUALS.findall(line):
            value = float(raw if not raw.startswith(".") else "0" + raw)
            if value < alpha:
                print(f"  {path}:{line_no}: p = {raw} is below alpha — report "
                      f"'{format_p(value, alpha)}' instead (Gorman, 2026-10-13)")
                bad += 1
                continue
            if not any(abs(value - c) <= 0.05 * max(value, c) for c in computed):
                print(f"  {path}:{line_no}: p = {raw} is not a value this "
                      f"script computes")
                bad += 1

        for raw in P_LESS.findall(line):
            threshold = float(raw if not raw.startswith(".") else "0" + raw)
            if smallest >= threshold:
                print(f"  {path}:{line_no}: claims p < {raw}, but the smallest "
                      f"p this script computes is {smallest:.2g}")
                bad += 1

        if bounds:
            for low, high in CI_BRACKET.findall(line):
                for raw in (low, high):
                    value = float(raw)
                    if not any(abs(value - b) <= 0.051 for b in bounds):
                        print(f"  {path}:{line_no}: the interval bound {raw} "
                              f"is not one this script computes")
                        bad += 1

    if not bad:
        print(f"  {path}: p-values and intervals check out")
    return bad


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", nargs="*", default=[],
                        help="documents whose p-values must match what is computed")
    args = parser.parse_args()

    report = Report()
    section_gradient(report)
    section_error(report)
    section_robustness(report)
    section_cot(report)
    section_fewshot(report)
    section_domain(report)
    section_stability(report)
    section_first_probe(report)
    section_polish(report)
    section_polish_vs_russian(report)
    section_german(report)
    section_case(report)
    section_halb(report)
    section_bare(report)
    section_frequency(report)

    if args.check:
        report.head("Checking documents against the computed values")
        bad = sum(check_document(path, report.p_values, bounds=report.bounds)
                  for path in args.check)
        if bad:
            print(f"\n  {bad} p-value(s) in the documents are not reproducible.")
            return 1
        print("\n  All reported p-values are reproducible from the logs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
