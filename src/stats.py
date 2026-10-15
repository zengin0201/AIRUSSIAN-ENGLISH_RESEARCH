"""
Every statistic the paper reports, regenerated from the logs.

    python -m src.stats
    python -m src.stats --check paper/paper.md README.md

Why this file exists.

Until 2026-08-21 the counts in the paper came from the logs but the p-values
came from nowhere. Nothing in this repository computed them. Five of them were
wrong by between three and nine orders of magnitude, and one value, 2.0e-12,
was printed for two different contingency tables, which cannot happen. They
were all conservative, so the reported effect was understated rather than
overstated, but a reviewer with R and thirty seconds would have found it.

The rule this file enforces: a number appears in the paper only if this script
prints it. `--check` reads a document and complains about any p-value in it
that is not one of the values computed here.

See NOTES.md, F32.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .metrics import fisher_exact

# Alpha level for null hypothesis testing, and how a p-value gets reported.
#
# Kyle Gorman (CUNY), 2026-10-13, reviewing this draft:
#
#   "If you are using Fisher for null hypothesis testing, you shouldn't report
#    miniscule p-values. Rather you should set an alpha-level (.05, or .01 are
#    normal) and then report p < .01 [...] it doesn't matter HOW MUCH SMALLER
#    THAN THE ALPHA-LEVEL a p-value is, it just matters WHETHER it's smaller."
#
# He is right, and the draft he read printed things like p = 1.4e-22 on 68
# items. Under frequentist NHST that number carries no information the
# threshold does not, and quoting it to two significant figures invites a
# reviewer to check it — which is how a tiny arithmetic difference turns into
# a question about whether the author understands their own test.
#
# The exact value is still computed and still printed by this script, because
# the lab notebook wants it and because `--check` needs it to verify that a
# "p < .01" claim in a document is actually true. It just does not go in the
# paper.
ALPHA = 0.01


def format_p(p: float, alpha: float = ALPHA) -> str:
    """
    The reportable form. Linguistics drops the leading zero: `p < .01`.

    Above the threshold the value itself is the information, so it is printed
    to two digits. Below it, only the threshold is.
    """
    if p < alpha:
        return f"p < {('%.2f' % alpha).lstrip('0')}"
    if p >= 1:
        return "p = 1.0"
    # Two significant digits, not two decimal places: at two decimals a p of
    # .0114 prints as ".01", which is the alpha level itself and reads as if
    # the test were exactly on the threshold.
    return f"p = {('%.2g' % p).lstrip('0')}"

# The Russian forms that count into the coming hour. `into_hour` is the
# undivided category from the first probe; the other three are its split.
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

# The main probe, one run per model, greedy.
MAIN_RUNS = [
    ("Qwen2.5-3B", "results/v2_qwen3b.jsonl"),
    ("Phi-3.5-mini", "results/v2_phi.jsonl"),
    ("qwen3.6-27b", "results/v2_qwen27b.jsonl"),
    ("gpt-oss-120b", "results/v2_gptoss120b.jsonl"),
]
# Chain of thought, paired with the plain run of the same model.
COT_RUNS = [
    ("Qwen2.5-3B", "results/v2_qwen3b.jsonl", "results/cot_qwen3b.jsonl"),
    ("Phi-3.5-mini", "results/v2_phi.jsonl", "results/cot_phi.jsonl"),
    ("gpt-oss-120b", "results/v2_gptoss120b.jsonl", "results/cot_gptoss120b.jsonl"),
]
# The second domain: the same 24 expressions asked as calendar events.
EVENT_RUNS = [
    ("Qwen2.5-3B", "results/v2_qwen3b.jsonl", "results/event_qwen3b.jsonl"),
    ("Phi-3.5-mini", "results/v2_phi.jsonl", "results/event_phi.jsonl"),
    ("gpt-oss-120b", "results/v2_gptoss120b.jsonl", "results/event_gptoss120b.jsonl"),
]
# Few-shot, paired with the plain run of the same model. Asked for by both
# reviewers of the August draft; see NOTES.md, session 15.
# qwen3.6-27b left the Groq catalogue before few-shot could be run on it
# (F41), so qwen3.8-27b stands in — run on all four arms in one pass, which is
# why its baseline and few-shot paths are the same file.
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

# The first, 18-item probe. Superseded by the expanded one above, but the lab
# notebook quotes its p-values, so they have to be checkable too.
FIRST_PROBE_RUNS = [
    ("Qwen2.5-3B", "results/time_probe.jsonl"),
    ("Phi-3.5-mini", "results/time_phi.jsonl"),
    ("qwen3.6-27b", "results/time_qwen27b.jsonl"),
    ("gpt-oss-120b", "results/time_gptoss120b.jsonl"),
]
RUSSIAN_ARMS = ("ru", "ru_sys_ru", "ru_translate_safe")

# The Polish crossed design. Polish counts into the coming hour for «wpół do»
# and from the completed hour for «po», so it carries both constructions at
# once and splits them inside a single language.
TASKS_PL = "data/tasks_time_pl.json"
POLISH_RUNS = [
    ("Qwen2.5-3B", "results/pl_qwen3b.jsonl"),
    ("Phi-3.5-mini", "results/pl_phi.jsonl"),
    ("qwen3.6-27b", "results/pl_qwen27b.jsonl"),
    ("gpt-oss-120b", "results/pl_gptoss120b.jsonl"),
]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load(path: str) -> List[Dict[str, Any]]:
    """Episodes only. A missing file is not fatal — say so and carry on."""
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
        total += 1
        ok += bool(row["success"])
    return ok, total


def minutes(value: str) -> int:
    hours, mins = value.split(":")
    return int(hours) * 60 + int(mins)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

class Report:
    """Collects every p-value printed, so `--check` can look for strays."""

    def __init__(self) -> None:
        self.p_values: List[float] = []

    def head(self, title: str) -> None:
        print()
        print("=" * 74)
        print(title)
        print("=" * 74)

    def compare(self, label: str, en: Tuple[int, int], other: Tuple[int, int],
                lang: str = "ru") -> float:
        """
        Compare English against the other language and print both rates.

        `lang` is printed rather than assumed. The first version of this method
        hardcoded "ru" in the format string, so the Polish rows came out
        labelled Russian — the exact substitution that `ARM_LANGUAGE` was added
        to prevent, reintroduced two hours later in the reporting layer.
        """
        en_ok, en_n = en
        other_ok, other_n = other
        p = fisher_exact(en_ok, en_n - en_ok, other_ok, other_n - other_ok)
        self.p_values.append(p)
        en_pct = 100 * en_ok / en_n if en_n else 0.0
        other_pct = 100 * other_ok / other_n if other_n else 0.0
        # Reportable form first, because that is what goes in the paper. The
        # exact value follows in brackets for the notebook only.
        print(f"  {label:<26} en {en_ok:>3}/{en_n:<3} ({en_pct:5.1f}%)   "
              f"{lang} {other_ok:>3}/{other_n:<3} ({other_pct:5.1f}%)   "
              f"{format_p(p):<9} [exact {p:.2e}]")
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
    """
    Does handing the model three worked examples repair the productive form?

    The demonstrations are at clock times that appear in no probe, so a score
    here cannot come from recalling what was shown. English gets the same three
    times, so an improvement in Russian alone is attributable to the
    demonstrations and not to the examples teaching the output format.
    """
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
            # Sample standard deviation, n-1. Three repeats is a sample, not a
            # population, and the paper reports it that way.
            spread = statistics.stdev(values) if len(values) > 1 else 0.0
            shown = ", ".join(f"{v:.0f}%" for v in values)
            print(f"  {name:<14} {arm:<3} {shown:<22} sd = {spread:.1f} points")
        print()


def section_first_probe(report: Report) -> None:
    """The 18-item probe of sessions 5 to 7. Kept so NOTES.md stays checkable."""
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
    """
    Polish holds the language fixed and varies the construction.

    Russian counts into the coming hour for all three of «половина»,
    «четверть» and «N минут». Polish counts into the hour for «wpół do» and
    from the completed hour for «kwadrans po» and «N po». So the same speaker,
    the same probe and the same clock times carry both constructions, and the
    two surviving explanations of the Russian result disagree about what
    happens:

      the construction reading -> «wpół do» fails, «po» is answered
      the "bad at Slavic" reading -> both fail

    `literal` is the health check. If Polish digital times are not near
    perfect, our Polish wording is broken and nothing below means anything.
    """
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
              f"   p = {p:.2f}")

    literal = pooled["digital"]["pl"]
    if literal[1] and literal[0] / literal[1] < 0.9:
        print()
        print(f"  !! Polish digital times are only {literal[0]}/{literal[1]}."
              f" The wording is suspect; do not read the rows above as a"
              f" result about the construction.")


# Polish and Russian ask for the same clock times, but they do not always use
# the same construction to do it. That mismatch is the experiment.
#
#   :30  wpół do ósmej   INTO the hour   половина восьмого   INTO the hour
#   :15  kwadrans po siódmej  FROM        четверть восьмого  INTO
#   :20  dwadzieścia po siódmej  FROM     двадцать минут восьмого  INTO
#
# So the half form is a matched pair — two Slavic languages, same construction —
# and the other two are crossed. A language-level explanation ("models are worse
# at Slavic", "Polish is lower-resource") predicts one language below the other
# everywhere. A construction-level explanation predicts a tie on the matched
# form and a split on the crossed ones.
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
    """Two Slavic languages, the same clock times, construction held or crossed."""
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
                  f"  p = {p:.2e}")
        print()

    print("  Polish is the lower-resource of the two languages and is the one that")
    print("  succeeds, so this cannot be read as a resource effect.")


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


# --------------------------------------------------------------------------
# checking a document against what we just computed
# --------------------------------------------------------------------------

P_EQUALS = re.compile(r"p\s*=\s*(\.?[0-9]+\.?[0-9]*(?:e-?[0-9]+)?)", re.IGNORECASE)
P_LESS = re.compile(r"p\s*<\s*(\.?[0-9]+\.?[0-9]*)", re.IGNORECASE)


def check_document(path: str, computed: List[float], alpha: float = ALPHA) -> int:
    """
    Check a document's p-values against what this script computed.

    Two jobs now. The original one: catch a number no code here produced,
    which is why this exists at all (F32 — five fabricated p-values).

    The second, added after Kyle Gorman's review: enforce the reporting
    convention. A document that prints `p = 1.4e-22` is not wrong, it is
    reporting something a frequentist test does not license. That is a
    defect the author should see before a reviewer does.
    """
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
            # Two significant figures is what gets printed, so compare there.
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

    if not bad:
        print(f"  {path}: p-values check out")
    return bad


def main() -> int:
    # Cyrillic category labels have to survive a Windows console.
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
    section_frequency(report)

    if args.check:
        report.head("Checking documents against the computed values")
        bad = sum(check_document(path, report.p_values) for path in args.check)
        if bad:
            print(f"\n  {bad} p-value(s) in the documents are not reproducible.")
            return 1
        print("\n  All reported p-values are reproducible from the logs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
