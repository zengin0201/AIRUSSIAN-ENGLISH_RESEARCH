"""
Generate data/tasks_time_de.json from data/tasks_time_v2.json.

    python tools/make_german_probe.py

Why German, and why it is the first probe that can prove us wrong.

Everything in this project so far has been a prediction of failure. Russian
counts into the coming hour and fails; Polish counts into the hour for one form
and fails there and not elsewhere. A theory that only ever predicts failure is
cheap. German is where ours predicts success, and can be caught.

German counts into the coming hour too:

    07:30   halb acht           half of the eighth hour     -> counts INTO
    07:15   viertel acht        a quarter of the eighth      -> counts INTO
    07:45   dreiviertel acht    three quarters of the eighth -> counts INTO

    07:15   viertel nach sieben a quarter past seven         -> counts FROM
    07:20   zwanzig nach sieben twenty past seven            -> counts FROM
    07:45   viertel vor acht    a quarter before eight       -> subtractive

So the direction is the same as Russian. **The productivity is not.** German's
counting-into forms are a closed set of three: halb, viertel, dreiviertel. There
is no German way to say 07:20 by counting into the eighth hour — *zwanzig acht*
is not a thing. Russian's construction is productive: any minute value takes it,
and «двадцать минут восьмого» is built by rule, not retrieved.

That is the whole experiment. German holds the direction of counting constant
and varies whether the form is built or memorised.

    our reading — the models memorise instances of the counting-into
    construction and never learn the rule
        -> German INTO forms are instances, three of them, all frequent.
           They should be read CORRECTLY. German ends up looking like English.

    the rival — "counting into the coming hour is simply hard, whatever
    the language"
        -> German INTO fails like Russian INTO.

If German fails, our account is wrong and the rival is right. Nothing else we
can run puts the thesis at risk this directly, and that is the reason to run it.

German also kills the resource explanation outright rather than by argument.
Russian-vs-Polish needed a paragraph explaining why corpus share cannot be the
variable. German is one of the best-resourced languages in every corpus anyone
trains on. If the construction's direction were the problem, German has no
excuse left.

A second, smaller thing the probe carries: the minimal pair.

    g06  05:15  viertel nach fünf    FROM
    g07  08:15  viertel neun         INTO
    g08  22:15  viertel nach zehn    FROM
    g09  13:15  viertel zwei         INTO

Same word, same clock position, one word of difference, opposite readings. The
+60 analogue is visible in the errors: reading *viertel neun* as the quarter
mark of a ninth hour already reached gives 09:15, and reading it as the standard
*viertel vor neun* gives 08:45. Both are diagnostic, and they are different
mistakes.

**Two things about this probe that a reader should be told, not discover.**

*viertel neun* and *dreiviertel neun* are regionally restricted — Saxony,
Thuringia, Bavaria, Austria, much of the east and south. A speaker from Hamburg
may not use them and may not parse them. That is worse than an unverified
wording: it is a wording whose gold label native speakers disagree about. The
two items carrying it are flagged `de_regional` so they can never enter a
primary comparison without someone deciding to let them in, and they need a
verifier from a region that actually uses the form, not just any German speaker.

*viertel* is written lowercase throughout, including in *viertel nach fünf*
where the noun would normally be capitalised. This is deliberate. With *Viertel
nach fünf* against *viertel neun* the pair differs in capitalisation as well as
in the presence of *nach*, and a model could key off the case instead of the
construction. Lowercase everywhere leaves exactly one word of difference. The
carrier is an informal instruction to an assistant, which is a register where
people type it that way, but it is a choice a native speaker should be told
about rather than left to find.

**The wording has not been checked by a native speaker.** Same status as Polish.
The forms are the textbook ones and the probe is partly self-checking — if the
German is simply bad, `digital` drops too and we know the file is worthless
rather than drawing a conclusion from it. But a clean `digital` score does not
make the rest idiomatic.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SOURCE = os.path.join(ROOT, "data", "tasks_time_v2.json")
TARGET = os.path.join(ROOT, "data", "tasks_time_de.json")

# Written out per task rather than derived by substitution, so that a German
# speaker can check the whole probe by reading one table.
#
# `direction` is what the experiment turns on. `into_hour` is the closed German
# set, which our account says should be read correctly because three frequent
# phrases are memorisable where a rule is not. `into_hour_regional` is the same
# direction in a form much of the country does not use.
GERMAN = {
    # half_past — INTO the coming hour, exactly as Russian. Universal across
    # the German-speaking area, and the single most frequent clock phrase in
    # the language. This is the cell the thesis stands or falls on.
    "g01": ("Stelle den Wecker auf halb drei nachmittags.", "into_hour"),
    "g02": ("Stelle den Wecker auf halb sieben morgens.", "into_hour"),
    "g03": ("Stelle den Wecker auf halb zehn abends.", "into_hour"),
    "g04": ("Stelle den Wecker auf halb zwölf vormittags.", "into_hour"),
    "g05": ("Stelle den Wecker auf halb fünf morgens.", "into_hour"),

    # quarter_past — the minimal pair. Two FROM, two INTO, same clock position.
    "g06": ("Stelle den Wecker auf viertel nach fünf morgens.", "from_hour"),
    "g07": ("Stelle den Wecker auf viertel neun morgens.", "into_hour_regional"),
    "g08": ("Stelle den Wecker auf viertel nach zehn abends.", "from_hour"),
    "g09": ("Stelle den Wecker auf viertel zwei nachmittags.", "into_hour_regional"),

    # n_minutes — FROM the completed hour, and the cell where no German INTO
    # form exists at all. Russian is 0 of 32 here. German has only the English
    # option, so if our account is right this should look like English.
    "g10": ("Stelle den Wecker auf zwanzig nach sieben morgens.", "from_hour"),
    "g11": ("Stelle den Wecker auf fünf nach acht abends.", "from_hour"),
    "g12": ("Stelle den Wecker auf zehn nach zehn morgens.", "from_hour"),
    "g13": ("Stelle den Wecker auf fünfundzwanzig nach eins nachmittags.", "from_hour"),
    "g14": ("Stelle den Wecker auf zwanzig nach zwei nachmittags.", "from_hour"),
    "g15": ("Stelle den Wecker auf zehn nach sechs morgens.", "from_hour"),
    "g16": ("Stelle den Wecker auf fünf nach zwölf mittags.", "from_hour"),
    "g17": ("Stelle den Wecker auf fünfundzwanzig nach neun abends.", "from_hour"),

    # literal — the probe's health check. Must be near perfect or the German
    # carrier sentence is broken and nothing else in the file means anything.
    "g18": ("Stelle den Wecker auf 07:20.", "digital"),
    "g19": ("Stelle den Wecker auf 14:30.", "digital"),
    "g20": ("Stelle den Wecker auf 21:15.", "digital"),
    "g21": ("Stelle den Wecker auf 05:45.", "digital"),

    # subtractive — subtractive in all four languages, and a control that fails
    # in English too. Kept standard: 05:45 could be dreiviertel sechs, but a
    # contested label does not belong in a control.
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
