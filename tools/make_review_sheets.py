"""
Generate review sheets for native speakers, one per language.

    python tools/make_review_sheets.py

Why this exists. The Polish and German probes were written from reference
grammars by an author who speaks neither. That is the single outstanding risk
to the paper: a reviewer who speaks one of those languages and objects to a
wording can sink a result, and no amount of care in the code answers it.

The sheets are plain Markdown so they can be sent as a file, pasted into a
message, or printed. One row per item, two things to fill in: a verdict and a
comment. Nothing else is asked, because a sheet that takes an hour comes back
empty.

**What the paper is allowed to claim depends on what comes back**, and these
are different claims carrying different weight:

    "checked by an acquaintance who speaks Polish"
    "checked by a native speaker"
    "checked by a native speaker with linguistic training"
    "checked by <name>" — in Acknowledgements, with their permission

Record which one is true. Upgrading it later because it sounds better is the
same class of act as backdating a notebook.

**The German regional items need a speaker from the right region.** viertel
neun and dreiviertel neun are used in the east and south; a speaker from
Hamburg may not use them at all, and "a German speaker said it was odd" is not
evidence against them. Ask where the person is from and write it down.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "review")

SHEETS = [
    ("polish", "data/tasks_time_pl.json", "pl", "Polish",
     "Polish items, written from reference grammars by a non-speaker."),
    ("german", "data/tasks_time_de.json", "de", "German",
     "German items, first probe. Written from reference grammars by a non-speaker."),
    ("german2", "data/tasks_time_de2.json", "de", "German",
     "German items, expanded probe. Twelve regional items need a speaker from "
     "the east or south."),
    ("russian_case", "data/tasks_time_case.json", "ru", "Russian",
     "Russian case frames. The author is a native speaker, so this sheet asks "
     "for a second opinion rather than a first one."),
]

HEADER = """# Review sheet — {language}

{why}

**Two things we already suspect, so you are not asked to find them:**

- German: six items say *morgens* for a time between 02:00 and 05:00
  (h03, h04, h17, h23, h29, h45). We think *nachts* or *früh* is right there.
- Polish: the three *za* items (g22-g24) drop the *na* that every other item
  has, which may push *za dwadzieścia* toward "in twenty minutes".

Please say whether you agree, and flag anything else.

**What to do:** read each sentence and put one mark in the verdict column.
Add a comment only where it is not `ok`. That is the whole task.

    ok        a person would say this
    odd       grammatical, but nobody talks like that
    wrong     not correct {language}
    unsure    you would not use it but others might (say where you are from)

**The time in the second column is what the sentence is supposed to mean.**
If you read the sentence as a different time, that is the most important thing
you can tell us — write the time you read it as.

---

**About you**, so the paper describes the check accurately rather than
generously:

- Name (only if you are willing to be thanked in the paper):
- Native language:
- Where you grew up / which region:
- Any linguistic training:
- May we name you in the Acknowledgements?  yes / no

---

"""

FOOTER = """
---

**Anything else?** Especially: a sentence that is fine in isolation but that
nobody would say to a phone assistant, or a time-of-day word that is wrong for
that hour.

"""


def sheet(path: str, lang: str, language: str, why: str) -> str:
    with open(os.path.join(ROOT, path), "r", encoding="utf-8") as handle:
        data = json.load(handle)

    rows = []
    for task in data["tasks"]:
        sentence = task["instruction"].get(lang)
        if sentence is None:
            continue
        time = task["checks"][lang][0].get("time", "")
        flags = []
        if task.get("de_regional"):
            flags.append("REGIONAL")
        if task.get("marginal_wording"):
            flags.append("we already suspect this one")
        note = f" **({'; '.join(flags)})**" if flags else ""
        rows.append(f"| {task['id']} | {time} | {sentence}{note} | | |")

    out = [HEADER.format(language=language, why=why)]
    out.append(f"Items: {len(rows)}\n")
    out.append("| id | should mean | sentence | verdict | comment |")
    out.append("|---|---|---|---|---|")
    out += rows
    out.append(FOOTER)
    return "\n".join(out)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(OUT, exist_ok=True)
    for name, path, lang, language, why in SHEETS:
        if not os.path.exists(os.path.join(ROOT, path)):
            print(f"  skipped {name}: {path} not found")
            continue
        target = os.path.join(OUT, f"{name}.md")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(sheet(path, lang, language, why))
        print(f"  wrote review/{name}.md")
    print()
    print("Send one sheet per person. Ask them to fill the last two columns")
    print("and the 'about you' block, then send it back. Paste what comes back")
    print("verbatim — the paper will describe the check as whatever it was, and")
    print("a weaker check honestly described is worth more than a stronger one")
    print("claimed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
