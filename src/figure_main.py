

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List

CATEGORY_ORDER = [
    "literal",
    "into_hour",
    "half_past",
    "quarter_past",
    "n_minutes",
    "subtractive",
    "arithmetic",
]
CATEGORY_LABEL = {
    "literal": "07:20\ncontrol",
    "into_hour": "counting\ninto hour",
    "half_past": "«половина»\nfixed",
    "quarter_past": "«четверть»\nmid",
    "n_minutes": "«N минут»\nproductive",
    "subtractive": "«без N»\ncontrol",
    "arithmetic": "arithmetic",
}

INTO_HOUR_FAMILY = {"into_hour", "half_past", "quarter_past", "n_minutes"}


def load(path: str):
    meta, episodes = {}, []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("record_type") == "meta":
                meta = row
            else:
                episodes.append(row)
    return meta, episodes


def short_model(name: str) -> str:
    return str(name).split("/")[-1].replace("-Instruct", "").replace("-instruct", "")


def minutes(value: str) -> int:
    hours, mins = value.split(":")
    return int(hours) * 60 + int(mins)


def save(fig, out_path: str) -> None:
    """
    Write the figure twice: PNG to look at, PDF to typeset.

    A raster figure dropped into a two-column paper looks soft the moment a
    reviewer zooms in, and these are all line and bar plots, so there is no
    reason to rasterise them. LaTeX takes the PDF; GitHub renders the PNG.
    """
    fig.savefig(out_path, dpi=160)
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    fig.savefig(pdf_path)
    print(f"saved {out_path} and {pdf_path}")


def figure_success(runs, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [short_model(meta.get("model")) for meta, _ in runs]
    n = len(runs)
    if n <= 3:
        rows, cols = 1, n
        figsize = (5.4 * n if n <= 2 else 4.6 * n, 4.6)
    else:
        rows = 2
        cols = (n + 1) // 2
        figsize = (5.6 * cols, 4.3 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharey=True)
    axes = [axes] if n == 1 else list(axes.flat)
    for spare in axes[n:]:
        spare.axis("off")
    axes = axes[:n]

    for ax, (meta, episodes), model in zip(axes, runs, models):
        rates: Dict[str, Dict[str, float]] = defaultdict(dict)
        for arm in ("en", "ru"):
            for category in CATEGORY_ORDER:
                rows = [
                    r for r in episodes
                    if r["arm"] == arm and r.get("difficulty") == category
                ]
                if rows:
                    rates[arm][category] = 100 * sum(
                        1 for r in rows if r["success"]
                    ) / len(rows)

        present = [c for c in CATEGORY_ORDER if c in rates.get("en", {})]
        xs = range(len(present))
        width = 0.38
        ax.bar([x - width / 2 for x in xs], [rates["en"][c] for c in present],
               width, label="English", color="#4c72b0")
        ax.bar([x + width / 2 for x in xs], [rates["ru"][c] for c in present],
               width, label="Russian", color="#c44e52")

        ax.set_xticks(list(xs))
        ax.set_xticklabels([CATEGORY_LABEL[c] for c in present], fontsize=8)
        ax.set_title(model, fontsize=11)
        ax.set_ylim(0, 108)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        family = [c for c in present if c in INTO_HOUR_FAMILY]
        if family:
            centre = sum(present.index(c) for c in family) / len(family)
            ax.text(centre, 103, "counting into the hour", ha="center",
                    fontsize=8, color="#333333")

    for index, ax in enumerate(axes):
        if index % cols == 0:
            ax.set_ylabel("tasks solved, %")
    axes[0].legend(loc="lower left", fontsize=9)
    if len(runs) >= 3:
        caption = (
            "English is flat across every category. Russian is not, and the "
            "Russian bars fall with how fixed the phrase is:\n«половина» is "
            "recovered by the large models, «четверть» barely, and «N минут "
            "X-ого» — the fully productive form — is never solved at any scale."
        )
    else:
        caption = (
            "Success by time expression. The controls carry the argument: "
            "digital times are equal, subtractive forms fail in both languages."
        )
    fig.suptitle(caption, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90 if len(runs) >= 3 else 0.94))
    save(fig, out_path)


def figure_error(runs, tasks: Dict[str, Any], out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buckets: Counter = Counter()
    total = 0
    for _, episodes in runs:
        for row in episodes:
            if row.get("difficulty") not in INTO_HOUR_FAMILY or row["arm"] == "en":
                continue
            task = tasks.get(row["task_id"])
            if not task:
                continue
            want = task["checks"]["en"][0]["time"]
            alarms = row["final_state"]["alarms"]
            if not alarms:
                continue
            delta = minutes(alarms[0]["time"]) - minutes(want)
            total += 1
            if delta == 0:
                buckets["0"] += 1
            elif delta == 60:
                buckets["+60"] += 1
            elif delta == -60:
                buckets["-60"] += 1
            else:
                buckets["other"] += 1

    order = ["0", "+60", "-60", "other"]
    labels = ["correct", "+60 min\n(one hour late)", "−60 min", "other error"]
    values = [buckets.get(k, 0) for k in order]
    colors = ["#55a868", "#c44e52", "#dd8452", "#999999"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        if value:
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.15, str(value),
                    ha="center", fontsize=10)
    ax.set_ylabel(f"Russian answers (n = {total})")
    ax.set_title(
        "Russian counting-into-the-hour expressions,\n"
        f"pooled over {len(runs)} models",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(values) * 1.18 if values else 1)
    fig.tight_layout()
    save(fig, out_path)


def figure_frequency(runs, freq_path: str, out_path: str) -> None:
    """
    Accuracy against corpus frequency, one line per language.

    This is the figure that kills the easy reading. Both languages get rarer
    from left to right, and English falls further, yet only the Russian line
    drops. If rarity were the explanation, the two lines would be parallel.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(freq_path, "r", encoding="utf-8") as handle:
        freq = json.load(handle)["per_category"]

    order = ["half_past", "quarter_past", "n_minutes"]
    ticks = ["«половина»\nhalf past", "«четверть»\nquarter past", "«N минут»\nN past"]

    accuracy = {"en": [], "ru": []}
    for category in order:
        for arm in ("en", "ru"):
            hits = total = 0
            for _, episodes in runs:
                for row in episodes:
                    if row.get("difficulty") == category and row["arm"] == arm:
                        total += 1
                        hits += bool(row["success"])
            accuracy[arm].append(100 * hits / total if total else 0.0)

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    xs = range(len(order))

    left.plot(xs, [freq[c]["en"] for c in order], marker="o",
              color="#4c72b0", label="English")
    left.plot(xs, [freq[c]["ru"] for c in order], marker="o",
              color="#c44e52", label="Russian")
    left.set_yscale("log")
    left.set_xticks(list(xs))
    left.set_xticklabels(ticks, fontsize=9)
    left.set_ylabel("corpus frequency (Google Books, 1980-2019)")
    left.set_title("Both languages get rarer, English more so")
    left.grid(alpha=0.3)
    left.legend()

    right.plot(xs, accuracy["en"], marker="o", color="#4c72b0", label="English")
    right.plot(xs, accuracy["ru"], marker="o", color="#c44e52", label="Russian")
    right.set_xticks(list(xs))
    right.set_xticklabels(ticks, fontsize=9)
    right.set_ylabel("tasks solved, % (four models pooled)")
    right.set_ylim(-4, 104)
    right.set_title("Only Russian accuracy follows")
    right.grid(alpha=0.3)
    right.legend()

    fig.suptitle(
        "Rarity alone does not explain the failure. English frequency falls by "
        "20x across these three categories and English accuracy does not move.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", nargs="+", required=True,
                        help="One result file per model, run on the time probe.")
    parser.add_argument("--tasks", default="data/tasks_time.json")
    parser.add_argument("--frequency", default=None,
                        help="results/frequency.json, to draw the third figure.")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    runs = [load(p) for p in args.probe]
    with open(args.tasks, "r", encoding="utf-8") as handle:
        tasks = {t["id"]: t for t in json.load(handle)["tasks"]}

    os.makedirs(args.out, exist_ok=True)
    figure_success(runs, os.path.join(args.out, "main.png"))
    figure_error(runs, tasks, os.path.join(args.out, "error.png"))
    if args.frequency:
        figure_frequency(runs, args.frequency,
                         os.path.join(args.out, "frequency.png"))


if __name__ == "__main__":
    main()
