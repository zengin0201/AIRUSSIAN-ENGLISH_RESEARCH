

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



LF = chr(10)


def figure_polish(pl_runs, ru_runs, out_path: str, large_only: bool = True) -> None:
    """
    The crossed design, which had no picture until now.

    Polish and Russian ask for the same clock times. At :30 both count into the
    coming hour. At :15 and :20 Polish counts from the completed hour, as
    English does, and Russian still counts into it. So the language is held
    roughly fixed — two Slavic languages — and the construction varies.

    Every bar is annotated with which way its language counts, because that is
    the whole argument and a reader should not have to take it on trust.

    This exists because `figure_main.py` was hardcoded to ("en", "ru") in three
    places and knew nothing about Polish, so the strongest result in the
    project had tables and no figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["half_past", "quarter_past", "n_minutes"]
    labels = ["«половина» / «wpół do»" + LF + "07:30",
              "«четверть» / «kwadrans po»" + LF + "07:15",
              "«N минут» / «N po»" + LF + "07:20"]
    # how each language counts, per clock form
    counts = {
        "English": ["from", "from", "from"],
        "Polish": ["INTO", "from", "from"],
        "Russian": ["INTO", "INTO", "INTO"],
    }
    colours = {"English": "#4c72b0", "Polish": "#55a868", "Russian": "#c44e52"}

    def rate(runs, arm, category):
        hits = total = 0
        for _, episodes in runs:
            for row in episodes:
                if row.get("difficulty") == category and row["arm"] == arm:
                    total += 1
                    hits += bool(row["success"])
        return (100 * hits / total if total else 0.0), hits, total

    series = {
        "English": [rate(pl_runs, "en", c) for c in order],
        "Polish": [rate(pl_runs, "pl", c) for c in order],
        "Russian": [rate(ru_runs, "ru", c) for c in order],
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = list(range(len(order)))
    width = 0.26
    offsets = {"English": -width, "Polish": 0.0, "Russian": width}

    for name, values in series.items():
        bars = ax.bar([x + offsets[name] for x in xs], [v[0] for v in values],
                      width, label=name, color=colours[name],
                      edgecolor="white", linewidth=0.6)
        for bar, (pct, hits, total), how in zip(bars, values, counts[name]):
            centre = bar.get_x() + bar.get_width() / 2
            ax.text(centre, bar.get_height() + 2, f"{hits}/{total}",
                    ha="center", va="bottom", fontsize=8)
            # A zero bar has no inside, so the construction label goes above the
            # count instead of on top of it.
            inside = pct > 20
            ax.text(centre, 3 if inside else bar.get_height() + 11, how,
                    ha="center", va="bottom", fontsize=7, rotation=90,
                    color="white" if inside else "#333333",
                    fontweight="bold" if how == "INTO" else "normal")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 132)
    ax.set_ylabel("tasks solved, %")
    ax.legend(loc="upper center", ncol=3, fontsize=9, frameon=False,
              bbox_to_anchor=(0.5, 1.02))
    ax.spines[["top", "right"]].set_visible(False)
    scope = "27B and 120B" if large_only else "all models"
    ax.set_title(
        f"Same clock times, two Slavic languages, {scope}." + LF
        + "The label on each bar is how that language counts.",
        fontsize=10, pad=26)
    ax.text(0.5, -0.22,
            "Where Polish and Russian share the construction (07:30) they tie."
            + LF +
            "Where it differs, Polish tracks English and Russian collapses.",
            transform=ax.transAxes, ha="center", va="top", fontsize=9)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save(fig, out_path)


def figure_german(de_runs, ru_runs, out_path: str) -> None:
    """
    The scale axis, which is what German added.

    German counts into the coming hour as Russian does, but with a closed set
    of three phrases and no productive form. So the picture is three lines:

      German counting-from   flat at ceiling, every size
      German counting-into   climbs from floor to ceiling with scale
      Russian productive     flat on the floor, every size

    The gap between the second and third lines is the paper's claim in one
    image: scale supplies the missing units and never the missing rule.

    `de_runs` and `ru_runs` are lists of (label, episodes) ordered smallest
    model first. Russian may be empty, in which case its line is omitted.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def rate(rows, arm, difficulty):
        ok = n = 0
        for row in rows:
            # The first line of a result file is the run header, which has
            # no arm. Other figures reach it through helpers that strip it.
            if row.get("arm") != arm or row.get("difficulty") != difficulty:
                continue
            n += 1
            ok += bool(row["success"])
        return (100.0 * ok / n if n else None), ok, n

    sizes = [label for label, _ in de_runs]
    x = list(range(len(sizes)))

    series = []
    into = [rate(rows, "de", "half_past") for _, rows in de_runs]
    frm = [rate(rows, "de", "n_minutes") for _, rows in de_runs]
    series.append(("German counting-from" + LF + "N nach X", frm, "#4c72b0", "o", "-"))
    series.append(("German counting-into" + LF + "halb X", into, "#dd8452", "s", "-"))
    if ru_runs:
        ru = [rate(rows, "ru", "n_minutes") for _, rows in ru_runs]
        if any(v[0] is not None for v in ru):
            series.append(("Russian productive" + LF + "N минут X-ого",
                           ru, "#c44e52", "^", "--"))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for rank, (name, values, colour, marker, style) in enumerate(series):
        ys = [v[0] for v in values]
        ax.plot(x, ys, marker=marker, color=colour, linestyle=style,
                linewidth=2, markersize=8, label=name)
        # The Russian line is flat at the floor and its per-point labels
        # collided with the German counting-into ones, which sit just above it
        # at the small sizes. It is constant, so one label says everything.
        points = list(zip(x, values))
        if style == "--":
            points = points[-1:]
        for xi, (pct, ok, n) in points:
            if pct is None:
                continue
            above = pct < 90
            dy = (9 + 11 * rank) if above else (-17 - 11 * rank)
            ax.annotate(f"{ok}/{n}" + (" at every size" if style == "--" else ""),
                        (xi, pct), textcoords="offset points",
                        xytext=(0, dy), ha="right" if style == "--" else "center",
                        fontsize=8, color=colour)

    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_xlabel("model size")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(-6, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
              fontsize=8)
    ax.set_title("Scale repairs the closed set and never the productive rule",
                 fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_path.rsplit(".", 1)[0] + "." + ext, dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", nargs="+", required=True,
                        help="One result file per model, run on the time probe.")
    parser.add_argument("--tasks", default="data/tasks_time.json")
    parser.add_argument("--frequency", default=None,
                        help="results/frequency.json, to draw the third figure.")
    parser.add_argument("--polish", nargs="*", default=None,
                        help="Result files from the Polish probe. Paired with "
                             "--probe, which must then be the Russian runs of "
                             "the same models, to draw the crossed-design figure.")
    parser.add_argument("--german", nargs="*", default=None,
                        help="German probe result files, smallest model first, "
                             "to draw the scale figure.")
    parser.add_argument("--german-labels", nargs="*", default=None,
                        help="Size labels matching --german, e.g. 3B 3.8B 27B 120B.")
    parser.add_argument("--german-russian", nargs="*", default=None,
                        help="Russian probe runs in the same size order, to draw "
                             "the productive-form reference line.")
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
    if args.polish:
        figure_polish([load(p) for p in args.polish], runs,
                      os.path.join(args.out, "polish.png"))
    if args.german:
        labels = args.german_labels or [f"model {i+1}" for i in range(len(args.german))]
        if len(labels) != len(args.german):
            raise SystemExit("--german-labels must match --german in length")
        ru_scale = []
        if args.german_russian:
            if len(args.german_russian) != len(args.german):
                raise SystemExit("--german-russian must match --german in length")
            ru_scale = [(lab, load(path)[1])
                        for lab, path in zip(labels, args.german_russian)]
        figure_german([(lab, load(path)[1])
                       for lab, path in zip(labels, args.german)],
                      ru_scale, os.path.join(args.out, "german.png"))


if __name__ == "__main__":
    main()
