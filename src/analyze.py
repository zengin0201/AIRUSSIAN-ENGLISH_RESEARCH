from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

from .metrics import PRICES, aggregate_arm, bootstrap_ci, decompose


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


def table(stats_by_arm: Dict[str, Dict[str, Any]]) -> str:
    columns = [
        ("arm", "arm", "{:<13}"),
        ("n", "episodes", "{:>4}"),
        ("succ%", "success_rate", "{:>6}"),
        ("steps", "mean_steps", "{:>6}"),
        ("tok/ep", "mean_tokens_per_episode", "{:>8}"),
        ("tok/step", "tokens_per_step", "{:>9}"),
        ("CPST", "cpst_tokens", "{:>9}"),
        ("cut%", "rate_any_truncation", "{:>7}"),
        ("loop%", "rate_looping", "{:>7}"),
        ("parse_err", "mean_parse_errors", "{:>10}"),
        ("tool_err", "mean_tool_errors", "{:>9}"),
    ]
    head = "".join(fmt.format(label) for label, _, fmt in columns)
    lines = [head, "-" * len(head)]
    for arm, stats in stats_by_arm.items():
        cells = []
        for label, key, fmt in columns:
            if key == "arm":
                value = arm
            elif key in ("success_rate", "rate_any_truncation", "rate_looping"):
                value = f'{(stats.get(key) or 0) * 100:.0f}'
            else:
                value = stats.get(key)
                value = "n/a" if value is None else value
            cells.append(fmt.format(str(value)))
        lines.append("".join(cells))
    return "\n".join(lines)


def figure(stats_by_arm, factors, fert_ratio, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping figure")
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    arms = list(stats_by_arm.keys())
    cpst = [stats_by_arm[a].get("cpst_tokens") or 0 for a in arms]
    left.bar(arms, cpst, color="#4c72b0")
    left.set_ylabel("tokens per successful task")
    left.set_title("Cost per successful task")
    left.tick_params(axis="x", rotation=20)

    names, values = [], []
    if fert_ratio:
        names.append("fertility\n(predicted)")
        values.append(fert_ratio)
    for key, label in [
        ("density_factor", "density"),
        ("step_factor", "steps"),
        ("success_factor", "success"),
        ("cpst_ratio_measured", "TOTAL\n(measured)"),
    ]:
        if factors.get(key):
            names.append(label)
            values.append(factors[key])

    colors = ["#999999"] + ["#4c72b0"] * (len(values) - 2) + ["#c44e52"]
    right.bar(names, values, color=colors[: len(values)])
    right.axhline(1.0, color="black", linewidth=0.8)
    right.set_ylabel("ratio RU / EN")
    right.set_title("Where the gap comes from")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"figure saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--figure", default="figures/main.png")
    parser.add_argument("--price", default="gpt-4o-mini", choices=sorted(PRICES))
    args = parser.parse_args()

    meta, episodes = load(args.results)
    by_arm: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        by_arm[row["arm"]].append(row)

    stats_by_arm = {
        arm: aggregate_arm(rows, args.price) for arm, rows in by_arm.items()
    }

    print(f"\nmodel: {meta.get('model')}   tasks: {meta.get('n_tasks')}   "
          f"repeats: {meta.get('repeats')}")
    fert = (meta.get("fertility") or {}).get("fertility_ratio")
    l2 = (meta.get("fertility") or {}).get("language") or "l2"
    print(f"static fertility ratio {l2.upper()}/EN: {fert}")
    print("\n" + table(stats_by_arm) + "\n")

    factors = {}
    if "en" in stats_by_arm and l2 in stats_by_arm:
        factors = decompose(stats_by_arm["en"], stats_by_arm[l2])
        print(f"decomposition of the CPST gap ({l2.upper()} / EN):")
        for key, value in factors.items():
            print(f"  {key:<24} {value}")

        ci = bootstrap_ci(by_arm["en"], by_arm[l2])
        if ci:
            print(f"\nbootstrap 95% CI for CPST ratio: "
                  f"{ci['cpst_ratio_median']} [{ci['ci95_low']}, {ci['ci95_high']}]")

        if fert and factors.get("cpst_ratio_measured"):
            print(f"\nHEADLINE: tokenizer fertility predicts {fert}x, "
                  f"the real agentic cost is {factors['cpst_ratio_measured']}x "
                  f"({factors['cpst_ratio_measured'] / fert:.2f}x more than predicted)")

    print("\nharness audit (these must be small, or the numbers are ours not the model's):")
    for arm, stats in stats_by_arm.items():
        print(f"  {arm:<20} episodes with a cut-off answer "
              f"{stats.get('rate_any_truncation', 0) * 100:>5.1f}%   "
              f"turns lost to cut-off {stats.get('mean_truncated_lost', 0):>5.2f}   "
              f"killed as looping {stats.get('rate_looping', 0) * 100:>5.1f}%")

    kinds = sorted({r.get("difficulty", "unknown") for r in episodes})
    if len(kinds) > 1:
        print("\nsuccess rate by difficulty:")
        header = f'  {"difficulty":<14}' + "".join(f"{a:>20}" for a in stats_by_arm)
        print(header)
        for kind in kinds:
            cells = []
            for arm in stats_by_arm:
                rows = [r for r in by_arm[arm] if r.get("difficulty") == kind]
                if rows:
                    ok = sum(1 for r in rows if r["success"])
                    cells.append(f"{ok}/{len(rows)}")
                else:
                    cells.append("-")
            print(f'  {kind:<14}' + "".join(f"{c:>20}" for c in cells))

    for arm, stats in stats_by_arm.items():
        if stats.get("placeholders_total"):
            print(f"\nplaceholder protection in {arm}: "
                  f"{stats['placeholders_total']} inserted, "
                  f"{stats['placeholders_lost']} lost, "
                  f"{stats['placeholders_mangled']} came back damaged")

    if "ru_translate" in stats_by_arm and "ru" in stats_by_arm:
        plain = stats_by_arm["ru"]
        trans = stats_by_arm["ru_translate"]
        print("\ntranslate-then-act policy:")
        print(f"  CPST ru            = {plain.get('cpst_tokens')}")
        print(f"  CPST ru_translate  = {trans.get('cpst_tokens')}")
        print(f"  mean translation overhead = "
              f"{trans.get('mean_translation_tokens')} tokens per episode")
        if plain.get("cpst_tokens") and trans.get("cpst_tokens"):
            saving = plain["cpst_tokens"] - trans["cpst_tokens"]
            verdict = "WORTH IT" if saving > 0 else "NOT worth it"
            print(f"  saving per successful task = {saving:.0f} tokens -> {verdict}")

    figure(stats_by_arm, factors, fert, args.figure)


if __name__ == "__main__":
    main()
