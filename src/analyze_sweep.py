from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def load(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
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


def figure(cells: Dict[Tuple[int, str], List[Dict[str, Any]]], path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping figure")
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    arms = sorted({arm for _, arm in cells})
    budgets = sorted({budget for budget, _ in cells})
    colors = {"en": "#4c72b0", "ru": "#c44e52", "ru_sys_ru": "#55a868",
              "ru_translate": "#8172b2", "ru_translate_safe": "#ccb974"}

    for arm in arms:
        xs, success, lost = [], [], []
        for budget in budgets:
            rows = cells.get((budget, arm))
            if not rows:
                continue
            xs.append(budget)
            success.append(100 * sum(1 for r in rows if r["success"]) / len(rows))
            lost.append(
                100 * sum(1 for r in rows if (r.get("truncated_lost_turns") or 0) > 0)
                / len(rows)
            )
        if not xs:
            continue
        style = dict(marker="o", color=colors.get(arm, "#777777"), label=arm)
        left.plot(xs, success, **style)
        right.plot(xs, lost, **style)

    left.set_xlabel("output budget (max_new_tokens)")
    left.set_ylabel("success rate, %")
    left.set_title("Where each language stops fitting")
    left.set_ylim(-3, 103)
    left.grid(alpha=0.3)
    left.legend()

    right.set_xlabel("output budget (max_new_tokens)")
    right.set_ylabel("episodes losing a turn to a cut-off, %")
    right.set_title("The cliff itself")
    right.set_ylim(-3, 103)
    right.grid(alpha=0.3)
    right.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"\nfigure saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True,
                        help="Result files, or a glob like 'results/sweep_*.jsonl'.")
    parser.add_argument("--figure", default="figures/budget_cliff.png")
    args = parser.parse_args()

    paths: List[str] = []
    for pattern in args.results:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    cells: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    models = set()
    for path in paths:
        meta, episodes = load(path)
        budget = meta.get("max_new_tokens")
        if budget is None:
            print(f"{path}: no max_new_tokens in meta, skipped "
                  f"(it predates the sweep support)")
            continue
        models.add(meta.get("model"))
        for row in episodes:
            cells[(int(budget), row["arm"])].append(row)

    if not cells:
        print("nothing to plot")
        return

    print(f"model(s): {', '.join(str(m) for m in sorted(models, key=str))}")
    arms = sorted({arm for _, arm in cells})
    budgets = sorted({budget for budget, _ in cells})

    head = f'{"budget":>8}' + "".join(f"{a:>22}" for a in arms)
    print("\nsuccess rate, and share of episodes that lost a turn to a cut-off:")
    print(head)
    print("-" * len(head))
    for budget in budgets:
        cellsline = []
        for arm in arms:
            rows = cells.get((budget, arm))
            if not rows:
                cellsline.append("-")
                continue
            ok = sum(1 for r in rows if r["success"])
            cut = sum(1 for r in rows if (r.get("truncated_lost_turns") or 0) > 0)
            cellsline.append(f"{ok}/{len(rows)}  cut {cut}/{len(rows)}")
        print(f"{budget:>8}" + "".join(f"{c:>22}" for c in cellsline))

    print("\nsmallest budget at which each arm reaches its own best success rate:")
    for arm in arms:
        best = 0.0
        first_at_best = None
        for budget in budgets:
            rows = cells.get((budget, arm))
            if not rows:
                continue
            rate = sum(1 for r in rows if r["success"]) / len(rows)
            if rate > best:
                best = rate
                first_at_best = budget
        if first_at_best is not None:
            print(f"  {arm:<20} {best * 100:>5.1f}% at {first_at_best} output tokens")
    print("\n  The gap between the English and the Russian row is the finding.")

    figure(cells, args.figure)


if __name__ == "__main__":
    main()
