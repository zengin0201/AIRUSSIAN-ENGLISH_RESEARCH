"""
Look at what actually went wrong.

    python -m src.inspect --results results/smoke.jsonl
    python -m src.inspect --results results/smoke.jsonl --arm ru --task t01
    python -m src.inspect --results results/smoke.jsonl --all

Numbers tell you how big the gap is. Traces tell you why it exists.
Read traces before you write a single sentence of the paper.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("record_type") != "meta":
                    rows.append(row)
    return rows


def show(record, max_turns=12):
    head = (
        f"=== {record['task_id']} / {record['arm']} / repeat {record.get('repeat', 0)} "
        f"=== {'SUCCESS' if record['success'] else 'FAIL'} "
        f"({record['checks_passed']}/{record['checks_total']} checks, "
        f"kind={record['failure_kind']})"
    )
    print("\n" + head)
    print(f"instruction: {record.get('instruction_used', '')[:200]}")
    if record.get("translation_text"):
        print(f"translation: {record['translation_text'][:200]}")
    print(f"tokens={record['total_tokens']}  steps={record['steps_used']}  "
          f"parse_errors={record['parse_errors']}  tool_errors={record['tool_errors']}")
    print("-" * 70)

    for turn in record.get("turns", [])[:max_turns]:
        raw = turn["raw"].replace("\n", " ")[:150]
        tools = turn.get("tools") or ([turn["tool"]] if turn.get("tool") else [])
        statuses = turn.get("tool_statuses") or [turn.get("tool_status")]
        print(f"  step {turn['step']}: parse={turn['parse_status']:<14} "
              f"n_actions={turn.get('n_actions', len(tools))}")
        for name, status in zip(tools, statuses):
            print(f"      {str(name):<18} status={status}")
        if not tools:
            print(f"      (no action)        status={statuses[0] if statuses else 'n/a'}")
        if turn["parse_status"] != "clean":
            print(f"    raw : {raw}")

    print(f"  final state: {json.dumps(record.get('final_state', {}), ensure_ascii=False)[:400]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--arm", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--all", action="store_true", help="Show successes too.")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    rows = load(args.results)
    print("failure summary by arm:")
    by_arm = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    for arm, group in by_arm.items():
        fails = [r for r in group if not r["success"]]
        kinds = Counter(r["failure_kind"] for r in fails)
        parse = Counter()
        for r in group:
            for t in r.get("turns", []):
                parse[t["parse_status"]] += 1
        print(f"  {arm:<13} {len(fails)}/{len(group)} failed   "
              f"kinds={dict(kinds)}   parse={dict(parse)}")

    selected = rows
    if args.arm:
        selected = [r for r in selected if r["arm"] == args.arm]
    if args.task:
        selected = [r for r in selected if r["task_id"] == args.task]
    if not args.all:
        selected = [r for r in selected if not r["success"]]

    if not selected:
        print("\nnothing to show with these filters.")
        return

    for record in selected[: args.limit]:
        show(record)


if __name__ == "__main__":
    main()
