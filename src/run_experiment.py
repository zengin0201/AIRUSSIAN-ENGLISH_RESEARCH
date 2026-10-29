from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

from .agent import ARMS, ARM_LANGUAGE, run_episode
from .backends import OracleBackend, build_backend
from .energy import EnergyMeter
from .metrics import fertility


def load_tasks(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agentic Language Tax runner")
    parser.add_argument("--backend", choices=["local", "api", "oracle"], required=True)
    parser.add_argument("--model", default="oracle")
    parser.add_argument("--tasks", default="data/tasks_seed.json")
    parser.add_argument("--arms", nargs="+", default=["en", "ru"], choices=ARMS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--context-budget", type=int, default=None,
                        help="Token budget for the conversation. Set it low "
                             "(e.g. 1500) to study forced compaction.")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for sampling. Only matters when "
                             "--temperature > 0.")
    parser.add_argument("--no-truncation-hint", action="store_true",
                        help="Do not tell the agent when its answer was cut "
                             "off by the output budget. Use this to measure "
                             "how much the hint is worth.")
    parser.add_argument("--energy", action="store_true", help="Measure GPU energy.")
    parser.add_argument("--limit", type=int, default=None, help="Use only N tasks.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if args.repeats > 1 and args.temperature == 0.0 and args.backend != "oracle":
        raise SystemExit(
            f"--repeats {args.repeats} at --temperature 0.0 runs every task "
            f"{args.repeats} times for the same answer every time (F55). "
            f"Pass --repeats 1, or raise the temperature if you want variation."
        )

    data = load_tasks(args.tasks)
    tasks = data["tasks"]
    if args.limit:
        tasks = tasks[: args.limit]
    today = data["meta"]["today"]
    weekday = data["meta"]["weekday"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.backend == "oracle":
        backend = OracleBackend([])
    elif args.backend == "local":
        backend = build_backend(
            "local",
            args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            load_in_4bit=not args.no_4bit,
            seed=args.seed,
        )
    else:
        backend = build_backend(
            "api",
            args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    fert = fertility(tasks, backend.count_tokens)
    print(f"[fertility] {fert}", flush=True)

    meter = EnergyMeter() if args.energy else None
    idle = None
    if meter and meter.available:
        print("[energy] measuring idle baseline, do not touch the GPU ...", flush=True)
        idle = meter.idle_baseline(5.0)
        print(f"[energy] idle = {idle:.1f} W", flush=True)

    header = {
        "record_type": "meta",
        "model": args.model,
        "backend": args.backend,
        "arms": args.arms,
        "repeats": args.repeats,
        "max_steps": args.max_steps,
        "context_budget": args.context_budget,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "truncation_hint": not args.no_truncation_hint,
        "tasks_file": args.tasks,
        "n_tasks": len(tasks),
        "fertility": fert,
        "idle_watts": idle,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    total = len(tasks) * len(args.arms) * args.repeats
    done = 0

    with open(args.out, "w", encoding="utf-8") as out:
        out.write(json.dumps(header, ensure_ascii=False) + "\n")

        for repeat in range(args.repeats):
            for task in tasks:
                for arm in args.arms:
                    if isinstance(backend, OracleBackend):
                        backend.reset(task["gold"][ARM_LANGUAGE[arm]])

                    if meter and meter.available:
                        meter.start()
                    n_gold = len(task.get("gold", {}).get("en", []))
                    step_budget = max(args.max_steps, n_gold + 3)

                    try:
                        record = run_episode(
                            backend=backend,
                            task=task,
                            arm=arm,
                            today=today,
                            weekday=weekday,
                            max_steps=step_budget,
                            context_budget=args.context_budget,
                            truncation_hint=not args.no_truncation_hint,
                        )
                        record["step_budget"] = step_budget
                    except Exception as exc:
                        record = {
                            "task_id": task["id"],
                            "arm": arm,
                            "error": f"{type(exc).__name__}: {exc}",
                            "success": False,
                            "steps_used": 0,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "parse_errors": 0,
                            "recovered_parses": 0,
                            "tool_errors": 0,
                            "wasted_steps": 0,
                            "called_done": False,
                            "failure_kind": "crash",
                            "latency_s": 0.0,
                        }

                    if meter and meter.available:
                        record["energy"] = meter.stop()

                    record["record_type"] = "episode"
                    record["repeat"] = repeat
                    record["model"] = args.model
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()

                    done += 1
                    flag = "OK " if record.get("success") else "-- "
                    cut = record.get("truncated_lost_turns") or 0
                    print(
                        f"[{done}/{total}] {flag} {task['id']:>4} {arm:<18} "
                        f"steps={record.get('steps_used')} "
                        f"tok={record.get('total_tokens')} "
                        f"chk={record.get('checks_passed')}/{record.get('checks_total')} "
                        f"stop={record.get('stopped_reason')}"
                        + (f" CUT={cut}" if cut else ""),
                        flush=True,
                    )

    if meter:
        meter.shutdown()

    print(f"\nDone. Results written to {args.out}")
    print("Now run:  python -m src.analyze --results " + args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
