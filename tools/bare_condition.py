"""
Ask the models to resolve the clock phrase with no agent machinery at all.

    python tools/bare_condition.py --model openai/gpt-oss-120b \
        --tasks data/tasks_time_v2.json --out results/bare_gptoss120b.jsonl

**Why.** Every probe in this project is an agent task: a system prompt listing
ten tools, an instruction, a JSON tool call with arguments. Resolving the clock
phrase is one step buried inside that. We have never asked the same models the
same sentences without any of it, so we do not know whether the construction is
unresolvable for them or unresolvable *under the load of the task*.

Those are different claims. The paper currently assumes the first and has never
checked.

- **Correct bare, wrong in the agent frame** -> the agentic setting is what
  breaks it, which is a sharper result than the paper claims and lands exactly
  on its framing.
- **Wrong both ways** -> the construction simply is not resolved, and the agent
  frame is incidental.

The sentence is passed through unchanged, carrier and all, so the linguistic
material is identical to the agent condition. Only the machinery is removed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PROMPT = (
    "What time is being asked for in this sentence? "
    "Answer with the 24-hour clock time only, in HH:MM format, and nothing else.\n\n"
    "Sentence: {sentence}"
)

TIME = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")


def ask(model: str, base: str, key: str, sentence: str, retries: int = 4) -> tuple:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(sentence=sentence)}],
        # Reasoning models spend completion tokens on a reasoning trace
        # before emitting content. At 32 the trace ate the budget,
        # finish_reason came back "length", content was empty, and the
        # whole run read as 0 of 48 -- a harness artifact, not a result.
        "max_tokens": 512,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=payload,
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json",
                 "User-Agent": "agentic-language-tax/1.0"})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                body = json.loads(response.read())
            choice = body["choices"][0]
            content = choice["message"].get("content") or ""
            if not content.strip():
                return None, f"empty content, finish={choice.get('finish_reason')}"
            return content, None
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(2 * (attempt + 1))
    return None, last


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = os.environ.get("ALT_API_BASE", "")
    key = os.environ.get("ALT_API_KEY", "")
    if not base or not key:
        raise SystemExit("Set ALT_API_BASE and ALT_API_KEY first.")

    with open(os.path.join(ROOT, args.tasks), "r", encoding="utf-8") as handle:
        tasks = json.load(handle)["tasks"]

    total = len(tasks) * len(args.arms)
    done = ok = 0
    with open(os.path.join(ROOT, args.out), "w", encoding="utf-8") as out:
        out.write(json.dumps({"record_type": "meta", "model": args.model,
                              "tasks": args.tasks, "arms": args.arms,
                              "condition": "bare"}, ensure_ascii=False) + "\n")
        for task in tasks:
            for arm in args.arms:
                sentence = task["instruction"].get(arm)
                if sentence is None:
                    continue
                done += 1
                gold = task["checks"][arm][0].get("time")
                text, error = ask(args.model, base, key, sentence)
                match = TIME.search(text or "")
                answer = (f"{int(match.group(1)):02d}:{match.group(2)}"
                          if match else None)
                success = answer == gold
                ok += success
                out.write(json.dumps({
                    "record_type": "bare", "task_id": task["id"], "arm": arm,
                    "difficulty": task.get("difficulty"), "sentence": sentence,
                    "gold": gold, "answer": answer, "raw": text,
                    "success": success, "error": error,
                }, ensure_ascii=False) + "\n")
                out.flush()
                flag = "OK" if success else "--"
                print(f"[{done}/{total}] {flag} {task['id']} {arm}  "
                      f"gold {gold}  got {answer}"
                      + (f"  ERROR {error[:50]}" if error else ""))

    print(f"\n{ok}/{done} correct. Written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
