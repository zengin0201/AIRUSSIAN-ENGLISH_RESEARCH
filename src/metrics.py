

from __future__ import annotations

import statistics
from typing import Any, Dict, Iterable, List, Optional


PRICES: Dict[str, Dict[str, float]] = {
    "local": {"in": 0.0, "out": 0.0},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "claude-haiku": {"in": 0.80, "out": 4.00},
    "llama-3.1-8b-hosted": {"in": 0.05, "out": 0.08},
}
PRICES_CHECKED_ON = "REPLACE WITH THE DATE YOU VERIFIED THESE"


def money(prompt_tokens: int, completion_tokens: int, price_key: str) -> float:
    price = PRICES.get(price_key, PRICES["local"])
    return (prompt_tokens * price["in"] + completion_tokens * price["out"]) / 1_000_000


def second_language(tasks: List[Dict[str, Any]]) -> str:
    """
    The non-English language of a task file, read from the file itself.

    Task files hold English plus exactly one other language. Which one is not
    hardcoded anywhere, so a Polish file reports Polish numbers under a Polish
    heading instead of borrowing Russian's.
    """
    codes = set()
    for task in tasks:
        codes |= set(task.get("instruction", {}))
    codes.discard("en")
    return codes.pop() if len(codes) == 1 else "ru"


def fertility(tasks: List[Dict[str, Any]], count_tokens) -> Dict[str, float]:
    """
    Tokens needed for the same request in the second language and in English.

    `count_tokens` is a callable, normally backend.count_tokens, so the number
    is measured with the exact tokenizer used in the experiment.
    """
    lang = second_language(tasks)
    en_tokens, ru_tokens = [], []
    for task in tasks:
        en_tokens.append(count_tokens(task["instruction"]["en"]))
        ru_tokens.append(count_tokens(task["instruction"][lang]))
    mean_en = statistics.mean(en_tokens)
    mean_ru = statistics.mean(ru_tokens)
    return {
        "language": lang,
        "mean_tokens_en": round(mean_en, 2),
        "mean_tokens_l2": round(mean_ru, 2),
        "fertility_ratio": round(mean_ru / mean_en, 3) if mean_en else 0.0,
    }


def scaffold_share(tasks: List[Dict[str, Any]], count_tokens, system_text: str) -> Dict[str, float]:
    """
    How much of the first prompt is the shared English scaffold?

    This is the number that explains why `density_factor` comes out near 1.0
    while the tokenizer fertility ratio is 1.4. The system prompt and the tool
    schema are identical English text in every arm. Only the user request
    changes language, and it is a small slice of the context. The tokenizer
    tax is real but it is diluted, and the dilution factor is measurable, so
    we measure it instead of hand-waving.
    """
    lang = second_language(tasks)
    scaffold = count_tokens(system_text)
    en = _safe_mean(count_tokens(t["instruction"]["en"]) for t in tasks)
    ru = _safe_mean(count_tokens(t["instruction"][lang]) for t in tasks)
    return {
        "language": lang,
        "scaffold_tokens": scaffold,
        "mean_instruction_tokens_en": round(en, 1),
        "mean_instruction_tokens_l2": round(ru, 1),
        "instruction_share_en": round(en / (scaffold + en), 4) if scaffold + en else 0.0,
        "predicted_first_prompt_ratio": (
            round((scaffold + ru) / (scaffold + en), 4) if scaffold + en else 0.0
        ),
    }


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def aggregate_arm(records: List[Dict[str, Any]], price_key: str = "gpt-4o-mini") -> Dict[str, Any]:
    """Aggregate all episodes of one arm."""
    if not records:
        return {}

    n = len(records)
    successes = [r for r in records if r["success"]]
    n_success = len(successes)

    total_prompt = sum(r["prompt_tokens"] for r in records)
    total_completion = sum(r["completion_tokens"] for r in records)
    total_tokens = total_prompt + total_completion
    total_cost = money(total_prompt, total_completion, price_key)

    mean_steps = _safe_mean(r["steps_used"] for r in records)
    mean_tokens = total_tokens / n
    tokens_per_step = mean_tokens / mean_steps if mean_steps else 0.0

    return {
        "episodes": n,
        "successes": n_success,
        "success_rate": round(n_success / n, 4),
        "mean_steps": round(mean_steps, 3),
        "mean_tokens_per_episode": round(mean_tokens, 1),
        "tokens_per_step": round(tokens_per_step, 1),
        "total_tokens": total_tokens,
        "cpst_tokens": round(total_tokens / n_success, 1) if n_success else None,
        "cpst_usd": round(total_cost / n_success, 6) if n_success else None,
        "mean_parse_errors": round(_safe_mean(r["parse_errors"] for r in records), 3),
        "mean_truncated_turns": round(
            _safe_mean(r.get("truncated_turns", 0) for r in records), 3
        ),
        "mean_truncated_lost": round(
            _safe_mean(r.get("truncated_lost_turns", 0) for r in records), 3
        ),
        "rate_any_truncation": round(
            sum(1 for r in records if (r.get("truncated_turns") or 0) > 0) / n, 4
        ),
        "mean_recovered_parses": round(_safe_mean(r["recovered_parses"] for r in records), 3),
        "mean_tool_errors": round(_safe_mean(r["tool_errors"] for r in records), 3),
        "mean_wasted_steps": round(_safe_mean(r["wasted_steps"] for r in records), 3),
        "rate_no_done": round(
            sum(1 for r in records if not r["called_done"]) / n, 4
        ),
        "rate_stalled": round(
            sum(1 for r in records if r.get("stopped_reason") == "stalled") / n, 4
        ),
        "rate_max_steps": round(
            sum(1 for r in records if r.get("stopped_reason") == "max_steps") / n, 4
        ),
        "rate_looping": round(
            sum(1 for r in records if r.get("stopped_reason") == "looping") / n, 4
        ),
        "failure_empty": sum(1 for r in records if r["failure_kind"] == "empty"),
        "failure_wrong_value": sum(
            1 for r in records if r["failure_kind"] == "wrong_value"
        ),
        "mean_latency_s": round(_safe_mean(r["latency_s"] for r in records), 2),
        "mean_translation_tokens": round(
            _safe_mean(r.get("translation_tokens", 0) for r in records), 1
        ),
        "placeholders_total": sum(r.get("placeholders_total", 0) for r in records),
        "placeholders_lost": sum(r.get("placeholders_lost", 0) for r in records),
        "placeholders_mangled": sum(r.get("placeholders_mangled", 0) for r in records),
    }


def decompose(
    stats_en: Dict[str, Any], stats_ru: Dict[str, Any]
) -> Dict[str, Optional[float]]:
    """Split the CPST gap into three multiplicative factors."""
    if not stats_en or not stats_ru:
        return {}

    step_factor = (
        stats_ru["mean_steps"] / stats_en["mean_steps"] if stats_en["mean_steps"] else None
    )
    density_factor = (
        stats_ru["tokens_per_step"] / stats_en["tokens_per_step"]
        if stats_en["tokens_per_step"]
        else None
    )
    success_factor = (
        stats_en["success_rate"] / stats_ru["success_rate"]
        if stats_ru["success_rate"]
        else None
    )
    cpst_ratio = (
        stats_ru["cpst_tokens"] / stats_en["cpst_tokens"]
        if stats_en.get("cpst_tokens") and stats_ru.get("cpst_tokens")
        else None
    )

    product = None
    if None not in (step_factor, density_factor, success_factor):
        product = step_factor * density_factor * success_factor

    return {
        "step_factor": round(step_factor, 3) if step_factor else None,
        "density_factor": round(density_factor, 3) if density_factor else None,
        "success_factor": round(success_factor, 3) if success_factor else None,
        "product_of_factors": round(product, 3) if product else None,
        "cpst_ratio_measured": round(cpst_ratio, 3) if cpst_ratio else None,
    }


def bootstrap_ci(
    records_en: List[Dict[str, Any]],
    records_ru: List[Dict[str, Any]],
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    95% confidence interval for the CPST ratio, by resampling episodes.

    Reviewers will ask for this. It costs nothing to compute and it is the
    difference between "a number" and "a result".
    """
    import random

    rng = random.Random(seed)
    ratios: List[float] = []

    for _ in range(n_boot):
        sample_en = [rng.choice(records_en) for _ in records_en]
        sample_ru = [rng.choice(records_ru) for _ in records_ru]
        s_en = sum(1 for r in sample_en if r["success"])
        s_ru = sum(1 for r in sample_ru if r["success"])
        if s_en == 0 or s_ru == 0:
            continue
        cpst_en = sum(r["total_tokens"] for r in sample_en) / s_en
        cpst_ru = sum(r["total_tokens"] for r in sample_ru) / s_ru
        if cpst_en > 0:
            ratios.append(cpst_ru / cpst_en)

    if not ratios:
        return {}
    ratios.sort()
    lo = ratios[int(0.025 * len(ratios))]
    hi = ratios[int(0.975 * len(ratios)) - 1]
    return {
        "cpst_ratio_median": round(ratios[len(ratios) // 2], 3),
        "ci95_low": round(lo, 3),
        "ci95_high": round(hi, 3),
        "n_bootstrap_used": len(ratios),
    }


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """
    Two-tailed Fisher exact test on the 2x2 table

        [[a, b],
         [c, d]]

    typically a = English successes, b = English failures, c = Russian
    successes, d = Russian failures.

    Exact rational arithmetic: no approximation, no floating point in the
    summation, so the answer is the same at any table size. The two-tailed
    p-value is the total probability of every table, at these margins, that is
    no more likely than the observed one.

    This function exists because the p-values in every draft up to 2026-08-21
    were not computed by anything in this repository. Two of them were shared
    verbatim between two different contingency tables, which is impossible.
    They were conservative rather than flattering, but they were wrong by up to
    nine orders of magnitude. Every statistic in the paper now comes from
    `python -m src.stats`, which reads the logs. See NOTES.md, F32.
    """
    from fractions import Fraction
    from math import comb

    if min(a, b, c, d) < 0:
        raise ValueError("counts must not be negative")

    n = a + b + c + d
    if n == 0:
        raise ValueError("the table is empty")

    row1, row2 = a + b, c + d
    col1 = a + c
    if row1 == 0 or row2 == 0 or col1 == 0 or (b + d) == 0:
        return 1.0

    def prob(x: int) -> Fraction:
        return Fraction(comb(row1, x) * comb(row2, col1 - x), comb(n, col1))

    observed = prob(a)
    lo, hi = max(0, col1 - row2), min(row1, col1)
    return float(sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed))
