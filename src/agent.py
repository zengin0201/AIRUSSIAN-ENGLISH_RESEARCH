from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .backends import BaseBackend, Generation
from .environment import Environment, render_tools, verify

ARMS = ["en", "ru", "ru_sys_ru", "ru_translate", "ru_translate_safe",
        "en_cot", "ru_cot",
        "pl", "pl_cot",
        "de", "de_cot",
        "en_fewshot", "ru_fewshot", "pl_fewshot", "de_fewshot"]
ARM_LANGUAGE = {
    "en": "en",
    "en_cot": "en",
    "ru": "ru",
    "ru_cot": "ru",
    "ru_sys_ru": "ru",
    "ru_translate": "ru",
    "ru_translate_safe": "ru",
    "pl": "pl",
    "pl_cot": "pl",
    "de": "de",
    "de_cot": "de",
    "en_fewshot": "en",
    "ru_fewshot": "ru",
    "pl_fewshot": "pl",
    "de_fewshot": "de",
}


SYSTEM_EN = """You are a personal assistant agent. Today is {today} ({weekday}).

You cannot talk to the user. You can only act by calling tools.

Available tools:
{tools}

Answer with JSON only. No explanation, no markdown, no code fences.
For one action:
{{"tool": "<tool_name>", "args": {{"<arg>": "<value>"}}}}
For several actions at once, use a JSON array:
[{{"tool": "...", "args": {{...}}}}, {{"tool": "...", "args": {{...}}}}]

Rules:
- Time must be 24-hour "HH:MM". Date must be "YYYY-MM-DD".
- Keep the user's own words for labels, titles, notes and list items. Do not
  translate them and do not transliterate them.
- Tool names and argument names are always English.
- When the whole request is complete, call done."""


FEWSHOT_TIMES = ("09:05", "19:10", "08:25")

FEWSHOT = {
    "en": """

Examples:
User: Set an alarm for five past nine in the morning.
You: {{"tool": "set_alarm", "args": {{"time": "09:05", "label": "alarm"}}}}
User: Set an alarm for ten past seven in the evening.
You: {{"tool": "set_alarm", "args": {{"time": "19:10", "label": "alarm"}}}}
User: Set an alarm for twenty five past eight in the morning.
You: {{"tool": "set_alarm", "args": {{"time": "08:25", "label": "alarm"}}}}""",

    "ru": """

Examples:
User: Поставь будильник на пять минут десятого утра.
You: {{"tool": "set_alarm", "args": {{"time": "09:05", "label": "будильник"}}}}
User: Поставь будильник на десять минут восьмого вечера.
You: {{"tool": "set_alarm", "args": {{"time": "19:10", "label": "будильник"}}}}
User: Поставь будильник на двадцать пять минут девятого утра.
You: {{"tool": "set_alarm", "args": {{"time": "08:25", "label": "будильник"}}}}""",

    "pl": """

Examples:
User: Ustaw budzik na pięć po dziewiątej rano.
You: {{"tool": "set_alarm", "args": {{"time": "09:05", "label": "budzik"}}}}
User: Ustaw budzik na dziesięć po siódmej wieczorem.
You: {{"tool": "set_alarm", "args": {{"time": "19:10", "label": "budzik"}}}}
User: Ustaw budzik na dwadzieścia pięć po ósmej rano.
You: {{"tool": "set_alarm", "args": {{"time": "08:25", "label": "budzik"}}}}""",

    "de": """

Examples:
User: Stelle den Wecker auf fünf nach neun morgens.
You: {{"tool": "set_alarm", "args": {{"time": "09:05", "label": "Wecker"}}}}
User: Stelle den Wecker auf zehn nach sieben abends.
You: {{"tool": "set_alarm", "args": {{"time": "19:10", "label": "Wecker"}}}}
User: Stelle den Wecker auf fünfundzwanzig nach acht morgens.
You: {{"tool": "set_alarm", "args": {{"time": "08:25", "label": "Wecker"}}}}""",
}


SYSTEM_COT = """You are a personal assistant agent. Today is {today} ({weekday}).

You cannot talk to the user. You can only act by calling tools.

Available tools:
{tools}

First think step by step. Work out what the user means, and if the request
mentions a time or a date, state explicitly how you converted it before you
act. Put that reasoning between <think> and </think>.

After </think>, answer with JSON only. No markdown, no code fences.
For one action:
{{"tool": "<tool_name>", "args": {{"<arg>": "<value>"}}}}
For several actions at once, use a JSON array:
[{{"tool": "...", "args": {{...}}}}, {{"tool": "...", "args": {{...}}}}]

Rules:
- Time must be 24-hour "HH:MM". Date must be "YYYY-MM-DD".
- Keep the user's own words for labels, titles, notes and list items. Do not
  translate them and do not transliterate them.
- Tool names and argument names are always English.
- When the whole request is complete, call done."""


SYSTEM_RU = """Ты агент-персональный ассистент. Сегодня {today} ({weekday}).

Ты не можешь разговаривать с пользователем. Ты можешь действовать только
через вызовы инструментов.

Доступные инструменты:
{tools}

Отвечай только в формате JSON. Без объяснений, без markdown, без блоков кода.
Для одного действия:
{{"tool": "<имя_инструмента>", "args": {{"<аргумент>": "<значение>"}}}}
Для нескольких действий сразу используй JSON-массив:
[{{"tool": "...", "args": {{...}}}}, {{"tool": "...", "args": {{...}}}}]

Правила:
- Время в 24-часовом формате "HH:MM". Дата в формате "YYYY-MM-DD".
- Сохраняй собственные слова пользователя для названий, заголовков, заметок и
  элементов списка. Не переводи их и не транслитерируй.
- Имена инструментов и имена аргументов всегда на английском.
- Когда весь запрос выполнен, вызови done."""


TRANSLATE_PROMPT = """Translate the user request below from Russian into English.

Very important: any name, title, label, note text or list item must stay in
the original Russian, exactly as written. Translate only the instruction
around them.

The request is between the markers. Return the English translation only, with
no comment, no markers and no repeated header.

<<<REQUEST
{instruction}
REQUEST>>>"""


TRANSLATE_SAFE_PROMPT = """Translate the user request below from Russian into English.

The request contains placeholders that look like __V1__, __V2__ and so on.
Copy every placeholder into your translation exactly as it is. Never change
them, never translate them, never remove them.

The request is between the markers. Return the English translation only, with
no comment, no markers and no repeated header.

<<<REQUEST
{instruction}
REQUEST>>>"""


QUOTED_RE = re.compile(r"«([^»]+)»|\"([^\"]+)\"|'([^']+)'")

_ECHO_RE = re.compile(
    r"^\s*(?:<<<REQUEST|REQUEST>>>|Request:|Translation:|English:|"
    r"Translated request:)\s*", re.IGNORECASE
)


def strip_translation_echo(text: str) -> str:
    cleaned = text.strip()
    for _ in range(4):
        new = _ECHO_RE.sub("", cleaned).strip()
        if new == cleaned:
            break
        cleaned = new
    cleaned = cleaned.replace("<<<REQUEST", " ").replace("REQUEST>>>", " ")
    return re.sub(r"[ 	]+", " ", cleaned).strip()


def protect_values(text: str) -> Tuple[str, Dict[str, str]]:
    mapping: Dict[str, str] = {}
    counter = [0]

    def _swap(match: "re.Match") -> str:
        value = next(g for g in match.groups() if g is not None)
        counter[0] += 1
        key = f"__V{counter[0]}__"
        mapping[key] = value
        return f"«{key}»"

    return QUOTED_RE.sub(_swap, text), mapping


_MANGLED_RE = re.compile(r"[^\s«»\"']?__V(\d+)__[^\s«»\"'.,;:!?]+")


def restore_values(text: str, mapping: Dict[str, str]) -> Tuple[str, int, int]:
    mangled = len(set(_MANGLED_RE.findall(text)))
    lost = 0
    for key, value in mapping.items():
        if key in text:
            text = text.replace(key, value)
        else:
            lost += 1
    return text, lost, mangled


FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
THINK_OPEN_RE = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)


def split_reasoning(text: str) -> Tuple[str, str]:
    match = THINK_RE.search(text)
    if match:
        return match.group(1).strip(), THINK_OPEN_RE.sub("", text, count=1).strip()
    if "</think>" in text.lower():
        cut = THINK_OPEN_RE.sub("", text, count=1).strip()
        return text[: len(text) - len(cut)].strip(), cut
    return "", text


def _as_action(obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    tool = obj.get("tool") or obj.get("name") or obj.get("tool_name")
    if not isinstance(tool, str) or not tool.strip():
        return None
    args = obj.get("args") or obj.get("arguments") or obj.get("parameters") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return {"tool": tool.strip(), "args": args}


def _flatten(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        actions = []
        for item in value:
            action = _as_action(item)
            if action:
                actions.append(action)
        return actions
    action = _as_action(value)
    return [action] if action else []


def _scan_json_blocks(text: str) -> Tuple[List[Any], bool]:
    blocks: List[Any] = []
    saw_brackets = False
    depth = 0
    start = -1
    opener = ""
    pairs = {"{": "}", "[": "]"}

    for i, ch in enumerate(text):
        if depth == 0 and ch in pairs:
            depth = 1
            start = i
            opener = ch
            saw_brackets = True
        elif depth > 0:
            if ch == opener:
                depth += 1
            elif ch == pairs[opener]:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        blocks.append(json.loads(text[start : i + 1]))
                    except json.JSONDecodeError:
                        pass
                    start = -1
    return blocks, saw_brackets


def extract_actions(text: str) -> Tuple[List[Dict[str, Any]], str]:
    raw = text.strip()

    try:
        parsed = json.loads(raw)
        actions = _flatten(parsed)
        return (actions, "clean") if actions else ([], "bad_schema")
    except json.JSONDecodeError:
        pass

    collected: List[Dict[str, Any]] = []
    parsed_something = False
    saw_brackets = False

    for fence in FENCE_RE.findall(raw):
        values, brackets = _scan_json_blocks(fence)
        saw_brackets = saw_brackets or brackets
        for value in values:
            parsed_something = True
            collected.extend(_flatten(value))
    if collected:
        return collected, "recovered"

    values, brackets = _scan_json_blocks(raw)
    saw_brackets = saw_brackets or brackets
    for value in values:
        parsed_something = True
        collected.extend(_flatten(value))
    if collected:
        return collected, "recovered"

    if parsed_something:
        return [], "bad_schema"
    return [], ("malformed_json" if saw_brackets else "no_json")


def compact(
    messages: List[Dict[str, str]],
    backend: BaseBackend,
    budget: Optional[int],
) -> Tuple[List[Dict[str, str]], int]:
    if budget is None or len(messages) <= 3:
        return messages, 0

    dropped = 0
    while len(messages) > 3:
        total = sum(backend.count_tokens(m["content"]) for m in messages)
        if total <= budget:
            break
        messages = messages[:2] + messages[3:]
        dropped += 1
    return messages, dropped


def run_episode(
    backend: BaseBackend,
    task: Dict[str, Any],
    arm: str,
    today: str,
    weekday: str,
    max_steps: int = 8,
    context_budget: Optional[int] = None,
    stall_limit: int = 3,
    repeat_limit: int = 2,
    truncation_hint: bool = True,
) -> Dict[str, Any]:

    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")

    check_lang = ARM_LANGUAGE[arm]
    if check_lang not in task["instruction"]:
        raise ValueError(
            f"arm '{arm}' needs a '{check_lang}' wording, and task "
            f"{task.get('id')} does not have one"
        )
    env = Environment(task.get("initial_state"))

    if arm.endswith("_cot"):
        template = SYSTEM_COT
    elif arm == "ru_sys_ru":
        template = SYSTEM_RU
    else:
        template = SYSTEM_EN
    if arm.endswith("_fewshot"):
        template = template + FEWSHOT[check_lang]
    system_text = template.format(
        today=today, weekday=weekday, tools=render_tools()
    )

    instruction = task["instruction"][check_lang]

    translation_prompt_tokens = 0
    translation_completion_tokens = 0
    translation_text = None
    translation_truncated = False
    placeholders_total = 0
    placeholders_lost = 0
    placeholders_mangled = 0

    if arm in ("ru_translate", "ru_translate_safe"):
        if arm == "ru_translate_safe":
            protected, mapping = protect_values(instruction)
            placeholders_total = len(mapping)
            prompt = TRANSLATE_SAFE_PROMPT.format(instruction=protected)
        else:
            mapping = {}
            prompt = TRANSLATE_PROMPT.format(instruction=instruction)

        gen: Generation = backend.generate([{"role": "user", "content": prompt}])
        translated = strip_translation_echo(gen.text)
        translation_prompt_tokens = gen.prompt_tokens
        translation_completion_tokens = gen.completion_tokens
        translation_truncated = gen.truncated

        if arm == "ru_translate_safe" and translated:
            translated, placeholders_lost, placeholders_mangled = restore_values(
                translated, mapping
            )

        translation_text = translated
        instruction = translated or instruction

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": instruction},
    ]

    turns: List[Dict[str, Any]] = []
    total_prompt = translation_prompt_tokens
    total_completion = translation_completion_tokens
    total_latency = 0.0
    compactions = 0
    total_actions = 0
    finished = False
    truncated_turns = 0
    truncated_lost_turns = 0
    stopped_reason = "max_steps"
    last_signature = json.dumps(env.state, sort_keys=True, ensure_ascii=False)
    stalled_turns = 0
    seen_actions: set = set()
    repeating_turns = 0

    for step in range(max_steps):
        messages, dropped = compact(messages, backend, context_budget)
        compactions += dropped

        gen = backend.generate(messages)
        total_prompt += gen.prompt_tokens
        total_completion += gen.completion_tokens
        total_latency += gen.latency_s

        reasoning, answer_text = split_reasoning(gen.text)
        actions, parse_status = extract_actions(answer_text)

        if gen.truncated:
            truncated_turns += 1
            if not actions:
                parse_status = "truncated"
                truncated_lost_turns += 1
            elif parse_status == "recovered":
                parse_status = "truncated_partial"

        observations: List[str] = []
        statuses: List[str] = []
        executed: List[Dict[str, Any]] = []

        if not actions:
            if gen.truncated and truncation_hint:
                observations.append(
                    "Error: your answer was cut off because it was too long, "
                    "so nothing was executed. Send at most two actions in one "
                    "answer, then wait for the result."
                )
            else:
                observations.append(
                    "Error: your answer contained no valid tool call. "
                    'Answer only with JSON, like {"tool": "set_alarm", '
                    '"args": {"time": "07:30", "label": "x"}}'
                )
            statuses.append("parse_error")
        else:
            for action in actions:
                tool_name = action["tool"]
                tool_args = action["args"]
                if tool_name == "done":
                    finished = True
                    observations.append("Task marked as finished.")
                    statuses.append("ok")
                    executed.append(action)
                    break
                obs, status = env.call(tool_name, tool_args)
                observations.append(f"{tool_name}: {obs}")
                statuses.append(status)
                executed.append(action)
            total_actions += len(executed)

        worst = "ok"
        for status in statuses:
            if status != "ok":
                worst = status
                break

        turns.append(
            {
                "step": step,
                "prompt_tokens": gen.prompt_tokens,
                "completion_tokens": gen.completion_tokens,
                "latency_s": round(gen.latency_s, 3),
                "parse_status": parse_status,
                "finish_reason": gen.finish_reason,
                "truncated": gen.truncated,
                "reasoning": reasoning[:600],
                "n_actions": len(executed),
                "batched": len(executed) > 1,
                "tool": executed[0]["tool"] if executed else None,
                "args": executed[0]["args"] if executed else {},
                "tools": [a["tool"] for a in executed],
                "tool_status": worst,
                "tool_statuses": statuses,
                "raw": gen.text[:400],
            }
        )

        messages.append({"role": "assistant", "content": gen.text})
        messages.append(
            {"role": "user", "content": "Observation: " + " | ".join(observations)}
        )

        if finished:
            stopped_reason = "done"
            break

        if executed:
            keys = [
                json.dumps([a["tool"], a["args"]], sort_keys=True, ensure_ascii=False)
                for a in executed
            ]
            if all(k in seen_actions for k in keys):
                repeating_turns += 1
            else:
                repeating_turns = 0
            seen_actions.update(keys)

            if repeating_turns >= repeat_limit:
                stopped_reason = "looping"
                break

        signature = json.dumps(env.state, sort_keys=True, ensure_ascii=False)
        if signature == last_signature:
            stalled_turns += 1
        else:
            stalled_turns = 0
            last_signature = signature

        if stalled_turns >= stall_limit:
            stopped_reason = "stalled"
            break

    result = verify(env.state, task["checks"][check_lang])

    parse_errors = sum(
        1 for t in turns
        if t["parse_status"] in ("no_json", "bad_schema", "malformed_json")
    )
    recovered = sum(
        1 for t in turns
        if t["parse_status"] in ("recovered", "truncated_partial")
    )
    batched_turns = sum(1 for t in turns if t["batched"])
    tool_errors = sum(
        1 for t in turns for s in t["tool_statuses"] if s in ("unknown_tool", "bad_args")
    )
    wasted = sum(1 for t in turns if t["tool_status"] != "ok")

    return {
        "task_id": task["id"],
        "difficulty": task.get("difficulty", "unknown"),
        "n_steps_gold": task.get("n_steps", 0),
        "arm": arm,
        "check_lang": check_lang,
        "success": result["success"],
        "checks_passed": result["passed"],
        "checks_total": result["total"],
        "failure_kind": result["failure_kind"],
        "called_done": finished,
        "stopped_reason": stopped_reason,
        "steps_used": len(turns),
        "actions_used": total_actions,
        "batched_turns": batched_turns,
        "parse_errors": parse_errors,
        "recovered_parses": recovered,
        "truncated_turns": truncated_turns,
        "truncated_lost_turns": truncated_lost_turns,
        "truncation_hint": truncation_hint,
        "tool_errors": tool_errors,
        "wasted_steps": wasted,
        "compactions": compactions,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "translation_tokens": translation_prompt_tokens + translation_completion_tokens,
        "placeholders_total": placeholders_total,
        "placeholders_lost": placeholders_lost,
        "placeholders_mangled": placeholders_mangled,
        "translation_truncated": translation_truncated,
        "latency_s": round(total_latency, 3),
        "instruction_used": instruction,
        "translation_text": translation_text,
        "turns": turns,
        "final_state": env.state,
    }
