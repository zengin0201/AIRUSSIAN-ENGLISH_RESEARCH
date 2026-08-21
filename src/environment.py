"""
Simulated personal-assistant environment.

The environment holds a small state (alarms, calendar, lists, music, notes).
The agent changes this state by calling tools. Success is decided by
comparing the final state against a list of checks. No LLM judge is used,
so the same task is scored the same way in every language.

All tool names and argument names are in English on purpose. This is how
real APIs work, and the mismatch between an English schema and a non-English
user request is one of the things we want to measure.
"""

from __future__ import annotations

import copy
import re
import unicodedata
from typing import Any, Dict, List, Tuple




def norm(value: Any) -> str:
    """Normalise a string for comparison (case, unicode form, spaces)."""
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = text.replace("\u0451", "\u0435")  # Russian ё -> е
    text = re.sub(r"[^\w\s:.-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------

def initial_state() -> Dict[str, Any]:
    return {
        "alarms": [],       # [{"time": "07:30", "label": "..."}]
        "calendar": [],     # [{"date": "2026-09-01", "time": "10:00", "title": "..."}]
        "lists": {},        # {"shopping": ["milk", ...]}
        "music": {"playing": None},
        "notes": [],        # ["..."]
    }


# --------------------------------------------------------------------------
# tool specifications (rendered into the system prompt)
# --------------------------------------------------------------------------

TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "set_alarm",
        "args": {"time": "string, 24-hour format HH:MM", "label": "string"},
        "desc": "Create a new alarm.",
    },
    {
        "name": "delete_alarm",
        "args": {"time": "string, 24-hour format HH:MM"},
        "desc": "Delete an existing alarm by its time.",
    },
    {
        "name": "add_event",
        "args": {
            "date": "string, YYYY-MM-DD",
            "time": "string, HH:MM",
            "title": "string",
        },
        "desc": "Add an event to the calendar.",
    },
    {
        "name": "add_to_list",
        "args": {"list_name": "string", "item": "string"},
        "desc": "Add one item to a named list. Creates the list if it does not exist.",
    },
    {
        "name": "remove_from_list",
        "args": {"list_name": "string", "item": "string"},
        "desc": "Remove one item from a named list.",
    },
    {
        "name": "play_music",
        "args": {"artist": "string"},
        "desc": "Start playing music by an artist.",
    },
    {
        "name": "stop_music",
        "args": {},
        "desc": "Stop the music.",
    },
    {
        "name": "create_note",
        "args": {"text": "string"},
        "desc": "Save a short text note.",
    },
    {
        "name": "read_state",
        "args": {},
        "desc": "Read the current state of alarms, calendar, lists, music and notes.",
    },
    {
        "name": "done",
        "args": {},
        "desc": "Call this when the whole request is finished.",
    },
]

TOOL_NAMES = {spec["name"] for spec in TOOL_SPECS}


def render_tools() -> str:
    lines = []
    for spec in TOOL_SPECS:
        if spec["args"]:
            args = ", ".join(f'"{k}": <{v}>' for k, v in spec["args"].items())
        else:
            args = ""
        lines.append(f'- {spec["name"]}({args}) : {spec["desc"]}')
    return "\n".join(lines)


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------

class ToolError(Exception):
    pass


class Environment:
    """Holds state and executes tool calls."""

    def __init__(self, state: Dict[str, Any] | None = None):
        self.state = copy.deepcopy(state) if state else initial_state()
        self.history: List[Dict[str, Any]] = []

    # -- individual tools ---------------------------------------------------

    def _set_alarm(self, time: str = "", label: str = "", **_) -> str:
        if not TIME_RE.match(str(time)):
            raise ToolError(
                f"invalid time '{time}'. Time must look like 07:30 (24-hour)."
            )
        self.state["alarms"].append({"time": str(time), "label": str(label)})
        return f"Alarm created at {time} with label '{label}'."

    def _delete_alarm(self, time: str = "", **_) -> str:
        before = len(self.state["alarms"])
        self.state["alarms"] = [
            a for a in self.state["alarms"] if norm(a["time"]) != norm(time)
        ]
        if len(self.state["alarms"]) == before:
            raise ToolError(f"no alarm found at {time}.")
        return f"Alarm at {time} deleted."

    def _add_event(self, date: str = "", time: str = "", title: str = "", **_) -> str:
        if not DATE_RE.match(str(date)):
            raise ToolError(f"invalid date '{date}'. Date must look like 2026-09-01.")
        if not TIME_RE.match(str(time)):
            raise ToolError(f"invalid time '{time}'. Time must look like 10:00.")
        if not str(title).strip():
            raise ToolError("title is empty.")
        self.state["calendar"].append(
            {"date": str(date), "time": str(time), "title": str(title)}
        )
        return f"Event '{title}' added on {date} at {time}."

    def _add_to_list(self, list_name: str = "", item: str = "", **_) -> str:
        if not str(list_name).strip() or not str(item).strip():
            raise ToolError("list_name and item must both be non-empty.")
        key = norm(list_name)
        self.state["lists"].setdefault(key, []).append(str(item))
        return f"Added '{item}' to list '{list_name}'."

    def _remove_from_list(self, list_name: str = "", item: str = "", **_) -> str:
        key = norm(list_name)
        items = self.state["lists"].get(key)
        if items is None:
            raise ToolError(f"list '{list_name}' does not exist.")
        for i, existing in enumerate(items):
            if norm(existing) == norm(item):
                items.pop(i)
                return f"Removed '{item}' from list '{list_name}'."
        raise ToolError(f"'{item}' is not in list '{list_name}'.")

    def _play_music(self, artist: str = "", **_) -> str:
        if not str(artist).strip():
            raise ToolError("artist is empty.")
        self.state["music"]["playing"] = str(artist)
        return f"Now playing music by {artist}."

    def _stop_music(self, **_) -> str:
        self.state["music"]["playing"] = None
        return "Music stopped."

    def _create_note(self, text: str = "", **_) -> str:
        if not str(text).strip():
            raise ToolError("text is empty.")
        self.state["notes"].append(str(text))
        return "Note saved."

    def _read_state(self, **_) -> str:
        alarms = ", ".join(f'{a["time"]} ({a["label"]})' for a in self.state["alarms"])
        events = ", ".join(
            f'{e["date"]} {e["time"]} {e["title"]}' for e in self.state["calendar"]
        )
        lists = "; ".join(f"{k}: {v}" for k, v in self.state["lists"].items())
        return (
            f"alarms: [{alarms}] | calendar: [{events}] | lists: [{lists}] | "
            f'music: {self.state["music"]["playing"]} | notes: {len(self.state["notes"])}'
        )

    def _done(self, **_) -> str:
        return "Task marked as finished."

    # -- dispatch -----------------------------------------------------------

    def call(self, tool: str, args: Dict[str, Any]) -> Tuple[str, str]:
        """
        Execute a tool call.

        Returns (observation, status) where status is one of:
        "ok", "unknown_tool", "bad_args".
        """
        handler = getattr(self, f"_{tool}", None)
        if tool not in TOOL_NAMES or handler is None:
            obs = (
                f"Error: unknown tool '{tool}'. "
                f"Available tools: {', '.join(sorted(TOOL_NAMES))}."
            )
            self.history.append({"tool": tool, "args": args, "status": "unknown_tool"})
            return obs, "unknown_tool"

        if not isinstance(args, dict):
            self.history.append({"tool": tool, "args": args, "status": "bad_args"})
            return "Error: args must be a JSON object.", "bad_args"

        try:
            obs = handler(**{str(k): v for k, v in args.items()})
        except ToolError as exc:
            self.history.append({"tool": tool, "args": args, "status": "bad_args"})
            return f"Error: {exc}", "bad_args"
        except TypeError as exc:
            self.history.append({"tool": tool, "args": args, "status": "bad_args"})
            return f"Error: wrong arguments for {tool}: {exc}", "bad_args"

        self.history.append({"tool": tool, "args": args, "status": "ok"})
        return obs, "ok"


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------

def _check_one(state: Dict[str, Any], check: Dict[str, Any]) -> Tuple[bool, str]:
    kind = check["kind"]

    if kind == "alarm_exists":
        for alarm in state["alarms"]:
            if norm(alarm["time"]) == norm(check["time"]):
                if "label_contains" in check:
                    if norm(check["label_contains"]) in norm(alarm["label"]):
                        return True, ""
                    return False, "wrong_value"
                return True, ""
        return False, ("empty" if not state["alarms"] else "wrong_value")

    if kind == "alarm_absent":
        for alarm in state["alarms"]:
            if norm(alarm["time"]) == norm(check["time"]):
                return False, "wrong_value"
        return True, ""

    if kind == "event_exists":
        for event in state["calendar"]:
            same_date = norm(event["date"]) == norm(check["date"])
            same_time = norm(event["time"]) == norm(check["time"])
            has_title = norm(check["title_contains"]) in norm(event["title"])
            if same_date and same_time and has_title:
                return True, ""
        return False, ("empty" if not state["calendar"] else "wrong_value")

    if kind == "list_contains":
        items = state["lists"].get(norm(check["list_name"]), [])
        for item in items:
            if norm(check["item"]) in norm(item):
                return True, ""
        return False, ("empty" if not items else "wrong_value")

    if kind == "list_not_contains":
        items = state["lists"].get(norm(check["list_name"]), [])
        for item in items:
            if norm(check["item"]) in norm(item):
                return False, "wrong_value"
        return True, ""

    if kind == "list_size":
        items = state["lists"].get(norm(check["list_name"]), [])
        return (len(items) == int(check["size"])), "wrong_value"

    if kind == "music_playing":
        playing = state["music"]["playing"]
        if playing is None:
            return False, "empty"
        return (norm(check["artist"]) in norm(playing)), "wrong_value"

    if kind == "music_stopped":
        return (state["music"]["playing"] is None), "wrong_value"

    if kind == "note_contains":
        for note in state["notes"]:
            if norm(check["text"]) in norm(note):
                return True, ""
        return False, ("empty" if not state["notes"] else "wrong_value")

    raise ValueError(f"unknown check kind: {kind}")


def verify(state: Dict[str, Any], checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run all checks against the final state.

    Returns a dict with:
      success        : True only if every check passed
      passed         : number of checks passed
      total          : number of checks
      failure_kind   : "none" | "empty" (agent did nothing in that slot)
                       | "wrong_value" (agent acted but wrote the wrong value)
    """
    passed = 0
    reasons = []
    for check in checks:
        ok, reason = _check_one(state, check)
        if ok:
            passed += 1
        else:
            reasons.append(reason)

    if not reasons:
        failure_kind = "none"
    elif all(r == "empty" for r in reasons):
        failure_kind = "empty"
    elif any(r == "wrong_value" for r in reasons):
        failure_kind = "wrong_value"
    else:
        failure_kind = "mixed"

    return {
        "success": passed == len(checks),
        "passed": passed,
        "total": len(checks),
        "failure_kind": failure_kind,
    }
