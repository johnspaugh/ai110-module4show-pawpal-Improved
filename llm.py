"""
The model call for PawPal+.

Two interchangeable backends share one method, complete(system, user) -> text:

* AnthropicLLM  - the live model, used when ANTHROPIC_API_KEY is set.
* DemoLLM       - a rule-based stand-in that needs no key. It reads the same
                  prompt a real model would see, so what it proposes still
                  depends on the guidance that was retrieved.

This file also owns the prompt format (SYSTEM_PROMPT, build_user_prompt), so
the format and the demo parser that reads it stay in one place.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional

from rag import TASK_LINE

logger = logging.getLogger("pawpal.llm")

# Override with the PAWPAL_MODEL environment variable (for example a cheaper
# model while developing).
DEFAULT_MODEL = "claude-opus-5-5"
MAX_TOKENS = 8000

SYSTEM_PROMPT = """You are the planning step of PawPal+, a pet care planner.

You are given the owner's daily time budget, optional owner notes, and for each
pet some retrieved care guidance. Propose care tasks for each pet.

Rules:
- Build tasks ONLY from the guidance provided. Copy durations, priorities,
  frequencies and the essential flag from the guidance line you used. You may
  shorten a duration or move a start time to fit the day, but do not invent
  tasks the guidance does not support.
- Every task must set "source" to the exact label inside "(source: ...)" of the
  guidance it came from.
- If a pet has no guidance, propose nothing for that pet.
- Do not repeat a task the pet already has (see EXISTING TASKS).
- Never choose or change a medication dose. Medication start times are
  placeholders for the owner to replace with the vet's schedule.
- Give each task a start_time as "HH:MM" (24-hour). The owner does one thing at
  a time, so tasks for all pets must not overlap each other or the existing
  timed tasks, and the total minutes should fit the owner's budget.
- If you receive FEEDBACK about a previous answer, fix every problem it lists.

Reply with ONLY one JSON object, no other text, in this shape:
{"tasks": [{"pet": "<pet name>", "title": "<task title>", "category": "<category>",
"duration_minutes": <int>, "priority": "low|medium|high",
"frequency": "daily|weekly|monthly|once", "start_time": "HH:MM",
"essential": <true|false>, "source": "<exact guidance label>"}],
"notes": "<one short sentence about your choices>"}"""


class LLMError(Exception):
    """The model call failed or was declined; the caller should fall back."""


def build_user_prompt(
    minutes_available: int,
    minutes_committed: int,
    notes: str,
    pet_blocks: List[Dict],
    feedback: Optional[List[str]] = None,
    previous_reply: str = "",
) -> str:
    """Build the user message: budget, notes, per-pet guidance and optional checker feedback.

    Each item of pet_blocks has: name, species, age, existing (list of str),
    and guidance (list of (label, text) pairs; empty means none was found).
    """
    lines = [
        f"OWNER MINUTES AVAILABLE PER DAY: {minutes_available}",
        f"MINUTES ALREADY COMMITTED TO EXISTING TASKS: {minutes_committed}",
        f"OWNER NOTES: {notes.strip() or '(none)'}",
    ]
    for block in pet_blocks:
        lines.append("")
        lines.append(f"PET: {block['name']} | species: {block['species']} | age: {block['age']}")
        lines.append("EXISTING TASKS: " + (", ".join(block["existing"]) or "none"))
        if not block["guidance"]:
            lines.append("NO GUIDANCE FOUND for this pet. Do not suggest tasks for it.")
        for label, text in block["guidance"]:
            lines.append(f"GUIDANCE (source: {label}):")
            lines.append(text)
    if feedback:
        lines.append("")
        lines.append("FEEDBACK FROM THE CHECKER ON YOUR PREVIOUS ANSWER (fix every problem):")
        lines.extend(f"- {item}" for item in feedback)
        lines.append("PREVIOUS ANSWER:")
        lines.append(previous_reply)
    return "\n".join(lines)


class AnthropicLLM:
    """Live model backend using the official Anthropic SDK."""

    mode = "live"

    def __init__(self, model: Optional[str] = None) -> None:
        import anthropic  # imported here so demo mode works without the package

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY; never log it
        self.model = model or os.getenv("PAWPAL_MODEL", DEFAULT_MODEL)

    def complete(self, system: str, user: str) -> str:
        """Send one request and return the model's text; raise LLMError on any failure."""
        logger.info("llm call model=%s prompt_chars=%d", self.model, len(user))
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": "low"},
            )
        except self._anthropic.APIError as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

        logger.info(
            "llm reply request_id=%s stop=%s tokens_in=%s tokens_out=%s",
            getattr(response, "_request_id", "?"), response.stop_reason,
            response.usage.input_tokens, response.usage.output_tokens,
        )
        if response.stop_reason == "refusal":
            raise LLMError("the model declined this request")
        text = "".join(b.text for b in response.content if b.type == "text")
        if not text.strip():
            raise LLMError("the model returned no text")
        return text


# ---------------------------------------------------------------------------
# Demo backend
# ---------------------------------------------------------------------------

_PET_LINE = re.compile(r"^PET: (?P<name>.+?) \| species: (?P<species>\w+) \| age: (?P<age>\d+)$")
_SOURCE_LINE = re.compile(r"^GUIDANCE \(source: (?P<label>.+)\):$")
_BUSY = re.compile(r"\((\d{2}:\d{2})-(\d{2}:\d{2})\)")
_PRIORITY_RANK = {"low": 1, "medium": 2, "high": 3}


def _mins(hhmm: str) -> int:
    return int(hhmm[:2]) * 60 + int(hhmm[3:])


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


class DemoLLM:
    """Rule-based stand-in for the model, used when no API key is set.

    It parses the same prompt a real model would get and turns the guidance
    task lines into a proposal, so output changes with what was retrieved.
    If the prompt contains checker feedback it trims non-essential tasks
    (SKIPPED_ESSENTIAL) and staggers start times (CONFLICT) - a simple version
    of what the live model is asked to do.
    """

    mode = "demo"

    def complete(self, system: str, user: str) -> str:
        """Return a JSON proposal built from the guidance text inside the prompt."""
        budget = self._number(user, "OWNER MINUTES AVAILABLE PER DAY") - self._number(
            user, "MINUTES ALREADY COMMITTED TO EXISTING TASKS"
        )
        main, _, feedback = user.partition("FEEDBACK FROM THE CHECKER")
        tasks, busy = self._tasks_from_guidance(main)

        if feedback.strip():
            # A real model sees its previous answer, so earlier fixes persist. Keep
            # staggering on every retry so a fix is not undone by the next one.
            if "SKIPPED_ESSENTIAL" in feedback:
                tasks = self._trim_to_budget(tasks, budget)
            tasks = self._stagger(tasks, busy)

        notes = "Demo mode: rule-based proposal built from the retrieved guidance."
        return json.dumps({"tasks": tasks, "notes": notes})

    @staticmethod
    def _number(text: str, label: str) -> int:
        m = re.search(rf"{label}: (\d+)", text)
        return int(m.group(1)) if m else 0

    @staticmethod
    def _tasks_from_guidance(prompt: str):
        tasks: List[Dict] = []
        busy: List[tuple] = []
        pet = None
        label = ""
        seen = set()
        existing = set()
        for raw in prompt.splitlines():
            line = raw.strip()
            m = _PET_LINE.match(line)
            if m:
                pet, label = m["name"], ""
                seen = set()
                existing = set()
                continue
            if line.startswith("EXISTING TASKS:"):
                for part in line[len("EXISTING TASKS:"):].split(", "):
                    title = _BUSY.sub("", part).strip().lower()
                    if title and title != "none":
                        existing.add(title)
                    window = _BUSY.search(part)
                    if window:
                        busy.append((_mins(window[1]), _mins(window[2])))
                continue
            m = _SOURCE_LINE.match(line)
            if m:
                label = m["label"]
                continue
            m = TASK_LINE.match(line)
            if m and pet and label:
                title = m["title"].strip()
                if title.lower() in seen or title.lower() in existing:
                    continue
                seen.add(title.lower())
                tasks.append({
                    "pet": pet,
                    "title": title,
                    "category": (m["category"] or "other").lower(),
                    "duration_minutes": int(m["minutes"]),
                    "priority": m["priority"].lower(),
                    "frequency": m["frequency"].lower(),
                    "start_time": m["start"],
                    "essential": m["essential"].lower() == "yes",
                    "source": label,
                })
        return tasks, busy

    @staticmethod
    def _trim_to_budget(tasks: List[Dict], budget: int) -> List[Dict]:
        """Drop non-essential tasks (lowest priority, longest first) until the total fits."""
        kept = list(tasks)
        droppable = sorted(
            (t for t in kept if not t["essential"]),
            key=lambda t: (_PRIORITY_RANK[t["priority"]], -t["duration_minutes"]),
        )
        for task in droppable:
            if sum(t["duration_minutes"] for t in kept) <= budget:
                break
            kept.remove(task)
        return kept

    @staticmethod
    def _stagger(tasks: List[Dict], busy: List[tuple]) -> List[Dict]:
        """Chain tasks one after another, starting no earlier than each task's own time."""
        ordered = sorted(tasks, key=lambda t: (t["start_time"], -_PRIORITY_RANK[t["priority"]]))
        cursor = 0
        result = []
        for task in ordered:
            start = max(_mins(task["start_time"]), cursor)
            moved = True
            while moved:  # step past any existing timed task that is in the way
                moved = False
                for b_start, b_end in busy:
                    if start < b_end and b_start < start + task["duration_minutes"]:
                        start = b_end
                        moved = True
            end = start + task["duration_minutes"]
            if end > 23 * 60 + 59:
                continue  # would run past midnight; leave it out
            result.append({**task, "start_time": _hhmm(start)})
            cursor = end
        return result


def get_llm(force_demo: Optional[bool] = None):
    """Return the live model if a key is configured, else the demo backend.

    Set PAWPAL_DEMO=1 (or pass force_demo=True) to use the demo backend even
    when a key is present.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if force_demo is None:
        force_demo = os.getenv("PAWPAL_DEMO", "") == "1"
    has_key = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
    if force_demo or not has_key:
        logger.info("using demo model (%s)", "forced" if force_demo else "no API key")
        return DemoLLM()
    try:
        return AnthropicLLM()
    except Exception as exc:  # missing package or bad configuration
        logger.warning("live model unavailable (%s: %s); using demo model", type(exc).__name__, exc)
        return DemoLLM()
