"""
PawPal+ planning agent: plan, retrieve, act, check, retry.

For the owner's pets the agent
  1. PLAN      decides what to look up (one query per pet),
  2. RETRIEVE  fetches care guidance (rag.py),
  3. ACT       asks the model for tasks, validates the reply, and adds the tasks
               to a SANDBOX COPY of the owner (the real owner is untouched),
  4. CHECK     runs the Scheduler on the sandbox and tests its own result:
               no essential task skipped for lack of time, no overlaps among
               planned tasks,
  5. RETRY     sends any problems back to the model, up to max_iterations.

Nothing is saved until the human accepts the proposal (commit_proposal).
If the model call fails, the agent falls back to scheduling the owner's
existing tasks the normal way.
"""

import copy
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from llm import SYSTEM_PROMPT, build_user_prompt, get_llm
from pawpal_system import PRIORITY_ORDER, RECURRENCE_DAYS, Owner, Scheduler, Task
from rag import Retriever, build_query, notes_for_pet

logger = logging.getLogger("pawpal.agent")

MAX_ITERATIONS = 3
MAX_TASKS = 30
MAX_NOTES_CHARS = 1000  # owner notes go into the prompt; cap them to limit cost and prompt stuffing
VALID_FREQUENCIES = set(RECURRENCE_DAYS) | {"once"}
HHMM = re.compile(r"([01]\d|2[0-3]):[0-5]\d")


@dataclass
class ProposedTask:
    """A task the model suggested, after validation."""

    pet: str
    title: str
    category: str
    duration_minutes: int
    priority: str
    frequency: str
    start_time: str
    essential: bool
    source: str


@dataclass
class AgentResult:
    """Everything the UI needs to show: the proposal, the plan, and how the agent got there."""

    mode: str = ""  # "live" or "demo"
    tasks: List[ProposedTask] = field(default_factory=list)
    plan: List[Task] = field(default_factory=list)
    skipped: List[Task] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)  # unresolved checker problems
    guidance: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    no_guidance: List[str] = field(default_factory=list)  # pets with nothing retrieved
    iterations: int = 0
    converged: bool = False  # True when the checker found no problems
    committed: bool = False  # True once the owner accepted and the tasks were added
    confidence: float = 0.0  # 0-1 heuristic from observable signals; see _confidence
    used_fallback: bool = False
    explanation: str = ""
    model_notes: str = ""
    trace: List[str] = field(default_factory=list)


def confidence_label(score: float) -> str:
    """Words for a confidence score: high (0.8 and up), medium (0.5 and up), else low."""
    return "high" if score >= 0.8 else "medium" if score >= 0.5 else "low"


def parse_reply(text: str) -> dict:
    """Pull the JSON object out of a model reply; raise ValueError if there isn't one."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("reply did not contain a JSON object")
    try:
        data = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"reply was not valid JSON ({exc.msg})") from exc
    if not isinstance(data, dict):
        raise ValueError("reply JSON must be an object")
    return data


def validate_proposal(
    data: dict,
    pet_names: Set[str],
    allowed_sources: Set[str],
    allowed_titles: Optional[Dict[str, Set[str]]] = None,
) -> Tuple[List[ProposedTask], List[str]]:
    """Check a parsed reply against the rules; return (tasks, errors).

    All-or-nothing: if any task is invalid, no tasks are returned, so a bad
    reply can never change the owner. When allowed_titles is given (guidance
    label -> lowercase task titles in that guide), a task's title must be one
    of the tasks in the guide it cites, so the model cannot cite a real guide
    and still write its own instruction as the title.
    """
    items = data.get("tasks")
    if not isinstance(items, list):
        return [], ["'tasks' must be a list"]
    if len(items) > MAX_TASKS:
        return [], [f"too many tasks ({len(items)}); the limit is {MAX_TASKS}"]

    errors: List[str] = []
    tasks: List[ProposedTask] = []
    seen: Set[Tuple[str, str]] = set()
    for n, item in enumerate(items, start=1):
        where = f"task {n}"
        if not isinstance(item, dict):
            errors.append(f"{where}: must be an object")
            continue
        pet = item.get("pet")
        title = item.get("title")
        minutes = item.get("duration_minutes")
        priority = item.get("priority")
        frequency = item.get("frequency")
        start = item.get("start_time", "")
        essential = item.get("essential", False)
        source = item.get("source")

        if pet not in pet_names:
            errors.append(f"{where}: unknown pet {pet!r}")
        if not isinstance(title, str) or not title.strip() or len(title) > 80:
            errors.append(f"{where}: title must be 1-80 characters")
        if isinstance(minutes, bool) or not isinstance(minutes, int) or not 1 <= minutes <= 240:
            errors.append(f"{where}: duration_minutes must be a whole number from 1 to 240")
        if priority not in PRIORITY_ORDER:
            errors.append(f"{where}: priority must be low, medium or high")
        if frequency not in VALID_FREQUENCIES:
            errors.append(f"{where}: frequency must be one of {sorted(VALID_FREQUENCIES)}")
        if not isinstance(start, str) or (start and not HHMM.fullmatch(start)):
            errors.append(f"{where}: start_time must look like 08:30 (24-hour HH:MM)")
        if not isinstance(essential, bool):
            errors.append(f"{where}: essential must be true or false")
        if source not in allowed_sources:
            errors.append(f"{where}: source {source!r} is not one of the retrieved guidance labels")
        elif (allowed_titles is not None and isinstance(title, str)
              and title.strip().lower() not in allowed_titles.get(source, set())):
            errors.append(
                f"{where}: title {title!r} is not a task in the guidance {source!r}; "
                f"use the exact title of one of its tasks"
            )
        if errors and errors[-1].startswith(where):
            continue
        key = (pet, title.strip().lower())
        if key in seen:
            errors.append(f"{where}: duplicate task {title!r} for {pet}")
            continue
        seen.add(key)
        tasks.append(ProposedTask(
            pet=pet, title=title.strip(), category=str(item.get("category", "other")),
            duration_minutes=minutes, priority=priority, frequency=frequency,
            start_time=start, essential=essential, source=source,
        ))
    return ([], errors) if errors else (tasks, [])


def commit_proposal(owner: Owner, result: AgentResult) -> List[Task]:
    """Add the accepted proposal to the real owner and return the new Task objects.

    Safe to call twice: a proposal that was already accepted adds nothing the second time.
    """
    if result.committed:
        logger.warning("proposal already accepted; ignoring a second commit")
        return []
    added: List[Task] = []
    for pt in result.tasks:
        pet = next((p for p in owner.pets if p.name == pt.pet), None)
        if pet is None:
            logger.warning("commit skipped %r: pet %r no longer exists", pt.title, pt.pet)
            continue
        task = Task(
            title=pt.title, category=pt.category, duration_minutes=pt.duration_minutes,
            priority=pt.priority, frequency=pt.frequency, start_time=pt.start_time,
            source=pt.source, essential=pt.essential,
        )
        pet.add_task(task)
        added.append(task)
    result.committed = True
    logger.info("human accepted proposal: %d task(s) added", len(added))
    return added


class PlanningAgent:
    """Plan, retrieve, act, check and retry over an owner's pets."""

    def __init__(
        self,
        owner: Owner,
        llm=None,
        retriever: Optional[Retriever] = None,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        self.owner = owner
        self.llm = llm if llm is not None else get_llm()
        self.retriever = retriever if retriever is not None else Retriever()
        self.max_iterations = max_iterations

    # -- tools: thin, logged wrappers over code that already exists ------------

    def _tool(self, name: str, fn: Callable, *args, **kwargs):
        # Truncate: prompts are long and can contain the owner's notes.
        shown = [str(a)[:60] for a in args if isinstance(a, (str, int))]
        logger.info("tool=%s args=%s", name, shown)
        return fn(*args, **kwargs)

    # -- the loop ---------------------------------------------------------------

    def run(self, notes: str = "") -> AgentResult:
        """Run the agent for the owner's pets and return a proposal (nothing is saved)."""
        result = self._run(notes)
        result.confidence = self._confidence(result)
        logger.info("agent finished: attempts=%d converged=%s fallback=%s confidence=%.2f (%s)",
                    result.iterations, result.converged, result.used_fallback,
                    result.confidence, confidence_label(result.confidence))
        return result

    @staticmethod
    def _confidence(result: AgentResult) -> float:
        """Heuristic 0-1 confidence built only from things the agent can observe.

        It is NOT the model rating itself. Start at 1.0, then subtract 0.15 for each
        retry that was needed and 0.2 for each pet with no retrieved guidance. If the
        checks still fail at the end, the score is capped at 0.4 (low), because an
        answer that failed its own checks should never read as medium or high. No
        suggestions, or a fallback, scores 0.
        """
        if result.used_fallback or not result.tasks:
            return 0.0
        score = 1.0 - 0.15 * (result.iterations - 1) - 0.2 * len(result.no_guidance)
        if not result.converged:
            score = min(score, 0.4)
        return round(max(score, 0.0), 2)

    def _run(self, notes: str) -> AgentResult:
        result = AgentResult(mode=getattr(self.llm, "mode", "unknown"))

        def step(message: str) -> None:
            result.trace.append(message)
            logger.info(message)

        if len(notes) > MAX_NOTES_CHARS:
            step(f"GUARD: owner notes were {len(notes)} characters; only the first "
                 f"{MAX_NOTES_CHARS} are used")
            notes = notes[:MAX_NOTES_CHARS]

        owner = self.owner
        if not owner.pets:
            result.problems = ["Add at least one pet first."]
            step("PLAN: owner has no pets; nothing to do")
            return result

        names = [p.name.strip().lower() for p in owner.pets]
        if len(set(names)) != len(names):
            result.problems = ["Two pets share the same name. Rename one so tasks can be "
                               "assigned to the right pet."]
            step("STOP: duplicate pet names; tasks could not be assigned unambiguously")
            return result

        step(f"PLAN: {len(owner.pets)} pet(s), {owner.available_minutes_per_day} min/day, "
             f"model mode={result.mode}")

        # RETRIEVE -----------------------------------------------------------
        pet_blocks: List[Dict] = []
        allowed_sources: Set[str] = set()
        allowed_titles: Dict[str, Set[str]] = {}
        for pet in owner.pets:
            query = build_query(pet.species, pet.age)
            pet_notes = notes_for_pet(notes, pet.name, [p.name for p in owner.pets])
            hits = self._tool("retrieve_guidance", self.retriever.search, query,
                              species=pet.species, notes=pet_notes)
            result.guidance[pet.name] = [(c.label, score) for c, score in hits]
            allowed_sources.update(c.label for c, _ in hits)
            for c, _ in hits:
                allowed_titles[c.label] = {t["title"].lower() for t in c.task_suggestions()}
            if hits:
                step(f"RETRIEVE: {pet.name} ({pet.species}, {pet.age}) -> "
                     + ", ".join(c.label for c, _ in hits))
            else:
                result.no_guidance.append(pet.name)
                step(f"RETRIEVE: no guidance found for {pet.name} ({pet.species}); "
                     "no tasks will be suggested for this pet")
            pet_blocks.append({
                "name": pet.name, "species": pet.species, "age": pet.age,
                "existing": [
                    f"{t.title} ({t.start_time}-{t.end_time})" if t.start_time else t.title
                    for t in pet.tasks if not t.completed
                ],
                "guidance": [(c.label, c.text) for c, _ in hits],
            })

        if len(result.no_guidance) == len(owner.pets):
            result.problems = ["No guidance was found for any of your pets, so nothing was suggested."]
            step("STOP: nothing retrieved, so the model was not called")
            return result

        committed = sum(t.duration_minutes for t in owner.get_all_tasks() if not t.completed)
        feedback: List[str] = []
        previous_reply = ""

        for attempt in range(1, self.max_iterations + 1):
            result.iterations = attempt
            prompt = build_user_prompt(
                owner.available_minutes_per_day, committed, notes, pet_blocks,
                feedback or None, previous_reply,
            )

            # ACT 1: ask the model ----------------------------------------------
            try:
                reply = self._tool("llm.complete", self.llm.complete, SYSTEM_PROMPT, prompt)
            except Exception as exc:  # any model failure must fall back, not crash the app
                step(f"ERROR: model call failed ({type(exc).__name__}: {exc}); using fallback")
                return self._fallback(result, f"The AI step failed ({type(exc).__name__}).")

            try:
                data = parse_reply(reply)
                tasks, errors = validate_proposal(
                    data, {p.name for p in owner.pets}, allowed_sources, allowed_titles
                )
            except ValueError as exc:
                tasks, errors, data = [], [str(exc)], {}
            previous_reply = reply

            if errors:
                step(f"VALIDATE (attempt {attempt}): rejected, {len(errors)} error(s): "
                     + "; ".join(errors[:3]))
                feedback = [f"INVALID: {e}" for e in errors]
                result.tasks, result.problems = [], errors
                continue
            step(f"VALIDATE (attempt {attempt}): accepted {len(tasks)} task(s)")
            result.model_notes = str(data.get("notes", ""))

            # ACT 2: add to a sandbox copy and schedule ---------------------------
            sandbox = copy.deepcopy(owner)
            added_ids: Set[int] = set()
            for pt in tasks:
                pet = next(p for p in sandbox.pets if p.name == pt.pet)
                task = Task(
                    title=pt.title, category=pt.category, duration_minutes=pt.duration_minutes,
                    priority=pt.priority, frequency=pt.frequency, start_time=pt.start_time,
                    source=pt.source, essential=pt.essential,
                )
                self._tool("add_task", pet.add_task, task)
                added_ids.add(id(task))
            scheduler = Scheduler(sandbox)
            plan = self._tool("generate_daily_plan", scheduler.generate_daily_plan)

            # CHECK -------------------------------------------------------------
            problems = self._check(scheduler, sandbox)
            result.tasks = tasks
            result.plan, result.skipped = plan, list(scheduler.skipped)
            result.problems = problems
            result.explanation = scheduler.explain_schedule(self._source_notes())
            if not problems:
                result.converged = True
                step(f"CHECK (attempt {attempt}): passed (time budget, essential tasks, no overlaps)")
                break
            step(f"CHECK (attempt {attempt}): {len(problems)} problem(s): "
                 + "; ".join(p.split(": ", 1)[1] for p in problems[:3]))
            feedback = problems
            if attempt < self.max_iterations:
                step("RETRY: sending the problems back to the model")

        if not result.converged:
            step(f"STOP: reached {result.iterations} attempt(s) with unresolved problems; "
                 "showing the last proposal for the owner to review")
        return result

    def _source_notes(self) -> Dict[str, str]:
        """Map each guide label to its first sentence, for the plain-language explanation."""
        notes = {}
        for chunk in self.retriever.chunks:
            sentence = re.split(r"(?<=[.!?])\s", chunk.prose.strip(), maxsplit=1)[0]
            notes[chunk.label] = sentence.strip()
        return notes

    # -- checks -----------------------------------------------------------------

    def _check(self, scheduler: Scheduler, sandbox: Owner) -> List[str]:
        """Test the agent's own result; return one string per problem (empty means it passed)."""
        problems: List[str] = []
        used = sum(t.duration_minutes for t in scheduler.plan)
        budget = sandbox.available_minutes_per_day
        if used > budget:  # generate_daily_plan prevents this; kept as a safety net
            problems.append(f"OVER_BUDGET: plan uses {used} of {budget} minutes")
        skipped_essential = [t for t in scheduler.skipped if t.essential]
        for task in skipped_essential:
            problems.append(
                f"SKIPPED_ESSENTIAL: {task.pet_name}'s {task.title} ({task.duration_minutes} min) "
                f"did not fit in the {budget}-minute day"
            )
        if skipped_essential:
            # Say how much time is really needed, but only when the day is simply too short
            # for the essential tasks. Otherwise they were crowded out by other tasks.
            needed = sum(t.duration_minutes for t in sandbox.get_all_tasks()
                         if t.essential and not t.completed)
            if needed > budget:
                problems.append(
                    f"NEEDS_MORE_TIME: The essential tasks alone need {needed} minutes but the day "
                    f"has {budget}. Raise the daily minutes to at least {needed}, or remove an "
                    f"essential task."
                )
        planned = {id(t) for t in scheduler.plan}
        for a, b in scheduler.conflicting_pairs():
            if id(a) in planned and id(b) in planned:
                problems.append(
                    f"CONFLICT: {a.pet_name}'s {a.title} ({a.start_time}-{a.end_time}) overlaps "
                    f"{b.pet_name}'s {b.title} ({b.start_time}-{b.end_time})"
                )
        return problems

    def _fallback(self, result: AgentResult, reason: str) -> AgentResult:
        """Schedule the owner's existing tasks the normal (non-AI) way."""
        scheduler = Scheduler(self.owner)
        result.used_fallback = True
        result.tasks = []
        result.plan = scheduler.generate_daily_plan()
        result.skipped = list(scheduler.skipped)
        result.explanation = scheduler.explain_plan()
        result.problems = [f"{reason} Showing your existing tasks scheduled normally instead."]
        result.trace.append("FALLBACK: scheduled existing tasks without AI suggestions")
        logger.info("fallback used: %s", reason)
        return result
