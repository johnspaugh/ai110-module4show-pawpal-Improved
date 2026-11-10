"""
Reliability evaluation for the PawPal+ AI layer.

    python evaluate.py

Prints text tables (markdown) that can be pasted into the README:

  1. Retrieval accuracy on a labeled set: for each pet, which guide sections
     SHOULD come back, compared with what does.
  2. Agent scenarios: each one has a criterion and a Pass/Fail result.
  3. Consistency: the same request run several times gives the same proposal.

Everything here runs without an API key (the demo model and scripted fake
models), so the results are reproducible. It does not measure the live model.
"""

import json
import logging
from typing import Callable, Dict, List, Set, Tuple

from agent import PlanningAgent, commit_proposal, confidence_label
from llm import DemoLLM, LLMError
from pawpal_system import Owner, Pet, Task
from rag import Retriever, build_query, notes_for_pet

SENIOR_DOG = "dog_care.md: Senior dogs"
ADULT_DOG = "dog_care.md: Adult dogs"
PUPPIES = "dog_care.md: Puppies"
ADULT_CAT = "cat_care.md: Adult cats"
SENIOR_CAT = "cat_care.md: Senior cats"
KITTENS = "cat_care.md: Kittens"
SENIOR_ANY = "senior_pets.md: Senior pets"
MEDICATION = "medication_basics.md: Medication"
OTHER = "other_pets.md: Small and other pets"

# (description, species, age, notes, labels that should be returned, no more, no fewer)
RETRIEVAL_CASES: List[Tuple[str, str, int, str, Set[str]]] = [
    ("senior dog (age 10)", "dog", 10, "", {SENIOR_DOG, SENIOR_ANY}),
    ("adult dog (age 3)", "dog", 3, "", {ADULT_DOG}),
    ("puppy (age 0)", "dog", 0, "", {PUPPIES}),
    ("adult cat (age 5)", "cat", 5, "", {ADULT_CAT}),
    ("senior cat (age 12)", "cat", 12, "", {SENIOR_CAT, SENIOR_ANY}),
    ("kitten (age 0)", "cat", 0, "", {KITTENS}),
    ("small pet (rabbit-type)", "other", 2, "", {OTHER}),
    ("adult dog + 'flea pills'", "dog", 3, "flea pills", {ADULT_DOG, MEDICATION}),
    ("senior dog + 'arthritis medicine'", "dog", 10, "arthritis medicine",
     {SENIOR_DOG, SENIOR_ANY, MEDICATION}),
    ("adult dog + common words only", "dog", 3, "every morning walks", {ADULT_DOG}),
    ("species not covered (fish)", "fish", 2, "", set()),
    ("species typed 'Dog' (capital)", "Dog", 3, "", {ADULT_DOG}),
    ("adult cat + 'insulin shots'", "cat", 5, "insulin shots", {ADULT_CAT}),
]


class FakeLLM:
    """Scripted stand-in model for scenarios: returns canned replies in order."""

    mode = "test"

    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


def _task(pet: str, title: str, start: str, source: str = SENIOR_DOG, minutes: int = 10) -> Dict:
    return {"pet": pet, "title": title, "category": "care", "duration_minutes": minutes,
            "priority": "high", "frequency": "daily", "start_time": start,
            "essential": True, "source": source}


def _owner(minutes: int = 120, pets=(("Biscuit", "dog", 10), ("Mochi", "cat", 5))) -> Owner:
    owner = Owner(name="Sally", available_minutes_per_day=minutes)
    for name, species, age in pets:
        owner.add_pet(Pet(name=name, species=species, age=age))
    return owner


def retrieval_eval(retriever: Retriever) -> List[Dict]:
    """Run the labeled retrieval cases; return one result row per case."""
    rows = []
    for description, species, age, notes, expected in RETRIEVAL_CASES:
        pet_notes = notes_for_pet(notes, "Pet", ["Pet"])
        got = {c.label for c, _ in retriever.search(build_query(species, age), species, pet_notes)}
        wrong_species = [
            c.label for c in retriever.chunks
            if c.label in got and c.species not in ("any", species.strip().lower())
        ]
        rows.append({
            "input": description,
            "expected": expected,
            "got": got,
            "exact": got == expected,
            "wrong_species": wrong_species,
        })
    return rows


def agent_eval(retriever: Retriever) -> List[Dict]:
    """Run the agent through scenarios; each row has input, criterion, passed, detail."""
    rows: List[Dict] = []

    def case(inp: str, criterion: str, check: Callable[[], Tuple[bool, str]]) -> None:
        try:
            passed, detail = check()
        except Exception as exc:  # a crash is a failed case, not a crashed evaluation
            passed, detail = False, f"crashed: {type(exc).__name__}: {exc}"
        rows.append({"input": inp, "criterion": criterion, "passed": bool(passed), "detail": detail})

    def demo_run():
        owner = _owner()
        r = PlanningAgent(owner, DemoLLM(), retriever).run(
            "Biscuit takes arthritis medicine every morning.")
        retrieved = {label for hits in r.guidance.values() for label, _ in hits}
        ok = (r.converged and r.iterations <= 2 and r.tasks
              and all(t.source in retrieved for t in r.tasks))
        return bool(ok), (f"{len(r.tasks)} tasks, {r.iterations} attempt(s), "
                          f"confidence {r.confidence:.2f}")

    case("2 pets + medicine note (demo model)",
         "converges in <= 2 attempts; every task cites a retrieved guide", demo_run)

    def overlap_then_fix():
        llm = FakeLLM({"tasks": [_task("Biscuit", "Feeding", "08:00"),
                                 _task("Mochi", "Feeding", "08:00", ADULT_CAT)]},
                      {"tasks": [_task("Biscuit", "Feeding", "08:00"),
                                 _task("Mochi", "Feeding", "08:10", ADULT_CAT)]})
        r = PlanningAgent(_owner(), llm, retriever).run()
        return r.converged and r.iterations == 2, f"{r.iterations} attempt(s)"

    case("model overlaps two tasks, then fixes it",
         "checker catches overlap, retry converges", overlap_then_fix)

    def never_fixes():
        llm = FakeLLM({"tasks": [_task("Biscuit", "Feeding", "08:00"),
                                 _task("Mochi", "Feeding", "08:00", ADULT_CAT)]})
        r = PlanningAgent(_owner(), llm, retriever, max_iterations=3).run()
        return (not r.converged and r.iterations == 3 and llm.calls == 3 and r.problems,
                f"stopped after {llm.calls} calls, {len(r.problems)} problem(s) reported")

    case("model never fixes the overlap", "stops at the iteration limit and reports it", never_fixes)

    def garbage_then_valid():
        llm = FakeLLM("Sure! Here is a plan.",
                      {"tasks": [_task("Biscuit", "Feeding", "08:00")]})
        r = PlanningAgent(_owner(pets=(("Biscuit", "dog", 10),)), llm, retriever).run()
        return r.converged and llm.calls == 2, f"{llm.calls} calls"

    case("model replies with prose, not JSON", "rejected, retried, recovers", garbage_then_valid)

    def uncited():
        llm = FakeLLM({"tasks": [_task("Biscuit", "Feeding", "08:00", "vet_blog.md: Tips")]},
                      {"tasks": [_task("Biscuit", "Feeding", "08:00")]})
        r = PlanningAgent(_owner(pets=(("Biscuit", "dog", 10),)), llm, retriever).run()
        return r.converged and r.iterations == 2, f"{r.iterations} attempt(s)"

    case("model cites a source that was never retrieved",
         "first reply rejected as ungrounded", uncited)

    def model_down():
        owner = _owner(pets=(("Biscuit", "dog", 10),))
        owner.pets[0].add_task(Task("Morning walk", "exercise", 20, "high"))
        r = PlanningAgent(owner, FakeLLM(LLMError("boom")), retriever).run()
        return (r.used_fallback and [t.title for t in r.plan] == ["Morning walk"]
                and r.confidence == 0.0), "fallback plan used, confidence 0.00"

    case("model call raises an error", "falls back to normal scheduling, no crash", model_down)

    def no_guidance():
        owner = _owner(pets=(("Bubbles", "fish", 2),))
        llm = FakeLLM(RuntimeError("must not be called"))
        r = PlanningAgent(owner, llm, retriever).run()
        return llm.calls == 0 and not r.tasks, f"model calls: {llm.calls}"

    case("pet with no guidance (fish)", "model is never called, nothing invented", no_guidance)

    def empty_notes():
        r = PlanningAgent(_owner(), DemoLLM(), retriever).run("   ")
        return r.converged, f"{len(r.tasks)} tasks"

    case("whitespace-only notes", "handled like no notes", empty_notes)

    def duplicate_names():
        owner = _owner(pets=(("Rex", "dog", 3), ("Rex", "cat", 3)))
        r = PlanningAgent(owner, DemoLLM(), retriever).run()
        return (not r.tasks and "same name" in r.problems[0]), r.problems[0][:60]

    case("two pets with the same name",
         "refused with a clear message (was: silently proposed 0 tasks)", duplicate_names)

    def sandbox():
        owner = _owner()
        PlanningAgent(owner, DemoLLM(), retriever).run()
        return owner.get_all_tasks() == [], "owner has 0 tasks after run()"

    case("run() before the owner accepts", "real owner is not changed", sandbox)

    def double_accept():
        owner = _owner(pets=(("Biscuit", "dog", 10),))
        r = PlanningAgent(owner, DemoLLM(), retriever).run()
        commit_proposal(owner, r)
        first = len(owner.get_all_tasks())
        commit_proposal(owner, r)
        return first > 0 and len(owner.get_all_tasks()) == first, f"{first} tasks after 2 accepts"

    case("owner accepts the same proposal twice",
         "tasks added once (was: duplicated)", double_accept)

    def past_midnight():
        end = Task("x", "c", 90, "high", start_time="23:30").end_time
        return end == "01:00", f"23:30 + 90 min -> {end}"

    case("task from 23:30 for 90 minutes", "end time wraps to 01:00 (was: 25:00)", past_midnight)

    return rows


def consistency_eval(retriever: Retriever, runs: int = 10) -> Tuple[bool, int]:
    """Run the same demo request several times; return (all identical, number of runs)."""
    seen = set()
    for _ in range(runs):
        r = PlanningAgent(_owner(), DemoLLM(), retriever).run("Biscuit takes arthritis medicine.")
        seen.add(tuple((t.pet, t.title, t.start_time, t.duration_minutes) for t in r.tasks))
    return len(seen) == 1, runs


def main() -> None:
    logging.disable(logging.CRITICAL)  # keep the tables readable; logs are in pawpal.log otherwise
    retriever = Retriever()

    rrows = retrieval_eval(retriever)
    print("### Retrieval accuracy (labeled set)\n")
    print("| Test Input | Expected guides | Returned | Result |")
    print("|---|---|---|---|")
    for r in rrows:
        fmt = lambda s: ", ".join(sorted(s)) or "(none)"  # noqa: E731
        result = "Pass" if r["exact"] and not r["wrong_species"] else "Fail"
        print(f"| {r['input']} | {fmt(r['expected'])} | {fmt(r['got'])} | {result} |")
    exact = sum(r["exact"] for r in rrows)
    print(f"\nRetrieval: {exact} of {len(rrows)} exact matches; "
          f"{sum(len(r['wrong_species']) for r in rrows)} wrong-species results.\n")

    arows = agent_eval(retriever)
    print("### Agent scenarios\n")
    print("| Test Input | Evaluation Criteria | Result | Detail |")
    print("|---|---|---|---|")
    for r in arows:
        print(f"| {r['input']} | {r['criterion']} | {'Pass' if r['passed'] else 'Fail'} | "
              f"{r['detail']} |")
    passed = sum(r["passed"] for r in arows)
    print(f"\nAgent scenarios: {passed} of {len(arows)} passed.\n")

    same, runs = consistency_eval(retriever)
    print(f"Consistency: {runs} identical requests produced "
          f"{'the same proposal every time' if same else 'different proposals'} "
          "(demo model; the live model is not deterministic).")


if __name__ == "__main__":
    main()
