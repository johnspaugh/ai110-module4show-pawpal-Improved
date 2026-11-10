"""
Terminal demo of the AI planning agent.

    python agent_demo.py            # live model if ANTHROPIC_API_KEY is set, else demo mode
    PAWPAL_DEMO=1 python agent_demo.py   # force the keyless demo model

Builds a sample household, asks the agent for a plan, prints every step it
took, and shows the proposal. Like the app, it saves nothing until accepted;
here it just prints.
"""

from agent import PlanningAgent, confidence_label
from logging_setup import setup_logging
from pawpal_system import Owner, Pet, Task


def build_sample_owner() -> Owner:
    owner = Owner(name="Sally", available_minutes_per_day=90)
    owner.add_pet(Pet(name="Biscuit", species="dog", age=10))
    owner.add_pet(Pet(name="Mochi", species="cat", age=5))
    owner.pets[1].add_task(Task("Vet visit reminder", "health", 10, "medium",
                                frequency="once", start_time="12:00"))
    return owner


def print_result(result) -> None:
    print(f"\nModel mode: {result.mode}   attempts: {result.iterations}   "
          f"checks passed: {result.converged}   "
          f"confidence: {result.confidence:.2f} ({confidence_label(result.confidence)})")
    print("\nWhat the agent did:")
    for line in result.trace:
        print(f"  - {line}")

    print("\nProposed tasks:")
    for t in result.tasks:
        when = t.start_time or "--:--"
        print(f"  {when}  {t.pet}: {t.title} ({t.duration_minutes} min, {t.priority})  "
              f"[{t.source}]")

    if result.problems:
        print("\nUnresolved problems:")
        for p in result.problems:
            print(f"  ! {p}")
    print("\nWhy this plan?")
    print(result.explanation or "  (no plan)")


def main() -> None:
    setup_logging()
    owner = build_sample_owner()
    notes = "Biscuit takes arthritis medicine every morning. Keep walks gentle."
    print(f"Owner notes: {notes}")
    agent = PlanningAgent(owner)
    print_result(agent.run(notes))


if __name__ == "__main__":
    main()
