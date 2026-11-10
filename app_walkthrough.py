"""
Automated walkthrough of the Streamlit app for the scenarios in the README's
human-evaluation table.

    python app_walkthrough.py

It drives the real app.py headlessly (Streamlit's AppTest), in demo mode, the
same way a person would: set the minutes, press "Generate AI plan", and read
what the app shows. It prints what the app produced so the results can be
pasted into the README. This is an automated run of the app, not a human
review: it records what happened, and a person still has to judge it.
"""

import logging
import os
from pathlib import Path

os.environ["PAWPAL_DEMO"] = "1"  # always demo mode: no key needed, no cost

from streamlit.testing.v1 import AppTest  # noqa: E402

from pawpal_system import Owner, Pet  # noqa: E402

APP = str(Path(__file__).resolve().parent / "app.py")


def run_app(pets, minutes=120, notes="", species_override=None):
    """Open the app with the given owner, press Generate AI plan, return (app, proposal)."""
    owner = Owner(name="Jordan", available_minutes_per_day=minutes)
    for name, species, age in pets:
        owner.add_pet(Pet(name, species, age))
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["owner"] = owner
    at.run()
    if notes:
        next(t for t in at.text_area if t.label.startswith("Notes for the assistant")).set_value(notes)
    next(b for b in at.button if b.label == "Generate AI plan").click().run()
    return at, at.session_state["proposal"]


def show(title, at, proposal):
    print(f"\n===== {title} =====")
    print(f"confidence {proposal.confidence:.2f} | attempts {proposal.iterations} | "
          f"checks passed {proposal.converged} | fallback {proposal.used_fallback}")
    print("on-screen messages:")
    for kind, items in (("success", at.success), ("warning", at.warning), ("info", at.info),
                        ("error", at.error)):
        for item in items:
            print(f"  [{kind}] {item.value}")
    print("trace:")
    for line in proposal.trace:
        print(f"  - {line}")
    print("proposed tasks:")
    for t in proposal.tasks:
        print(f"  {t.start_time or '--:--'} {t.pet}: {t.title} ({t.duration_minutes} min, "
              f"{t.priority}{', essential' if t.essential else ''})")
    if proposal.skipped:
        print("would not fit: " + ", ".join(f"{t.pet_name}'s {t.title}" for t in proposal.skipped))


def main() -> None:
    logging.disable(logging.CRITICAL)
    two_pets = [("Biscuit", "dog", 10), ("Mochi", "cat", 5)]

    at, p = run_app(two_pets, notes="Biscuit takes arthritis medicine every morning.")
    show("A. Medication note (120 min)", at, p)
    meds = [(t.pet, t.title) for t in p.tasks if "medication" in t.title.lower()]
    print(f"medication tasks: {meds}")
    print(f"'dose' or 'mg' appears in any task or explanation: "
          f"{any(w in (p.explanation + ' '.join(t.title for t in p.tasks)).lower() for w in (' mg', 'dose'))}")
    print("--- explanation excerpt ---")
    print("\n".join(p.explanation.splitlines()[:6]))

    at, p = run_app(two_pets, minutes=30)
    show("B. 30-minute day, no notes", at, p)
    print("problems:", p.problems)
    print("plan (what fits):", [f"{t.pet_name}: {t.title}" for t in p.plan])

    at, p = run_app([("Bubbles", "other", 2)])
    show("C. Species 'other' (the only non-dog/cat choice the app offers)", at, p)


if __name__ == "__main__":
    main()
