"""Automated tests for the PawPal+ logic layer (pawpal_system.py)."""

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

# Let `import pawpal_system` work when pytest is run from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pawpal_system import Owner, Pet, Scheduler, Task  # noqa: E402


def make_task(title="Walk", start_time="", duration=30, priority="medium",
              frequency="daily"):
    return Task(
        title=title,
        category="general",
        duration_minutes=duration,
        priority=priority,
        frequency=frequency,
        start_time=start_time,
    )


@pytest.fixture
def owner():
    """An owner with two pets and no tasks yet."""
    o = Owner(name="Sam", available_minutes_per_day=120)
    o.add_pet(Pet(name="Biscuit", species="dog", age=3))
    o.add_pet(Pet(name="Mochi", species="cat", age=5))
    return o


# --- Sorting correctness ---------------------------------------------------

def test_sort_by_time_returns_chronological_order(owner):
    biscuit, mochi = owner.pets
    biscuit.add_task(make_task("Evening walk", "18:00"))
    biscuit.add_task(make_task("Breakfast", "07:30"))
    mochi.add_task(make_task("Lunch", "12:15"))

    result = Scheduler(owner).sort_by_time()

    assert [t.start_time for t in result] == ["07:30", "12:15", "18:00"]


def test_sort_by_time_puts_unscheduled_tasks_last(owner):
    pet = owner.pets[0]
    pet.add_task(make_task("Anytime", ""))
    pet.add_task(make_task("Morning", "08:00"))

    result = Scheduler(owner).sort_by_time()

    assert [t.title for t in result] == ["Morning", "Anytime"]


def test_sort_by_time_does_not_modify_input_list(owner):
    tasks = [make_task("B", "10:00"), make_task("A", "09:00")]

    Scheduler(owner).sort_by_time(tasks)

    assert [t.title for t in tasks] == ["B", "A"]


def test_sort_by_time_with_no_tasks_returns_empty_list(owner):
    assert Scheduler(owner).sort_by_time() == []


# --- Recurrence logic ------------------------------------------------------

def test_completing_daily_task_creates_task_for_next_day(owner):
    pet = owner.pets[0]
    task = make_task("Feed", "08:00", frequency="daily")
    pet.add_task(task)

    next_task = pet.complete_task(task)

    assert task.completed is True
    assert next_task is not None
    assert next_task.completed is False
    assert next_task.due_date == date.today() + timedelta(days=1)
    assert next_task in pet.tasks
    assert next_task.pet_name == "Biscuit"
    assert next_task.title == "Feed"


def test_completing_weekly_task_is_due_in_seven_days(owner):
    pet = owner.pets[0]
    task = make_task("Bath", frequency="weekly")
    pet.add_task(task)

    next_task = pet.complete_task(task)

    assert next_task.due_date == date.today() + timedelta(days=7)


def test_completing_once_task_creates_no_new_task(owner):
    pet = owner.pets[0]
    task = make_task("Vet visit", frequency="once")
    pet.add_task(task)

    next_task = pet.complete_task(task)

    assert next_task is None
    assert task.completed is True
    assert len(pet.tasks) == 1


# --- Conflict detection ----------------------------------------------------

def test_detect_conflicts_flags_duplicate_times_same_pet(owner):
    pet = owner.pets[0]
    pet.add_task(make_task("Walk", "09:00"))
    pet.add_task(make_task("Groom", "09:00"))

    warnings = Scheduler(owner).detect_conflicts()

    assert len(warnings) == 1
    assert "same pet" in warnings[0]
    assert "Walk" in warnings[0] and "Groom" in warnings[0]


def test_detect_conflicts_flags_duplicate_times_different_pets(owner):
    biscuit, mochi = owner.pets
    biscuit.add_task(make_task("Walk", "09:00"))
    mochi.add_task(make_task("Feed", "09:00"))

    warnings = Scheduler(owner).detect_conflicts()

    assert len(warnings) == 1
    assert "different pets" in warnings[0]


def test_detect_conflicts_back_to_back_tasks_do_not_conflict(owner):
    pet = owner.pets[0]
    pet.add_task(make_task("Walk", "09:00", duration=30))  # ends 09:30
    pet.add_task(make_task("Feed", "09:30", duration=15))

    assert Scheduler(owner).detect_conflicts() == []


def test_detect_conflicts_ignores_completed_tasks(owner):
    pet = owner.pets[0]
    done = make_task("Walk", "09:00")
    done.completed = True
    pet.add_task(done)
    pet.add_task(make_task("Groom", "09:00"))

    assert Scheduler(owner).detect_conflicts() == []


def test_detect_conflicts_with_no_tasks_returns_empty_list(owner):
    assert Scheduler(owner).detect_conflicts() == []


# --- generate_daily_plan ---------------------------------------------------

def titles(tasks):
    return [t.title for t in tasks]


def test_plan_orders_by_priority_high_first(owner):
    pet = owner.pets[0]
    pet.add_task(make_task("Low", duration=10, priority="low"))
    pet.add_task(make_task("High", duration=10, priority="high"))
    pet.add_task(make_task("Medium", duration=10, priority="medium"))

    plan = Scheduler(owner).generate_daily_plan()

    assert titles(plan) == ["High", "Medium", "Low"]


def test_plan_puts_shorter_tasks_first_within_a_priority(owner):
    pet = owner.pets[0]
    pet.add_task(make_task("Long", duration=40, priority="high"))
    pet.add_task(make_task("Short", duration=5, priority="high"))
    pet.add_task(make_task("Mid", duration=20, priority="high"))

    assert titles(Scheduler(owner).generate_daily_plan()) == ["Short", "Mid", "Long"]


def test_plan_never_exceeds_the_time_budget(owner):
    owner.available_minutes_per_day = 50
    pet = owner.pets[0]
    for n, minutes in enumerate([30, 25, 20, 15, 10], start=1):
        pet.add_task(make_task(f"T{n}", duration=minutes, priority="medium"))

    scheduler = Scheduler(owner)
    plan = scheduler.generate_daily_plan()

    assert sum(t.duration_minutes for t in plan) <= 50
    assert sorted(titles(plan + scheduler.skipped)) == ["T1", "T2", "T3", "T4", "T5"]


def test_plan_skips_a_task_that_does_not_fit_but_keeps_filling_with_smaller_ones(owner):
    owner.available_minutes_per_day = 30
    pet = owner.pets[0]
    pet.add_task(make_task("Big", duration=25, priority="high"))
    pet.add_task(make_task("Too long", duration=20, priority="medium"))  # 5 min left: skipped
    pet.add_task(make_task("Tiny", duration=5, priority="low"))  # still fits afterwards

    scheduler = Scheduler(owner)
    plan = scheduler.generate_daily_plan()

    assert titles(plan) == ["Big", "Tiny"]
    assert titles(scheduler.skipped) == ["Too long"]


def test_plan_allows_a_task_that_exactly_fills_the_remaining_time(owner):
    owner.available_minutes_per_day = 30
    owner.pets[0].add_task(make_task("A", duration=10, priority="high"))
    owner.pets[0].add_task(make_task("B", duration=20, priority="medium"))

    scheduler = Scheduler(owner)

    assert titles(scheduler.generate_daily_plan()) == ["A", "B"]
    assert scheduler.skipped == []


def test_plan_with_a_zero_minute_budget_skips_everything(owner):
    owner.available_minutes_per_day = 0
    owner.pets[0].add_task(make_task("Walk", duration=10))

    scheduler = Scheduler(owner)

    assert scheduler.generate_daily_plan() == []
    assert titles(scheduler.skipped) == ["Walk"]


def test_plan_with_no_tasks_is_empty(owner):
    scheduler = Scheduler(owner)

    assert scheduler.generate_daily_plan() == []
    assert scheduler.skipped == []


def test_plan_with_no_pets_is_empty():
    scheduler = Scheduler(Owner(name="Nobody", available_minutes_per_day=60))

    assert scheduler.generate_daily_plan() == []


def test_plan_excludes_completed_tasks(owner):
    done = make_task("Done", duration=10, priority="high")
    done.completed = True
    owner.pets[0].add_task(done)
    owner.pets[0].add_task(make_task("Todo", duration=10, priority="low"))

    scheduler = Scheduler(owner)

    assert titles(scheduler.generate_daily_plan()) == ["Todo"]
    assert scheduler.skipped == []


def test_plan_pools_tasks_from_every_pet(owner):
    biscuit, mochi = owner.pets
    biscuit.add_task(make_task("Walk", duration=20, priority="high"))
    mochi.add_task(make_task("Feed", duration=10, priority="high"))

    plan = Scheduler(owner).generate_daily_plan()

    assert {(t.pet_name, t.title) for t in plan} == {("Biscuit", "Walk"), ("Mochi", "Feed")}


def test_plan_is_rebuilt_not_accumulated_when_run_twice(owner):
    owner.available_minutes_per_day = 20
    owner.pets[0].add_task(make_task("A", duration=15, priority="high"))
    owner.pets[0].add_task(make_task("B", duration=15, priority="low"))
    scheduler = Scheduler(owner)

    scheduler.generate_daily_plan()
    second = scheduler.generate_daily_plan()

    assert titles(second) == ["A"] and titles(scheduler.plan) == ["A"]
    assert titles(scheduler.skipped) == ["B"]  # not ["B", "B"]


def test_plan_treats_an_unknown_priority_as_lowest(owner):
    pet = owner.pets[0]
    pet.add_task(make_task("Odd", duration=10, priority="urgent"))  # not a known level
    pet.add_task(make_task("Low", duration=10, priority="low"))

    assert titles(Scheduler(owner).generate_daily_plan()) == ["Low", "Odd"]


def test_plan_does_not_change_the_tasks_themselves(owner):
    task = make_task("Walk", "09:00", duration=10)
    owner.pets[0].add_task(task)

    Scheduler(owner).generate_daily_plan()

    assert (task.completed, task.start_time, task.duration_minutes) == (False, "09:00", 10)


def test_explain_plan_lists_used_minutes_order_and_skipped_tasks(owner):
    owner.available_minutes_per_day = 20
    owner.pets[0].add_task(make_task("Walk", duration=15, priority="high"))
    owner.pets[0].add_task(make_task("Groom", duration=30, priority="low"))
    scheduler = Scheduler(owner)
    scheduler.generate_daily_plan()

    text = scheduler.explain_plan()

    assert "15 of 20 minutes used" in text
    assert "1. Walk for Biscuit" in text
    assert "Skipped" in text and "Groom" in text


def test_explain_plan_cites_the_source_of_ai_suggestions(owner):
    task = make_task("Gentle walk", duration=10, priority="high")
    task.source = "dog_care.md: Senior dogs"
    owner.pets[0].add_task(task)
    scheduler = Scheduler(owner)
    scheduler.generate_daily_plan()

    assert "[source: dog_care.md: Senior dogs]" in scheduler.explain_plan()


# --- filter_tasks ----------------------------------------------------------

@pytest.fixture
def stocked_owner(owner):
    biscuit, mochi = owner.pets
    biscuit.add_task(make_task("Walk"))
    biscuit.add_task(make_task("Groom"))
    mochi.add_task(make_task("Feed"))
    biscuit.tasks[1].completed = True  # Groom is done
    return owner


def test_filter_with_no_arguments_returns_every_task(stocked_owner):
    assert titles(Scheduler(stocked_owner).filter_tasks()) == ["Walk", "Groom", "Feed"]


def test_filter_by_pet_name(stocked_owner):
    result = Scheduler(stocked_owner).filter_tasks(pet_name="Mochi")

    assert titles(result) == ["Feed"]


def test_filter_by_pending_and_by_completed(stocked_owner):
    scheduler = Scheduler(stocked_owner)

    assert titles(scheduler.filter_tasks(completed=False)) == ["Walk", "Feed"]
    assert titles(scheduler.filter_tasks(completed=True)) == ["Groom"]


def test_filter_by_pet_and_status_together(stocked_owner):
    scheduler = Scheduler(stocked_owner)

    assert titles(scheduler.filter_tasks(pet_name="Biscuit", completed=False)) == ["Walk"]
    assert scheduler.filter_tasks(pet_name="Mochi", completed=True) == []


def test_filter_by_unknown_pet_returns_empty_list(stocked_owner):
    assert Scheduler(stocked_owner).filter_tasks(pet_name="Nobody") == []


def test_filter_is_case_sensitive_on_pet_name(stocked_owner):
    # Documents current behavior: "mochi" does not match "Mochi".
    assert Scheduler(stocked_owner).filter_tasks(pet_name="mochi") == []


def test_filter_returns_the_same_task_objects_not_copies(stocked_owner):
    result = Scheduler(stocked_owner).filter_tasks(pet_name="Mochi")

    assert result[0] is stocked_owner.pets[1].tasks[0]


def test_filter_on_an_owner_with_no_tasks_returns_empty_list(owner):
    assert Scheduler(owner).filter_tasks() == []


# --- input validation in the logic layer -------------------------------------

@pytest.mark.parametrize("bad", ["25:99", "24:00", "12:60", "8:30", "0830", "ab:cd", "08:30pm"])
def test_task_rejects_an_invalid_start_time(bad):
    with pytest.raises(ValueError, match="start_time"):
        make_task("Walk", bad)


@pytest.mark.parametrize("good", ["", "00:00", "08:30", "23:59"])
def test_task_accepts_valid_or_empty_start_times(good):
    assert make_task("Walk", good).start_time == good


def test_task_rejects_a_negative_duration():
    with pytest.raises(ValueError, match="duration"):
        make_task("Walk", duration=-5)


def test_completing_a_recurring_task_still_produces_a_valid_task(owner):
    pet = owner.pets[0]
    task = make_task("Walk", "08:00", duration=20, frequency="daily")
    pet.add_task(task)

    next_task = pet.complete_task(task)

    assert next_task.start_time == "08:00" and next_task.completed is False


def test_zero_duration_tasks_are_allowed_and_never_conflict(owner):
    # Documents current behavior: a task that takes no time occupies no time.
    pet = owner.pets[0]
    pet.add_task(make_task("A", "08:00", duration=0))
    pet.add_task(make_task("B", "08:00", duration=0))

    assert Scheduler(owner).detect_conflicts() == []


def test_task_running_past_midnight_does_not_conflict_with_early_morning_tasks(owner):
    # Documents a known limit: times are assumed to be on the same day.
    pet = owner.pets[0]
    pet.add_task(make_task("Late", "23:30", duration=90))  # ends 01:00 next day
    pet.add_task(make_task("Early", "00:30", duration=30))

    assert Scheduler(owner).detect_conflicts() == []


# --- explain_schedule (plain-language explanation) ---------------------------

def planned(owner, *tasks):
    for pet, task in tasks:
        owner.pets[pet].add_task(task)
    scheduler = Scheduler(owner)
    scheduler.generate_daily_plan()
    return scheduler


def test_explain_schedule_lists_tasks_in_time_order_with_times(owner):
    scheduler = planned(
        owner,
        (0, make_task("Evening walk", "18:00", duration=30, priority="high")),
        (1, make_task("Breakfast", "07:30", duration=10, priority="high")),
    )

    text = scheduler.explain_schedule()

    assert text.index("07:30-07:40  Mochi: Breakfast") < text.index("18:00-18:30  Biscuit: Evening walk")


def test_explain_schedule_puts_unscheduled_tasks_after_timed_ones(owner):
    scheduler = planned(
        owner,
        (0, make_task("Anytime", "", duration=10)),
        (0, make_task("Morning", "08:00", duration=10)),
    )

    text = scheduler.explain_schedule()

    assert text.index("Morning") < text.index("any time  Biscuit: Anytime")


def test_explain_schedule_gives_a_reason_for_every_task(owner):
    essential = make_task("Meds", "08:00", duration=5, priority="high")
    essential.essential = True
    scheduler = planned(owner, (0, essential), (1, make_task("Play", "09:00", priority="low")))

    text = scheduler.explain_schedule()

    assert text.count("Why:") == 2
    assert "high priority; essential" in text
    assert "a task you added yourself" in text


def test_explain_schedule_numbers_each_cited_guide_once_and_quotes_it(owner):
    walk = make_task("Walk", "08:00", duration=10)
    feed = make_task("Feed", "09:00", duration=10)
    meds = make_task("Meds", "10:00", duration=10)
    walk.source = feed.source = "dog_care.md: Senior dogs"
    meds.source = "medication_basics.md: Medication"
    scheduler = planned(owner, (0, walk), (0, feed), (0, meds))

    text = scheduler.explain_schedule({
        "dog_care.md: Senior dogs": "Older dogs do better with gentle walks.",
        "medication_basics.md: Medication": "Follow the vet's instructions exactly.",
    })

    assert text.count("suggested by guide [1]") == 2  # walk and feed share one footnote
    assert text.count("suggested by guide [2]") == 1
    assert '[1] dog_care.md: Senior dogs: "Older dogs do better with gentle walks."' in text
    assert "[2] medication_basics.md: Medication" in text


def test_explain_schedule_still_names_a_guide_that_has_no_note(owner):
    task = make_task("Walk", "08:00", duration=10)
    task.source = "mystery.md: Unknown"
    scheduler = planned(owner, (0, task))

    text = scheduler.explain_schedule()  # no notes supplied

    assert "[1] mystery.md: Unknown" in text and '"' not in text.split("What the guides say:")[1]


def test_explain_schedule_lists_skipped_tasks_and_budget(owner):
    owner.available_minutes_per_day = 20
    scheduler = planned(
        owner,
        (0, make_task("Walk", "08:00", duration=15, priority="high")),
        (0, make_task("Groom", "09:00", duration=30, priority="low")),
    )

    text = scheduler.explain_schedule()

    assert "15 of 20 minutes used" in text
    assert "Left out because there was not enough time left" in text
    assert "Biscuit: Groom (30 min, low priority)" in text


def test_explain_schedule_with_an_empty_plan_still_reads_sensibly(owner):
    scheduler = Scheduler(owner)
    scheduler.generate_daily_plan()

    text = scheduler.explain_schedule()

    assert "0 of 120 minutes used" in text and "Why:" not in text


def test_explain_schedule_does_not_change_the_existing_explain_plan(owner):
    scheduler = planned(owner, (0, make_task("Walk", "08:00", duration=10)))

    assert "1. Walk for Biscuit" in scheduler.explain_plan()  # the original format is intact
