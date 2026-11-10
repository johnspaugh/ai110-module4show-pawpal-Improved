"""Reliability tests: bugs found by probing, confidence scoring, logging, and the evaluation harness."""

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evaluate  # noqa: E402
from agent import PlanningAgent, commit_proposal, confidence_label  # noqa: E402
from evaluate import FakeLLM  # noqa: E402
from llm import DemoLLM, LLMError  # noqa: E402
from pawpal_system import Owner, Pet, Task  # noqa: E402
from rag import Retriever, build_query  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(scope="module")
def retriever():
    return Retriever()


def owner_with(*pets, minutes=120):
    owner = Owner(name="Sally", available_minutes_per_day=minutes)
    for name, species, age in pets:
        owner.add_pet(Pet(name, species, age))
    return owner


class ListHandler(logging.Handler):
    """Collects log records from the pawpal loggers so tests can assert on them."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def pawpal_logs():
    handler = ListHandler()
    logger = logging.getLogger("pawpal")
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    yield handler.records
    logger.removeHandler(handler)
    logger.setLevel(old_level)


# --- bugs found by probing (each was a real failure before the fix) ----------------

def test_task_running_past_midnight_wraps_the_end_time():
    assert Task("x", "c", 90, "high", start_time="23:30").end_time == "01:00"
    assert Task("x", "c", 30, "high", start_time="08:00").end_time == "08:30"  # unchanged


def test_duplicate_pet_names_are_refused_with_a_clear_message(retriever):
    owner = owner_with(("Rex", "dog", 3), ("rex", "cat", 3))  # case-insensitive match
    llm = FakeLLM(RuntimeError("must not be called"))
    result = PlanningAgent(owner, llm, retriever).run()
    assert llm.calls == 0 and result.tasks == []
    assert "same name" in result.problems[0]


def test_accepting_the_same_proposal_twice_adds_tasks_once(retriever):
    owner = owner_with(("Biscuit", "dog", 10))
    result = PlanningAgent(owner, DemoLLM(), retriever).run()
    first = commit_proposal(owner, result)
    count = len(owner.get_all_tasks())
    assert first and commit_proposal(owner, result) == []
    assert len(owner.get_all_tasks()) == count


def test_species_is_matched_case_insensitively(retriever):
    lower = {c.label for c, _ in retriever.search(build_query("dog", 3), "dog")}
    upper = {c.label for c, _ in retriever.search(build_query("Dog", 3), "Dog")}
    assert lower == upper and lower


def test_app_refuses_a_duplicate_pet_name():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    owner = owner_with(("Biscuit", "dog", 10))
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["owner"] = owner
    at.run()
    next(t for t in at.text_input if t.label == "Pet name").set_value("biscuit")
    next(b for b in at.button if b.label == "Add pet").click().run()
    assert len(owner.pets) == 1
    assert any("already have a pet named" in e.value for e in at.error)


# --- confidence score ----------------------------------------------------------------

def good_reply(start="08:00"):
    return {"tasks": [{
        "pet": "Biscuit", "title": "Feeding", "category": "feeding", "duration_minutes": 10,
        "priority": "high", "frequency": "daily", "start_time": start, "essential": True,
        "source": "dog_care.md: Senior dogs",
    }]}


def test_first_try_success_has_full_confidence(retriever):
    owner = owner_with(("Biscuit", "dog", 10))
    result = PlanningAgent(owner, FakeLLM(good_reply()), retriever).run()
    assert result.confidence == 1.0 and confidence_label(result.confidence) == "high"


def test_each_retry_lowers_confidence(retriever):
    owner = owner_with(("Biscuit", "dog", 10))
    result = PlanningAgent(owner, FakeLLM("not json", good_reply()), retriever).run()
    assert result.iterations == 2 and result.confidence == 0.85


def test_unresolved_problems_and_missing_guidance_lower_confidence(retriever):
    big = {"tasks": [{**good_reply()["tasks"][0], "duration_minutes": 200}]}
    owner = owner_with(("Biscuit", "dog", 10), minutes=30)
    result = PlanningAgent(owner, FakeLLM(big), retriever, max_iterations=2).run()
    assert not result.converged
    assert result.confidence == 0.4 and confidence_label(result.confidence) == "low"


def test_missing_guidance_for_one_pet_lowers_confidence(retriever):
    owner = owner_with(("Biscuit", "dog", 10), ("Bubbles", "fish", 2))
    result = PlanningAgent(owner, FakeLLM(good_reply()), retriever).run()
    assert result.no_guidance == ["Bubbles"] and result.confidence == 0.8


def test_long_prompts_are_not_dumped_into_the_log(retriever, pawpal_logs):
    owner = owner_with(("Biscuit", "dog", 10))
    PlanningAgent(owner, FakeLLM(good_reply()), retriever).run("my secret note " * 20)
    assert all(len(m) < 600 for m in messages(pawpal_logs) if m.startswith("tool=llm"))
    assert not any("secret note secret note secret note" in m for m in messages(pawpal_logs))


def test_no_suggestions_and_fallback_have_zero_confidence(retriever):
    owner = owner_with(("Bubbles", "fish", 2))
    assert PlanningAgent(owner, FakeLLM(good_reply()), retriever).run().confidence == 0.0
    owner = owner_with(("Biscuit", "dog", 10))
    result = PlanningAgent(owner, FakeLLM(LLMError("down")), retriever).run()
    assert result.used_fallback and result.confidence == 0.0


@pytest.mark.parametrize("score, label", [(1.0, "high"), (0.8, "high"), (0.79, "medium"),
                                          (0.5, "medium"), (0.49, "low"), (0.0, "low")])
def test_confidence_labels(score, label):
    assert confidence_label(score) == label


# --- logging: failures are recorded with what and why ---------------------------------

def messages(records, level=None):
    return [r.getMessage() for r in records if level is None or r.levelno >= level]


def test_model_failure_is_logged_with_its_reason(retriever, pawpal_logs):
    owner = owner_with(("Biscuit", "dog", 10))
    PlanningAgent(owner, FakeLLM(LLMError("rate limited")), retriever).run()
    text = " | ".join(messages(pawpal_logs))
    assert "model call failed" in text and "rate limited" in text
    assert "fallback" in text.lower()


def test_rejected_replies_and_retries_are_logged(retriever, pawpal_logs):
    owner = owner_with(("Biscuit", "dog", 10))
    PlanningAgent(owner, FakeLLM("not json", good_reply()), retriever).run()
    text = " | ".join(messages(pawpal_logs))
    assert "VALIDATE (attempt 1): rejected" in text
    assert "VALIDATE (attempt 2): accepted" in text


def test_retrieval_and_tool_calls_are_logged(retriever, pawpal_logs):
    owner = owner_with(("Biscuit", "dog", 10))
    PlanningAgent(owner, FakeLLM(good_reply()), retriever).run()
    text = " | ".join(messages(pawpal_logs))
    assert "tool=retrieve_guidance" in text and "tool=generate_daily_plan" in text
    assert "retrieve query=" in text and "senior" in text


def test_final_summary_with_confidence_is_logged(retriever, pawpal_logs):
    owner = owner_with(("Biscuit", "dog", 10))
    PlanningAgent(owner, FakeLLM(good_reply()), retriever).run()
    assert any("agent finished" in m and "confidence=1.00" in m for m in messages(pawpal_logs))


def test_a_second_accept_is_logged_as_a_warning(retriever, pawpal_logs):
    owner = owner_with(("Biscuit", "dog", 10))
    result = PlanningAgent(owner, DemoLLM(), retriever).run()
    commit_proposal(owner, result)
    commit_proposal(owner, result)
    assert any("already accepted" in m for m in messages(pawpal_logs, logging.WARNING))


def test_logging_setup_writes_to_a_file_and_is_idempotent(tmp_path):
    from logging_setup import setup_logging

    logger = logging.getLogger("pawpal")
    saved = list(logger.handlers)
    for h in saved:
        logger.removeHandler(h)
    try:
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file)
        setup_logging(log_file=tmp_path / "other.log")  # second call must not add handlers
        assert len(logger.handlers) == 2  # one file, one console
        logging.getLogger("pawpal.test").info("hello log file")
        for h in logger.handlers:
            h.flush()
        assert "hello log file" in log_file.read_text(encoding="utf-8")
    finally:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()
        for h in saved:
            logger.addHandler(h)


# --- the evaluation harness is itself tested --------------------------------------------

def test_every_labeled_retrieval_case_matches(retriever):
    rows = evaluate.retrieval_eval(retriever)
    failures = [(r["input"], r["expected"], r["got"]) for r in rows
                if not r["exact"] or r["wrong_species"]]
    assert failures == []


def test_every_agent_scenario_passes(retriever):
    rows = evaluate.agent_eval(retriever)
    failures = [(r["input"], r["detail"]) for r in rows if not r["passed"]]
    assert failures == []


def test_demo_proposals_are_consistent_across_runs(retriever):
    same, runs = evaluate.consistency_eval(retriever, runs=5)
    assert same and runs == 5
