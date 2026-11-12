"""Tests for choosing and calling the live model backends. No test here touches the network."""

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import PlanningAgent  # noqa: E402
from llm import (  # noqa: E402
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_GEMINI_MODEL,
    AnthropicLLM,
    DemoLLM,
    GeminiLLM,
    LLMError,
    get_llm,
)
from pawpal_system import Owner, Pet  # noqa: E402
from rag import Retriever  # noqa: E402

SECRET = "AQ.fake-gemini-key-for-tests-0123456789"


# --- which backend get_llm picks --------------------------------------------------

def test_a_gemini_key_alone_selects_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    llm = get_llm()
    assert llm.provider == "gemini" and llm.mode == "live" and llm.model == DEFAULT_GEMINI_MODEL


def test_google_api_key_is_accepted_as_a_gemini_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", SECRET)
    assert get_llm().provider == "gemini"


def test_an_anthropic_key_alone_selects_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    llm = get_llm()
    assert llm.provider == "anthropic" and llm.model == DEFAULT_ANTHROPIC_MODEL


def test_with_both_keys_anthropic_wins_unless_a_provider_is_chosen(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    assert get_llm().provider == "anthropic"
    monkeypatch.setenv("PAWPAL_PROVIDER", "gemini")
    assert get_llm().provider == "gemini"


def test_a_chosen_provider_without_its_own_key_falls_back_to_demo(monkeypatch):
    # The Gemini key must never be sent to Anthropic (or the reverse).
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    monkeypatch.setenv("PAWPAL_PROVIDER", "anthropic")
    assert get_llm().mode == "demo"


def test_an_unknown_provider_name_falls_back_to_demo(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    monkeypatch.setenv("PAWPAL_PROVIDER", "openai")
    assert get_llm().mode == "demo"


def test_no_key_means_demo():
    assert isinstance(get_llm(), DemoLLM)


def test_forcing_demo_beats_any_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    monkeypatch.setenv("PAWPAL_DEMO", "1")
    assert get_llm().mode == "demo"


def test_every_backend_reports_a_provider_name():
    assert DemoLLM.provider == "demo" and GeminiLLM.provider == "gemini"
    assert AnthropicLLM.provider == "anthropic"


# --- model name handling ------------------------------------------------------------

def test_a_custom_model_name_is_used(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    monkeypatch.setenv("PAWPAL_MODEL", "gemini-2.5-flash-lite")
    assert get_llm().model == "gemini-2.5-flash-lite"


def test_a_claude_model_name_is_ignored_for_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    monkeypatch.setenv("PAWPAL_MODEL", "claude-opus-5-5")
    assert get_llm().model == DEFAULT_GEMINI_MODEL


def test_a_gemini_model_name_is_ignored_for_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    monkeypatch.setenv("PAWPAL_MODEL", "gemini-2.5-flash")
    assert get_llm().model == DEFAULT_ANTHROPIC_MODEL


# --- GeminiLLM.complete, with a fake client ---------------------------------------------

class FakeModels:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.error:
            raise self.error
        return self.reply


def gemini_with(monkeypatch, reply=None, error=None):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    llm = GeminiLLM()
    llm.client = SimpleNamespace(models=FakeModels(reply, error))
    return llm


def reply_with(text, finish="STOP", block=None):
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=finish)],
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=34),
        prompt_feedback=SimpleNamespace(block_reason=block),
    )


def test_gemini_returns_the_model_text_and_asks_for_json(monkeypatch):
    llm = gemini_with(monkeypatch, reply_with('{"tasks": []}'))

    assert llm.complete("system rules", "user message") == '{"tasks": []}'
    call = llm.client.models.calls[0]
    assert call["model"] == DEFAULT_GEMINI_MODEL and call["contents"] == "user message"
    assert call["config"].system_instruction == "system rules"
    assert call["config"].response_mime_type == "application/json"


def test_gemini_api_errors_become_llm_errors_with_the_key_removed(monkeypatch):
    from google.genai import errors

    boom = errors.ClientError(400, {"error": {"message": f"API key not valid: {SECRET}"}})
    llm = gemini_with(monkeypatch, error=boom)

    with pytest.raises(LLMError) as caught:
        llm.complete("s", "u")
    assert SECRET not in str(caught.value) and "ClientError" in str(caught.value)


def test_gemini_empty_or_blocked_replies_become_llm_errors(monkeypatch):
    llm = gemini_with(monkeypatch, reply_with(None, finish="SAFETY", block="PROHIBITED_CONTENT"))

    with pytest.raises(LLMError, match="no text"):
        llm.complete("s", "u")


def test_gemini_whitespace_only_reply_is_an_error(monkeypatch):
    with pytest.raises(LLMError):
        gemini_with(monkeypatch, reply_with("   \n ")).complete("s", "u")


def test_the_gemini_key_never_reaches_the_log(monkeypatch):
    llm = gemini_with(monkeypatch, reply_with('{"tasks": []}'))
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger = logging.getLogger("pawpal")
    old = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        llm.complete("s", "u")
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old)

    assert records and not any(SECRET in r.getMessage() for r in records)
    assert any("provider=gemini" in r.getMessage() for r in records)


# --- the whole agent on the Gemini backend ----------------------------------------------

def test_the_agent_runs_end_to_end_on_the_gemini_backend(monkeypatch):
    task = {
        "pet": "Biscuit", "title": "Feeding", "category": "feeding", "duration_minutes": 10,
        "priority": "high", "frequency": "daily", "start_time": "08:00", "essential": True,
        "source": "dog_care.md: Senior dogs",
    }
    llm = gemini_with(monkeypatch, reply_with(json.dumps({"tasks": [task], "notes": "ok"})))
    owner = Owner("Sally", 60)
    owner.add_pet(Pet("Biscuit", "dog", 10))

    result = PlanningAgent(owner, llm, Retriever()).run()

    assert result.mode == "live" and result.converged
    assert [t.title for t in result.tasks] == ["Feeding"]
    assert owner.get_all_tasks() == []  # still nothing saved before the owner accepts


def test_a_gemini_failure_triggers_the_normal_fallback(monkeypatch):
    from google.genai import errors

    llm = gemini_with(monkeypatch, error=errors.ServerError(503, {"error": {"message": "busy"}}))
    owner = Owner("Sally", 60)
    owner.add_pet(Pet("Biscuit", "dog", 10))

    result = PlanningAgent(owner, llm, Retriever()).run()

    assert result.used_fallback and result.tasks == [] and result.confidence == 0.0


# --- retrying temporary Gemini errors -----------------------------------------------------

class SequencedModels:
    """Fake models endpoint that raises or returns each item in order."""

    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    def generate_content(self, model, contents, config):
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def sequenced_gemini(monkeypatch, *outcomes):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    monkeypatch.setattr("llm.time.sleep", lambda seconds: None)  # do not really wait
    llm = GeminiLLM()
    llm.client = SimpleNamespace(models=SequencedModels(*outcomes))
    return llm


def test_a_busy_server_is_retried_and_then_succeeds(monkeypatch):
    from google.genai import errors

    busy = errors.ServerError(503, {"error": {"message": "high demand"}})
    llm = sequenced_gemini(monkeypatch, busy, busy, reply_with('{"tasks": []}'))

    assert llm.complete("s", "u") == '{"tasks": []}'
    assert llm.client.models.calls == 3


def test_a_rate_limit_error_is_also_retried(monkeypatch):
    from google.genai import errors

    limited = errors.ClientError(429, {"error": {"message": "slow down"}})
    llm = sequenced_gemini(monkeypatch, limited, reply_with('{"tasks": []}'))

    assert llm.complete("s", "u") == '{"tasks": []}' and llm.client.models.calls == 2


def test_it_gives_up_after_the_last_attempt_with_a_clear_error(monkeypatch):
    from google.genai import errors

    from llm import GEMINI_ATTEMPTS

    busy = errors.ServerError(503, {"error": {"message": "high demand"}})
    llm = sequenced_gemini(monkeypatch, busy)  # busy forever

    with pytest.raises(LLMError, match="ServerError"):
        llm.complete("s", "u")
    assert llm.client.models.calls == GEMINI_ATTEMPTS


def test_errors_that_retrying_cannot_fix_are_not_retried(monkeypatch):
    from google.genai import errors

    bad_key = errors.ClientError(400, {"error": {"message": "API key not valid"}})
    not_found = errors.ClientError(404, {"error": {"message": "model not available"}})
    for error in (bad_key, not_found):
        llm = sequenced_gemini(monkeypatch, error)
        with pytest.raises(LLMError):
            llm.complete("s", "u")
        assert llm.client.models.calls == 1  # failed fast, no pointless retries


def test_retries_are_logged_so_a_slow_run_can_be_explained(monkeypatch):
    from google.genai import errors

    busy = errors.ServerError(503, {"error": {"message": "high demand"}})
    llm = sequenced_gemini(monkeypatch, busy, reply_with('{"tasks": []}'))
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger = logging.getLogger("pawpal")
    old = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        llm.complete("s", "u")
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old)

    assert any("temporary error 503" in r.getMessage() and "retrying" in r.getMessage() for r in records)
