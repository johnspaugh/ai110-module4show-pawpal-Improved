# Its presence at the project root makes pytest add this folder to sys.path, so tests can
# `import pawpal_system` whether you run `pytest` or `python -m pytest`.

import os
import tempfile
from pathlib import Path

import pytest

# Tests must never write into the real pawpal.log: fake model calls there would look like
# live ones. Set at import so it is in place before any test or app run configures logging.
os.environ["PAWPAL_LOG_FILE"] = str(Path(tempfile.gettempdir()) / "pawpal-tests.log")


@pytest.fixture(autouse=True)
def isolate_tests_from_real_credentials(monkeypatch):
    """Keep every test independent of the developer's real .env and API key.

    Without this, a key in .env would make get_llm() return the live model, so
    results would depend on whose machine runs them and a test could make a paid
    API call. Tests that need a key or demo mode set their own values.
    """
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                 "PAWPAL_MODEL", "PAWPAL_DEMO", "PAWPAL_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
