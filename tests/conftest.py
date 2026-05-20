"""
Shared pytest fixtures — applied automatically to all tests in this directory.
"""
import pytest


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch):
    """Inject dummy env vars so modules that read os.environ don't KeyError."""
    monkeypatch.setenv("DEEPSEEK_API_KEY",  "test-deepseek-key")
    monkeypatch.setenv("VOLC_ACCESSKEY",    "test-ak")
    monkeypatch.setenv("VOLC_SECRETKEY",    "test-sk")
    monkeypatch.setenv("PUBLIC_BASE_URL",   "http://localhost:8000")
