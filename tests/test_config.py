from __future__ import annotations

import pytest

from multi_agent.config import Settings, normalize_log_level


def test_normalize_log_level_accepts_case_and_blank():
    assert normalize_log_level("debug") == "DEBUG"
    assert normalize_log_level(" Critical ") == "CRITICAL"
    assert normalize_log_level("") == "WARNING"
    assert normalize_log_level(None) == "WARNING"


def test_normalize_log_level_rejects_unknown():
    with pytest.raises(RuntimeError) as excinfo:
        normalize_log_level("verbose")
    message = str(excinfo.value)
    assert "verbose" in message
    assert "CRITICAL" in message


def test_settings_from_env_rejects_non_finite_timeout(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    for raw in ("nan", "inf", "-1", "0"):
        monkeypatch.setenv("OPENAI_TIMEOUT", raw)
        with pytest.raises(RuntimeError, match="OPENAI_TIMEOUT"):
            Settings.from_env()


def test_settings_from_env_validates_log_level(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("MULTI_AGENT_LOG_LEVEL", "noisy")
    with pytest.raises(RuntimeError, match="日志级别非法"):
        Settings.from_env()

    monkeypatch.setenv("MULTI_AGENT_LOG_LEVEL", "info")
    assert Settings.from_env().log_level == "INFO"
