"""The .env loader must never beat the real environment."""

from __future__ import annotations

import os

import pytest

from evals.dotenv import load


@pytest.fixture
def env_file(tmp_path):
    def write(text):
        path = tmp_path / ".env"
        path.write_text(text, encoding="utf-8")
        return path

    return write


def test_reads_key_value_pairs(env_file, monkeypatch):
    monkeypatch.delenv("EVAL_DEMO_A", raising=False)
    load(env_file("EVAL_DEMO_A=hello\n"))
    assert os.environ["EVAL_DEMO_A"] == "hello"


def test_a_real_environment_variable_wins(env_file, monkeypatch):
    """An export or a CI secret must override the file, never the other way."""
    monkeypatch.setenv("EVAL_DEMO_B", "from-the-shell")
    applied = load(env_file("EVAL_DEMO_B=from-the-file\n"))
    assert os.environ["EVAL_DEMO_B"] == "from-the-shell"
    assert "EVAL_DEMO_B" not in applied


def test_comments_and_blank_lines_are_skipped(env_file, monkeypatch):
    monkeypatch.delenv("EVAL_DEMO_C", raising=False)
    applied = load(env_file("# a comment\n\n   \nEVAL_DEMO_C=3\n"))
    assert applied == {"EVAL_DEMO_C": "3"}


def test_quotes_and_export_prefix_are_stripped(env_file, monkeypatch):
    for name in ("EVAL_DEMO_D", "EVAL_DEMO_E"):
        monkeypatch.delenv(name, raising=False)
    load(env_file('export EVAL_DEMO_D="quoted"\nEVAL_DEMO_E=\'single\'\n'))
    assert os.environ["EVAL_DEMO_D"] == "quoted"
    assert os.environ["EVAL_DEMO_E"] == "single"


def test_a_missing_file_is_not_an_error(tmp_path):
    assert load(tmp_path / "nope.env") == {}


def test_lines_without_an_equals_sign_are_ignored(env_file, monkeypatch):
    monkeypatch.delenv("EVAL_DEMO_F", raising=False)
    assert load(env_file("not a pair\nEVAL_DEMO_F=ok\n")) == {"EVAL_DEMO_F": "ok"}
