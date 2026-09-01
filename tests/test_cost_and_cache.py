"""Cost accounting and response-cache tests."""

import time

import pytest

from evals import cache as cache_mod
from evals.cost import PRICING, Spend, over_budget, unknown_models


class TestSpend:
    def test_prices_input_and_output_at_the_model_rate(self):
        spend = Spend(input_tokens=1_000_000, output_tokens=1_000_000)
        in_rate, out_rate = PRICING["claude-opus-5"]
        assert spend.usd("claude-opus-5") == pytest.approx(in_rate + out_rate)

    def test_cache_reads_are_a_tenth_of_input(self):
        cached = Spend(cache_read_tokens=1_000_000).usd("claude-opus-5")
        fresh = Spend(input_tokens=1_000_000).usd("claude-opus-5")
        assert cached == pytest.approx(fresh * 0.10)

    def test_cache_writes_cost_a_premium(self):
        written = Spend(cache_write_tokens=1_000_000).usd("claude-opus-5")
        fresh = Spend(input_tokens=1_000_000).usd("claude-opus-5")
        assert written > fresh

    def test_addition_accumulates_every_field(self):
        total = Spend(input_tokens=10, api_calls=1) + Spend(output_tokens=5, cached_calls=2)
        assert (total.input_tokens, total.output_tokens) == (10, 5)
        assert (total.api_calls, total.cached_calls) == (1, 2)

    def test_unknown_model_prices_at_zero_and_is_reported(self):
        assert Spend(input_tokens=1_000_000).usd("some-future-model") == 0.0
        assert unknown_models(["claude-opus-5", "some-future-model"]) == ["some-future-model"]


class TestBudget:
    def test_under_budget(self):
        exceeded, usd = over_budget(Spend(input_tokens=1000), "claude-opus-5", budget=1.0)
        assert not exceeded
        assert usd < 1.0

    def test_over_budget(self):
        exceeded, usd = over_budget(Spend(output_tokens=10_000_000), "claude-opus-5", budget=1.0)
        assert exceeded
        assert usd > 1.0


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cache_mod, "ENABLED", True)
    return tmp_path


def _entry(text="hello", stored_at=None):
    return cache_mod.CachedCompletion(
        text=text,
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=0,
        latency_ms=100,
        stop_reason="end_turn",
        stored_at=time.time() if stored_at is None else stored_at,
    )


class TestCache:
    def test_roundtrip(self, tmp_cache):
        digest = cache_mod.key(model="m", effort="low", system="s", user="u", max_tokens=10, repeat=0)
        cache_mod.put(digest, _entry("cached answer"))
        assert cache_mod.get(digest).text == "cached answer"

    def test_miss_returns_none(self, tmp_cache):
        assert cache_mod.get("0" * 64) is None

    def test_any_prompt_change_misses(self, tmp_cache):
        args = dict(model="m", effort="low", system="s", user="u", max_tokens=10, repeat=0)
        original = cache_mod.key(**args)
        for field, value in [
            ("model", "other"),
            ("effort", "high"),
            ("system", "s "),
            ("user", "u2"),
            ("max_tokens", 11),
            ("repeat", 1),
        ]:
            assert cache_mod.key(**{**args, field: value}) != original, field

    def test_repeats_are_stored_separately(self, tmp_cache):
        args = dict(model="m", effort="low", system="s", user="u", max_tokens=10)
        cache_mod.put(cache_mod.key(**args, repeat=0), _entry("first"))
        cache_mod.put(cache_mod.key(**args, repeat=1), _entry("second"))
        assert cache_mod.get(cache_mod.key(**args, repeat=0)).text == "first"
        assert cache_mod.get(cache_mod.key(**args, repeat=1)).text == "second"

    def test_stale_entry_is_a_miss(self, tmp_cache, monkeypatch):
        monkeypatch.setattr(cache_mod, "TTL_SECONDS", 60)
        digest = cache_mod.key(model="m", effort="low", system="s", user="u", max_tokens=10, repeat=0)
        cache_mod.put(digest, _entry(stored_at=time.time() - 3600))
        assert cache_mod.get(digest) is None

    def test_corrupt_entry_is_a_miss_not_a_crash(self, tmp_cache):
        digest = "ab" + "0" * 62
        path = tmp_cache / digest[:2] / f"{digest}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ not json", encoding="utf-8")
        assert cache_mod.get(digest) is None

    def test_disabled_cache_never_hits(self, tmp_cache, monkeypatch):
        digest = cache_mod.key(model="m", effort="low", system="s", user="u", max_tokens=10, repeat=0)
        cache_mod.put(digest, _entry())
        monkeypatch.setattr(cache_mod, "ENABLED", False)
        assert cache_mod.get(digest) is None

    def test_clear_removes_entries(self, tmp_cache):
        for i in range(3):
            cache_mod.put(
                cache_mod.key(model="m", effort="low", system="s", user=f"u{i}", max_tokens=10, repeat=0),
                _entry(),
            )
        assert cache_mod.stats()["entries"] == 3
        assert cache_mod.clear() == 3
        assert cache_mod.stats()["entries"] == 0
