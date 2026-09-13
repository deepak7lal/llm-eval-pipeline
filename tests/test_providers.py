"""Provider selection tests. No provider is ever actually constructed here."""

import pytest

from evals import providers


@pytest.fixture(autouse=True)
def clean_cache():
    providers.reset()
    yield
    providers.reset()


class TestPresets:
    def test_every_free_provider_is_listed(self):
        assert set(providers.available()) >= {
            "anthropic", "gemini", "groq", "ollama", "openrouter", "openai"
        }

    def test_each_preset_has_url_keyvar_and_model(self):
        for name, preset in providers.PRESETS.items():
            base_url, key_var, model = preset
            assert base_url.startswith("http"), name
            assert isinstance(key_var, str), name
            assert model, name

    def test_ollama_needs_no_key_and_is_local(self):
        base_url, key_var, _ = providers.PRESETS["ollama"]
        assert key_var == ""
        assert "localhost" in base_url


class TestSelection:
    def test_unknown_provider_is_rejected_by_name(self, monkeypatch):
        monkeypatch.setenv("EVAL_PROVIDER", "not-a-provider")
        with pytest.raises(ValueError, match="unknown provider"):
            providers.get_provider()

    def test_missing_key_names_the_variable_to_set(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("EVAL_PROVIDER", "gemini")
        with pytest.raises(RuntimeError) as exc:
            providers.get_provider()
        message = str(exc.value)
        # The error has to be actionable, not just "auth failed".
        assert "GEMINI_API_KEY" in message
        assert "ollama" in message  # points at the no-key fallback

    def test_reset_clears_the_cached_provider(self):
        providers._cached = object()
        providers.reset()
        assert providers._cached is None


class TestResponse:
    def test_defaults_are_zero_not_none(self):
        r = providers.Response(text="hi", input_tokens=1, output_tokens=2)
        assert (r.cache_read_tokens, r.cache_write_tokens) == (0, 0)
        assert r.refused is False
        assert r.stop_reason is None

    def test_refusal_is_explicit(self):
        r = providers.Response(text="", input_tokens=1, output_tokens=0, refused=True)
        assert r.refused


class TestConfigIntegration:
    """Switching provider must switch the default model with it."""

    @pytest.mark.parametrize(
        "provider,expected",
        [
            ("gemini", "gemini-2.0-flash"),
            ("groq", "llama-3.3-70b-versatile"),
            ("ollama", "llama3.2"),
            ("anthropic", "claude-opus-5"),
        ],
    )
    def test_default_model_follows_provider(self, provider, expected, monkeypatch):
        monkeypatch.setenv("EVAL_PROVIDER", provider)
        monkeypatch.delenv("EVAL_MODEL", raising=False)
        import importlib

        from evals import config

        reloaded = importlib.reload(config)
        try:
            assert reloaded.MODEL == expected
        finally:
            monkeypatch.delenv("EVAL_PROVIDER", raising=False)
            importlib.reload(config)

    def test_explicit_model_wins_over_the_preset(self, monkeypatch):
        monkeypatch.setenv("EVAL_PROVIDER", "gemini")
        monkeypatch.setenv("EVAL_MODEL", "gemini-2.5-flash")
        import importlib

        from evals import config

        reloaded = importlib.reload(config)
        try:
            assert reloaded.MODEL == "gemini-2.5-flash"
        finally:
            monkeypatch.delenv("EVAL_PROVIDER", raising=False)
            monkeypatch.delenv("EVAL_MODEL", raising=False)
            importlib.reload(config)


def test_free_tier_models_are_priced_at_zero_not_unknown():
    """A free model must read $0.00, not silently fall out of cost reporting."""
    from evals.cost import PRICING, unknown_models

    for _, _, model in providers.PRESETS.values():
        if model.startswith("gpt-"):
            continue  # OpenAI is not free; deliberately unpriced here
        assert model in PRICING, model
    assert unknown_models(["gemini-2.0-flash", "llama3.2"]) == []
