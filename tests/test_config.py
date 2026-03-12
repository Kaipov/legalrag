from src import config as config_mod


def test_openrouter_model_slug_prefers_openrouter_when_key_present(monkeypatch) -> None:
    monkeypatch.setattr(config_mod, 'LLM_PROVIDER', '')
    monkeypatch.setattr(config_mod, 'OPENAI_API_KEY', 'openai-key')
    monkeypatch.setattr(config_mod, 'OPENROUTER_API_KEY', 'openrouter-key')

    assert config_mod.model_uses_openrouter('openai/gpt-oss-120b:nitro') is True
    assert config_mod.get_llm_api_key('openai/gpt-oss-120b:nitro') == 'openrouter-key'
    assert config_mod.get_llm_api_base('openai/gpt-oss-120b:nitro') == 'https://openrouter.ai/api/v1'


def test_openai_model_stays_on_openai_when_both_keys_present(monkeypatch) -> None:
    monkeypatch.setattr(config_mod, 'LLM_PROVIDER', '')
    monkeypatch.setattr(config_mod, 'OPENAI_API_KEY', 'openai-key')
    monkeypatch.setattr(config_mod, 'OPENROUTER_API_KEY', 'openrouter-key')

    assert config_mod.model_uses_openrouter('gpt-4.1-mini') is False
    assert config_mod.get_llm_api_key('gpt-4.1-mini') == 'openai-key'
    assert config_mod.get_llm_api_base('gpt-4.1-mini') == 'https://api.openai.com/v1'


def test_llm_provider_override_forces_openrouter(monkeypatch) -> None:
    monkeypatch.setattr(config_mod, 'LLM_PROVIDER', 'openrouter')
    monkeypatch.setattr(config_mod, 'OPENAI_API_KEY', 'openai-key')
    monkeypatch.setattr(config_mod, 'OPENROUTER_API_KEY', 'openrouter-key')

    assert config_mod.model_uses_openrouter('gpt-4.1-mini') is True
    assert config_mod.get_llm_api_key('gpt-4.1-mini') == 'openrouter-key'
