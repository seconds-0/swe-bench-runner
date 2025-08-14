import types


def make_fake_keyring():
    store = {}

    def get_password(service, username):
        return store.get((service, username))

    def set_password(service, username, password):
        store[(service, username)] = password

    def delete_password(service, username):
        store.pop((service, username), None)

    fake = types.SimpleNamespace(
        get_password=get_password,
        set_password=set_password,
        delete_password=delete_password,
    )
    return fake, store


def reload_with_fake_keyring(monkeypatch):
    fake, store = make_fake_keyring()
    # Ensure module is reloaded fresh
    import sys
    sys.modules.pop("src.swebench_runner.secrets_store", None)
    sys.modules.pop("swebench_runner.secrets_store", None)
    # Load module and patch its keyring attribute
    from swebench_runner import secrets_store
    monkeypatch.setattr(secrets_store, "keyring", fake, raising=True)
    return secrets_store, store


def test_env_var_precedence_over_keyring(monkeypatch):
    secrets_store, store = reload_with_fake_keyring(monkeypatch)

    # Put value in keyring first
    store[("swebench-runner", "openai")] = "keyring-openai"
    # Env var should take precedence
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai")

    assert secrets_store.get_api_key("openai") == "env-openai"


def test_set_and_clear_key(monkeypatch):
    secrets_store, store = reload_with_fake_keyring(monkeypatch)

    ok = secrets_store.set_api_key("anthropic", "  abc123\n")
    assert ok is True
    assert store[("swebench-runner", "anthropic")] == "abc123"

    ok2 = secrets_store.clear_api_key("anthropic")
    assert ok2 is True
    assert ("swebench-runner", "anthropic") not in store


def test_get_returns_none_when_missing(monkeypatch):
    secrets_store, _ = reload_with_fake_keyring(monkeypatch)
    assert secrets_store.get_api_key("openrouter") is None


def test_sanitize_env_key(monkeypatch):
    secrets_store, _ = reload_with_fake_keyring(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", " sk-proj-XYZ\n\n")
    assert secrets_store.get_api_key("openai") == "sk-proj-XYZ"
