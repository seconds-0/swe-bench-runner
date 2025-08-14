from __future__ import annotations

import json
import os
from pathlib import Path

try:
    import keyring  # type: ignore
except Exception:  # pragma: no cover
    keyring = None  # type: ignore

_SERVICE = "swebench-runner"


def _config_dir() -> Path:
    base = Path.home() / ".swebench_runner"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _index_path() -> Path:
    return _config_dir() / "secrets.json"


def _load_index() -> dict:
    path = _index_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_index(data: dict) -> None:
    try:
        _index_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def _sanitize_secret(secret: str | None) -> str | None:
    if secret is None:
        return None
    # Trim surrounding whitespace and remove internal newlines/tabs/spaces
    cleaned = secret.strip()
    if not cleaned:
        return None
    # Collapse all whitespace characters which commonly appear when pasting
    cleaned = "".join(ch for ch in cleaned if ch not in " \t\r\n")
    return cleaned or None


def _env_var_name(provider: str, profile: str | None = None) -> str | None:
    provider = provider.lower()
    if provider == "openai":
        base = "OPENAI_API_KEY"
    elif provider == "anthropic":
        base = "ANTHROPIC_API_KEY"
    elif provider == "openrouter":
        base = "OPENROUTER_API_KEY"
    elif provider == "huggingface":
        base = "HUGGINGFACE_TOKEN"
    else:
        return None
    if profile:
        suffix = "__" + "".join(ch for ch in profile.upper() if ch.isalnum() or ch == "_")
        return base + suffix
    return base


def get_active_profile(provider: str) -> str:
    data = _load_index()
    return (
        data.get("providers", {})
        .get(provider, {})
        .get("active", "")
    )


def set_active_profile(provider: str, profile: str) -> None:
    data = _load_index()
    providers = data.setdefault("providers", {})
    entry = providers.setdefault(provider, {})
    entry["active"] = profile
    profiles = set(entry.get("profiles", []))
    profiles.add(profile)
    entry["profiles"] = sorted(profiles)
    _save_index(data)


def list_profiles(provider: str) -> list[str]:
    data = _load_index()
    return data.get("providers", {}).get(provider, {}).get("profiles", [])


def get_api_key(provider: str, profile: str | None = None) -> str | None:
    """Resolve API key with precedence: env > keyring > None."""
    target_profile = profile or get_active_profile(provider)
    # 1) Profile-specific env var
    env_name = _env_var_name(provider, target_profile)
    if env_name:
        val = _sanitize_secret(os.getenv(env_name))
        if val:
            return val
    # 2) Generic env var (backward compatibility)
    generic_env = _env_var_name(provider)
    if generic_env:
        val2 = _sanitize_secret(os.getenv(generic_env))
        if val2:
            return val2
    # Skip keyring access in test mode to avoid prompts
    if os.getenv("SWEBENCH_TEST_MODE") or os.getenv("PYTEST_CURRENT_TEST"):
        return None
    if keyring is not None:
        try:
            # Prefer profile-specific account, fall back to legacy single-account
            if target_profile:
                account = f"{provider}:{target_profile}"
                val = _sanitize_secret(keyring.get_password(_SERVICE, account))
                if val:
                    return val
            legacy = _sanitize_secret(keyring.get_password(_SERVICE, provider))
            if legacy:
                return legacy
            # Also consider other profiles if active not found
            data = _load_index()
            profs = data.get("providers", {}).get(provider, {}).get("profiles", [])
            for prof in profs:
                if prof == target_profile:
                    continue
                v = _sanitize_secret(keyring.get_password(_SERVICE, f"{provider}:{prof}"))
                if v:
                    return v
        except Exception:
            return None
    return None


def set_api_key(provider: str, secret: str, profile: str = "default") -> bool:
    """Store API key in OS keyring. Returns True on success."""
    secret_clean = _sanitize_secret(secret)
    if not secret_clean:
        return False
    # Skip keyring access in test mode
    if os.getenv("SWEBENCH_TEST_MODE") or os.getenv("PYTEST_CURRENT_TEST"):
        return False
    if keyring is None:
        return False
    try:
        # Store under profile-specific account; keep legacy account for default to ease transition
        account = f"{provider}:{profile}"
        keyring.set_password(_SERVICE, account, secret_clean)
        if profile == "default":
            try:
                keyring.set_password(_SERVICE, provider, secret_clean)
            except Exception:
                pass
        set_active_profile(provider, profile)
        return True
    except Exception:
        return False


def clear_api_key(provider: str, profile: str = "default") -> bool:
    # Skip keyring access in test mode
    if os.getenv("SWEBENCH_TEST_MODE") or os.getenv("PYTEST_CURRENT_TEST"):
        return False
    if keyring is None:
        return False
    try:
        # Remove profile-specific account; do not error if missing
        try:
            keyring.delete_password(_SERVICE, f"{provider}:{profile}")
        except Exception:
            pass
        # Clean legacy account for default
        if profile == "default":
            try:
                keyring.delete_password(_SERVICE, provider)
            except Exception:
                pass
        return True
    except Exception:
        return False


def has_key_for_profile(provider: str, profile: str) -> bool:
    """Strictly check if a key is set for the given provider/profile.

    Does not consider generic env vars or other profiles. Returns True if either
    a profile-specific environment variable is set, or a key exists in the keyring
    for that exact provider:profile account.
    """
    # Profile-specific env var
    env_name = _env_var_name(provider, profile)
    if env_name and _sanitize_secret(os.getenv(env_name)):
        return True
    # Skip keyring access in test mode
    if os.getenv("SWEBENCH_TEST_MODE") or os.getenv("PYTEST_CURRENT_TEST"):
        return False
    # Keyring account for this exact profile
    if keyring is not None:
        try:
            account = f"{provider}:{profile}"
            val = _sanitize_secret(keyring.get_password(_SERVICE, account))
            if val:
                return True
        except Exception:
            return False
    return False


def get_key_source(provider: str, profile: str) -> str:
    """Return the source of a key for provider/profile.

    Values: 'env:PROFILE' | 'env:GENERIC' | 'keychain' | 'none'.
    """
    # Profile-specific env var
    env_name = _env_var_name(provider, profile)
    if env_name and _sanitize_secret(os.getenv(env_name)):
        return "env:PROFILE"
    # Generic env var
    generic_env = _env_var_name(provider)
    if generic_env and _sanitize_secret(os.getenv(generic_env)):
        return "env:GENERIC"
    # Skip keyring access in test mode
    if os.getenv("SWEBENCH_TEST_MODE") or os.getenv("PYTEST_CURRENT_TEST"):
        return "none"
    # Keychain
    if keyring is not None:
        try:
            account = f"{provider}:{profile}"
            if _sanitize_secret(keyring.get_password(_SERVICE, account)):
                return "keychain"
        except Exception:
            pass
    return "none"


def is_keychain_available() -> bool:
    """Best-effort check if keyring backend is available."""
    return keyring is not None


def probe_keychain_usable() -> tuple[bool, str]:
    """Attempt a real set/get/delete to verify keychain usability.

    Returns (ok, message). If not ok, message contains a brief reason.

    Notes:
      - On macOS, this may trigger a GUI prompt the first time when run in a GUI
        session. In headless/SSH sessions, prompts won't appear and operations may fail.
      - We keep the probe lightweight and best-effort; any exception returns False.
    """
    # Skip keyring access in test mode
    if os.getenv("SWEBENCH_TEST_MODE") or os.getenv("PYTEST_CURRENT_TEST"):
        return False, "keyring disabled in test mode"
    if keyring is None:
        return False, "python-keyring not installed"
    service = _SERVICE
    account = "__probe__"
    try:
        keyring.set_password(service, account, "ok")
        val = keyring.get_password(service, account)
        # Always attempt cleanup
        try:
            keyring.delete_password(service, account)
        except Exception:
            pass
        if val == "ok":
            return True, "ok"
        return False, "get returned None"
    except Exception as e:  # pragma: no cover - environment specific
        return False, str(e)


def get_keyring_backend_label() -> str:
    """Return a short label describing the active keyring backend."""
    try:
        if keyring is None:
            return "(none)"
        kr = keyring.get_keyring()
        # Prefer class name with module for clarity
        cls = getattr(kr, "__class__", None)
        if cls is not None:
            return f"{cls.__module__}.{cls.__name__}"
        return repr(kr)
    except Exception:
        return "(unknown)"


def add_profile(provider: str, profile: str) -> None:
    data = _load_index()
    providers = data.setdefault("providers", {})
    entry = providers.setdefault(provider, {})
    profiles = set(entry.get("profiles", []))
    profiles.add(profile)
    entry["profiles"] = sorted(profiles)
    _save_index(data)


def remove_profile(provider: str, profile: str) -> None:
    data = _load_index()
    providers = data.setdefault("providers", {})
    entry = providers.setdefault(provider, {})
    profiles = [p for p in entry.get("profiles", []) if p != profile]
    entry["profiles"] = profiles
    if entry.get("active") == profile:
        entry["active"] = ""
    _save_index(data)


def rename_profile(provider: str, old: str, new: str) -> bool:
    try:
        # Skip keyring operations in test mode
        if os.getenv("SWEBENCH_TEST_MODE") or os.getenv("PYTEST_CURRENT_TEST"):
            # Still update the index even in test mode
            data = _load_index()
            providers = data.setdefault("providers", {})
            entry = providers.setdefault(provider, {})
            profiles = entry.get("profiles", [])
            if old in profiles:
                profiles = [new if p == old else p for p in profiles]
                entry["profiles"] = profiles
                if entry.get("active") == old:
                    entry["active"] = new
                _save_index(data)
                return True
            return False
        # Move key if present
        if keyring is not None:
            old_account = f"{provider}:{old}"
            new_account = f"{provider}:{new}"
            existing = _sanitize_secret(keyring.get_password(_SERVICE, old_account))
            if existing:
                keyring.set_password(_SERVICE, new_account, existing)
                try:
                    keyring.delete_password(_SERVICE, old_account)
                except Exception:
                    pass
        # Update index
        data = _load_index()
        providers = data.setdefault("providers", {})
        entry = providers.setdefault(provider, {})
        profiles = set(entry.get("profiles", []))
        if old in profiles:
            profiles.remove(old)
        profiles.add(new)
        entry["profiles"] = sorted(profiles)
        if entry.get("active") == old:
            entry["active"] = new
        _save_index(data)
        return True
    except Exception:
        return False
