"""The one place the proxy key and endpoint are read from."""

from __future__ import annotations

from pathlib import Path

import pytest

from aieng.advanced_rag.proxy import (
    PROXY_BASE_URL,
    MissingProxyKeyError,
    find_env_file,
    is_placeholder,
    points_at_proxy,
    read_env_file,
    resolve_api_key,
    resolve_base_url,
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ("OPENAI_API_KEY", "VECTOR_PROXY_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    # A repository root of its own, so the walk never reaches a real .env.
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)


def test_the_platform_key_wins_over_the_alias(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", PROXY_BASE_URL)
    monkeypatch.setenv("OPENAI_API_KEY", " platform-key ")
    monkeypatch.setenv("VECTOR_PROXY_API_KEY", "alias-key")

    assert resolve_api_key() == "platform-key"


def test_an_exported_openai_key_without_the_proxy_endpoint_is_never_used(monkeypatch):
    # A laptop with a personal OpenAI key exported must not send it to the proxy.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-personal")

    with pytest.raises(MissingProxyKeyError):
        resolve_api_key()

    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("VECTOR_PROXY_API_KEY", "alias-key")

    assert resolve_api_key() == "alias-key"


def test_an_openai_key_in_the_env_file_still_counts(tmp_path):
    # Older .env files used that name for a Vector proxy key.
    (tmp_path / ".env").write_text("OPENAI_API_KEY=old-proxy-key\n")

    assert resolve_api_key() == "old-proxy-key"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (PROXY_BASE_URL, True),
        ("https://proxy.vectorinstitute.ai/v1/ ", True),
        ("https://api.openai.com/v1", False),
        ("https://proxy.vectorinstitute.ai.example.com/v1", False),
        ("", False),
    ],
)
def test_only_the_proxy_host_points_at_the_proxy(url, expected):
    assert points_at_proxy(url) is expected


def test_the_alias_alone_is_enough(monkeypatch):
    monkeypatch.setenv("VECTOR_PROXY_API_KEY", "alias-key")

    assert resolve_api_key() == "alias-key"


def test_an_exported_key_wins_over_the_env_file(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("VECTOR_PROXY_API_KEY=stale\n")
    monkeypatch.setenv("OPENAI_BASE_URL", PROXY_BASE_URL)
    monkeypatch.setenv("OPENAI_API_KEY", "fresh")

    assert resolve_api_key() == "fresh"


def test_the_env_file_is_read_past_spaces_quotes_and_export(tmp_path):
    (tmp_path / ".env").write_text(
        "# comment\nOTHER=1\nexport VECTOR_PROXY_API_KEY = 'from-file'\n"
    )

    assert resolve_api_key() == "from-file"


def test_an_explicit_env_file_is_read(tmp_path):
    env = tmp_path / "elsewhere.env"
    env.write_text('OPENAI_API_KEY="explicit"\n')

    assert resolve_api_key(env) == "explicit"


def test_a_placeholder_counts_as_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "<your_key>")
    (tmp_path / ".env").write_text("VECTOR_PROXY_API_KEY=real\n")

    assert resolve_api_key() == "real"


def test_no_key_anywhere_names_both_variables():
    with pytest.raises(MissingProxyKeyError, match="VECTOR_PROXY_API_KEY"):
        resolve_api_key()


def test_the_missing_key_error_is_a_runtime_error():
    assert issubclass(MissingProxyKeyError, RuntimeError)


def test_a_caller_can_put_its_own_name_first(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "override")
    monkeypatch.setenv("OPENAI_API_KEY", "platform")

    assert resolve_api_key(names=("LLM_API_KEY", "OPENAI_API_KEY")) == "override"


def test_the_walk_finds_a_env_file_above_the_start(tmp_path):
    (tmp_path / ".env").write_text("A=1\n")
    nested = tmp_path / "implementations" / "getting_started"
    nested.mkdir(parents=True)

    assert find_env_file(nested) == tmp_path / ".env"


def test_the_walk_stops_at_the_repository_root(tmp_path):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=outer\n")
    worktree = tmp_path / "worktrees" / "feature"
    worktree.mkdir(parents=True)
    (worktree / ".git").write_text("gitdir: elsewhere\n")

    assert find_env_file(worktree) is None


def test_the_base_url_defaults_to_the_proxy():
    assert resolve_base_url() == PROXY_BASE_URL


def test_the_injected_base_url_wins_and_loses_its_trailing_slash(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example.org/v1/")

    assert resolve_base_url() == "https://proxy.example.org/v1"


def test_the_base_url_falls_back_to_the_env_file(tmp_path):
    (tmp_path / ".env").write_text("OPENAI_BASE_URL=https://proxy.example.org/v1\n")

    assert resolve_base_url() == "https://proxy.example.org/v1"


@pytest.mark.parametrize("value", ["", "  ", "<your_key>", "your_key", "sk-..."])
def test_template_values_are_placeholders(value):
    assert is_placeholder(value)


def test_a_real_looking_key_is_not_a_placeholder():
    assert not is_placeholder("abc123")


def test_a_later_line_wins_in_the_env_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text("A=1\nA=2\n")

    assert read_env_file(env) == {"A": "2"}
