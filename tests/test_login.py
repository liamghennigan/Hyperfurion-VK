"""`voice-keyboard login` refuses, and does not contact a host.

`_set_toml_value` is what setup and the installers use to edit config.toml
in place. These tests pin that editor against a real tomllib round-trip.
"""

import io
from contextlib import redirect_stderr

import pytest
import tomllib

from voice_keyboard.client import _run_login, _set_toml_value


def _parsed(text: str) -> dict:
    return tomllib.loads(text)


def test_login_refuses_without_contacting_a_host(monkeypatch) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("login must not use the network")

    monkeypatch.setattr("requests.post", boom, raising=False)
    monkeypatch.setattr("requests.get", boom, raising=False)
    err = io.StringIO()
    with redirect_stderr(err), pytest.raises(SystemExit) as exc:
        _run_login({}, ["someone@example.com"])
    assert exc.value.code == 2
    text = err.getvalue().lower()
    assert "not available" in text
    assert "http" not in text


class TestSetTomlValue:
    def test_inserts_missing_table(self) -> None:
        out = _set_toml_value("[a]\nx = 1\n", "b", "y", "z")
        cfg = _parsed(out)
        assert cfg["a"]["x"] == 1 and cfg["b"]["y"] == "z"

    def test_replaces_without_duplicating(self) -> None:
        out = _set_toml_value('[providers.xai]\napi_key = "old"\n', "providers.xai", "api_key", "new")
        assert _parsed(out)["providers"]["xai"]["api_key"] == "new"
        assert out.count("api_key") == 1
