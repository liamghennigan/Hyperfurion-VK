"""The Windows tray menu in setup mode (no key yet).

A new user is pointed at the settings file (a key, or their own server).
The Win32 calls are stubbed; only the labels are checked.
"""

from unittest import mock

from voice_keyboard.windows import shell as shell_mod
from voice_keyboard.windows.shell import ShellCallbacks, WinShell


def _menu_labels(monkeypatch, setup_message: str, status=None) -> list:
    user32 = mock.Mock()
    user32.TrackPopupMenu.return_value = 0  # dismissed: nothing chosen
    monkeypatch.setattr(shell_mod, "_load_api", lambda: (user32, None, None, None, None))
    callbacks = ShellCallbacks(status=lambda: dict(status or {}))
    shell = WinShell(callbacks, version="2.4.0", assistant_hotkey="Right Ctrl")
    shell._setup_message = setup_message
    monkeypatch.setattr(shell, "_focusable", lambda hwnd: False)
    monkeypatch.setattr(shell, "_app_window", lambda hwnd: None)
    shell._show_menu()
    return [c.args[3] for c in user32.AppendMenuW.call_args_list if c.args[3]]


def test_setup_menu_offers_the_settings_file(monkeypatch):
    labels = _menu_labels(monkeypatch, "Add your speech provider API key to get started")
    assert "Open settings file…" in labels
    assert not any(text.lower().startswith("sign in") for text in labels)


def test_ready_menu_has_no_sign_in(monkeypatch):
    labels = _menu_labels(monkeypatch, "")
    assert not any(text.lower().startswith("sign in") for text in labels)


def test_kai_off_offers_only_turn_on(monkeypatch):
    labels = _menu_labels(monkeypatch, "", {"assistant": False})
    assert "Turn on Kai…" in labels  # the exact words Kai's hints point to
    assert not any(text.startswith("Ask Kai") for text in labels)
    assert "Show Kai orb" not in labels
    assert "Turn off Kai" not in labels


def test_no_daemon_counts_as_off(monkeypatch):
    labels = _menu_labels(monkeypatch, "", {})
    assert "Turn on Kai…" in labels
    assert not any(text.startswith("Ask Kai") for text in labels)


def test_kai_on_offers_ask_orb_and_turn_off(monkeypatch):
    labels = _menu_labels(monkeypatch, "", {"assistant": True})
    assert "Ask Kai\tHold Right Ctrl" in labels
    assert "Show Kai orb" in labels
    assert "Turn off Kai" in labels
    assert "Turn on Kai…" not in labels


def test_the_tray_label_matches_the_hint():
    from voice_keyboard.assistant.locality import turn_off_hint, turn_on_hint

    assert turn_on_hint("win32").endswith("→ Turn on Kai…")
    assert turn_off_hint("win32").endswith("→ Turn off Kai")
