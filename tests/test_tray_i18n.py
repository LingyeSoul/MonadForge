"""Tests for the tray's self-contained i18n + language prefs.

No pystray / GUI needed — these cover the string table lookup, fallback,
formatting, and the on-disk preference round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path


from scripts.tray.i18n import (
    DEFAULT_LANGUAGE,
    LANGUAGES,
    STRINGS,
    language_label,
    normalize_lang,
    tr,
)


# ── tr() lookup / fallback / formatting ──────────────────────────────


def test_tr_translates_known_key_to_chinese():
    assert tr("Open WebUI", "cn") == "打开 WebUI"


def test_tr_falls_back_to_english_key_for_missing_translation():
    # "English" has no "cn" entry → falls back to the English key.
    assert tr("English", "cn") == "English"


def test_tr_falls_back_to_raw_label_for_unknown_key():
    assert tr("Completely Unknown", "cn") == "Completely Unknown"


def test_tr_formats_placeholders():
    out = tr("running {label}{step}", "cn", label="lora", step="（步 450）")
    assert out == "正在运行 lora（步 450）"


def test_tr_leaves_unknown_placeholders_verbatim():
    # A placeholder not in fmt stays as {name} rather than raising KeyError.
    out = tr("running {label}{step}", "cn", label="lora")
    assert "{step}" in out


def test_tr_no_fmt_args_returns_plain_translation():
    assert tr("Quit", "cn") == "退出"


def test_every_menu_string_has_cn_translation():
    """All user-facing menu keys must be localized for the default (cn) locale."""
    menu_keys = [
        "Open WebUI",
        "Pause queue",
        "Resume queue",
        "Stop active job",
        "Restart daemon",
        "Quit",
        "Language",
        # "English" / "Chinese" are intentionally left as-is (language names).
    ]
    for key in menu_keys:
        assert key in STRINGS, f"missing string key: {key!r}"
        assert "cn" in STRINGS[key], f"missing cn translation for {key!r}"
        assert STRINGS[key]["cn"], f"empty cn translation for {key!r}"


# ── language helpers ─────────────────────────────────────────────────


def test_default_language_is_chinese():
    assert DEFAULT_LANGUAGE == "cn"


def test_languages_include_all_supported():
    assert set(LANGUAGES) == {"en", "cn", "ko", "ja"}


def test_normalize_lang_accepts_supported():
    assert normalize_lang("en") == "en"
    assert normalize_lang("cn") == "cn"


def test_normalize_lang_falls_back_for_unsupported():
    assert normalize_lang("fr") == "en"
    assert normalize_lang("xyz") == "en"


def test_normalize_lang_uses_default_for_empty():
    assert normalize_lang(None) == DEFAULT_LANGUAGE
    assert normalize_lang("") == DEFAULT_LANGUAGE


def test_language_label():
    assert language_label("cn") == "中文"
    assert language_label("en") == "English"
    assert language_label("??") == "??"


# ── prefs persistence (disk round-trip) ──────────────────────────────


def test_prefs_save_and_load_roundtrip(tmp_path: Path, monkeypatch):
    from scripts.tray import prefs

    # Redirect the prefs file into the tmp dir so the test is hermetic.
    prefs_file = tmp_path / "tray-prefs.json"
    monkeypatch.setattr(prefs, "PREFS_DIR", tmp_path)
    monkeypatch.setattr(prefs, "PREFS_FILE", prefs_file)

    prefs.save_language("cn")
    assert prefs_file.is_file()
    assert prefs.load_language() == "cn"

    prefs.save_language("en")
    assert prefs.load_language() == "en"


def test_prefs_load_returns_default_when_missing(tmp_path: Path, monkeypatch):
    from scripts.tray import prefs

    monkeypatch.setattr(prefs, "PREFS_DIR", tmp_path)
    monkeypatch.setattr(prefs, "PREFS_FILE", tmp_path / "tray-prefs.json")
    assert prefs.load_language() == DEFAULT_LANGUAGE


def test_prefs_load_returns_default_when_corrupt(tmp_path: Path, monkeypatch):
    from scripts.tray import prefs

    prefs_file = tmp_path / "tray-prefs.json"
    prefs_file.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(prefs, "PREFS_DIR", tmp_path)
    monkeypatch.setattr(prefs, "PREFS_FILE", prefs_file)
    assert prefs.load_language() == DEFAULT_LANGUAGE


def test_prefs_ignores_unsupported_language_on_load(tmp_path: Path, monkeypatch):
    from scripts.tray import prefs

    prefs_file = tmp_path / "tray-prefs.json"
    prefs_file.write_text(json.dumps({"language": "klingon"}), encoding="utf-8")
    monkeypatch.setattr(prefs, "PREFS_DIR", tmp_path)
    monkeypatch.setattr(prefs, "PREFS_FILE", prefs_file)
    assert prefs.load_language() == DEFAULT_LANGUAGE


def test_prefs_save_ignores_unsupported_language(tmp_path: Path, monkeypatch):
    from scripts.tray import prefs

    prefs_file = tmp_path / "tray-prefs.json"
    monkeypatch.setattr(prefs, "PREFS_DIR", tmp_path)
    monkeypatch.setattr(prefs, "PREFS_FILE", prefs_file)
    prefs.save_language("klingon")  # no-op
    assert not prefs_file.exists()



# --- TrayApp: language switch reflows the menu + updates the icon ---


def test_set_language_reflows_menu_and_updates_icon(tmp_path, monkeypatch):
    """Switching language must swap the icon.menu + call update_menu() so
    the next open shows the new language (Win32 caches the displayed menu —
    update_menu rebuilds the HMENU for next-show, swapping the descriptor
    forces a clean rebuild)."""
    from scripts.tray import prefs
    from scripts.tray.app import TrayApp

    prefs_file = tmp_path / "tray-prefs.json"
    prefs_file.write_text('{"language": "cn"}', encoding="utf-8")
    monkeypatch.setattr(prefs, "PREFS_DIR", tmp_path)
    monkeypatch.setattr(prefs, "PREFS_FILE", prefs_file)

    class _StubIcon:
        def __init__(self):
            self.menu = None
            self.updates = 0

        def update_menu(self):
            self.updates += 1

    t = TrayApp()
    t._icon = _StubIcon()
    t._icon.menu = t._build_menu()

    initial_menu = t._icon.menu
    t._set_language("en")
    assert t._lang == "en"
    assert t._icon.menu is not initial_menu, "icon.menu should be replaced"
    assert t._icon.updates >= 1
    labels = [it.text for it in t._icon.menu.items]
    assert "Open WebUI" in labels
    assert prefs.load_language() == "en"


def test_set_language_is_idempotent(tmp_path, monkeypatch):
    """Calling _set_language with the current lang is a no-op (no menu swap)."""
    from scripts.tray import prefs
    from scripts.tray.app import TrayApp

    prefs_file = tmp_path / "tray-prefs.json"
    prefs_file.write_text('{"language": "en"}', encoding="utf-8")
    monkeypatch.setattr(prefs, "PREFS_DIR", tmp_path)
    monkeypatch.setattr(prefs, "PREFS_FILE", prefs_file)

    class _StubIcon:
        def __init__(self):
            self.menu = "unchanged"
            self.updates = 0

        def update_menu(self):
            self.updates += 1

    t = TrayApp()
    t._icon = _StubIcon()
    t._set_language("en")
    assert t._icon.menu == "unchanged"
    assert t._icon.updates == 0


def test_set_language_rejects_unsupported(tmp_path, monkeypatch):
    from scripts.tray import prefs
    from scripts.tray.app import TrayApp

    prefs_file = tmp_path / "tray-prefs.json"
    prefs_file.write_text('{"language": "cn"}', encoding="utf-8")
    monkeypatch.setattr(prefs, "PREFS_DIR", tmp_path)
    monkeypatch.setattr(prefs, "PREFS_FILE", prefs_file)

    t = TrayApp()
    t._set_language("fr")  # not in LANGUAGES
    assert t._lang == "cn"


def _tray_for_tick():
    from scripts.tray.app import TrayApp

    tray = TrayApp()
    tray._icon = None
    tray._last_error = None
    return tray


def test_tick_ignores_stale_active_job_that_is_already_terminal(monkeypatch):
    from webui.services import daemon_client as daemon_module

    class Client:
        def health_sync(self):
            return {"paused": False, "active_job": "job-1"}

        def get_job_sync(self, _job_id):
            return {
                "id": "job-1",
                "state": "done",
                "pid": 123,
                "create_time": 456.0,
            }

    monkeypatch.setattr(daemon_module, "daemon_client", Client())
    tray = _tray_for_tick()
    tray._state = "running"
    tray._active_job = {"id": "previous-job", "state": "running"}

    tray._tick()

    assert tray._state == "idle"
    assert tray._active_job is None


def test_tick_requires_live_process_identity_before_showing_running(monkeypatch):
    from scripts.daemon import proc
    from webui.services import daemon_client as daemon_module

    class Client:
        def health_sync(self):
            return {"paused": False, "active_job": "job-1"}

        def get_job_sync(self, _job_id):
            return {
                "id": "job-1",
                "state": "running",
                "pid": 123,
                "create_time": 456.0,
                "method": "lora",
            }

    monkeypatch.setattr(daemon_module, "daemon_client", Client())
    monkeypatch.setattr(proc, "is_alive", lambda pid, create_time: False)
    tray = _tray_for_tick()

    tray._tick()

    assert tray._state == "idle"
    assert tray._active_job is None

    monkeypatch.setattr(proc, "is_alive", lambda pid, create_time: True)
    tray._tick()
    assert tray._state == "running"
    assert tray._active_job["id"] == "job-1"
