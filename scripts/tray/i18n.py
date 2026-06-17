"""Self-contained i18n for the MonadForge tray.

The tray keeps its own language choice (default Chinese) — it does **not**
follow the WebUI, so there's no cross-process sync: the choice lives in
``output/daemon/tray-prefs.json`` and is read/written only by the tray.

Keys are the English strings; each value is a per-language dict. ``tr()``
looks up the key, falls back to the English key (and then the raw label) if a
translation is missing, and applies ``str.format`` for the small handful of
templated tooltips.
"""

from __future__ import annotations

from typing import Optional

# The two languages the WebUI frontend exposes. Keep in sync with
# webui/frontend/src/i18n/index.ts.
LANGUAGES = ["en", "cn"]
LANGUAGE_LABELS = {"en": "English", "cn": "中文"}
DEFAULT_LANGUAGE = "cn"  # the maintainer is a Chinese user

# Keys = English. Values = {lang: translation}. A key absent from a language
# falls back to the English key (i.e. the dict key itself).
STRINGS: dict[str, dict[str, str]] = {
    # ── menu items ─────────────────────────────────────────────────────
    "Open WebUI": {"cn": "打开 WebUI"},
    "Pause queue": {"cn": "暂停队列"},
    "Resume queue": {"cn": "恢复队列"},
    "Stop active job": {"cn": "停止当前作业"},
    "Restart daemon": {"cn": "重启守护进程"},
    "Quit": {"cn": "退出"},
    "Language": {"cn": "语言"},
    "English": {"cn": "English"},
    "Chinese": {"en": "中文", "cn": "中文"},
    # ── tooltip suffixes (the "MonadForge — " brand prefix is added by the
    # caller and never translated) ──────────────────────────────────────
    "daemon not running": {"cn": "守护进程未运行"},
    "running {label}{step}": {"cn": "正在运行 {label}{step}"},
    "last job errored: {err}": {"cn": "上一个作业出错：{err}"},
    "idle": {"cn": "空闲"},
    "(paused)": {"cn": "（已暂停）"},
    # The "(step N)" fragment appended inside the running tooltip.
    "(step {n})": {"cn": "（步 {n}）"},
}


def tr(key: str, lang: str, **fmt: object) -> str:
    """Translate ``key`` for ``lang`` and apply ``str.format(**fmt)``.

    Lookup order: ``STRINGS[key][lang]`` → ``key`` (the English key) → ``key``
    unchanged. Missing-format-field safety: ``str.format_map`` with a default
    dict so an unknown placeholder doesn't raise.

    ``key`` (not ``label``) is the param name so callers can pass a ``label=``
    format field without colliding with the positional argument.
    """
    entry = STRINGS.get(key, {})
    text: str = entry.get(lang) or entry.get("en") or key
    if not fmt:
        return text
    return text.format_map(_SafeDict(fmt))


class _SafeDict(dict):
    """``str.format_map`` dict that leaves unknown placeholders verbatim."""

    def __missing__(self, key: str) -> str:  # noqa: D401
        return "{" + key + "}"


def language_label(lang: str) -> str:
    """Display name for a language code (for the Language submenu)."""
    return LANGUAGE_LABELS.get(lang, lang)


def normalize_lang(lang: Optional[str]) -> str:
    """Coerce a stored/persisted value to a supported code (fallback en)."""
    if lang in LANGUAGES:
        return lang
    return "en" if lang not in (None, "") else DEFAULT_LANGUAGE
