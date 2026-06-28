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

# Languages the tray supports. Keep in sync with webui/frontend/src/i18n/index.ts.
LANGUAGES = ["en", "cn", "ko", "ja"]
LANGUAGE_LABELS = {"en": "English", "cn": "中文", "ko": "한국어", "ja": "日本語"}
DEFAULT_LANGUAGE = "cn"  # the maintainer is a Chinese user

# Keys = English. Values = {lang: translation}. A key absent from a language
# falls back to the English key (i.e. the dict key itself).
STRINGS: dict[str, dict[str, str]] = {
    # ── menu items ─────────────────────────────────────────────────────
    "Open WebUI": {"cn": "打开 WebUI", "ko": "WebUI 열기", "ja": "WebUIを開く"},
    "Pause queue": {
        "cn": "暂停队列",
        "ko": "대기열 일시정지",
        "ja": "キューを一時停止",
    },
    "Resume queue": {"cn": "恢复队列", "ko": "대기열 재개", "ja": "キューを再開"},
    "Stop active job": {
        "cn": "停止当前作业",
        "ko": "현재 작업 중지",
        "ja": "実行中ジョブを停止",
    },
    "Restart daemon": {
        "cn": "重启守护进程",
        "ko": "데몬 재시작",
        "ja": "デーモンを再起動",
    },
    "Quit": {"cn": "退出", "ko": "종료", "ja": "終了"},
    "Language": {"cn": "语言", "ko": "언어", "ja": "言語"},
    "English": {"cn": "English", "ko": "English", "ja": "English"},
    "Chinese": {"en": "中文", "cn": "中文", "ko": "中文", "ja": "中文"},
    "Korean": {"en": "한국어", "cn": "한국어", "ko": "한국어", "ja": "한국어"},
    "Japanese": {"en": "日本語", "cn": "日本語", "ko": "日本語", "ja": "日本語"},
    # ── tooltip suffixes (the "MonadForge — " brand prefix is added by the
    # caller and never translated) ──────────────────────────────────────
    "daemon not running": {
        "cn": "守护进程未运行",
        "ko": "데몬이 실행되지 않음",
        "ja": "デーモン未起動",
    },
    "running {label}{step}": {
        "cn": "正在运行 {label}{step}",
        "ko": "실행 중 {label}{step}",
        "ja": "実行中 {label}{step}",
    },
    "last job errored: {err}": {
        "cn": "上一个作业出错：{err}",
        "ko": "이전 작업 오류: {err}",
        "ja": "前回ジョブエラー: {err}",
    },
    "idle": {"cn": "空闲", "ko": "대기", "ja": "アイドル"},
    "(paused)": {"cn": "（已暂停）", "ko": " (일시정지)", "ja": " (一時停止)"},
    # The "(step N)" fragment appended inside the running tooltip.
    "(step {n})": {"cn": "（步 {n}）", "ko": " (단계 {n})", "ja": " (ステップ {n})"},
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
