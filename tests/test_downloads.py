"""The model catalog and the ``make download-*`` targets that drive it.

The behaviour worth pinning is the *wiring*: rows land where loaders look, every
target resolves to real ids, the GUI's buttons run registered tasks, and a
present file is not re-fetched. Downloading itself is ``anime_tools``' tested
code — mock at the ``Asset.fetch`` boundary rather than at the hub.
"""

from __future__ import annotations

import urllib.error
from io import BytesIO
from pathlib import Path

import pytest

from scripts.tasks import _legacy_downloads as legacy
from scripts.tasks import downloads

from library import downloads as DL
from library.runtime.ms_download import fetch_file as real_ms_fetch_file


def test_the_two_catalog_halves_have_disjoint_ids():
    """``by_id`` merges them; a collision would silently shadow a row."""
    trainer = {a.id for a in DL.catalog()}
    curation = {a.id for a in DL.curation_catalog()}
    assert not (trainer & curation)
    assert len(DL.by_id()) == len(trainer) + len(curation)


def test_every_group_and_the_default_set_name_real_rows():
    """A typo in ``GROUPS`` is a make target that downloads nothing."""
    known = set(DL.by_id())
    for group, ids in DL.GROUPS.items():
        assert set(ids) <= known, group
    assert set(DL.DEFAULT_SET) <= known
    assert "mit" not in DL.GROUPS


def test_every_alias_and_pack_resolves():
    """``resolve`` precedence is alias → pack → row; every token of the first
    two kinds must land on rows, and the two ``pe`` meanings must differ."""
    for name in (*DL.GROUP_ALIASES, *DL.by_pack()):
        assert DL.resolve([name]), name
    # The make target `pe` is both towers; the pack `pe` is PE-Core alone.
    assert [a.id for a in DL.resolve(["pe"])] == ["pe_core", "pe_spatial"]
    assert [a.id for a in DL.by_pack()["pe"]] == ["pe_core"]
    # `tagger` the alias (checkpoint only) shadows `tagger` the pack.
    assert [a.id for a in DL.resolve(["tagger"])] == ["tagger"]
    assert len(DL.by_pack()["tagger"]) > 1


def test_every_visible_row_belongs_to_a_listed_pack():
    """The GUI renders by pack: a row with a pack id that is not in ``PACKS``
    would silently vanish from both tabs."""
    pack_ids = {p.id for p in DL.PACKS}
    for a in DL.full_catalog():
        assert a.pack in pack_ids, a.id
    assert DL.PACKS[: len(DL.TRAINER_PACKS)] == DL.TRAINER_PACKS
    assert [p.id for p in DL.TRAINER_PACKS] == [
        "anima",
        "pe",
        "cjk",
        "moss-base",
        "moss-sft",
    ]
    assert not (set(DL.HIDDEN_PACKS) & pack_ids)


def test_hidden_pack_rows_are_not_on_the_trainer_side(monkeypatch):
    """A pack in ``HIDDEN_PACKS`` is neither listed, resolved nor downloadable
    from here. The set is empty since the package dropped the MIT text
    segmenter (``text_mask``) itself — pin that the rows are really gone and
    that the seam still filters when a pack is named."""
    from library.curation_compat.downloads import catalog as package_catalog

    assert DL.HIDDEN_PACKS == ()
    assert not {"mit_text", "ctd_onnx"} & {a.id for a in package_catalog()}
    with pytest.raises(KeyError):
        DL.resolve(["mit_text"])
    with pytest.raises(KeyError):
        DL.resolve(["text_mask"])

    monkeypatch.setattr(DL, "HIDDEN_PACKS", ("grouping",))
    hidden = {a.id for a in package_catalog() if a.pack == "grouping"}
    assert hidden == {"pe_spatial"}
    assert not (hidden & {a.id for a in DL.curation_catalog()})


def test_the_first_run_set_is_the_mandatory_set():
    """The first-run set includes CJK support while gated masking stays opt-in."""
    assert "sam3" not in DL.DEFAULT_SET
    assert "vocab_pack" in DL.DEFAULT_SET
    assert "mit_text" not in DL.DEFAULT_SET
    assert "mit_text" not in DL.by_id()
    # …and the rows a default preprocess actually needs are.
    assert {"anima_dit", "anima_te", "anima_vae", "tagger"} <= set(DL.DEFAULT_SET)


def test_resolve_expands_groups_and_dedupes_in_catalog_order():
    ids = [a.id for a in DL.resolve(["pe", "pe_core", "anima"])]
    assert ids == ["anima_dit", "anima_te", "anima_vae", "pe_core", "pe_spatial"]


def test_resolve_expands_a_pack_id():
    """``make download-model ARGS=ocr`` is the pack, in catalog order."""
    ids = [a.id for a in DL.resolve(["ocr"])]
    assert ids == [a.id for a in DL.by_pack()["ocr"]]
    assert ids and all(DL.by_id()[i].pack == "ocr" for i in ids)


def test_by_pack_keeps_pack_order_and_drops_empty_packs():
    grouped = DL.by_pack()
    assert list(grouped) == [p.id for p in DL.PACKS if p.id in grouped]
    assert all(rows for rows in grouped.values())
    trainer_only = DL.by_pack(DL.catalog())
    assert list(trainer_only) == ["anima", "pe", "cjk", "moss-base", "moss-sft"]


def test_resolve_raises_naming_the_unknown_token():
    with pytest.raises(KeyError) as exc:
        DL.resolve(["sam3", "nope"])
    assert "nope" in str(exc.value)


def test_rows_land_where_the_loaders_look():
    """The catalog's one rule: a path a loader owns separately is a Download
    button that writes where the loader will not look."""
    from library.anima import vocab_pack
    from library.vision import encoders

    rows = DL.by_id()
    assert encoders._default_pe_model_id() == str(DL.default_pe_core_path())
    assert rows["pe_core"].dest / DL.PE_CORE_FILENAME == DL.default_pe_core_path()

    assert vocab_pack.PACK_REPO == rows["vocab_pack"].repo
    assert vocab_pack.DEFAULT_PACK_PREFIX.startswith(f"models/{DL.VOCAB_PACK_DIR}/")
    assert rows["vocab_pack"].dest == DL.default_vocab_pack_dir()
    from library.env import anima_home, resolve_under_home

    assert (
        resolve_under_home(vocab_pack.DEFAULT_PACK_PREFIX)
        == DL.default_vocab_pack_prefix()
        == anima_home() / "models" / DL.VOCAB_PACK_DIR / DL.VOCAB_PACK_STEM
    )


def test_moss_rows_land_where_library_moss_looks():
    """``library/moss`` reads every part from ``<variant>/<subfolder>``; the TE,
    VAE and tokenizer only ever from Base, so SFT ships its DiT alone."""
    from library.moss import requests as moss

    rows = DL.by_id()
    for variant in moss.VARIANTS:
        root = moss.variant_dir(None, variant)
        assert rows[f"moss_{variant}_root"].dest == root
        assert rows[f"moss_{variant}_transformer"].dest == root / "transformer"
    base = moss.variant_dir(None, "base")
    for part in ("text_encoder", "vae", "tokenizer", "scheduler"):
        assert rows[f"moss_base_{part}"].dest == base / part
        assert f"moss_sft_{part}" not in rows
    assert not set(DL.GROUPS["moss-base"]) & set(DL.DEFAULT_SET)
    assert not set(DL.GROUPS["moss-sft"]) & set(DL.DEFAULT_SET)


def test_anima_rows_land_on_the_model_config_defaults():
    """``configs/model.toml`` owns these shipped model paths after migration."""
    import tomllib
    from library.env import anima_home

    cfg = tomllib.loads((anima_home() / "configs" / "model.toml").read_text("utf-8"))
    rows = DL.by_id()
    configured = {
        "anima_dit": cfg["pretrained_model_name_or_path"],
        "anima_te": cfg["qwen3"],
        "anima_vae": cfg["vae"],
    }
    for row_id, rel in configured.items():
        row = rows[row_id]
        landed = row.dest / row.files[0].rsplit("/", 1)[-1]
        assert landed == anima_home() / rel, row_id
    # Model-path migration leaves the CJK pack opt-in. Its advertised enable
    # prefix and its loader must still agree with both downloaded files.
    from library.anima.vocab_pack import DEFAULT_PACK_PREFIX

    assert DEFAULT_PACK_PREFIX == f"models/{DL.VOCAB_PACK_DIR}/{DL.VOCAB_PACK_STEM}"
    assert anima_home() / DEFAULT_PACK_PREFIX == DL.default_vocab_pack_prefix()
    for name in rows["vocab_pack"].files:
        assert (rows["vocab_pack"].dest / name).with_suffix("") == (
            DL.default_vocab_pack_prefix()
        )


def test_prune_empty_drops_the_split_files_scaffolding(tmp_path):
    """``hf`` mirrors the repo layout under ``--local-dir``; after the flatten
    the empty tree must go, but a real file must not."""
    (tmp_path / "split_files" / "diffusion_models").mkdir(parents=True)
    (tmp_path / "keep").mkdir()
    (tmp_path / "keep" / "f.bin").write_bytes(b"x")

    DL._prune_empty(tmp_path)

    assert not (tmp_path / "split_files").exists()
    assert (tmp_path / "keep" / "f.bin").exists()


def test_fetch_skips_an_installed_row(monkeypatch):
    """Idempotency contract (GH #21): a re-run verifies, it does not re-fetch.

    Rows that move files out of ``hf``'s ``--local-dir`` layout would otherwise
    re-pull the whole repo, because the hub no longer sees them where it looks.
    """
    asset = DL.by_id()["sam3"]
    monkeypatch.setattr(type(asset), "installed", property(lambda _self: True))
    monkeypatch.setattr(
        type(asset), "fetch", lambda *_a, **_k: pytest.fail("re-fetched")
    )

    assert DL.fetch(asset, log=lambda _m: None) is False


def test_fetch_all_continues_past_a_failure_and_reports_it():
    """One gated repo without granted access must not abort the row beside it."""
    fetched: list[str] = []

    class _Row:
        def __init__(self, title, ok):
            self.title, self._ok, self.installed = title, ok, False
            self.repo = self.location = "-"
            self.dest = None

        def fetch(self, _log):
            if not self._ok:
                raise FileNotFoundError("gated")
            fetched.append(self.title)

    failed = DL.fetch_all(
        [_Row("a", True), _Row("b", False), _Row("c", True)], log=lambda _m: None
    )

    assert fetched == ["a", "c"]
    assert failed == ["b"]


def test_every_download_target_resolves(monkeypatch):
    """Each ``make download-<x>`` must name rows that exist — the failure mode
    is a target that prints nothing and exits 0."""
    import tasks

    picked: list[list[str]] = []
    monkeypatch.setattr(
        DL,
        "fetch_all",
        lambda assets, **_kw: picked.append([a.id for a in assets]) or [],
    )
    targets = [
        n
        for n in tasks.COMMANDS
        if n.startswith("download-")
        and n
        not in (
            "download-anima-variant",
            "download-list",
            "download-model",
            "download-ms",
            "download-mit",
        )
    ]
    assert len(targets) >= 8
    # These two preserved commands are not catalog selectors. Their own
    # argument/transport tests below cover them at the legacy boundary.
    assert tasks.COMMANDS["download-mit"][0].resolve() is legacy.cmd_download_mit
    assert tasks.COMMANDS["download-ms"][0].resolve() is legacy.cmd_download_ms
    for name in targets:
        tasks.COMMANDS[name][0]([])
    assert all(ids for ids in picked)


def test_download_model_accepts_pack_ids(monkeypatch):
    """The GUI's pack buttons and ``make download-model ARGS=ocr`` go through the
    same front door; a pack id must expand there, not only in ``resolve``."""
    import tasks

    picked: list[list[str]] = []
    monkeypatch.setattr(
        DL,
        "fetch_all",
        lambda assets, **_kw: picked.append([a.id for a in assets]) or [],
    )
    tasks.COMMANDS["download-model"][0](["ocr", "anima"])
    assert picked == [[a.id for a in DL.resolve(["anima", "ocr"])]]


def test_models_dialog_rows_come_from_the_catalog():
    """Both tabs render Assets and run ``download-model <id>``; the id has to be
    one the task can resolve, and every labelled row has to be a real one."""
    pytest.importorskip("PySide6")
    import tasks
    from gui.dialogs.system import _TITLE_KEYS

    assert "download-model" in tasks.COMMANDS
    assert set(_TITLE_KEYS) <= set(DL.by_id())


def test_the_two_tabs_show_the_two_catalog_halves():
    """One modal, one QProcess, one log — the split is a tab, so neither list
    grows long enough to push the log pane off the dialog."""
    import os

    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QGroupBox

    from gui.dialogs.system import ModelsDialog, _pack_title

    app = QApplication.instance() or QApplication([])
    assert app is not None
    dlg = ModelsDialog()
    try:
        assert dlg.tabs.count() == 2
        anima, curation = dlg._panels
        assert [a.id for a in anima.assets()] == [a.id for a in DL.catalog()]
        assert [a.id for a in curation.assets()] == [
            a.id for a in DL.curation_catalog()
        ]
        assert "mit_text" not in {a.id for a in curation.assets()}
        # Rows are grouped by pack: one QGroupBox per pack the tab's rows
        # populate, in PACKS order, each with a pack-level button that names
        # exactly that pack's rows.
        for panel in dlg._panels:
            expected = DL.by_pack(panel.assets())
            assert [pid for pid, _ids, _b in panel._packs] == list(expected)
            for pid, ids, _btn in panel._packs:
                assert ids == tuple(a.id for a in expected[pid])
            boxes = panel.findChildren(QGroupBox)
            assert [b.title() for b in boxes] == [
                _pack_title(DL.PACK_BY_ID[pid]) for pid in expected
            ]
        assert [pid for pid, _i, _b in anima._packs] == [
            "anima",
            "pe",
            "cjk",
            "moss-base",
            "moss-sft",
        ]
        # Busy disables every button on *both* tabs: the dialog runs one job.
        dlg._set_busy(True)
        assert not any(b.isEnabled() for p in dlg._panels for _a, _s, b in p._rows)
        assert not any(b.isEnabled() for p in dlg._panels for _p, _i, b in p._packs)
        assert not any(p.all_btn.isEnabled() for p in dlg._panels)
        dlg._set_busy(False)
        assert all(b.isEnabled() for p in dlg._panels for _p, _i, b in p._packs)
    finally:
        dlg.close()


def test_every_title_key_exists_in_every_language():
    """A missing key silently falls back to English — worst for the KR/JA/ZH
    users who are most of the base."""
    pytest.importorskip("PySide6")
    from gui.i18n import TRANSLATIONS
    from gui.dialogs.system import _PACK_KEYS, _TITLE_KEYS

    extra = (
        "curation_models_intro",
        "models_tab_anima",
        "models_tab_curation",
        "models_download_missing",
        "models_all_installed",
        "models_used_by",
        "models_download_pack",
        "models_redownload_pack",
    )
    pack_keys = [k for key in _PACK_KEYS.values() for k in (key, f"{key}_desc")]
    for lang, table in TRANSLATIONS.items():
        for key in (*_TITLE_KEYS.values(), *pack_keys, *extra):
            assert key in table, f"{lang}: {key}"
        assert "model_mit" not in table, lang
    # Every pack the trainer shows has a translated title (the fallback is the
    # catalog's English title, which the KR/JA/ZH users should not see).
    assert set(_PACK_KEYS) == {p.id for p in DL.PACKS}


@pytest.fixture(autouse=True)
def forbid_real_downloads(monkeypatch):
    """Each test must opt into a fake transport; restored tests stay offline."""
    from library.runtime import ms_download
    from library.curation_compat import _hf

    def unexpected(*_args, **_kwargs):
        pytest.fail("A download test reached a real transport or child command")

    monkeypatch.delenv("ANIMA_DOWNLOAD_SOURCE", raising=False)
    monkeypatch.setattr("urllib.request.urlopen", unexpected)
    monkeypatch.setattr(_hf, "hf_download", unexpected)
    monkeypatch.setattr(ms_download, "snapshot_download", unexpected)
    monkeypatch.setattr(ms_download, "fetch_file", unexpected)
    monkeypatch.setattr(legacy, "run", unexpected)
    monkeypatch.setattr(downloads, "run", unexpected)


# ModelScope and legacy tag-table coverage retained from the local merge parent.


def test_danbooru_tags_download_url_points_to_source_repo():
    assert legacy.DANBOORU_TAGS_URLS == (
        "https://raw.githubusercontent.com/Localsmile/danbooru_KR_wiki_tag_search/main/danbooru_tags_classified.csv",
    )


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self) -> bytes:
        return self._payload


def test_download_danbooru_tags_writes_models_file(tmp_path, monkeypatch):
    dest = tmp_path / "models" / "danbooru_tags_classified.csv"
    monkeypatch.setattr(legacy, "DANBOORU_TAGS_PATH", dest)
    monkeypatch.setattr(legacy, "DANBOORU_TAGS_EN_PATH", dest.with_suffix(".en.csv"))
    monkeypatch.setattr(
        legacy, "DANBOORU_TAGS_URLS", ("https://example.test/tags.csv",)
    )
    monkeypatch.setattr(
        legacy.urllib.request,
        "urlopen",
        lambda _req, timeout=60: _FakeResponse(
            b"name,category,post_count,description\n1girl,0,1,test\n"
        ),
    )

    builds = []
    monkeypatch.setattr(legacy, "run", lambda argv, **kwargs: builds.append(argv))
    legacy.cmd_download_danbooru_tags([])

    assert builds and "scripts.anima_tagger.build_english_tag_csv" in builds[0]
    assert dest.read_text(encoding="utf-8").startswith("name,category")


def test_download_danbooru_tags_skips_existing_without_force(tmp_path, monkeypatch):
    dest = tmp_path / "models" / "danbooru_tags_classified.csv"
    dest.parent.mkdir(parents=True)
    dest.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(legacy, "DANBOORU_TAGS_PATH", dest)
    called = False

    def _fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True
        return _FakeResponse(BytesIO().read())

    monkeypatch.setattr(legacy.urllib.request, "urlopen", _fail_if_called)

    monkeypatch.setattr(legacy, "DANBOORU_TAGS_EN_PATH", dest)
    legacy.cmd_download_danbooru_tags([])

    assert not called
    assert dest.read_text(encoding="utf-8") == "existing"


def test_download_danbooru_tags_failure_names_source_repo(tmp_path, monkeypatch):
    dest = tmp_path / "models" / "danbooru_tags_classified.csv"
    monkeypatch.setattr(legacy, "DANBOORU_TAGS_PATH", dest)
    monkeypatch.setattr(
        legacy, "DANBOORU_TAGS_URLS", ("https://example.test/tags.csv",)
    )
    monkeypatch.setattr(
        legacy.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(urllib.error.URLError("nope")),
    )

    with pytest.raises(SystemExit) as exc:
        legacy.cmd_download_danbooru_tags([])

    assert "Localsmile/danbooru_KR_wiki_tag_search" in str(exc.value)


def _no_hf_run(*_args, **_kwargs):
    raise AssertionError("hf CLI must not run under --source modelscope")


def test_anima_modelscope_source_uses_snapshot_and_moves(tmp_path, monkeypatch):
    monkeypatch.setattr(legacy, "ROOT", tmp_path)
    monkeypatch.setattr(legacy, "run", _no_hf_run)
    calls: list[tuple[str, str, tuple[str, ...]]] = []

    def fake_snapshot(repo_id, local_dir, *, filenames=None, include=None, force=False):
        calls.append((repo_id, str(local_dir), tuple(filenames or ())))
        models = Path(local_dir)
        for name in filenames or ():
            p = models / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")

    monkeypatch.setattr(legacy.ms_download, "snapshot_download", fake_snapshot)

    legacy.cmd_download_anima(["--source", "modelscope"])

    models = tmp_path / "models"
    assert calls and calls[0][0] == "circlestone-labs/Anima"
    assert "split_files/diffusion_models/anima-base-v1.0.safetensors" in calls[0][2]
    assert (models / "diffusion_models" / "anima-base-v1.0.safetensors").exists()
    assert (models / "text_encoders" / "qwen_3_06b_base.safetensors").exists()
    assert (models / "vae" / "qwen_image_vae.safetensors").exists()
    assert not (models / "split_files").exists()


def test_sam3_modelscope_env_routes_without_flag(tmp_path, monkeypatch):
    """``ANIMA_DOWNLOAD_SOURCE=modelscope`` alone must select the ms branch —
    no ``--source`` flag needed."""
    monkeypatch.setattr(legacy, "ROOT", tmp_path)
    monkeypatch.setenv("ANIMA_DOWNLOAD_SOURCE", "modelscope")
    monkeypatch.setattr(legacy, "run", _no_hf_run)
    monkeypatch.setattr(
        legacy.ms_download,
        "snapshot_download",
        lambda *a, **k: None,
    )

    legacy.cmd_download_sam3([])


def test_anima_modelscope_env_routes_without_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(legacy, "ROOT", tmp_path)
    monkeypatch.setenv("ANIMA_DOWNLOAD_SOURCE", "modelscope")
    monkeypatch.setattr(legacy, "run", _no_hf_run)
    called = {}

    def fake_snapshot(repo_id, local_dir, *, filenames=None, include=None, force=False):
        called["repo"] = repo_id
        called["force"] = force

    monkeypatch.setattr(legacy.ms_download, "snapshot_download", fake_snapshot)

    legacy.cmd_download_anima([])

    assert called["repo"] == "circlestone-labs/Anima"
    assert called["force"] is False


def test_anima_modelscope_force_passes_through(tmp_path, monkeypatch):
    """``--force`` must reach the ms snapshot instead of being dropped."""
    monkeypatch.setattr(legacy, "ROOT", tmp_path)
    monkeypatch.setattr(legacy, "run", _no_hf_run)
    called = {}

    def fake_snapshot(repo_id, local_dir, *, filenames=None, include=None, force=False):
        called["force"] = force

    monkeypatch.setattr(legacy.ms_download, "snapshot_download", fake_snapshot)

    legacy.cmd_download_anima(["--source", "modelscope", "--force"])

    assert called["force"] is True


def test_sam3_modelscope_force_passes_through(tmp_path, monkeypatch):
    monkeypatch.setattr(legacy, "ROOT", tmp_path)
    monkeypatch.setattr(legacy, "run", _no_hf_run)
    called = {}

    def fake_snapshot(repo_id, local_dir, *, force=False, **_k):
        called["force"] = force

    monkeypatch.setattr(legacy.ms_download, "snapshot_download", fake_snapshot)

    legacy.cmd_download_sam3(["--source", "modelscope", "--force"])

    assert called["force"] is True


def test_pe_core_modelscope_force_passes_through(tmp_path, monkeypatch):
    monkeypatch.setattr(legacy, "ROOT", tmp_path)
    monkeypatch.setattr(legacy, "run", _no_hf_run)
    called = {}

    def fake_fetch_file(*, ms_repo, filename, local_dir, what="asset", force=False):
        called["force"] = force
        return Path(local_dir) / filename

    monkeypatch.setattr(legacy.ms_download, "fetch_file", fake_fetch_file)

    legacy.cmd_download_pe(["--source", "modelscope", "--force"])

    assert called["force"] is True


def test_anima_default_source_still_uses_hf(tmp_path, monkeypatch):
    monkeypatch.setattr(legacy, "ROOT", tmp_path)
    seen: list[list[str]] = []
    monkeypatch.setattr(legacy, "run", lambda cmd, **_k: seen.append(cmd))

    legacy.cmd_download_anima([])

    assert seen and seen[0][:3] == ["hf", "download", "circlestone-labs/Anima"]


def test_pe_spatial_modelscope_without_mirror_skips(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(legacy, "ROOT", tmp_path)
    monkeypatch.setattr(legacy, "run", _no_hf_run)

    legacy.cmd_download_pe_spatial(["--source", "modelscope"])

    out = capsys.readouterr().out
    assert "no ModelScope mirror" in out
    assert not (tmp_path / "models" / "pe" / "PE-Spatial-B16-512.pt").exists()


def test_download_ms_requires_repo_id():
    with pytest.raises(SystemExit, match="usage"):
        legacy.cmd_download_ms([])
    with pytest.raises(SystemExit, match="org/name"):
        legacy.cmd_download_ms(["justname"])


def test_download_ms_rejects_unknown_option(tmp_path, monkeypatch):
    """A typo'd flag must fail loudly, not silently become a repo id/file."""
    monkeypatch.setattr(legacy, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="unknown option --localdir"):
        legacy.cmd_download_ms(["org/name", "--localdir", "x"])


def test_download_ms_passes_repo_and_default_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(legacy, "ROOT", tmp_path)
    calls: dict = {}

    def fake_snapshot(repo_id, local_dir, *, include=None, force=False, **_k):
        calls["args"] = (repo_id, str(local_dir), include, force)

    monkeypatch.setattr(legacy.ms_download, "snapshot_download", fake_snapshot)

    legacy.cmd_download_ms(["org/name", "--include", "*.safetensors"])

    repo_id, local_dir, include, force = calls["args"]
    assert repo_id == "org/name"
    assert local_dir == str(tmp_path / "models" / "name")
    assert include == ["*.safetensors"]
    assert force is False

    legacy.cmd_download_ms(["org/other", "--local-dir", str(tmp_path / "x")])
    assert calls["args"][1] == str(tmp_path / "x")


@pytest.fixture
def catalog_mirror_transport(tmp_path, monkeypatch):
    """Exercise real mirror selection and filename lookup with local tiny bytes."""
    from library.runtime import ms_download

    monkeypatch.setenv("ANIMA_HOME", str(tmp_path))
    monkeypatch.setenv("ANIME_TOOLS_HOME", str(tmp_path))
    monkeypatch.setattr(ms_download, "fetch_file", real_ms_fetch_file)
    listing = {}
    for asset in DL.full_catalog():
        mirror = ms_download.mirror_repo(asset.repo)
        if mirror:
            listing.setdefault(mirror, []).extend(
                (f"{asset.subfolder}/{name}" if asset.subfolder else name, 4)
                for name in asset.files
            )
    monkeypatch.setattr(ms_download, "repo_files", lambda repo, revision: listing[repo])
    calls = []

    def fake_download(url, dest, **kwargs):
        calls.append((url, dest, kwargs))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"tiny")
        return dest

    monkeypatch.setattr(ms_download, "download_file", fake_download)
    return tmp_path, calls


def test_catalog_anima_modelscope_lands_on_loader_paths(catalog_mirror_transport):
    root, calls = catalog_mirror_transport
    downloads.cmd_download_anima(["--source", "modelscope"])
    assets = DL.resolve(["anima"])
    assert len(calls) == len(assets) == 3
    for asset in assets:
        assert asset.installed
        assert (asset.dest / Path(asset.files[0]).name).read_bytes() == b"tiny"
    assert all("/split_files/" in url for url, _, _ in calls)
    assert not list(root.rglob("split_files"))


def test_catalog_modelscope_force_refetches_complete_files(catalog_mirror_transport):
    _, calls = catalog_mirror_transport
    args = ["--source", "modelscope"]
    downloads.cmd_download_anima(args)
    downloads.cmd_download_anima(args)
    assert len(calls) == 3
    downloads.cmd_download_anima([*args, "--force"])
    assert len(calls) == 6


@pytest.mark.parametrize("source", ["flag", "environment"])
def test_first_run_modelscope_never_falls_back_to_hf(monkeypatch, source):
    picked = []
    monkeypatch.setattr(
        DL,
        "fetch_all",
        lambda *a, **k: pytest.fail("ModelScope selection fell back to HF"),
    )
    monkeypatch.setattr(
        downloads,
        "_ms_fetch",
        lambda name, repo, filename, dest, extra: picked.append((name, extra)) or True,
    )
    args = ["--force"]
    if source == "flag":
        args += ["--source", "modelscope"]
    else:
        monkeypatch.setenv("ANIMA_DOWNLOAD_SOURCE", "modelscope")
    downloads.cmd_download_models(args)
    assert set(name for name, _ in picked) == set(DL.DEFAULT_SET)
    assert all("--force" in extra for _, extra in picked)


def test_catalog_modelscope_missing_mirror_reports_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        DL, "fetch_all", lambda *a, **k: pytest.fail("Missing mirror fell back to HF")
    )
    with pytest.raises(SystemExit, match="pe_spatial"):
        downloads.cmd_download_pe_spatial(["--source", "modelscope"])
    assert "no ModelScope mirror" in capsys.readouterr().out
