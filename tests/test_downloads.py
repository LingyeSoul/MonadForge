"""The model catalog and the ``make download-*`` targets that drive it.

import urllib.error
from io import BytesIO
from pathlib import Path

import pytest

from library import downloads as DL


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
    from anime_tools.downloads import catalog as package_catalog

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
    """v2 default: masking is opt-in, so SAM3 must not be in a fresh install's
    path — its gated repo was the first-run failure this removes. The CJK vocab
    pack is the opposite: base.toml enables it, so the set must install it."""
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


def test_download_danbooru_tags_writes_models_file(tmp_path, monkeypatch):
    dest = tmp_path / "models" / "danbooru_tags_classified.csv"
    monkeypatch.setattr(downloads, "DANBOORU_TAGS_PATH", dest)
    monkeypatch.setattr(
        downloads, "DANBOORU_TAGS_URLS", ("https://example.test/tags.csv",)
    )
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


def test_download_danbooru_tags_failure_names_source_repo(tmp_path, monkeypatch):
    dest = tmp_path / "models" / "danbooru_tags_classified.csv"
    monkeypatch.setattr(downloads, "DANBOORU_TAGS_PATH", dest)
    monkeypatch.setattr(
        downloads, "DANBOORU_TAGS_URLS", ("https://example.test/tags.csv",)
    )
    monkeypatch.setattr(
        downloads.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(urllib.error.URLError("nope")),
    )
    targets = [
        n
        for n in tasks.COMMANDS
        if n.startswith("download-")
        and n not in ("download-anima-variant", "download-list", "download-model")
    ]
    assert len(targets) >= 8
    assert "download-mit" not in tasks.COMMANDS
    for name in targets:
        tasks.COMMANDS[name][0]([])
    assert all(ids for ids in picked)


    assert "Localsmile/danbooru_KR_wiki_tag_search" in str(exc.value)


def _no_hf_run(*_args, **_kwargs):
    raise AssertionError("hf CLI must not run under --source modelscope")


def test_anima_modelscope_source_uses_snapshot_and_moves(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "ROOT", tmp_path)
    monkeypatch.setattr(downloads, "run", _no_hf_run)
    calls: list[tuple[str, str, tuple[str, ...]]] = []

    def fake_snapshot(repo_id, local_dir, *, filenames=None, include=None, force=False):
        calls.append((repo_id, str(local_dir), tuple(filenames or ())))
        models = Path(local_dir)
        for name in filenames or ():
            p = models / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")

    monkeypatch.setattr(downloads.ms_download, "snapshot_download", fake_snapshot)

    downloads.cmd_download_anima(["--source", "modelscope"])

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
    monkeypatch.setattr(downloads, "ROOT", tmp_path)
    monkeypatch.setenv("ANIMA_DOWNLOAD_SOURCE", "modelscope")
    monkeypatch.setattr(downloads, "run", _no_hf_run)
    monkeypatch.setattr(
        downloads.ms_download,
        "snapshot_download",
        lambda *a, **k: None,
    )

    downloads.cmd_download_sam3([])


def test_anima_modelscope_env_routes_without_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "ROOT", tmp_path)
    monkeypatch.setenv("ANIMA_DOWNLOAD_SOURCE", "modelscope")
    monkeypatch.setattr(downloads, "run", _no_hf_run)
    called = {}

    def fake_snapshot(repo_id, local_dir, *, filenames=None, include=None, force=False):
        called["repo"] = repo_id
        called["force"] = force

    monkeypatch.setattr(downloads.ms_download, "snapshot_download", fake_snapshot)

    downloads.cmd_download_anima([])

    assert called["repo"] == "circlestone-labs/Anima"
    assert called["force"] is False


def test_anima_modelscope_force_passes_through(tmp_path, monkeypatch):
    """``--force`` must reach the ms snapshot instead of being dropped."""
    monkeypatch.setattr(downloads, "ROOT", tmp_path)
    monkeypatch.setattr(downloads, "run", _no_hf_run)
    called = {}

    def fake_snapshot(repo_id, local_dir, *, filenames=None, include=None, force=False):
        called["force"] = force

    monkeypatch.setattr(downloads.ms_download, "snapshot_download", fake_snapshot)

    downloads.cmd_download_anima(["--source", "modelscope", "--force"])

    assert called["force"] is True


def test_sam3_modelscope_force_passes_through(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "ROOT", tmp_path)
    monkeypatch.setattr(downloads, "run", _no_hf_run)
    called = {}

    def fake_snapshot(repo_id, local_dir, *, force=False, **_k):
        called["force"] = force

    monkeypatch.setattr(downloads.ms_download, "snapshot_download", fake_snapshot)

    downloads.cmd_download_sam3(["--source", "modelscope", "--force"])

    assert called["force"] is True


def test_pe_core_modelscope_force_passes_through(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "ROOT", tmp_path)
    monkeypatch.setattr(downloads, "run", _no_hf_run)
    called = {}

    def fake_fetch_file(*, ms_repo, filename, local_dir, what="asset", force=False):
        called["force"] = force
        return Path(local_dir) / filename

    monkeypatch.setattr(downloads.ms_download, "fetch_file", fake_fetch_file)

    downloads.cmd_download_pe(["--source", "modelscope", "--force"])

    assert called["force"] is True


def test_anima_default_source_still_uses_hf(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "ROOT", tmp_path)
    seen: list[list[str]] = []
    monkeypatch.setattr(downloads, "run", lambda cmd, **_k: seen.append(cmd))

    downloads.cmd_download_anima([])

    assert seen and seen[0][:3] == ["hf", "download", "circlestone-labs/Anima"]


def test_pe_spatial_modelscope_without_mirror_skips(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(downloads, "ROOT", tmp_path)
    monkeypatch.setattr(downloads, "run", _no_hf_run)

    downloads.cmd_download_pe_spatial(["--source", "modelscope"])

    out = capsys.readouterr().out
    assert "no ModelScope mirror" in out
    assert not (tmp_path / "models" / "pe" / "PE-Spatial-B16-512.pt").exists()


def test_download_ms_requires_repo_id():
    with pytest.raises(SystemExit, match="usage"):
        downloads.cmd_download_ms([])
    with pytest.raises(SystemExit, match="org/name"):
        downloads.cmd_download_ms(["justname"])


def test_download_ms_rejects_unknown_option(tmp_path, monkeypatch):
    """A typo'd flag must fail loudly, not silently become a repo id/file."""
    monkeypatch.setattr(downloads, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="unknown option --localdir"):
        downloads.cmd_download_ms(["org/name", "--localdir", "x"])


def test_download_ms_passes_repo_and_default_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "ROOT", tmp_path)
    calls: dict = {}

    def fake_snapshot(repo_id, local_dir, *, include=None, force=False, **_k):
        calls["args"] = (repo_id, str(local_dir), include, force)

    monkeypatch.setattr(downloads.ms_download, "snapshot_download", fake_snapshot)

    downloads.cmd_download_ms(["org/name", "--include", "*.safetensors"])

    repo_id, local_dir, include, force = calls["args"]
    assert repo_id == "org/name"
    assert local_dir == str(tmp_path / "models" / "name")
    assert include == ["*.safetensors"]
    assert force is False

    downloads.cmd_download_ms(["org/other", "--local-dir", str(tmp_path / "x")])
    assert calls["args"][1] == str(tmp_path / "x")
