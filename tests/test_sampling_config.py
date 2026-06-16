"""Boundary normalization for the sample-image knobs.

``normalize_sample_args`` reconciles the loose shapes the GUI emits
(inline prompts, ``0`` cadence sentinels) into what the training core assumes.
These tests lock the four branches: cadence coercion, inline list / multi-line
text written to a file, existing-file passthrough, and the empty-input disable.
"""

import argparse

import pytest

from library.training.sampling_config import normalize_sample_args


def _ns(tmp_path, **kw):
    base = dict(
        output_dir=str(tmp_path),
        sample_prompts=None,
        sample_every_n_epochs=None,
        sample_every_n_steps=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.mark.parametrize("knob", ["sample_every_n_epochs", "sample_every_n_steps"])
@pytest.mark.parametrize("sentinel", [0, -1])
def test_nonpositive_cadence_coerced_to_none(tmp_path, knob, sentinel):
    args = _ns(tmp_path, **{knob: sentinel})
    normalize_sample_args(args)
    assert getattr(args, knob) is None


@pytest.mark.parametrize("knob", ["sample_every_n_epochs", "sample_every_n_steps"])
def test_positive_cadence_preserved(tmp_path, knob):
    args = _ns(tmp_path, **{knob: 5})
    normalize_sample_args(args)
    assert getattr(args, knob) == 5


def test_inline_list_written_to_file(tmp_path):
    args = _ns(tmp_path, sample_prompts=["a cat", "  a dog  ", "# comment", ""])
    normalize_sample_args(args)

    expected = tmp_path / "sample_prompts.txt"
    assert args.sample_prompts == str(expected)
    # Blank + comment lines dropped; surrounding whitespace stripped.
    assert expected.read_text(encoding="utf-8") == "a cat\na dog\n"


def test_inline_multiline_string_written_to_file(tmp_path):
    args = _ns(tmp_path, sample_prompts="a cat\n# comment\n\na dog\n")
    normalize_sample_args(args)

    expected = tmp_path / "sample_prompts.txt"
    assert args.sample_prompts == str(expected)
    assert expected.read_text(encoding="utf-8") == "a cat\na dog\n"


def test_existing_file_path_left_untouched(tmp_path):
    real = tmp_path / "my_prompts.txt"
    real.write_text("a cat\n", encoding="utf-8")
    args = _ns(tmp_path, sample_prompts=str(real))
    normalize_sample_args(args)

    # The CLI case: a real path is passed through verbatim, no rewrite.
    assert args.sample_prompts == str(real)
    assert not (tmp_path / "sample_prompts.txt").exists()


def test_empty_inline_disables_sampling(tmp_path):
    args = _ns(tmp_path, sample_prompts=["", "# only a comment"])
    normalize_sample_args(args)

    # Nothing usable → disabled rather than pointing at a phantom file.
    assert args.sample_prompts is None
    assert not (tmp_path / "sample_prompts.txt").exists()


def test_none_prompts_noop(tmp_path):
    args = _ns(tmp_path, sample_prompts=None)
    normalize_sample_args(args)
    assert args.sample_prompts is None
