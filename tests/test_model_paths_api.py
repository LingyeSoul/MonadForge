from __future__ import annotations

import toml

from webui.api import system


def test_update_preserves_custom_configs():
    from scripts import update

    assert update._is_preserved("configs/custom/model.toml") is True
    assert update._is_preserved("configs/custom/preprocess.toml") is True
    assert update._is_preserved("configs/model.toml") is False


def test_system_model_path_update_writes_custom_config(monkeypatch, tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    base_path = configs / "base.toml"
    base_path.write_text('output_dir = "output/ckpt"\n', encoding="utf-8")
    (configs / "model.toml").write_text(
        'pretrained_model_name_or_path = "default-dit"\n'
        'qwen3 = "default-te"\n'
        'vae = "default-vae"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(system, "ROOT", tmp_path)

    result = system.update_model_paths(
        [
            system.ModelPathUpdate(key="vae", value="D:/models/local-vae.safetensors"),
            system.ModelPathUpdate(key="not_a_model", value="ignored"),
        ]
    )

    assert result == {"ok": True}
    assert base_path.read_text(encoding="utf-8") == 'output_dir = "output/ckpt"\n'
    saved = toml.loads((configs / "custom" / "model.toml").read_text(encoding="utf-8"))
    assert saved == {"vae": "D:/models/local-vae.safetensors"}

    paths = {row["toml_key"]: row for row in system.get_model_paths()["paths"]}
    assert paths["vae"]["path"] == "D:/models/local-vae.safetensors"
    assert paths["qwen3"]["path"] == "default-te"
