import fnirsatlas.config as config_module


def test_config_defaults_work_outside_repository(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_module, "config", None)

    assert config_module.get_config()["logging"]["level"] == "INFO"


def test_local_config_overrides_packaged_defaults(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config_local.yaml").write_text("logging:\n  level: DEBUG\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_module, "config", None)

    assert config_module.get_config()["logging"]["level"] == "DEBUG"
