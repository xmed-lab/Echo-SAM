"""Release integration checks; no datasets, checkpoints, or GPU required."""

from importlib.resources import files
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from sam3.train.utils.train_utils import register_omegaconf_resolvers


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "sam3/train/configs"
CARDIAC_CONFIGS = sorted(CONFIGS.glob("*cardiac*/*.yaml"))


@pytest.fixture
def release_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("ECHO_SAM_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("SAM3_CHECKPOINT", str(tmp_path / "base.pt"))
    monkeypatch.setenv("ECHO_SAM_CHECKPOINT", str(tmp_path / "trained.pt"))
    monkeypatch.setenv("ECHO_SAM_OUTPUT_DIR", str(tmp_path / "outputs"))
    if not OmegaConf.has_resolver("get_method"):
        register_omegaconf_resolvers()
    return tmp_path


@pytest.mark.parametrize("path", CARDIAC_CONFIGS, ids=lambda p: p.stem)
def test_cardiac_config_and_loss_construct(path, release_environment):
    with initialize_config_dir(config_dir=str(CONFIGS), version_base="1.2"):
        cfg = compose(config_name=str(path.relative_to(CONFIGS)))
    OmegaConf.resolve(cfg)
    assert cfg.paths.ultrasound_cardiac_root == str(release_environment / "data")
    assert cfg.paths.checkpoint_path == str(release_environment / "base.pt")
    assert cfg.paths.experiment_log_dir == str(release_environment / "outputs")
    assert Path(cfg.paths.bpe_path).is_file()
    if "ultrasound_cardiac_knowledge" in cfg.paths:
        assert Path(cfg.paths.ultrasound_cardiac_knowledge).is_file()
    loss_cfg = cfg.ultrasound_cardiac_train.get("loss")
    if loss_cfg:
        loss = instantiate(loss_cfg)
        assert callable(loss)
    if "zero_shot" in path.stem:
        assert not cfg.few_shot.enable
        assert cfg.trainer.data.train.dataset.support_datasets == []
    if path.stem == "ultrasound_cardiac_zero_shot_eval":
        assert cfg.trainer.model.eval_mode
        assert cfg.trainer.checkpoint.resume_from == str(release_environment / "trained.pt")


def test_packaged_graph_and_vocabulary():
    import json
    import torch
    from sam3.model.medical_knowledge import MedicalKnowledgeBase

    package = files("sam3")
    assert (package / "assets/bpe_simple_vocab_16e6.txt.gz").is_file()
    graph_path = package / "train/configs/ultrasound_cardiac/ultrasound_cardiac_knowledge.json"
    graph = json.loads(graph_path.read_text())
    knowledge = MedicalKnowledgeBase(knowledge_config_path=str(graph_path))
    assert set(graph["structure_descriptions"]) == set(knowledge._concept_to_idx)
    prompts = ["left_ventricle", "myocardium"]
    embeddings = knowledge.fuse_knowledge_for_prompt(prompts, torch.device("cpu"))
    assert embeddings.shape == (2, 256)
    assert torch.isfinite(embeddings).all()


def test_launcher_preserves_visible_devices(monkeypatch):
    import importlib

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    module = importlib.import_module("sam3.train.train")
    importlib.reload(module)
    import os
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
