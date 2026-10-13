"""Integration coverage with synthetic proteins and random model parameters."""
import subprocess
import sys
from pathlib import Path

import hydra
import mdtraj as md
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from openfold.np import protein, residue_constants

from DyneTrion.eval_actor_pipeline import ActorEvaluator
from DyneTrion.supervised_pretrain import SupervisedTrainer
from src.data.se3_diffuser import SE3Diffuser
from src.model.network import DyneTrion

CONFIG_DIR = Path(__file__).resolve().parents[1] / "DyneTrion" / "config"


def _make_synthetic_data(tmp_path):
    n = 8
    positions = np.zeros((n, 37, 3), dtype=np.float32)
    mask = np.zeros((n, 37), dtype=np.float32)
    offsets = {"N": [-1.2, 0.5, 0.1], "CA": [0, 0, 0],
               "C": [1.3, 0.4, -0.1], "O": [1.7, 1.3, -0.1],
               "CB": [-0.1, -0.7, 1.3]}
    for i in range(n):
        center = np.array([3.7 * i, np.sin(i), np.cos(i)])
        for name, offset in offsets.items():
            j = residue_constants.atom_order[name]
            positions[i, j] = center + offset
            mask[i, j] = 1
    prot = protein.Protein(atom_positions=positions, aatype=np.zeros(n, dtype=int),
                           atom_mask=mask, residue_index=np.arange(n),
                           b_factors=np.zeros((n, 37)))
    pdb = tmp_path / "synthetic.pdb"
    pdb.write_text(protein.to_pdb(prot))
    initial = md.load(str(pdb))
    xyz = np.repeat(initial.xyz, 8, axis=0)
    # Vary internal geometry across time so history frames are distinct.
    xyz[:, 10, 1] += np.arange(8) * 0.005
    traj = md.Trajectory(xyz, initial.topology)
    xtc = tmp_path / "synthetic.xtc"
    traj.save_xtc(str(xtc))
    raw = tmp_path / "raw.csv"
    pd.DataFrame([dict(pdb_id="synthetic", seq_len=n, sequence="A"*n,
                       pdb_path=str(pdb), traj_path=str(xtc))]).to_csv(raw, index=False)
    fasta_dir = tmp_path / "fasta"
    subprocess.run([sys.executable, "-m", "datasets.toolbox.build_fasta",
                    "--input_csv", str(raw), "--outdir", str(fasta_dir)],
                   check=True, capture_output=True)
    assert (fasta_dir / "synthetic.fasta").read_text() == ">synthetic\nAAAAAAAA\n"
    embed_dir = tmp_path / "embeddings"
    embed_dir.mkdir()
    rng = np.random.default_rng(42)
    np.savez(embed_dir / "synthetic.npz", single=rng.normal(size=(n, 32)).astype("float32"),
             pair=rng.normal(size=(n, n, 128)).astype("float32"))
    coord_dir = tmp_path / "coordinates"
    subprocess.run([sys.executable, "-m", "datasets.toolbox.process_coordination",
                    "--datacsv", str(raw), "--outdir", str(coord_dir),
                    "--num_workers", "1", "--stride", "1"], check=True, capture_output=True)
    train_csv = tmp_path / "train.csv"
    subprocess.run([sys.executable, "-m", "datasets.toolbox.build_final_csv",
                    "--input_csv", str(raw), "--output_csv", str(train_csv),
                    "--coordination_npz_dir", str(coord_dir), "--embed_dir", str(embed_dir)],
                   check=True, capture_output=True)
    return train_csv, coord_dir / "synthetic.npz"


def test_preprocessing_supervised_step_and_guided_sampling(tmp_path, monkeypatch):
    # Test the plain PyTorch paths even on machines with optional extensions.
    from src.model import attention
    monkeypatch.setattr(attention, "FLASH_ATTN_2_AVAILABLE", False)
    monkeypatch.setattr(attention, "FLASH_ATTN_3_AVAILABLE", False)
    torch.manual_seed(42)
    train_csv, coordinates = _make_synthetic_data(tmp_path)
    with np.load(coordinates) as features:
        assert features["all_atom_positions"].shape == (8, 8, 37, 3)
        assert features["all_atom_mask"].shape == (8, 37)

    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="actor_train")
    OmegaConf.set_struct(cfg, False)
    cfg.data.update(frame_time=2, history_frames=2, motion_number=1, frame_sample_step=1,
                    max_res_len=16, train_csv_path=str(train_csv), test_csv_path=str(train_csv),
                    num_t=2, num_loader_workers=0)
    cfg.model.update(node_input_embed_size=32, node_embed_size=32, edge_embed_size=128,
                     attention_num_heads=4, num_frame_per_block=1)
    cfg.model.ipa.update(c_hidden=32, no_heads=4, no_qk_points=2, no_v_points=2, num_blocks=1)
    cfg.actor.update(c_hidden=32, no_heads=4, no_qk_points=2, no_v_points=2, num_blocks=1)
    cfg.diffuser.so3.update(num_sigma=16, num_omega=32, cache_dir=str(tmp_path / "cache"))
    cfg.experiment.update(num_loader_workers=0, use_recoder=False,
                          save_dir=str(tmp_path / "training"))
    diffuser = SE3Diffuser(cfg.diffuser)
    env = DyneTrion(cfg.model, diffuser)
    env_file = tmp_path / "env.pth"
    torch.save({"model": env.state_dict()}, env_file)
    cfg.model.pretrained_weights = str(env_file)
    trainer = SupervisedTrainer(cfg)
    batch = next(iter(trainer.train_dataloader))
    result = trainer._dispatch_train_step(batch)
    assert np.isfinite(result["loss"])
    assert np.isfinite(result["env_support_loss"])
    assert trainer.global_step == 1
    assert all(not p.requires_grad for p in trainer._env_model.parameters())
    actor_file = tmp_path / "actor.bin"
    torch.save(trainer.accelerator.unwrap_model(trainer.actor).state_dict(), actor_file)

    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        eval_cfg = hydra.compose(config_name="eval_actor")
    eval_cfg.model = cfg.model
    eval_cfg.actor = cfg.actor
    eval_cfg.actor.weight = str(actor_file)
    eval_cfg.diffuser = cfg.diffuser
    eval_cfg.data = cfg.data
    eval_cfg.data.dynamics = True
    eval_cfg.eval.update(extrapolation_time=2, output_dir=str(tmp_path / "samples"))
    evaluator = ActorEvaluator(conf=eval_cfg)
    eval_batch, _ = next(iter(evaluator._test_loader))
    sample = evaluator.sampler.sample(eval_batch)
    assert sample.atom_traj.shape == (1, 2, 2, 8, 37, 3)
    assert torch.isfinite(sample.atom_traj).all()
    evaluator._save_batch(sample.atom_traj, eval_batch, ["synthetic"])
    generated = Path(evaluator.runtime.save_dir) / "synthetic.pdb"
    assert len(md.load(str(generated))) == 4
    subprocess.run([sys.executable, "-m", "metrics.structural_metric",
                    "--input_csv", str(train_csv), "--gen_root", evaluator.runtime.save_dir,
                    "--save_dir", str(tmp_path / "metrics"), "--jobs_name", "smoke",
                    "--device", "cpu", "--md_stride", "1", "--batch_size", "2"],
                   check=True, capture_output=True)
    metrics = pd.read_csv(tmp_path / "metrics" / "smoke_mean.csv")
    assert len(metrics) == 1
    assert {"diversity_score", "precision", "recall"}.issubset(metrics.columns)
    assert np.isfinite(metrics.select_dtypes(include="number").to_numpy()).all()


def test_openfold_imports_without_deepspeed():
    code = """
import sys
sys.modules['deepspeed'] = None
import torch
from openfold.model.primitives import Linear, LayerNorm
from openfold.utils.checkpointing import get_checkpoint_fn
assert callable(get_checkpoint_fn())
value = torch.randn(2, 4, dtype=torch.bfloat16)
assert torch.isfinite(LayerNorm(4).to(torch.bfloat16)(value)).all()
assert torch.isfinite(Linear(4, 4).to(torch.bfloat16)(value)).all()
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_environment_loaders_without_workers(tmp_path, monkeypatch):
    from DyneTrion.train import Experiment

    train_csv, _ = _make_synthetic_data(tmp_path)
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="train")
    cfg.data.update(csv_path=str(train_csv), val_csv_path=str(train_csv),
                    frame_time=2, motion_number=1, frame_sample_step=1)
    cfg.diffuser.so3.update(num_sigma=16, num_omega=32,
                           cache_dir=str(tmp_path / "cache"))
    cfg.experiment.update(batch_size=1, eval_batch_size=1, num_loader_workers=0)

    # Exercise the real datasets and DataLoaders without requiring a GPU process group.
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    experiment = Experiment.__new__(Experiment)
    experiment._data_conf = cfg.data
    experiment._exp_conf = cfg.experiment
    experiment._diffuser = SE3Diffuser(cfg.diffuser)
    experiment.ddp_info = {"world_size": 1}
    train_loader, valid_loader = experiment.create_dataset()

    train_batch = next(iter(train_loader))
    valid_batch, _, _ = next(iter(valid_loader))
    for batch in [train_batch, valid_batch]:
        assert batch["aatype"].shape == (1, 2, 8)
        assert batch["rigids_t"].shape[-1] == 7
        assert torch.isfinite(batch["rigids_t"]).all()


def test_environment_ddp_uses_assigned_gpu(monkeypatch):
    import logging
    from DyneTrion.train import Experiment

    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="train")
    cfg.experiment.update(num_epoch=0, ckpt_dir=None)
    experiment = Experiment.__new__(Experiment)
    experiment._conf = cfg
    experiment._exp_conf = cfg.experiment
    experiment._use_ddp = True
    experiment._use_recoder = False
    experiment._available_gpus = ""
    experiment.ddp_info = {"local_rank": 0, "world_size": 1}
    experiment.trained_epochs = 0
    experiment._log = logging.getLogger(__name__)
    experiment._model = torch.nn.Linear(1, 1)
    assigned_devices = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(experiment._model, "to",
                        lambda device: assigned_devices.append(str(device)) or experiment._model)
    monkeypatch.setattr(experiment, "create_dataset", lambda: ([], []))

    experiment.start_training()
    assert assigned_devices == ["cuda:0"]
