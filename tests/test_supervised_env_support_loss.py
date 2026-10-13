import hydra
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from einops import rearrange

from DyneTrion.supervised_pretrain import SupervisedTrainer


CONFIG_DIR = Path(__file__).resolve().parent.parent / "DyneTrion" / "config"


def _compose_actor_train_cfg():
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return hydra.compose(config_name="actor_train")


def test_actor_train_exposes_env_support_fields():
    cfg = _compose_actor_train_cfg()
    assert "use_env_support_loss" in cfg.experiment
    assert "lambda_support" in cfg.experiment
    assert "history_frames" in cfg.data
    assert "frame_time" in cfg.data


def test_actor_train_exposes_env_support_loss_type():
    cfg = _compose_actor_train_cfg()
    assert cfg.experiment.env_support_loss_type == "kl_forward"






def test_prepare_history_batch_uses_history_torsion_targets():
    trainer = SupervisedTrainer.__new__(SupervisedTrainer)
    bsz, frame_t, hist_len, n_res, n_tor = 1, 2, 3, 4, 7

    history_torsion = torch.arange(
        bsz * hist_len * n_res * n_tor * 2, dtype=torch.float32
    ).reshape(bsz, hist_len, n_res, n_tor, 2)
    ref_torsion = torch.full((bsz, 1, n_res, n_tor, 2), -1.0)

    batch = {
        "score_rot_guided": torch.zeros(bsz, frame_t, hist_len, n_res, 3),
        "score_trans_guided": torch.zeros(bsz, frame_t, hist_len, n_res, 3),
        "rigids_t": torch.zeros(bsz, frame_t, n_res, 7),
        "res_mask": torch.ones(bsz, frame_t, n_res),
        "fixed_mask": torch.zeros(bsz, frame_t, n_res),
        "aatype": torch.zeros(bsz, frame_t, n_res, dtype=torch.long),
        "seq_idx": torch.zeros(bsz, frame_t, n_res, dtype=torch.long),
        "node_repr": torch.zeros(bsz, n_res, 8),
        "edge_repr": torch.zeros(bsz, n_res, n_res, 4),
        "history_rigids": torch.zeros(bsz, hist_len, n_res, 7),
        "motion_rigids_0": torch.zeros(bsz, 2, n_res, 7),
        "motion_node_mask": torch.ones(bsz, 2, n_res),
        "ref_node_mask": torch.ones(bsz, 1, n_res),
        "rot_score_scaling": torch.ones(bsz),
        "trans_score_scaling": torch.ones(bsz),
        "t": torch.ones(bsz),
        "torsion_angles_sin_cos": torch.zeros(bsz, frame_t, n_res, n_tor, 2),
        "alt_torsion_angles_sin_cos": torch.zeros(bsz, frame_t, n_res, n_tor, 2),
        "torsion_angles_mask": torch.ones(bsz, frame_t, n_res, n_tor),
        "ref_torsion_angles_sin_cos": ref_torsion,
        "ref_alt_torsion_angles_sin_cos": ref_torsion.clone(),
        "history_torsion_angles_sin_cos": history_torsion,
        "history_alt_torsion_angles_sin_cos": history_torsion.clone(),
    }

    out = trainer._prepare_history_batch(batch)

    expected = history_torsion.unsqueeze(1).expand(bsz, frame_t, hist_len, n_res, n_tor, 2)
    expected = expected.reshape(bsz * frame_t * hist_len, n_res, n_tor, 2)

    assert torch.equal(out["ref_torsion_angles_sin_cos"], expected)
    assert not torch.equal(out["ref_torsion_angles_sin_cos"], ref_torsion[:, 0].repeat(frame_t * hist_len, 1, 1, 1))


class _FakePredRigid:
    def __init__(self, trans):
        self._trans = trans

    def get_trans(self):
        return self._trans


def _extract_uniform_score_value(rot_score, trans_score, *, atol):
    rot_value = float(rot_score.reshape(-1)[0].item())
    trans_value = float(trans_score.reshape(-1)[0].item())
    if not torch.allclose(rot_score, torch.full_like(rot_score, rot_value), atol=atol, rtol=0.0):
        raise AssertionError("Expected `rot_score` to be uniform in this test setup.")
    if not torch.allclose(trans_score, torch.full_like(trans_score, trans_value), atol=atol, rtol=0.0):
        raise AssertionError("Expected `trans_score` to be uniform in this test setup.")
    if abs(rot_value - trans_value) > atol:
        raise AssertionError(
            f"Expected uniform rot/trans score values to match; got rot={rot_value:.6f}, "
            f"trans={trans_value:.6f}."
        )
    return rot_value


class _ScoreSensitiveConstantLogprobDiffuser:
    def __init__(self, logp_rot, logp_trans, expected_score_value, atol=1e-6):
        self._logp_rot = float(logp_rot)
        self._logp_trans = float(logp_trans)
        self._expected_score_value = float(expected_score_value)
        self._atol = float(atol)

    def reverse_torch(self, rigid_t, **kwargs):
        return rigid_t

    def transition_logprob_given_next(self, rot_score, trans_score, **kwargs):
        observed_score_value = _extract_uniform_score_value(
            rot_score, trans_score, atol=self._atol
        )
        if abs(observed_score_value - self._expected_score_value) > self._atol:
            raise AssertionError(
                f"Unexpected score value {observed_score_value:.6f}; "
                f"expected {self._expected_score_value:.6f}."
            )

        batch = rot_score.shape[0]
        logp_rot = torch.full(
            (batch,), self._logp_rot, dtype=rot_score.dtype, device=rot_score.device
        )
        logp_trans = torch.full(
            (batch,), self._logp_trans, dtype=trans_score.dtype, device=trans_score.device
        )
        return logp_rot, logp_trans, None


class _ScalarTaggedLogprobDiffuser:
    def __init__(
        self,
        guided_score_value,
        env_score_value,
        guided_logp,
        env_logp,
        atol=1e-6,
    ):
        self._guided_score_value = float(guided_score_value)
        self._env_score_value = float(env_score_value)
        self._guided_logp = float(guided_logp)
        self._env_logp = float(env_logp)
        self._atol = float(atol)

    def reverse_torch(self, rigid_t, **kwargs):
        return rigid_t

    def transition_logprob_given_next(self, rot_score, trans_score, **kwargs):
        score_value = _extract_uniform_score_value(rot_score, trans_score, atol=self._atol)
        if abs(score_value - self._guided_score_value) <= self._atol:
            logp_val = self._guided_logp
        elif abs(score_value - self._env_score_value) <= self._atol:
            logp_val = self._env_logp
        else:
            raise AssertionError(
                f"Unexpected score value {score_value:.6f}; "
                f"expected guided={self._guided_score_value:.6f} "
                f"or env={self._env_score_value:.6f}."
            )

        batch = rot_score.shape[0]
        logp_rot = torch.full((batch,), logp_val, dtype=rot_score.dtype, device=rot_score.device)
        logp_trans = torch.full((batch,), logp_val, dtype=trans_score.dtype, device=trans_score.device)
        return logp_rot, logp_trans, None


def _make_env_support_test_inputs(*, actor_hist_value, env_value):
    batch_size, frame_count, history_len, residue_count = 1, 1, 2, 3

    # Intentionally stub trainer internals via private fields so the test can
    # isolate `_compute_env_support_loss` without full trainer/runtime setup.
    trainer = SupervisedTrainer.__new__(SupervisedTrainer)
    alpha = 1.0
    history_weights = torch.tensor([0.5, 0.5], dtype=torch.float32)
    trainer._exp_conf = SimpleNamespace(
        actor_blend_alpha=alpha,
        env_support_loss_type="kl_forward",
    )
    trainer.conf = SimpleNamespace(data=SimpleNamespace(num_t=10))
    trainer._env_support_noise_scale = 0.0

    def _stub_history_decay_weights(t_tensor, *_args, **_kwargs):
        return history_weights.to(device=t_tensor.device, dtype=t_tensor.dtype).view(1, 1, -1)

    trainer._history_decay_weights = _stub_history_decay_weights

    actor_scores_h = torch.full(
        (batch_size, frame_count, history_len, residue_count, 3),
        actor_hist_value,
        dtype=torch.float32,
    )
    env_rot_score = torch.full(
        (batch_size, frame_count, residue_count, 3), env_value, dtype=torch.float32
    )
    env_trans_score = torch.full(
        (batch_size, frame_count, residue_count, 3), env_value, dtype=torch.float32
    )

    rigids_t = torch.zeros((batch_size * frame_count * history_len, residue_count, 7), dtype=torch.float32)
    rigids_t[..., 0] = 1.0

    formatted_batch = {
        "t": torch.full((batch_size * frame_count * history_len,), 0.5, dtype=torch.float32),
        "res_mask": torch.ones((batch_size * frame_count * history_len, residue_count), dtype=torch.float32),
        "rigids_t": rigids_t,
        "rot_score_scaling": torch.ones((batch_size * frame_count * history_len,), dtype=torch.float32),
        "trans_score_scaling": torch.ones((batch_size * frame_count * history_len,), dtype=torch.float32),
    }
    pred_ctx = {
        "pred_rot_guided": rearrange(actor_scores_h, "b f h n c -> (b f h) n c"),
        "pred_trans_guided": rearrange(actor_scores_h, "b f h n c -> (b f h) n c"),
        "pred_x_ref_rigids": _FakePredRigid(
            torch.zeros((batch_size * frame_count * history_len, residue_count, 3), dtype=torch.float32)
        ),
    }

    mean_actor_score = float((history_weights * actor_hist_value).sum().item())
    bias_value = env_value - mean_actor_score
    expected_guided_score_value = float(env_value + alpha * bias_value)
    return SimpleNamespace(
        trainer=trainer,
        formatted_batch=formatted_batch,
        pred_ctx=pred_ctx,
        env_rot_score=env_rot_score,
        env_trans_score=env_trans_score,
        bsz=batch_size,
        frame_t=frame_count,
        hist_len=history_len,
        guided_score_value=expected_guided_score_value,
        env_score_value=float(env_value),
    )


def test_env_support_forward_kl_zero_when_guided_equals_env():
    test_inputs = _make_env_support_test_inputs(actor_hist_value=0.0, env_value=0.0)
    trainer = test_inputs.trainer
    trainer._diffuser = _ScoreSensitiveConstantLogprobDiffuser(
        logp_rot=-2.0, logp_trans=-1.0, expected_score_value=test_inputs.env_score_value
    )

    guided_cache = trainer._compute_guided_scores(
        test_inputs.formatted_batch,
        test_inputs.pred_ctx,
        test_inputs.env_rot_score,
        test_inputs.env_trans_score,
        test_inputs.bsz,
        test_inputs.frame_t,
        test_inputs.hist_len,
    )
    loss = trainer._compute_env_support_loss(
        test_inputs.formatted_batch,
        test_inputs.env_rot_score,
        test_inputs.env_trans_score,
        guided_cache,
        test_inputs.bsz,
        test_inputs.frame_t,
        test_inputs.hist_len,
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_env_support_forward_kl_positive_when_guided_differs_from_env():
    test_inputs = _make_env_support_test_inputs(actor_hist_value=1.0, env_value=0.0)
    trainer = test_inputs.trainer

    trainer._diffuser = _ScalarTaggedLogprobDiffuser(
        guided_score_value=test_inputs.guided_score_value,
        env_score_value=test_inputs.env_score_value,
        guided_logp=-1.0,
        env_logp=-2.0,
    )

    guided_cache = trainer._compute_guided_scores(
        test_inputs.formatted_batch,
        test_inputs.pred_ctx,
        test_inputs.env_rot_score,
        test_inputs.env_trans_score,
        test_inputs.bsz,
        test_inputs.frame_t,
        test_inputs.hist_len,
    )
    loss = trainer._compute_env_support_loss(
        test_inputs.formatted_batch,
        test_inputs.env_rot_score,
        test_inputs.env_trans_score,
        guided_cache,
        test_inputs.bsz,
        test_inputs.frame_t,
        test_inputs.hist_len,
    )

    # Forward KL expectation under this fake setup:
    # (logq_rot - logp_rot) + (logq_trans - logp_trans) = [(-1)-(-2)] + [(-1)-(-2)] = 2.0.
    # Here B*F = 1, so this 2.0 is the single-element fake reduction result.
    assert loss.item() == pytest.approx(2.0, abs=1e-6)
