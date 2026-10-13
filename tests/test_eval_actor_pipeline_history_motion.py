import types
import unittest
from unittest import mock

import torch

from DyneTrion import eval_actor_pipeline


def _frame(value: float) -> torch.Tensor:
    return torch.full((1, 1, 7), float(value), dtype=torch.float32)


class _DummyEnvModel:
    def eval(self):
        return self


class _SingleStepEnvModel:
    def eval(self):
        return self

    def __call__(self, curr_feats, is_training=False):
        bsz, frame_t, n_res = curr_feats["aatype"].shape
        rigids = curr_feats["rigids_t"].clone()
        return {
            "rot_score": torch.zeros((bsz, frame_t, n_res, 3), dtype=torch.float32),
            "trans_score": torch.zeros((bsz, frame_t, n_res, 3), dtype=torch.float32),
            "rigids": rigids,
            "angles": torch.zeros((bsz, frame_t, n_res, 7, 2), dtype=torch.float32),
        }


class _DummyDiffuser:
    def score_scaling(self, t):
        return torch.tensor(1.0), torch.tensor(1.0)


class _FakeScoreCache:
    def store(self, frame_id, step_idx, rot_score, trans_score):
        return None

    def get_history_at_step(self, step_idx, exclude_frame):
        rot = torch.zeros((1, 1, 1, 1, 3), dtype=torch.float32)
        trans = torch.zeros((1, 1, 1, 1, 3), dtype=torch.float32)
        return rot, trans


class _FakeStructureCache:
    def get_history(self):
        hist = torch.zeros((1, 1, 1, 7), dtype=torch.float32)
        hist[..., 0] = 1.0
        return hist


class _FakeCorrectionNet:
    def __init__(self):
        self.calls = 0

    def eval(self):
        return self

    def __call__(self, x0_current_rigids, x0_history_rigids, *args, **kwargs):
        self.calls += 1
        bsz, frame_t, n_res, _ = x0_current_rigids.shape
        hist_len = x0_history_rigids.shape[1]
        bias_rot = torch.zeros((bsz, frame_t, n_res, 3), dtype=torch.float32)
        bias_trans = torch.zeros((bsz, frame_t, n_res, 3), dtype=torch.float32)
        attn = torch.zeros((bsz, frame_t, hist_len, n_res), dtype=torch.float32)
        return bias_rot, bias_trans, attn


class EvalActorPipelineHistoryMotionTests(unittest.TestCase):
    def test_sample_moves_previous_motion_into_history(self):
        sampler = eval_actor_pipeline.BiasedActorSampler.__new__(
            eval_actor_pipeline.BiasedActorSampler
        )
        sampler.env_model = _DummyEnvModel()
        sampler.runtime = types.SimpleNamespace(max_history=16, extrapolation_time=2)
        sampler.model_conf = types.SimpleNamespace(motion_number=2)

        history_inputs = []
        rigid_preds = [
            torch.stack([_frame(30), _frame(31), _frame(32), _frame(33)], dim=1),
            torch.stack([_frame(40), _frame(41), _frame(42), _frame(43)], dim=1),
        ]
        atom_pred = torch.zeros((1, 4, 1, 37, 3), dtype=torch.float32)

        def fake_prepare_init_feats(batch_feats):
            return {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in batch_feats.items()
            }

        def fake_single_rollout(curr_feats, history=None):
            history_inputs.append(None if history is None else history.clone())
            return atom_pred, rigid_preds[len(history_inputs) - 1], curr_feats

        def fake_update(curr_feats, atom_pred_arg, rigid_pred):
            motion_keep = int(sampler.model_conf.motion_number)
            curr_feats = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in curr_feats.items()
            }
            curr_feats["motion_rigids_0"] = rigid_pred[:, -motion_keep:].clone()
            return curr_feats

        sampler._prepare_init_feats = fake_prepare_init_feats
        sampler._single_rollout = fake_single_rollout
        sampler._update_ref_with_prediction = fake_update

        batch_feats = {
            "aatype": torch.zeros((1, 4, 1), dtype=torch.long),
            "ref_rigids_0": _frame(10).unsqueeze(1),
            "motion_rigids_0": torch.stack([_frame(20), _frame(21)], dim=1),
        }

        sampler.sample(batch_feats)

        self.assertEqual(history_inputs[0].shape, (1, 1, 1, 7))
        self.assertEqual(history_inputs[0][0, 0, 0, 0].item(), 10.0)

        self.assertEqual(history_inputs[1].shape, (1, 5, 1, 7))
        self.assertEqual(
            history_inputs[1][0, :, 0, 0].tolist(),
            [10.0, 20.0, 21.0, 30.0, 31.0],
        )

    def test_history_guided_sampler_uses_learned_correction_when_lambda_zero(self):
        sampler = eval_actor_pipeline.HistoryGuidedSampler.__new__(
            eval_actor_pipeline.HistoryGuidedSampler
        )
        sampler.env_model = _SingleStepEnvModel()
        sampler.correction_net = _FakeCorrectionNet()
        sampler.diffuser = _DummyDiffuser()
        sampler.runtime = types.SimpleNamespace(
            min_t=1.0,
            num_t=1,
            noise_scale=1.0,
            lambda_blend=0.0,
            rmsd_sigma=1.0,
        )
        sampler.device = torch.device("cpu")
        sampler.model_conf = types.SimpleNamespace(
            embed=types.SimpleNamespace(embed_self_conditioning=False)
        )

        curr_feats = {
            "aatype": torch.zeros((1, 1, 1), dtype=torch.long),
            "res_mask": torch.ones((1, 1, 1), dtype=torch.float32),
            "fixed_mask": torch.zeros((1, 1, 1), dtype=torch.float32),
            "rigids_t": torch.tensor([[[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]]),
            "node_repr": torch.zeros((1, 1, 4), dtype=torch.float32),
            "edge_repr": torch.zeros((1, 1, 1, 2), dtype=torch.float32),
        }

        fake_atom37 = torch.zeros((1, 1, 1, 37, 3), dtype=torch.float32)
        with mock.patch.object(
            eval_actor_pipeline.all_atom,
            "compute_backbone_atom37",
            return_value=(fake_atom37, None, None, None),
        ):
            sampler._single_rollout_guided(
                curr_feats=curr_feats,
                frame_id=1,
                score_cache=_FakeScoreCache(),
                structure_cache=_FakeStructureCache(),
            )

        self.assertEqual(sampler.correction_net.calls, 1)


if __name__ == "__main__":
    unittest.main()
