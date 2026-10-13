import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.data.so3_diffuser import SO3Diffuser


def _make_so3_conf(use_cached_score: bool):
    return SimpleNamespace(
        schedule="logarithmic",
        min_sigma=0.1,
        max_sigma=1.5,
        num_sigma=32,
        num_omega=64,
        use_cached_score=use_cached_score,
        cache_dir=".cache/unittest_so3",
    )


class SO3TorchIndexingTest(unittest.TestCase):
    def test_t_to_idx_torch_matches_numpy(self):
        diffuser = SO3Diffuser(_make_so3_conf(use_cached_score=False))
        t = torch.tensor([0.0, 0.01, 0.2, 0.5, 0.9, 1.0], dtype=torch.float32)
        expected = diffuser.t_to_idx(t.detach().cpu().numpy())
        actual = diffuser._t_to_idx_torch(t).detach().cpu().numpy()
        self.assertTrue(np.array_equal(actual, expected))

    def test_torch_score_matches_legacy_numpy_indexing_path(self):
        diffuser = SO3Diffuser(_make_so3_conf(use_cached_score=False))
        vec = torch.randn(6, 5, 3, dtype=torch.float32)
        t = torch.tensor([0.0, 0.01, 0.2, 0.5, 0.9, 1.0], dtype=torch.float32)

        expected_sigma = diffuser.discrete_sigma[diffuser.t_to_idx(t.detach().cpu().numpy())]
        expected_sigma = torch.tensor(expected_sigma, dtype=vec.dtype)
        omega = torch.linalg.norm(vec, dim=-1) + 1e-6
        omega_vals = diffuser.__class__.__dict__["torch_score"].__globals__["igso3_expansion"](
            omega, expected_sigma[:, None], use_torch=True
        )
        expected_scores = diffuser.__class__.__dict__["torch_score"].__globals__["score"](
            omega_vals, omega, expected_sigma[:, None], use_torch=True
        )
        expected = expected_scores[..., None] * vec / (omega[..., None] + 1e-6)

        actual = diffuser.torch_score(vec, t)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
