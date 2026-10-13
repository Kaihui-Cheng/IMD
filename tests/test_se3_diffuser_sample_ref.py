import torch
from types import SimpleNamespace

from src.data.se3_diffuser import SE3Diffuser


def _build_conf(frame_time: int = 4):
    return SimpleNamespace(
        diffuse_rot=True,
        diffuse_trans=True,
        dynamics=True,
        frame_time=frame_time,
        so3=SimpleNamespace(
            schedule="logarithmic",
            min_sigma=0.1,
            max_sigma=1.5,
            num_sigma=8,
            use_cached_score=True,
            num_omega=32,
            cache_dir=".cache/test_so3",
        ),
        r3=SimpleNamespace(
            min_b=0.1,
            max_b=20.0,
            coordinate_scaling=1.0,
        ),
    )


def test_sample_ref_can_return_tensor7_on_requested_device():
    diffuser = SE3Diffuser(_build_conf(frame_time=4))
    device = torch.device("cpu")

    out = diffuser.sample_ref(n_samples=12, as_tensor_7=True, device=device)

    assert torch.is_tensor(out["rigids_t"])
    assert out["rigids_t"].device == device
    assert out["rigids_t"].shape == (4, 3, 7)
