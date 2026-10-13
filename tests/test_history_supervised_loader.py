import torch

from src.data.history_supervised_loader import (
    split_history_query_frames,
    history_supervised_collate,
)


def test_split_history_query_frames_shapes():
    total = torch.randn(7, 11, 7)
    history, query = split_history_query_frames(total, history_frames=3, query_frame_time=4)
    assert history.shape == (3, 11, 7)
    assert query.shape == (4, 11, 7)


def test_history_supervised_collate_preserves_required_keys():
    batch = [{
        "history_rigids": torch.randn(3, 11, 7),
        "rigids_t": torch.randn(4, 11, 7),
        "res_mask": torch.ones(4, 11),
        "fixed_mask": torch.zeros(4, 11),
        "node_repr": torch.randn(11, 32),
        "edge_repr": torch.randn(11, 11, 16),
    }]
    out = history_supervised_collate(batch)
    assert "history_rigids" in out
    assert out["rigids_t"].shape == (1, 4, 11, 7)
