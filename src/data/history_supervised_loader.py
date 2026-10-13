"""History-aware supervised dynamic dataset for actor pretraining.

Draws contiguous windows of (H + F) frames from MD trajectories,
splitting them into history (context) and query (training targets).
Modeled on RLPdbDataset / ActorPretrainDataset sampling semantics.
"""
import os
import logging
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
import tree
from torch.utils.data import Dataset
from openfold.data import data_transforms
from openfold.np import residue_constants
from openfold.utils import rigid_utils

from src.data.DyneTrion_data_loader_dynamic import parse_dynamics_chain_feats

logger = logging.getLogger(__name__)


def split_history_query_frames(total, history_frames: int, query_frame_time: int):
    """Split a [W, ...] tensor into history [H, ...] and query [F, ...].

    Args:
        total: tensor of shape [W, ...] where W = history_frames + query_frame_time
        history_frames: number of history frames H
        query_frame_time: number of query frames F

    Returns:
        (history, query) tuple
    """
    assert total.shape[0] == history_frames + query_frame_time, (
        f"Expected {history_frames + query_frame_time} frames, got {total.shape[0]}"
    )
    history = total[:history_frames]
    query = total[history_frames:]
    return history, query


class HistoryAwareSupervisedDataset(Dataset):
    """Supervised dataset with history-aware contiguous window sampling.

    Each trajectory sample draws a window of W = H + F frames at uniform
    frame_sample_step intervals. The first H frames are history (context),
    the remaining F are query (training targets).
    """

    def __init__(
        self,
        csv_path: str,
        diffuser,
        data_conf,
        is_training: bool = True,
        max_res_len: int = 512,
    ):
        super().__init__()
        self.csv = pd.read_csv(csv_path)
        self.csv = self.csv[self.csv["seq_len"] < max_res_len]
        self.diffuser = diffuser
        self.is_training = is_training

        self.history_frames = int(data_conf.history_frames)
        self.motion_number = int(getattr(data_conf, "motion_number", 2))
        self.query_frame_time = int(data_conf.frame_time)
        self.ref_number = int(getattr(data_conf, "ref_number", 1))
        # Window layout: [H history | M motion | F query]
        self.window_size = self.history_frames + self.motion_number + self.query_frame_time
        self.frame_sample_step = int(data_conf.frame_sample_step)
        self.samples_per_traj = int(data_conf.samples_per_traj)
        self.min_t = float(getattr(data_conf, "min_t", 0.01))

        self._lengths = self.csv["seq_len"].astype(int).clip(lower=1).tolist()
        logger.info(
            f"HistoryAwareSupervisedDataset: {len(self.csv)} proteins, "
            f"H={self.history_frames}, M={self.motion_number}, F={self.query_frame_time}, "
            f"step={self.frame_sample_step}, window={self.window_size}"
        )

    @property
    def lengths(self):
        return self._lengths

    def __len__(self):
        return len(self.csv)

    def _sample_window_indices(self, total_frames: int):
        """Sample a contiguous window of W frames with uniform stride.

        Returns: np.ndarray of shape [W] with frame indices.
        """
        W = self.window_size
        step = self.frame_sample_step
        max_start = total_frames - 1 - (W - 1) * step
        if max_start < 0:
            raise ValueError(
                f"Trajectory too short ({total_frames} frames) for "
                f"window={W}, step={step}"
            )
        start = np.random.randint(0, max_start + 1)
        return np.array([start + i * step for i in range(W)])

    def _read_repr(self, feature_path):
        assert os.path.exists(feature_path), f"Feature file not found: {feature_path}"
        attr_dict = dict(np.load(feature_path, allow_pickle=True))
        node_data = attr_dict.get("node_repr", attr_dict.get("single"))
        edge_data = attr_dict.get("edge_repr", attr_dict.get("pair"))
        return {
            "node_repr": torch.tensor(node_data),
            "edge_repr": torch.tensor(edge_data),
        }

    def __getitem__(self, idx):
        csv_row = self.csv.iloc[idx]
        pos_path = csv_row["pos_path"]
        raw_feats = dict(np.load(pos_path, allow_pickle=True))
        all_pos = raw_feats["all_atom_positions"]
        total_frames = len(all_pos)

        repr_dict = self._read_repr(csv_row["embed_path"])

        H = self.history_frames
        M = self.motion_number
        F = self.query_frame_time
        R = self.ref_number
        W = self.window_size  # H + M + F

        # Sample one contiguous window of W = H + M + F frames
        window_indices = self._sample_window_indices(total_frames)  # [W]

        # Ref: always frame 0 of the trajectory (initial structure)
        ref_indices = np.arange(R)  # [0, ..., R-1]

        # Combine: ref frames + window frames for batch processing
        all_indices = np.concatenate([ref_indices, window_indices])  # [R + W]

        feats = raw_feats.copy()
        feats["all_atom_positions"] = feats["all_atom_positions"][all_indices]
        feats = parse_dynamics_chain_feats(feats)

        # Build OpenFold chain features
        aatype_static = feats["aatype"]
        if aatype_static.ndim == 2:
            aatype_static = np.argmax(aatype_static, axis=-1)
        aatype_tensor = torch.tensor(aatype_static).long()
        total_n = R + W
        chain_feats = {
            "aatype": aatype_tensor.unsqueeze(0).repeat(total_n, 1),
            "all_atom_positions": torch.tensor(feats["all_atom_positions"]),
            "all_atom_mask": torch.tensor(feats["all_atom_mask"]).unsqueeze(0).expand(total_n, -1, -1).clone(),
        }

        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)

        # Extract rigids: [R+W, N, 7]
        gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(
            chain_feats["rigidgroups_gt_frames"]
        )[:, :, 0]
        gt_bb_rigid_7 = gt_bb_rigid.to_tensor_7()  # [R+W, N, 7]
        N = gt_bb_rigid_7.shape[1]

        # Split: [R ref | H history | M motion | F query]
        ref_rigids_0 = gt_bb_rigid_7[:R].contiguous()                    # [R, N, 7]
        history_rigids = gt_bb_rigid_7[R : R + H].contiguous()           # [H, N, 7]
        motion_rigids_0 = gt_bb_rigid_7[R + H : R + H + M].contiguous()  # [M, N, 7]
        query_rigids_7 = gt_bb_rigid_7[R + H + M :].contiguous()         # [F, N, 7]

        # BB mask
        bb_mask = torch.tensor(feats["bb_mask"])  # [R+W, N] or [N]
        if bb_mask.ndim == 1:
            bb_mask = bb_mask.unsqueeze(0).repeat(total_n, 1)
        res_mask = bb_mask[R+H+M:].contiguous()  # [F, N]
        fixed_mask = 1.0 - res_mask

        # Torsion fields
        torsion_angles_sin_cos = chain_feats["torsion_angles_sin_cos"][R+H+M:].contiguous()
        alt_torsion_angles_sin_cos = chain_feats["alt_torsion_angles_sin_cos"][R+H+M:].contiguous()
        torsion_angles_mask = chain_feats["torsion_angles_mask"][R+H+M:].contiguous()
        ref_torsion_angles_sin_cos = chain_feats["torsion_angles_sin_cos"][:R].contiguous()     # [R, N, ...]
        ref_alt_torsion_angles_sin_cos = chain_feats["alt_torsion_angles_sin_cos"][:R].contiguous()  # [R, N, ...]
        history_torsion_angles_sin_cos = chain_feats["torsion_angles_sin_cos"][R:R+H].contiguous()  # [H, N, ...]
        history_alt_torsion_angles_sin_cos = chain_feats["alt_torsion_angles_sin_cos"][R:R+H].contiguous()  # [H, N, ...]

        # Sample one t, diffuse all F query frames together (same as DyneTrion_data_loader_dynamic)
        query_rigids = rigid_utils.Rigid.from_tensor_7(query_rigids_7)  # [F, N]
        t = np.random.uniform(self.min_t, 1.0)
        diffused = self.diffuser.forward_marginal(
            rigids_0=query_rigids, t=t, diffuse_mask=None,
        )

        # Compute guided scores: rigids_i_t [F, N] vs each history frame [H, N] → [F, H, N, 3]
        rigids_i_t = rigid_utils.Rigid.from_tensor_7(diffused["rigids_t"])  # [F, N]
        t_tensor = torch.tensor(t, dtype=torch.float32)

        # Base score: toward query GT
        score_trans_base = self.diffuser.calc_trans_score(
            rigids_i_t.get_trans(), query_rigids.get_trans(), t_tensor,
        )  # [F, N, 3]
        score_rot_base = self.diffuser.calc_rot_score(
            rigids_i_t.get_rots(), query_rigids.get_rots(), t_tensor,
        )  # [F, N, 3]

        # Guided score: rigids_i_t [F, N] vs history [H, N] → [F, H, N, 3]
        # Expand: rigids_i_t [F,1,N] x history [1,H,N] → [F,H,N]
        hist_rigids_7 = history_rigids.unsqueeze(0).expand(F, -1, -1, -1)  # [F, H, N, 7]
        query_i_t_7 = diffused["rigids_t"].unsqueeze(1).expand(-1, H, -1, -1)  # [F, H, N, 7]
        # Flatten to [F*H, N] for batch score computation
        FH = F * H
        rigids_i_t_fh = rigid_utils.Rigid.from_tensor_7(query_i_t_7.reshape(FH, N, 7))
        rigids_hist_fh = rigid_utils.Rigid.from_tensor_7(hist_rigids_7.reshape(FH, N, 7))
        score_trans_guided = self.diffuser.calc_trans_score(
            rigids_i_t_fh.get_trans(), rigids_hist_fh.get_trans(), t_tensor,
        ).reshape(F, H, N, 3)  # [F, H, N, 3]
        score_rot_guided = self.diffuser.calc_rot_score(
            rigids_i_t_fh.get_rots(), rigids_hist_fh.get_rots(), t_tensor,
        ).reshape(F, H, N, 3)  # [F, H, N, 3]

        bb_mask_1d = torch.tensor(feats["bb_mask"])
        if bb_mask_1d.ndim == 2:
            bb_mask_1d = bb_mask_1d[0]  # [N]

        final_feats = {
            # History context
            "history_rigids": history_rigids,  # [H, N, 7]
            # Ref & motion (same convention as DyneTrion_data_loader_dynamic)
            "ref_rigids_0": ref_rigids_0,        # [R, N, 7] — frame 0 of trajectory
            "motion_rigids_0": motion_rigids_0,  # [M, N, 7]
            # Diffused query (all F frames, same t)
            "rigids_t": torch.as_tensor(diffused["rigids_t"]),  # [F, N, 7]
            "rigids_0": query_rigids_7,  # [F, N, 7]
            "t": t_tensor,  # scalar
            "sc_ca_t": torch.zeros_like(query_rigids_7)[..., 4:],  # [F, N, 3]
            # Masks
            "res_mask": res_mask,  # [F, N]
            "fixed_mask": fixed_mask,  # [F, N]
            # Scores
            "rot_score": torch.as_tensor(diffused["rot_score"]),       # [F, N, 3]
            "trans_score": torch.as_tensor(diffused["trans_score"]),   # [F, N, 3]
            "rot_score_scaling": torch.as_tensor(diffused["rot_score_scaling"], dtype=torch.float32),    # scalar
            "trans_score_scaling": torch.as_tensor(diffused["trans_score_scaling"], dtype=torch.float32),  # scalar
            "score_rot_guided": score_rot_guided.detach(),     # [F, H, N, 3]
            "score_trans_guided": score_trans_guided.detach(), # [F, H, N, 3]
            "score_rot_base": score_rot_base.detach(),         # [F, N, 3]
            "score_trans_base": score_trans_base.detach(),     # [F, N, 3]
            # Sequence/structure
            "aatype": aatype_tensor.unsqueeze(0).expand(F, -1).contiguous(),  # [F, N]
            "seq_idx": torch.tensor(raw_feats["residue_index"]).unsqueeze(0).expand(F, -1).contiguous(),
            # Torsion
            "torsion_angles_sin_cos": torsion_angles_sin_cos,      # [F, N, 7, 2]
            "alt_torsion_angles_sin_cos": alt_torsion_angles_sin_cos,
            "torsion_angles_mask": torsion_angles_mask,            # [F, N, 7]
            "ref_torsion_angles_sin_cos": ref_torsion_angles_sin_cos,      # [R, N, ...]
            "ref_alt_torsion_angles_sin_cos": ref_alt_torsion_angles_sin_cos,  # [R, N, ...]
            "history_torsion_angles_sin_cos": history_torsion_angles_sin_cos,  # [H, N, ...]
            "history_alt_torsion_angles_sin_cos": history_alt_torsion_angles_sin_cos,  # [H, N, ...]
            # Masks for motion/ref
            "motion_node_mask": bb_mask_1d.unsqueeze(0).expand(M, -1).contiguous(),  # [M, N]
            "ref_node_mask": bb_mask_1d.unsqueeze(0).expand(R, -1).contiguous(),  # [R, N]
            # Representations (shared across frames)
            "node_repr": repr_dict["node_repr"],  # [N, D]
            "edge_repr": repr_dict["edge_repr"],  # [N, N, D]
        }

        # Ensure contiguous for DataLoader pin_memory
        final_feats = tree.map_structure(
            lambda x: x.contiguous().clone() if torch.is_tensor(x) else x,
            final_feats,
        )
        return final_feats


def history_supervised_collate(batch_list):
    """Collate for HistoryAwareSupervisedDataset.

    Each item has per-frame fields [F, N, ...], history [H, N, 7],
    and shared fields [N, D]. Stack along a new batch dim B.
    """
    return tree.map_structure(
        lambda *xs: torch.stack(xs, dim=0),
        *batch_list,
    )
