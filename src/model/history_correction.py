"""History-guided denoising with RMSD prior and learnable correction.

Provides:
- per_residue_rmsd_weights: per-residue distance-based prior weights
- HistoryCorrectionNetwork: lightweight f_θ for per-residue correction logits
- aggregate_history_scores: softmax(δ + log w) score aggregation
- ScoreCache / StructureCache: inference-time caches
"""

import math
import contextlib
import torch
import torch.nn as nn
from torch.nn.functional import silu
from einops import rearrange, repeat
from openfold.utils.rigid_utils import Rigid
from src.analysis.geometry import rmsdalign
from src.model import modules


# ---------------------------------------------------------------------------
# Task 1: Per-Residue RMSD Prior Weights
# ---------------------------------------------------------------------------

def per_residue_rmsd_weights(
    x0_current: torch.Tensor,
    x0_history: torch.Tensor,
    res_mask: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Per-residue distance-based prior weights via Kabsch-aligned structures.

    Aligns each history x̂₀ to the current x̂₀ via Kabsch, then computes
    per-residue distance → Gaussian kernel weight.

    Follows the same pattern as _kabsch_rmsd_weights in supervised_pretrain.py
    but returns per-residue weights instead of per-frame.

    Args:
        x0_current: [B, F, N, 3] env-predicted CA positions for query frames.
        x0_history: [B, H, N, 3] cached CA positions from historical frames.
        res_mask:   [B, N] residue mask (1 = valid, 0 = padding).
        sigma:      temperature for exp(-dist / sigma).

    Returns:
        [B, F, H, N] unnormalized per-residue weights. Masked residues are 0.
    """
    B, F, N, _ = x0_current.shape
    H = x0_history.shape[1]
    sigma = max(float(sigma), 1e-6)
    out_dtype = x0_current.dtype

    autocast_ctx = (
        torch.amp.autocast("cuda", enabled=False)
        if x0_current.is_cuda
        else contextlib.nullcontext()
    )
    with autocast_ctx:
        x0_current = x0_current.float()
        x0_history = x0_history.float()
        res_mask_f = res_mask.to(device=x0_current.device, dtype=x0_current.dtype)

        # Expand pairwise: [B,F,H,N,3]
        curr_exp = x0_current.unsqueeze(2).expand(-1, -1, H, -1, -1)
        hist_exp = x0_history.unsqueeze(1).expand(-1, F, -1, -1, -1)

        # Flatten to [B*F*H, N, 3] for batch Kabsch alignment
        curr_flat = rearrange(curr_exp, "b f h n c -> (b f h) n c")
        hist_flat = rearrange(hist_exp, "b f h n c -> (b f h) n c")
        mask_flat = repeat(res_mask_f, "b n -> (b f h) n", f=F, h=H)

        # Kabsch-align history to current
        aligned_hist = rmsdalign(curr_flat, hist_flat, weights=mask_flat)

        # Per-residue distance after alignment
        per_res_dist = (aligned_hist - curr_flat).pow(2).sum(dim=-1).sqrt()  # [B*F*H, N]
        per_res_dist = rearrange(per_res_dist, "(b f h) n -> b f h n", b=B, f=F, h=H)

        w = torch.exp(-per_res_dist / sigma)
        w = w * res_mask_f[:, None, None, :]

    return w.to(dtype=out_dtype)


# ---------------------------------------------------------------------------
# Task 2: HistoryCorrectionNetwork (f_θ)
# ---------------------------------------------------------------------------

class HistoryCorrectionNetwork(nn.Module):
    """IPA-based per-residue correction network f_θ.

    Processes both current (F) and history (H) frames through shared IPA layers
    using protein node/edge features and rigid body geometry, then computes
    per-residue correction logits via dot product between current and history
    frame representations.

    Architecture:
        1. Time embedding (sinusoidal → MLP) added to node features
        2. Shared IPA layers process all (F+H) frames jointly
        3. Split output into current/history representations
        4. Per-residue dot product → δ [B, F, H, N]

    Output projection initialized near zero so δ ≈ 0 at init → α ≈ w (pure RMSD prior).
    """

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        n_ipa_layers: int = 2,
        # IPA config
        c_hidden: int = 256,
        no_heads: int = 8,
        no_qk_points: int = 8,
        no_v_points: int = 12,
        # Cross attention Q/K projection dim
        d_k: int = 128,
        # Legacy kwargs (ignored, for backward compat with config)
        d_hidden: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.n_ipa_layers = n_ipa_layers

        # Time embedding: scalar t → [c_s], added to node features
        self.time_embed_dim = c_s
        self.time_mlp = nn.Sequential(
            nn.Linear(c_s, c_s // 2),
            nn.SiLU(),
            nn.Linear(c_s // 2, c_s),
        )

        # Shared IPA layers (process F+H frames through same weights)
        from types import SimpleNamespace
        ipa_conf = SimpleNamespace(
            c_s=c_s, c_z=c_z, c_hidden=c_hidden,
            no_heads=no_heads, no_qk_points=no_qk_points,
            no_v_points=no_v_points,
        )

        self.ipa_layers = nn.ModuleList()
        self.ipa_lns = nn.ModuleList()
        self.ipa_transitions = nn.ModuleList()
        for _ in range(n_ipa_layers):
            ipa = modules.InvariantPointAttention(ipa_conf)
            # Override zero-init on linear_out so IPA output is non-zero
            # and gradients can flow through the residual connection.
            nn.init.xavier_uniform_(ipa.linear_out.weight)
            self.ipa_layers.append(ipa)
            self.ipa_lns.append(nn.LayerNorm(c_s))
            self.ipa_transitions.append(modules.StructureModuleTransition(c_s))

        # Cross attention projections: Q from current frames, K from history frames
        # Small-init so QK ≈ 0 at init → attn ≈ softmax(w) ≈ pure RMSD prior
        self.d_k = d_k
        self.linear_q = nn.Linear(c_s, d_k)
        self.linear_k = nn.Linear(c_s, d_k)
        # Xavier uniform init (default) — no small-init, so QK is non-zero
        # from the start and entropy gradient can flow through softmax.

    def _sinusoidal_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        """Sinusoidal time embedding. t: [B] → [B, dim]."""
        half_dim = dim // 2
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(
            -emb_scale * torch.arange(half_dim, device=t.device, dtype=t.dtype)
        )
        args = t.unsqueeze(-1) * freqs
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(
        self,
        x0_current_rigids: torch.Tensor,
        x0_history_rigids: torch.Tensor,
        t: torch.Tensor,
        res_mask: torch.Tensor,
        node_repr: torch.Tensor,
        edge_repr: torch.Tensor,
        S_H_rot: torch.Tensor,
        S_H_trans: torch.Tensor,
        x0_ca_current: torch.Tensor,
        x0_ca_history: torch.Tensor,
        rmsd_sigma: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute bias scores via cross attention over history scores.

        Args:
            x0_current_rigids: [B, F, N, 7] predicted rigids from env model.
            x0_history_rigids: [B, H, N, 7] cached rigids from historical frames.
            t:          [B] diffusion time in [0, 1].
            res_mask:   [B, N] residue mask.
            node_repr:  [B, N, c_s] protein node features.
            edge_repr:  [B, N, N, c_z] protein edge features.
            S_H_rot:    [B, F, H, N, 3] history rotation scores.
            S_H_trans:  [B, F, H, N, 3] history translation scores.
            x0_ca_current: [B, F, N, 3] env-predicted CA positions.
            x0_ca_history: [B, H, N, 3] cached CA positions.
            rmsd_sigma: temperature for RMSD prior.

        Returns:
            bias_rot:   [B, F, N, 3] rotation score bias.
            bias_trans: [B, F, N, 3] translation score bias.
            attn:       [B, F, H, N] attention weights (for monitoring).
        """
        B, F, N, _ = x0_current_rigids.shape
        H = x0_history_rigids.shape[1]
        K = F + H

        # 1. Time embedding → add to node features
        t_emb = self._sinusoidal_embedding(t, self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)
        node_s = node_repr + t_emb[:, None, :]
        node_s = node_s.unsqueeze(1).expand(B, K, N, -1)
        node_s = rearrange(node_s, "b k n d -> (b k) n d")

        # 2. Edge features tiled
        edge_z = edge_repr.unsqueeze(1).expand(B, K, N, N, -1)
        edge_z = rearrange(edge_z, "b k i j d -> (b k) i j d")

        # 3. Rigids
        all_rigids_7 = torch.cat([x0_current_rigids, x0_history_rigids], dim=1)
        all_rigids_7 = rearrange(all_rigids_7, "b k n c -> (b k) n c")
        all_rigids = Rigid.from_tensor_7(all_rigids_7)

        # 4. Mask
        mask_flat = res_mask.unsqueeze(1).expand(B, K, N)
        mask_flat = rearrange(mask_flat, "b k n -> (b k) n")

        # 5. Shared IPA layers
        s = node_s
        for ipa, ln, transition in zip(self.ipa_layers, self.ipa_lns, self.ipa_transitions):
            s = s + ipa(s, edge_z, all_rigids, mask_flat)
            s = ln(s)
            s = transition(s)

        # 6. Split into current / history
        s = rearrange(s, "(b k) n d -> b k n d", b=B, k=K)
        f_r = s[:, :F]   # [B, F, N, c_s]
        f_h = s[:, F:]   # [B, H, N, c_s]

        # 7. Cross attention: Q from current, K from history, V = history scores
        Q = self.linear_q(f_r)      # [B, F, N, d_k]
        K_feat = self.linear_k(f_h)  # [B, H, N, d_k]

        # QK attention logits: [B, F, H, N]
        scale = self.d_k ** (-0.5)
        attn_logits = torch.einsum("bfnd,bhnd->bfhn", Q, K_feat) * scale

        # Apply the RMSD prior in logit space so softmax(QK + log w)
        # matches a multiplicative prior exp(QK) * w in probability space.
        w = per_residue_rmsd_weights(
            x0_ca_current, x0_ca_history, res_mask, sigma=rmsd_sigma,
        )
        attn_logits = attn_logits + torch.log(w.clamp_min(1e-12))

        # Softmax over H dimension
        attn = torch.softmax(attn_logits, dim=2)  # [B, F, H, N]
        attn = attn * res_mask[:, None, None, :]

        # V = cat(S_H_rot, S_H_trans) → [B, F, H, N, 6]
        V = torch.cat([S_H_rot, S_H_trans], dim=-1)

        # Weighted sum: [B, F, N, 6]
        bias_score = torch.einsum("bfhn,bfhnd->bfnd", attn, V)

        bias_rot = bias_score[..., :3]
        bias_trans = bias_score[..., 3:]
        return bias_rot, bias_trans, attn


# ---------------------------------------------------------------------------
# Task 3: Score Aggregation (moved into HistoryCorrectionNetwork.forward)
# ---------------------------------------------------------------------------
# Cross attention now handles aggregation internally.
# See HistoryCorrectionNetwork.forward() for the implementation.


# ---------------------------------------------------------------------------
# Task 6: ScoreCache and StructureCache (for inference)
# ---------------------------------------------------------------------------

class ScoreCache:
    """Cache env model scores indexed by (frame_id, step_idx) for inference.

    During sequential frame generation, stores env scores at each reverse
    diffusion step. Later frames read cached scores at the same step index.
    """

    def __init__(self, max_history: int):
        self.max_history = max_history
        self._scores: dict[int, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
        self._frame_order: list[int] = []  # tracks insertion order for eviction

    def store(
        self,
        frame_id: int,
        step_idx: int,
        rot_score: torch.Tensor,
        trans_score: torch.Tensor,
    ):
        """Store env scores for a given frame and step.

        Args:
            frame_id: which extrapolation frame (0, 1, 2, ...).
            step_idx: which reverse diffusion step (0 = t_max, T-1 = t_min).
            rot_score:   [B, F, N, 3] detached.
            trans_score: [B, F, N, 3] detached.
        """
        if frame_id not in self._scores:
            self._scores[frame_id] = {}
        self._scores[frame_id][step_idx] = (
            rot_score.detach(),
            trans_score.detach(),
        )

    def finalize_frame(self, frame_id: int):
        """Mark frame as complete and enforce max_history eviction."""
        if frame_id not in self._frame_order:
            self._frame_order.append(frame_id)
        while len(self._frame_order) > self.max_history:
            evict = self._frame_order.pop(0)
            self._scores.pop(evict, None)

    def get(self, frame_id: int, step_idx: int):
        """Retrieve cached scores or None."""
        frame_data = self._scores.get(frame_id)
        if frame_data is None:
            return None
        return frame_data.get(step_idx)

    def get_history_at_step(self, step_idx: int, exclude_frame: int):
        """Get all cached history scores at a given step index.

        Args:
            step_idx: reverse diffusion step.
            exclude_frame: current frame id (don't include self).

        Returns:
            (rot_scores, trans_scores): [B, F, H, N, 3] stacked, or (None, None).
        """
        rot_list, trans_list = [], []
        for fid in self._frame_order:
            if fid == exclude_frame:
                continue
            entry = self.get(fid, step_idx)
            if entry is not None:
                rot_list.append(entry[0])
                trans_list.append(entry[1])
        if len(rot_list) == 0:
            return None, None
        # Stack along new H dimension (dim=2)
        return torch.stack(rot_list, dim=2), torch.stack(trans_list, dim=2)


class StructureCache:
    """FIFO cache for final generated structures (for RMSD prior and f_θ input)."""

    def __init__(self, max_history: int):
        self.max_history = max_history
        self._frames: dict[int, torch.Tensor] = {}
        self._order: list[int] = []

    def push(self, frame_id: int, rigid: torch.Tensor):
        """Store final rigid [B, N, 7] for a completed frame."""
        self._frames[frame_id] = rigid.detach().clone()
        if frame_id not in self._order:
            self._order.append(frame_id)
        while len(self._order) > self.max_history:
            evict = self._order.pop(0)
            self._frames.pop(evict, None)

    def get_history(self) -> torch.Tensor | None:
        """Returns [B, H, N, 7] stacked history or None if empty."""
        if len(self._order) == 0:
            return None
        tensors = [self._frames[fid] for fid in self._order]
        return torch.stack(tensors, dim=1)

    def get_frame_ids(self) -> list[int]:
        return list(self._order)
