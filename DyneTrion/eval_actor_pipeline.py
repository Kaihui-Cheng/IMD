import contextlib
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

import hydra
import numpy as np
import torch
import tree
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange, repeat
from omegaconf import DictConfig
from openfold.utils.rigid_utils import Rigid
from torch.utils import data
from tqdm import tqdm

from src.analysis import utils as au
from src.data import all_atom, se3_diffuser
from src.data import utils as du
from src.data.sampling import ProteinSamplingDataset
from src.model import network
from src.model.actor import MetadynamicsActor
from src.analysis.geometry import rmsdalign
from icecream import ic
import torch.nn.functional as F

def _norm(x):
    return x.float().norm(dim=-1).mean().item()

def _cos(a, b):
    a = a.float().reshape(a.shape[0], -1)
    b = b.float().reshape(b.shape[0], -1)
    return F.cosine_similarity(a, b, dim=-1).mean().item()

@dataclass
class RuntimeConfig:
    mode: int
    save_dir: str
    num_t: int
    min_t: float
    noise_scale: float
    max_history: int
    history_decay_lambda: float
    history_decay_t_scale: float
    history_decay_mode: str
    history_distance_sigma: float
    history_distance_variant: str
    bias_gamma: float
    center: bool
    extrapolation_time: int
    actor_blend_alpha: float
    overwrite: bool
    lambda_blend: float = 0.0  # Option A: fixed RMSD-weighted history blending strength
    bias_scale: float = 1.0    # Multiplier on learned correction_net bias at eval time
    record_last_k: int = 0     # Girsanov collect: save last K reverse-step per-step data
    actor_condition_mode: str = "none"   # {none, target_anchor}: ω anchor source
    bias_direction: str = "away"          # {away, toward}: sign of (base, actor_h) bias
    inline_refine_t: float = 0.0          # >0: refine each extrap segment via env_model noise+denoise
    omega_method: str = "rmsd_kabsch"     # {rmsd_kabsch, contact_map, dihedral}: ω distance metric
    fkc_enabled: bool = False             # Feynman-Kac correction via SMC over the F frame dim
    fkc_resample_k: int = 10              # legacy: fixed resample interval (used if ess_frac<=0)
    fkc_beta: float = 1.0                 # scale on |bias|^2 * dt for log-weight accumulation
    fkc_ess_frac: float = 0.5             # ESS-adaptive trigger: resample when ESS < F * frac
    fkc_final_reweight: bool = True       # one final categorical reweight at end of each segment
    fkc_action_mode: str = "option_c"     # {option_c, split}: how to combine rot/trans in log_w
    fkc_num_particles: int = 1            # Skreta SMC particle count K (independent reverse SDEs per segment)
    fkc_particle_chunk: int = 0           # env/actor forward chunk along K (0 = no chunking)


@dataclass
class SampleResult:
    atom_traj: torch.Tensor  # [B, E, F, N, 37, 3]
    final_atom37: torch.Tensor  # [B, F, N, 37, 3]
    final_rigids: torch.Tensor  # [B, F, N, 7]


@dataclass
class ModelBundle:
    diffuser: se3_diffuser.SE3Diffuser
    actor: Optional[MetadynamicsActor]
    env_model: Optional[network.DyneTrion]


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def _resolve_weight_path(path_or_dir: str) -> str:
    if os.path.isdir(path_or_dir):
        candidate = os.path.join(path_or_dir, "pytorch_model.bin")
        if os.path.exists(candidate):
            return candidate
    return path_or_dir


def _conf_get(conf_obj, key: str, default):
    if conf_obj is None:
        return default
    if isinstance(conf_obj, dict):
        return conf_obj.get(key, default)
    if hasattr(conf_obj, "get"):
        try:
            return conf_obj.get(key, default)
        except TypeError:
            pass
    return getattr(conf_obj, key, default)


def _rmsd_kabsch_distance(
    pred_ca_h: torch.Tensor,
    res_mask: torch.Tensor,
    target_ca: Optional[torch.Tensor],
) -> torch.Tensor:
    """Kabsch-aligned CA RMSD between each history slot and the reference."""
    bsz, frame_t, hist_len, n_res, _ = pred_ca_h.shape
    ca_ref = target_ca if target_ca is not None else pred_ca_h[:, :, -1, :, :]
    ca_ref_flat = repeat(ca_ref, "b f n c -> (b f h) n c", h=hist_len)
    ca_h_flat = rearrange(pred_ca_h, "b f h n c -> (b f h) n c")
    mask_flat = repeat(res_mask, "b f n -> (b f h) n", h=hist_len)
    aligned = rmsdalign(ca_ref_flat, ca_h_flat, weights=mask_flat)
    sq = (aligned - ca_ref_flat).pow(2).sum(dim=-1)
    denom = mask_flat.sum(dim=-1).clamp_min(1.0)
    rmsd_flat = torch.sqrt((sq * mask_flat).sum(dim=-1) / denom)
    return rearrange(rmsd_flat, "(b f h) -> b f h", b=bsz, f=frame_t, h=hist_len)


def _contact_map_distance(
    pred_ca_h: torch.Tensor,
    res_mask: torch.Tensor,
    target_ca: Optional[torch.Tensor],
) -> torch.Tensor:
    """Pairwise CA-CA distance-map L2 between each history slot and the reference."""
    bsz, frame_t, hist_len, n_res, _ = pred_ca_h.shape
    ca_ref = target_ca if target_ca is not None else pred_ca_h[:, :, -1, :, :]
    d_ref = torch.cdist(ca_ref, ca_ref)  # [B,F,N,N]
    flat = rearrange(pred_ca_h, "b f h n c -> (b f h) n c")
    d_h_flat = torch.cdist(flat, flat)
    d_h = rearrange(d_h_flat, "(b f h) i j -> b f h i j", b=bsz, f=frame_t, h=hist_len)
    pair_mask = res_mask.unsqueeze(-1) * res_mask.unsqueeze(-2)  # [B,F,N,N]
    diff = (d_h - d_ref.unsqueeze(2)) * pair_mask.unsqueeze(2)
    n_valid = pair_mask.sum(dim=(-1, -2)).clamp_min(1.0)         # [B,F]
    d = diff.pow(2).sum(dim=(-1, -2)) / n_valid.unsqueeze(-1)    # [B,F,H]
    return d.sqrt()


# Ideal local backbone coords (Å), CA at origin; from openfold restype constants.
_BACKBONE_N_LOCAL = (-0.5272, 1.3593, 0.0)
_BACKBONE_C_LOCAL = (1.5260, 0.0, 0.0)


def _backbone_dihedrals(
    pos_n: torch.Tensor,
    pos_ca: torch.Tensor,
    pos_c: torch.Tensor,
) -> tuple:
    """Return (phi, psi) per residue for shape [..., N, 3]; boundaries are NaN."""
    eps = 1e-8

    def _dihedral(p1, p2, p3, p4):
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        b2n = b2 / (b2.norm(dim=-1, keepdim=True) + eps)
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)
        m1 = torch.cross(n1, b2n, dim=-1)
        x = (n1 * n2).sum(dim=-1)
        y = (m1 * n2).sum(dim=-1)
        return torch.atan2(y, x)

    nan = torch.full_like(pos_ca[..., 0], float("nan"))
    # phi_i uses (C_{i-1}, N_i, CA_i, C_i); defined for i=1..N-1
    phi_inner = _dihedral(pos_c[..., :-1, :], pos_n[..., 1:, :],
                          pos_ca[..., 1:, :], pos_c[..., 1:, :])
    phi = torch.cat([nan[..., :1], phi_inner], dim=-1)
    # psi_i uses (N_i, CA_i, C_i, N_{i+1}); defined for i=0..N-2
    psi_inner = _dihedral(pos_n[..., :-1, :], pos_ca[..., :-1, :],
                          pos_c[..., :-1, :], pos_n[..., 1:, :])
    psi = torch.cat([psi_inner, nan[..., -1:]], dim=-1)
    return phi, psi


def _dihedral_distance(
    pred_ca_h: torch.Tensor,
    pred_rot_h: torch.Tensor,
    res_mask: torch.Tensor,
    target_ca: Optional[torch.Tensor],
) -> torch.Tensor:
    """Mean (1−cos Δφ) + (1−cos Δψ) over residues with both angles defined.

    Reference frame uses the last history slot (or `target_ca` if provided —
    however φ/ψ need rotations, so target_ca on its own falls back to the slot's
    own rotations: i.e. `target_ca`-anchor mode is not supported for dihedral).
    """
    if target_ca is not None:
        raise ValueError("omega_method=dihedral does not support target_ca anchor")
    n_local = torch.tensor(_BACKBONE_N_LOCAL, device=pred_rot_h.device, dtype=pred_rot_h.dtype)
    c_local = torch.tensor(_BACKBONE_C_LOCAL, device=pred_rot_h.device, dtype=pred_rot_h.dtype)
    pos_ca = pred_ca_h  # [B,F,H,N,3]
    pos_n = torch.einsum("bfhnij,j->bfhni", pred_rot_h, n_local) + pos_ca
    pos_c = torch.einsum("bfhnij,j->bfhni", pred_rot_h, c_local) + pos_ca
    phi, psi = _backbone_dihedrals(pos_n, pos_ca, pos_c)  # [B,F,H,N]
    phi_ref = phi[:, :, -1:, :]
    psi_ref = psi[:, :, -1:, :]
    # Per-residue valid mask: both angles finite at this residue and ref, plus res_mask.
    finite = (
        torch.isfinite(phi) & torch.isfinite(psi) &
        torch.isfinite(phi_ref) & torch.isfinite(psi_ref)
    )
    valid = finite & (res_mask.unsqueeze(2) > 0.5)
    cos_dphi = torch.cos(phi - phi_ref)
    cos_dpsi = torch.cos(psi - psi_ref)
    per_res = (1.0 - cos_dphi) + (1.0 - cos_dpsi)
    per_res = torch.where(valid, per_res, torch.zeros_like(per_res))
    n_valid = valid.float().sum(dim=-1).clamp_min(1.0)  # [B,F,H]
    return per_res.sum(dim=-1) / n_valid


def _kabsch_rmsd_weights(
    pred_ca_h: torch.Tensor,
    res_mask: torch.Tensor,
    sigma: float,
    variant: str = "unnormalized",
    target_ca: Optional[torch.Tensor] = None,
    omega_method: str = "rmsd_kabsch",
    pred_rot_h: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return ω weights of shape [B,F,H] from a per-slot distance metric.

    Despite the legacy name, this dispatches by `omega_method`:
    `rmsd_kabsch` (default, Kabsch-aligned CA RMSD), `contact_map`
    (Frobenius distance between CA-CA distance maps), or `dihedral`
    (φ/ψ angular distance). The last slot of `pred_ca_h` is always the
    reference; `target_ca` swaps the reference for distance/contact methods.
    """
    if pred_ca_h.dim() != 5 or pred_ca_h.shape[-1] != 3:
        raise ValueError(f"pred_ca_h must be [B,F,H,N,3], got {tuple(pred_ca_h.shape)}")
    if res_mask.dim() != 3:
        raise ValueError(f"res_mask must be [B,F,N], got {tuple(res_mask.shape)}")
    out_dtype = pred_ca_h.dtype
    autocast_ctx = (
        torch.amp.autocast("cuda", enabled=False)
        if pred_ca_h.is_cuda
        else contextlib.nullcontext()
    )
    with autocast_ctx:
        pred_ca_h = pred_ca_h.float()
        res_mask = res_mask.to(device=pred_ca_h.device, dtype=pred_ca_h.dtype)
        bsz, frame_t, hist_len, n_res, _ = pred_ca_h.shape
        if tuple(res_mask.shape) != (bsz, frame_t, n_res):
            raise ValueError(
                "res_mask shape mismatch: "
                f"expected {(bsz, frame_t, n_res)}, got {tuple(res_mask.shape)}"
            )
        if hist_len == 1:
            return torch.ones((bsz, frame_t, 1), device=pred_ca_h.device, dtype=out_dtype)

        if target_ca is not None:
            if target_ca.shape != (bsz, frame_t, n_res, 3):
                raise ValueError(
                    f"target_ca shape mismatch: expected {(bsz, frame_t, n_res, 3)}, "
                    f"got {tuple(target_ca.shape)}"
                )
            target_ca = target_ca.to(device=pred_ca_h.device, dtype=pred_ca_h.dtype)

        method = str(omega_method).lower()
        if method == "rmsd_kabsch":
            d = _rmsd_kabsch_distance(pred_ca_h, res_mask, target_ca)
        elif method == "contact_map":
            d = _contact_map_distance(pred_ca_h, res_mask, target_ca)
        elif method == "dihedral":
            if pred_rot_h is None:
                raise ValueError("omega_method=dihedral requires pred_rot_h")
            d = _dihedral_distance(
                pred_ca_h, pred_rot_h.float(), res_mask, target_ca
            )
        else:
            raise ValueError(
                f"Unsupported omega_method={omega_method!r}; "
                "expected one of rmsd_kabsch, contact_map, dihedral"
            )

        sigma = max(float(sigma), 1e-6)
        w_raw = torch.exp(-d / sigma)

    occupancy = w_raw.sum(dim=-1, keepdim=True)
    variant = str(variant).lower()
    if variant == "normalized":
        weights = w_raw / (occupancy + 1e-8)
    elif variant == "unnormalized":
        weights = w_raw
    elif variant == "occupancy_scaled":
        direction = w_raw / (occupancy + 1e-8)
        occ_factor = occupancy / max(hist_len, 1)
        weights = direction * occ_factor
    else:
        raise ValueError(
            "Unsupported history_distance_variant="
            f"{variant!r}; expected one of normalized, unnormalized, occupancy_scaled"
        )
    return weights.to(dtype=out_dtype)


def _infer_lora_rank_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    ranks = {
        int(value.shape[1])
        for key, value in state_dict.items()
        if key.endswith(".lora_A") and torch.is_tensor(value) and value.ndim == 2
    }
    if not ranks:
        return None
    if len(ranks) != 1:
        raise ValueError(f"Inconsistent LoRA ranks found in checkpoint: {sorted(ranks)}")
    return next(iter(ranks))


def _maybe_prepare_actor_for_state_dict(
    actor: MetadynamicsActor,
    state_dict: Dict[str, torch.Tensor],
    exp_conf,
) -> Optional[Dict[str, float]]:
    rank = _infer_lora_rank_from_state_dict(state_dict)
    if rank is None:
        return None
    alpha_raw = _conf_get(exp_conf, "lora_alpha", None)
    alpha = 16.0 if alpha_raw in (None, "", "null") else float(alpha_raw)
    modules = int(actor.inject_lora(rank=rank, alpha=alpha))
    return {
        "rank": rank,
        "alpha": alpha,
        "modules": modules,
    }


def _move_to_device(batch_feats: Dict, device: torch.device) -> Dict:
    return tree.map_structure(
        lambda x: x.to(device) if torch.is_tensor(x) else x,
        batch_feats,
    )


class ReverseSampler:
    """Actor-only reverse sampler, aligned with supervised_pretrain.inference_fn."""

    def __init__(
        self,
        *,
        actor: MetadynamicsActor,
        diffuser: se3_diffuser.SE3Diffuser,
        runtime: RuntimeConfig,
        device: torch.device,
    ):
        self.actor = actor
        self.diffuser = diffuser
        self.runtime = runtime
        self.device = device

    def _flatten_for_actor(self, batch_feats: Dict[str, torch.Tensor]):
        rigids_t = batch_feats["rigids_t"]  # [B, F, N, 7]
        bsz, frame_t, n_res = rigids_t.shape[:3]

        ref_rigids = batch_feats["ref_rigids_0"]
        if ref_rigids.ndim == 4:
            ref_rigids = ref_rigids[:, 0]  # [B, N, 7]

        flat = {
            "rigids_t": rearrange(rigids_t, "b f n c -> (b f) n c"),
            "aatype": rearrange(batch_feats["aatype"], "b f n -> (b f) n"),
            "res_mask": rearrange(batch_feats["res_mask"], "b f n -> (b f) n").float(),
            "fixed_mask": rearrange(
                batch_feats["fixed_mask"], "b f n -> (b f) n"
            ).float(),
            "node_repr": repeat(batch_feats["node_repr"], "b n d -> (b f) n d", f=frame_t),
            "edge_repr": repeat(
                batch_feats["edge_repr"], "b i j d -> (b f) i j d", f=frame_t
            ),
            "ref_rigids_0": repeat(ref_rigids, "b n c -> (b f) n c", f=frame_t),
        }
        return flat, bsz, frame_t, n_res

    def sample(self, batch_feats: Dict[str, torch.Tensor]) -> SampleResult:
        self.actor.eval()
        flat_feats, bsz, frame_t, n_res = self._flatten_for_actor(batch_feats)
        batch_size = flat_feats["rigids_t"].shape[0]

        reverse_steps = np.linspace(self.runtime.min_t, 1.0, self.runtime.num_t)[::-1]
        dt = 1.0 / self.runtime.num_t
        last_pred_dict = None

        with torch.no_grad():
            for t in reverse_steps:
                t_tensor = torch.full(
                    (batch_size,),
                    float(t),
                    device=self.device,
                    dtype=flat_feats["rigids_t"].dtype,
                )
                flat_feats["t"] = t_tensor

                pred_dict = self.actor(
                    rigids_t=flat_feats["rigids_t"],
                    t=t_tensor,
                    node_repr=flat_feats["node_repr"],
                    edge_repr=flat_feats["edge_repr"],
                    ref_rigids_0=flat_feats["ref_rigids_0"],
                    res_mask=flat_feats["res_mask"],
                    memory_cache=None,
                    context_rigids=None,
                )
                last_pred_dict = pred_dict

                if t > self.runtime.min_t:
                    pred_x0 = pred_dict["final_rigids"]
                    x_t = Rigid.from_tensor_7(flat_feats["rigids_t"])

                    rot_score = self.diffuser.calc_rot_score(
                        x_t.get_rots(),
                        pred_x0.get_rots(),
                        t_tensor,
                    )
                    trans_score = self.diffuser.calc_trans_score(
                        x_t.get_trans(),
                        pred_x0.get_trans(),
                        t_tensor[:, None, None],
                        use_torch=True,
                    )
                    mask = flat_feats["res_mask"][..., None]
                    rot_score = rot_score * mask
                    trans_score = trans_score * mask
                    diffuse_mask = (1.0 - flat_feats["fixed_mask"]) * flat_feats["res_mask"]

                    next_rigid = self.diffuser.reverse(
                        rigid_t=x_t,
                        rot_score=rot_score.detach().cpu().numpy(),
                        trans_score=trans_score.detach().cpu().numpy(),
                        t=float(t),
                        dt=dt,
                        diffuse_mask=diffuse_mask.detach().cpu().numpy(),
                        center=self.runtime.center,
                        noise_scale=self.runtime.noise_scale,
                        device=self.device,
                    )
                else:
                    next_rigid = pred_dict["final_rigids"]

                flat_feats["rigids_t"] = next_rigid.to_tensor_7().to(self.device)

        if last_pred_dict is None:
            raise RuntimeError("ReverseSampler produced no prediction.")

        final_rigid = Rigid.from_tensor_7(flat_feats["rigids_t"])
        final_atom37 = all_atom.compute_backbone_atom37(
            bb_rigids=final_rigid,
            aatypes=flat_feats["aatype"].long(),
            torsions=last_pred_dict["angles"],
        )[0]

        final_atom37 = rearrange(
            final_atom37, "(b f) n a c -> b f n a c", b=bsz, f=frame_t
        )
        final_rigids = rearrange(
            flat_feats["rigids_t"], "(b f) n c -> b f n c", b=bsz, f=frame_t
        )
        atom_traj = final_atom37.unsqueeze(1)  # [B, 1, F, N, 37, 3]

        return SampleResult(
            atom_traj=atom_traj,
            final_atom37=final_atom37,
            final_rigids=final_rigids,
        )


class BiasedActorSampler:
    """Env reverse process with optional actor score blending and extrapolation."""

    def __init__(
        self,
        *,
        env_model: network.DyneTrion,
        diffuser: se3_diffuser.SE3Diffuser,
        runtime: RuntimeConfig,
        device: torch.device,
        model_conf: DictConfig,
        actor: Optional[MetadynamicsActor] = None,
    ):
        self.env_model = env_model
        self.actor = actor
        self.diffuser = diffuser
        self.runtime = runtime
        self.device = device
        self.model_conf = model_conf

    def _set_t_feats(self, feats: Dict[str, torch.Tensor], t: float, t_placeholder: torch.Tensor):
        feats["t"] = t * t_placeholder
        rot_score_scaling, trans_score_scaling = self.diffuser.score_scaling(t)
        feats["rot_score_scaling"] = rot_score_scaling * t_placeholder
        feats["trans_score_scaling"] = trans_score_scaling * t_placeholder
        return feats

    def _flatten_actor_inputs(
        self, feats: Dict[str, torch.Tensor], history: torch.Tensor | None = None
    ):
        bsz, frame_t, n_res = feats["aatype"].shape
        ref_rigids = feats["ref_rigids_0"]
        if ref_rigids.ndim == 4:   # [B, R, N, 7] -> first ref
            ref_rigids = ref_rigids[:, 0]  # [B, N, 7]

        rigids_t_actor = feats.get("rigids_t_actor", feats["rigids_t"])
        if rigids_t_actor.shape[:3] != (bsz, frame_t, n_res):
            raise ValueError(
                f"rigids_t_actor shape mismatch: got {tuple(rigids_t_actor.shape)}, "
                f"expected [B,F,N,7] with B={bsz}, F={frame_t}, N={n_res}"
            )

        # No-history path: standard [B*F, N, 7] flatten.
        if history is None:
            flat = {
                "rigids_t": rearrange(rigids_t_actor, "b f n c -> (b f) n c"),
                "res_mask": rearrange(feats["res_mask"], "b f n -> (b f) n").float(),
                "fixed_mask": rearrange(feats["fixed_mask"], "b f n -> (b f) n").float(),
                "node_repr": repeat(feats["node_repr"], "b n d -> (b f) n d", f=frame_t),
                "edge_repr": repeat(feats["edge_repr"], "b i j d -> (b f) i j d", f=frame_t),
                "ref_rigids_0": repeat(ref_rigids, "b n c -> (b f) n c", f=frame_t),
            }
            return flat, {"B": bsz, "F": frame_t, "H": 0, "paired": False}

        # History-pair path:
        # history: [B, H, N, 7]
        if history.ndim != 4:
            raise ValueError(f"history must be [B,H,N,7], got {tuple(history.shape)}")
        if history.shape[0] != bsz or history.shape[2] != n_res or history.shape[3] != 7:
            raise ValueError(
                f"history shape mismatch, got {tuple(history.shape)}; "
                f"expected [B,H,N,7] with B={bsz}, N={n_res}"
            )
        hist_len = history.shape[1]
        if hist_len < 1:
            raise ValueError("history length H must be >= 1")

        # Build pair tensors: [B, F, H, N, 7]
        query_pair = repeat(rigids_t_actor, "b f n c -> b f h n c", h=hist_len)
        hist_pair = repeat(history, "b h n c -> b f h n c", f=frame_t)

        flat = {
            "rigids_t": rearrange(query_pair, "b f h n c -> (b f h) n c"),
            "res_mask": repeat(feats["res_mask"], "b f n -> (b f h) n", h=hist_len).float(),
            "fixed_mask": repeat(
                feats["fixed_mask"], "b f n -> (b f h) n", h=hist_len
            ).float(),
            "node_repr": repeat(
                feats["node_repr"], "b n d -> (b f h) n d", f=frame_t, h=hist_len
            ),
            "edge_repr": repeat(
                feats["edge_repr"], "b i j d -> (b f h) i j d", f=frame_t, h=hist_len
            ),
            # Pair each query frame with each history rigid as pair reference.
            "ref_rigids_0": rearrange(hist_pair, "b f h n c -> (b f h) n c"),
        }
        return flat, {"B": bsz, "F": frame_t, "H": hist_len, "paired": True}

    def _history_decay_weights(
        self,
        t_tensor: torch.Tensor,
        frame_t: int,
        hist_len: int,
        pred_ca_h: Optional[torch.Tensor] = None,
        res_mask: Optional[torch.Tensor] = None,
        target_ca: Optional[torch.Tensor] = None,
        pred_rot_h: Optional[torch.Tensor] = None,
    ):
        if hist_len <= 0:
            raise ValueError(f"hist_len must be > 0, got {hist_len}")
        mode = str(getattr(self.runtime, "history_decay_mode", "time"))
        if mode == "distance":
            if pred_ca_h is None or res_mask is None:
                raise ValueError("distance mode requires pred_ca_h and res_mask")
            return _kabsch_rmsd_weights(
                pred_ca_h=pred_ca_h,
                res_mask=res_mask,
                sigma=float(getattr(self.runtime, "history_distance_sigma", 1.0)),
                variant=str(getattr(self.runtime, "history_distance_variant", "unnormalized")),
                target_ca=target_ca,
                omega_method=str(getattr(self.runtime, "omega_method", "rmsd_kabsch")),
                pred_rot_h=pred_rot_h,
            )

        t_bf = repeat(t_tensor, "b 1 -> b f", f=frame_t).clamp(0.0, 1.0)
        base_lambda = float(self.runtime.history_decay_lambda)
        t_scale = float(self.runtime.history_decay_t_scale)
        lam = base_lambda * (1.0 + t_scale * (1.0 - t_bf))

        hist_idx = torch.arange(hist_len, device=t_tensor.device, dtype=t_tensor.dtype)
        hist_age = (hist_len - 1) - hist_idx
        weights = torch.exp(-lam[..., None] * hist_age[None, None, :])
        return weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)


    def _compute_actor_guidance(
        self,
        feats: Dict[str, torch.Tensor],
        t_tensor: torch.Tensor,
        base_rot: torch.Tensor,
        base_trans: torch.Tensor,
        history=None,
    ):
        if self.actor is None:
            return None, None
        if history is not None:
            assert history.ndim == 4
            assert history.shape[2] == feats["aatype"].shape[-1]
        self.actor.eval()
        flat_feats, meta = self._flatten_actor_inputs(feats, history=history)
        bsz, frame_t, hist_len = meta["B"], meta["F"], meta["H"]
        if meta["paired"]:
            t_flat = repeat(t_tensor, "b 1 -> (b f h)", f=frame_t, h=hist_len).reshape(-1)
        else:
            t_flat = repeat(t_tensor, "b 1 -> (b f)", f=frame_t).reshape(-1)

        with torch.no_grad():
            actor_out = self.actor(
                rigids_t=flat_feats["rigids_t"],
                t=t_flat,
                node_repr=flat_feats["node_repr"],
                edge_repr=flat_feats["edge_repr"],
                ref_rigids_0=flat_feats["ref_rigids_0"],
                res_mask=flat_feats["res_mask"],
                memory_cache=None,
                context_rigids=None,
            )
            pred_x0 = actor_out["final_rigids"]
            x_t = Rigid.from_tensor_7(flat_feats["rigids_t"])

            actor_rot = self.diffuser.calc_rot_score(x_t.get_rots(), pred_x0.get_rots(), t_flat)
            actor_trans = self.diffuser.calc_trans_score(
                x_t.get_trans(), pred_x0.get_trans(), t_flat[:, None, None], use_torch=True
            )
            mask = flat_feats["res_mask"][..., None]
            actor_rot = actor_rot * mask
            actor_trans = actor_trans * mask

        cond_mode = str(getattr(self.runtime, "actor_condition_mode", "none"))
        target_ca = None
        if cond_mode == "target_anchor":
            tgt = feats.get("target_atom37_pos")
            if tgt is None:
                raise ValueError(
                    "actor_condition_mode=target_anchor requires "
                    "feats['target_atom37_pos'] (set holo_pdb_path or target_pdb_path in CSV)"
                )
            # tgt: [B,F,N,37,3] → CA = atom37 idx 1 → [B,F,N,3]
            target_ca = tgt[..., 1, :].to(device=base_rot.device, dtype=base_rot.dtype)

        toward = str(getattr(self.runtime, "bias_direction", "away")) == "toward"

        if meta["paired"]:
            actor_rot_h = rearrange(actor_rot, "(b f h) n c -> b f h n c", b=bsz, f=frame_t, h=hist_len)
            actor_trans_h = rearrange(actor_trans, "(b f h) n c -> b f h n c", b=bsz, f=frame_t, h=hist_len)
            pred_ca_h = rearrange(pred_x0.get_trans().detach(), "(b f h) n c -> b f h n c", b=bsz, f=frame_t, h=hist_len)
            omega_method = str(getattr(self.runtime, "omega_method", "rmsd_kabsch"))
            if omega_method == "dihedral":
                pred_rot_h = rearrange(
                    pred_x0.get_rots().get_rot_mats().detach(),
                    "(b f h) n i j -> b f h n i j", b=bsz, f=frame_t, h=hist_len,
                )
            else:
                pred_rot_h = None
            res_mask_bfh = rearrange(flat_feats["res_mask"].detach(), "(b f h) n -> b f h n", b=bsz, f=frame_t, h=hist_len)
            res_mask_bf = res_mask_bfh[:, :, 0, :]
            hist_w = self._history_decay_weights(
                t_tensor,
                frame_t=frame_t,
                hist_len=hist_len,
                pred_ca_h=pred_ca_h,
                res_mask=res_mask_bf,
                target_ca=target_ca,
                pred_rot_h=pred_rot_h,
            )
            if toward:
                bias_rot = ((actor_rot_h - base_rot.unsqueeze(2)) * hist_w[..., None, None]).sum(dim=2)
                bias_trans = ((actor_trans_h - base_trans.unsqueeze(2)) * hist_w[..., None, None]).sum(dim=2)
            else:
                bias_rot = ((base_rot.unsqueeze(2) - actor_rot_h) * hist_w[..., None, None]).sum(dim=2)
                bias_trans = ((base_trans.unsqueeze(2) - actor_trans_h) * hist_w[..., None, None]).sum(dim=2)
        else:
            actor_rot = rearrange(actor_rot, "(b f) n c -> b f n c", b=bsz, f=frame_t)
            actor_trans = rearrange(actor_trans, "(b f) n c -> b f n c", b=bsz, f=frame_t)
            if toward:
                bias_rot = actor_rot - base_rot
                bias_trans = actor_trans - base_trans
            else:
                bias_rot = base_rot - actor_rot
                bias_trans = base_trans - actor_trans

        return bias_rot, bias_trans

    def _prepare_init_feats(self, batch_feats: Dict[str, torch.Tensor]):
        curr_feats = {
            k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch_feats.items()
        }
        bsz, frame_t, n_res = curr_feats["aatype"].shape
        ref_sample = self.diffuser.sample_ref(n_samples=bsz * n_res * frame_t, as_tensor_7=True)
        ref_sample["rigids_t"] = rearrange(
            ref_sample["rigids_t"], "f (b n) c -> b f n c", b=bsz, f=frame_t, n=n_res
        )
        curr_feats["rigids_t"] = ref_sample["rigids_t"].to(self.device)
        if "rigids_t_actor" not in curr_feats:
            curr_feats["rigids_t_actor"] = curr_feats["rigids_t"].clone()
        curr_feats["fixed_mask"] = torch.zeros(
            (bsz, frame_t, n_res), device=self.device, dtype=curr_feats["res_mask"].dtype
        )
        curr_feats["sc_ca_t"] = torch.zeros((bsz, frame_t, n_res, 3), device=self.device)
        return curr_feats

    def _update_ref_with_prediction(
        self,
        curr_feats: Dict[str, torch.Tensor],
        atom_pred: torch.Tensor,
        rigid_pred: torch.Tensor,
    ):
        bsz, frame_t, n_res, _ = rigid_pred.shape
        motion_number = int(getattr(self.model_conf, "motion_number", 1))
        concat_rigids = torch.cat([
            curr_feats["motion_rigids_0"],
            rigid_pred,
        ], dim=1)

        concat_atoms = torch.cat([
            curr_feats["motion_atom37_pos"],
            atom_pred,
        ], dim=1)
        curr_feats["motion_rigids_0"] = concat_rigids[:, -motion_number:]
        curr_feats["motion_atom37_pos"] = concat_atoms[:, -motion_number:]

        ref_sample = self.diffuser.sample_ref(n_samples=bsz * n_res * frame_t, as_tensor_7=True)
        ref_sample["rigids_t"] = rearrange(
            ref_sample["rigids_t"], "f (b n) c -> b f n c", b=bsz, f=frame_t, n=n_res
        )
        curr_feats["rigids_t"] = ref_sample["rigids_t"].to(self.device)
        curr_feats["rigids_t_actor"] = curr_feats["rigids_t"].clone()
        return curr_feats

    def _single_rollout(self, curr_feats: Dict[str, torch.Tensor], history=None):
        bsz = curr_feats["aatype"].shape[0]
        t_placeholder = torch.ones((bsz, 1), device=self.device)
        reverse_steps = np.linspace(self.runtime.min_t, 1.0, self.runtime.num_t)[::-1]
        dt = 1.0 / self.runtime.num_t

        if self.model_conf.embed.embed_self_conditioning:
            with torch.no_grad():
                curr_feats = self._set_t_feats(curr_feats, float(reverse_steps[0]), t_placeholder)
                warm_out = self.env_model(curr_feats, is_training=False)
                curr_feats["sc_ca_t"] = warm_out["rigids"][..., 4:]

        final_prot = None
        final_rigid = None

        for step_idx, t in enumerate(reverse_steps):
            curr_feats = self._set_t_feats(curr_feats, float(t), t_placeholder)
            is_last_step = step_idx == len(reverse_steps) - 1

            with torch.no_grad():
                model_out = self.env_model(curr_feats, is_training=False)
                rot_score = model_out["rot_score"]
                trans_score = model_out["trans_score"]
                rigid_pred = model_out["rigids"]

                if self.model_conf.embed.embed_self_conditioning:
                    curr_feats["sc_ca_t"] = rigid_pred[..., 4:]

                bias_rot, bias_trans = self._compute_actor_guidance(
                    curr_feats,
                    curr_feats["t"],
                    base_rot=rot_score,
                    base_trans=trans_score,
                    history=history,
                )
                if bias_rot is not None and bias_trans is not None:
                    blend = float(self.runtime.actor_blend_alpha)
                    if os.environ.get("PURE_SCALE_ABLATION") == "1":
                        rot_score = (1 + blend) * rot_score
                        trans_score = (1 + blend) * trans_score
                    else:
                        rot_score = rot_score + blend * bias_rot
                        trans_score = trans_score + blend * bias_trans
                mask = curr_feats["res_mask"][..., None]
                rot_score = rot_score * mask
                trans_score = trans_score * mask

                if t > self.runtime.min_t:
                    diffuse_mask = (1.0 - curr_feats["fixed_mask"]) * curr_feats["res_mask"]
                    next_rigid, _, _, _ = self.diffuser.reverse_with_logprob(
                        rigid_t=Rigid.from_tensor_7(curr_feats["rigids_t"]),
                        rot_score=rot_score,
                        trans_score=trans_score,
                        t=float(t),
                        dt=dt,
                        diffuse_mask=diffuse_mask,
                        noise_scale=self.runtime.noise_scale,
                        device=self.device,
                    )
                else:
                    next_rigid = Rigid.from_tensor_7(model_out["rigids"])

                curr_feats["rigids_t"] = next_rigid.to_tensor_7()
                curr_feats["rigids_t_actor"] = curr_feats["rigids_t"].clone()

                if is_last_step:
                    final_rigid = model_out["rigids"]
                    final_prot = all_atom.compute_backbone_atom37(
                        bb_rigids=next_rigid,
                        aatypes=curr_feats["aatype"],
                        torsions=model_out["angles"],
                    )[0]

        if final_prot is None or final_rigid is None:
            raise RuntimeError("Biased rollout failed to produce final structures.")
        return final_prot, final_rigid, curr_feats

    def _init_history_chunk(self, curr_feats: Dict[str, torch.Tensor]) -> list[torch.Tensor]:
        history_chunk = []
        ref = curr_feats.get("ref_rigids_0", None)
        if ref is None:
            return history_chunk
        if ref.ndim == 4:  # [B, R, N, 7]
            for i in range(ref.shape[1]):
                history_chunk.append(ref[:, i].detach().clone())
        elif ref.ndim == 3:  # [B, N, 7]
            history_chunk.append(ref.detach().clone())
        return history_chunk

    def _build_history_tensor(
        self, history_chunk: list[torch.Tensor]
    ) -> torch.Tensor | None:
        if len(history_chunk) == 0:
            return None
        return torch.stack(history_chunk, dim=1)  # [B, H, N, 7]

    def _push_history(
        self,
        history_chunk: list[torch.Tensor],
        history_frames: torch.Tensor | None,  # [B, H_new, N, 7]
        max_history: int,
    ) -> list[torch.Tensor]:
        if max_history <= 0:
            return []
        if history_frames is None:
            return history_chunk[-max_history:]
        if history_frames.ndim == 3:
            history_frames = history_frames.unsqueeze(1)
        if history_frames.ndim != 4 or history_frames.shape[-1] != 7:
            raise ValueError(
                f"history_frames must be [B,H,N,7], got {tuple(history_frames.shape)}"
            )
        if history_frames.shape[1] == 0:
            return history_chunk[-max_history:]
        for frame_hist in history_frames.detach().clone().unbind(dim=1):
            history_chunk.append(frame_hist)  # each is [B, N, 7]
        if len(history_chunk) > max_history:
            history_chunk = history_chunk[-max_history:]
        return history_chunk

    def sample(self, batch_feats: Dict[str, torch.Tensor]) -> SampleResult:
        self.env_model.eval()
        curr_feats = self._prepare_init_feats(batch_feats)
        atom_steps, rigid_steps = [], []
        history_chunk = self._init_history_chunk(curr_feats) if self.runtime.max_history > 0 else []
        pbar = tqdm(
            range(self.runtime.extrapolation_time),
            total=self.runtime.extrapolation_time,
            desc="Extrapolation",
            leave=False,
        )
        for ext_idx in pbar:
            pbar.set_description(f"Extrapolation {ext_idx+1}/{self.runtime.extrapolation_time}")
            history = self._build_history_tensor(history_chunk)  # [B,H,N,7] or None
            atom_pred, rigid_pred, curr_feats = self._single_rollout(
                curr_feats, history=history
            )
            motion_keep = max(1, int(getattr(self.model_conf, "motion_number", 1)))
            motion_keep = min(motion_keep, rigid_pred.shape[1])
            to_history = torch.cat(
                [curr_feats["motion_rigids_0"], rigid_pred[:, :-motion_keep]],
                dim=1,
            )
            history_chunk = self._push_history(
                history_chunk,
                to_history,
                max_history=self.runtime.max_history,
            )
            atom_steps.append(atom_pred)
            rigid_steps.append(rigid_pred)
            curr_feats = self._update_ref_with_prediction(curr_feats, atom_pred, rigid_pred)

        atom_traj = torch.stack(atom_steps, dim=1)  # [B, E, F, N, 37, 3]
        rigid_traj = torch.stack(rigid_steps, dim=1)  # [B, E, F, N, 7]
        return SampleResult(
            atom_traj=atom_traj,
            final_atom37=atom_traj[:, -1],
            final_rigids=rigid_traj[:, -1],
        )


class FKCBiasedSampler(BiasedActorSampler):
    """BiasedActorSampler + Feynman-Kac correction via SMC over the F dimension.

    Per-step weight update (Skreta et al. 2025 Props 3.1-3.4, guidance-only):
        dlog w / dt = -0.5 * alpha(1+alpha) * sum_c[ g_c(t)^2 / std_c(score, t)^2 * |bias_c|^2 ]
    where c iterates over (rot, trans) tracks. The 1/std^2 factor makes
    rot/trans contributions commensurable in SE(3).

    Resampling policy: ESS-adaptive. At each reverse step compute
        ESS = (sum exp(log_w))^2 / sum exp(2 log_w)
    and resample (multinomial along F) when ESS < F * fkc_ess_frac. Set
    fkc_ess_frac<=0 to fall back to fixed-interval resampling every fkc_resample_k
    steps (legacy behaviour).

    Final reweight: one categorical resample at the end of each extrapolation
    segment if fkc_final_reweight is True, so the returned structures and the
    history fed to the next segment are unbiased w.r.t. the FK target.

    fkc_beta is a multiplicative knob defaulting to 1.0 (= principled). The
    omitted Laplacian/divergence term would require Hutchinson estimation.
    """

    @staticmethod
    def _gather_K(tensor, idx):
        """Gather along batch dim 0. tensor: [K, ...], idx: [K]. Returns [K, ...]."""
        return tensor[idx]

    def _multinomial_resample_idx(self, log_w):
        """log_w shape [K] (1D) → idx [K]. Zeros log_w in place."""
        probs = torch.softmax(log_w, dim=-1)
        K = log_w.shape[-1]
        idx = torch.multinomial(probs, K, replacement=True)
        log_w.zero_()
        return idx

    def _resample_state(self, curr_feats, log_w):
        idx = self._multinomial_resample_idx(log_w)
        for key in ("rigids_t", "rigids_t_actor", "sc_ca_t"):
            if key in curr_feats:
                curr_feats[key] = self._gather_K(curr_feats[key], idx)
        return curr_feats, idx

    @staticmethod
    def _tile_feats(curr_feats, K):
        """Tile every tensor value in curr_feats along dim 0 by K-fold."""
        return {
            k: (v.repeat_interleave(K, dim=0) if torch.is_tensor(v) else v)
            for k, v in curr_feats.items()
        }

    @staticmethod
    def _slice_feats(curr_feats, K_total, idx):
        """Slice every K-batch tensor to idx (scalar or slice). Non-K tensors pass through."""
        return {
            k: (v[idx] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == K_total else v)
            for k, v in curr_feats.items()
        }

    def _chunked_apply(self, fn, curr_feats, chunk):
        """Apply fn(sub_feats) → dict|tuple|tensor with K dim chunked. Concat outputs along dim 0."""
        K = next((v.shape[0] for v in curr_feats.values() if torch.is_tensor(v) and v.dim() > 0), 0)
        if chunk <= 0 or chunk >= K:
            return fn(curr_feats)
        outs = []
        for k0 in range(0, K, chunk):
            k1 = min(k0 + chunk, K)
            sub = self._slice_feats(curr_feats, K, slice(k0, k1))
            outs.append(fn(sub))
        if isinstance(outs[0], dict):
            return {k: torch.cat([o[k] for o in outs], dim=0) for k in outs[0].keys()}
        if isinstance(outs[0], tuple):
            return tuple(torch.cat([o[i] for o in outs], dim=0) for i in range(len(outs[0])))
        return torch.cat(outs, dim=0)

    def _single_rollout(self, curr_feats, history=None):
        # Tile to K particles along batch dim. Each particle = one independent reverse SDE
        # trajectory (Skreta 2025). All curr_feats values (incl. motion conditional) are
        # tiled K-fold; per-particle state (rigids_t etc.) is later gathered after resample.
        K_particles = max(1, int(self.runtime.fkc_num_particles))
        chunk = int(self.runtime.fkc_particle_chunk)
        orig_bsz = curr_feats["aatype"].shape[0]
        if K_particles > 1:
            curr_feats = self._tile_feats(curr_feats, K_particles)
            if history is not None and torch.is_tensor(history):
                history = history.repeat_interleave(K_particles, dim=0)
        bsz, frame_t, _ = curr_feats["aatype"].shape
        K_total = bsz  # = orig_bsz * K_particles
        t_placeholder = torch.ones((bsz, 1), device=self.device)
        reverse_steps = np.linspace(self.runtime.min_t, 1.0, self.runtime.num_t)[::-1]
        dt = 1.0 / self.runtime.num_t

        if self.model_conf.embed.embed_self_conditioning:
            with torch.no_grad():
                curr_feats = self._set_t_feats(curr_feats, float(reverse_steps[0]), t_placeholder)
                warm_out = self._chunked_apply(
                    lambda f: self.env_model(f, is_training=False), curr_feats, chunk
                )
                curr_feats["sc_ca_t"] = warm_out["rigids"][..., 4:]

        beta = float(self.runtime.fkc_beta)
        alpha = float(self.runtime.actor_blend_alpha)
        # Skreta 2025 Prop 3.1 (CFG + FKC):
        #   d log w / dt = (g²/2) · β(β-1) · ‖s_env − s_actor‖²
        # Our CFG (1+α)·s_env − α·s_actor maps to q¹^(1-β)·q²^β with β = −α
        beta_fkc = -alpha
        fkc_coef = 0.5 * beta_fkc * (beta_fkc - 1)   # = +α(α+1)/2 for extrapolation
        ess_frac = float(self.runtime.fkc_ess_frac)
        ess_threshold = K_total * ess_frac if ess_frac > 0 else None
        K_legacy = max(1, int(self.runtime.fkc_resample_k))  # fixed-K resample interval
        final_reweight = bool(self.runtime.fkc_final_reweight)
        action_mode = str(self.runtime.fkc_action_mode)
        # log_w shape [K_total] (1D, per-particle weight)
        log_w = torch.zeros((K_total,), device=self.device)
        log_w_rot = torch.zeros((K_total,), device=self.device) if action_mode == "split" else None
        log_w_trans = torch.zeros((K_total,), device=self.device) if action_mode == "split" else None

        final_prot = None
        final_rigid = None

        for step_idx, t in enumerate(reverse_steps):
            curr_feats = self._set_t_feats(curr_feats, float(t), t_placeholder)
            is_last_step = step_idx == len(reverse_steps) - 1

            with torch.no_grad():
                model_out = self._chunked_apply(
                    lambda f: self.env_model(f, is_training=False), curr_feats, chunk
                )
                rot_score = model_out["rot_score"]
                trans_score = model_out["trans_score"]
                rigid_pred = model_out["rigids"]

                if self.model_conf.embed.embed_self_conditioning:
                    curr_feats["sc_ca_t"] = rigid_pred[..., 4:]

                # Actor needs chunking too — IPA layer's q_pts × k_pts is O(N²·K) memory.
                if chunk > 0 and chunk < K_total:
                    bias_rot_parts, bias_trans_parts = [], []
                    actor_disabled = False
                    for k0 in range(0, K_total, chunk):
                        k1 = min(k0 + chunk, K_total)
                        sub_feats = self._slice_feats(curr_feats, K_total, slice(k0, k1))
                        sub_hist = history[k0:k1] if (history is not None and torch.is_tensor(history)) else history
                        br, bt = self._compute_actor_guidance(
                            sub_feats, sub_feats["t"],
                            base_rot=rot_score[k0:k1], base_trans=trans_score[k0:k1],
                            history=sub_hist,
                        )
                        if br is None:
                            actor_disabled = True
                            break
                        bias_rot_parts.append(br)
                        bias_trans_parts.append(bt)
                    if actor_disabled:
                        bias_rot, bias_trans = None, None
                    else:
                        bias_rot = torch.cat(bias_rot_parts, dim=0)
                        bias_trans = torch.cat(bias_trans_parts, dim=0)
                else:
                    bias_rot, bias_trans = self._compute_actor_guidance(
                        curr_feats, curr_feats["t"],
                        base_rot=rot_score, base_trans=trans_score, history=history,
                    )
                if bias_rot is not None and bias_trans is not None:
                    blend = float(self.runtime.actor_blend_alpha)
                    if os.environ.get("PURE_SCALE_ABLATION") == "1":
                        rot_score = (1 + blend) * rot_score
                        trans_score = (1 + blend) * trans_score
                    else:
                        rot_score = rot_score + blend * bias_rot
                        trans_score = trans_score + blend * bias_trans
                    g2_rot = float(self.diffuser._so3_diffuser.diffusion_coef(float(t))) ** 2
                    g2_trans = float(self.diffuser._r3_diffuser.diffusion_coef(float(t))) ** 2
                    coef = beta * fkc_coef * dt
                    # Sum bias² over all per-particle dims (F, N, rot/trans dim) → [K_total]
                    bias_rot_sq = bias_rot.pow(2).flatten(1).sum(dim=-1)
                    bias_trans_sq = bias_trans.pow(2).flatten(1).sum(dim=-1)
                    if action_mode == "split":
                        # Raw bias · g²(t), no score_scaling. Each component accumulates
                        # independently; combined via logsumexp-centered product at resample.
                        log_w_rot.add_(coef * g2_rot * bias_rot_sq)
                        log_w_trans.add_(coef * g2_trans * bias_trans_sq)
                        log_w = (
                            (log_w_rot - torch.logsumexp(log_w_rot, dim=-1, keepdim=True))
                            + (log_w_trans - torch.logsumexp(log_w_trans, dim=-1, keepdim=True))
                        )
                    else:
                        # option_c: per-component z-score. rot_score_scaling stores std(score),
                        # trans_score_scaling stores 1/std(score) (different conventions).
                        rot_sc = float(curr_feats["rot_score_scaling"].reshape(-1)[0])
                        trans_sc = float(curr_feats["trans_score_scaling"].reshape(-1)[0])
                        w_rot = g2_rot / max(rot_sc * rot_sc, 1e-8)
                        w_trans = g2_trans * (trans_sc * trans_sc)
                        bias_action = w_rot * bias_rot_sq + w_trans * bias_trans_sq
                        log_w = log_w + coef * bias_action

                mask = curr_feats["res_mask"][..., None]
                rot_score = rot_score * mask
                trans_score = trans_score * mask

                if t > self.runtime.min_t:
                    diffuse_mask = (1.0 - curr_feats["fixed_mask"]) * curr_feats["res_mask"]
                    next_rigid, _, _, _ = self.diffuser.reverse_with_logprob(
                        rigid_t=Rigid.from_tensor_7(curr_feats["rigids_t"]),
                        rot_score=rot_score,
                        trans_score=trans_score,
                        t=float(t),
                        dt=dt,
                        diffuse_mask=diffuse_mask,
                        noise_scale=self.runtime.noise_scale,
                        device=self.device,
                    )
                else:
                    next_rigid = Rigid.from_tensor_7(model_out["rigids"])

                curr_feats["rigids_t"] = next_rigid.to_tensor_7()
                curr_feats["rigids_t_actor"] = curr_feats["rigids_t"].clone()

                if not is_last_step:
                    did_resample = False
                    if ess_threshold is not None:
                        # ESS-adaptive trigger (across K particles)
                        log_w_norm = log_w - torch.logsumexp(log_w, dim=-1, keepdim=True)
                        ess = torch.exp(-torch.logsumexp(2.0 * log_w_norm, dim=-1))  # scalar
                        if os.environ.get("FKC_DEBUG_ESS") == "1":
                            ess_val = float(ess.item())
                            log_w_max = float(log_w.max().item())
                            log_w_min = float(log_w.min().item())
                            log_w_std = float(log_w.std().item()) if K_total > 1 else 0.0
                            triggered = bool((ess < ess_threshold).item())
                            print(
                                f"[ESS] step={step_idx:03d} t={float(t):.3f} "
                                f"ess={ess_val:.2f}/{K_total} thr={ess_threshold:.1f} "
                                f"trig={int(triggered)} "
                                f"log_w[min={log_w_min:.2f} max={log_w_max:.2f} std={log_w_std:.2f}]",
                                flush=True,
                            )
                        if ess < ess_threshold:
                            curr_feats, _ = self._resample_state(curr_feats, log_w)
                            did_resample = True
                    elif (step_idx + 1) % K_legacy == 0:
                        curr_feats, _ = self._resample_state(curr_feats, log_w)
                        did_resample = True
                    if did_resample and action_mode == "split":
                        log_w_rot.zero_()
                        log_w_trans.zero_()

                if is_last_step:
                    final_rigid = model_out["rigids"]
                    final_prot = all_atom.compute_backbone_atom37(
                        bb_rigids=next_rigid,
                        aatypes=curr_feats["aatype"],
                        torsions=model_out["angles"],
                    )[0]
                    if final_reweight and log_w.abs().max() > 1e-6:
                        curr_feats, idx = self._resample_state(curr_feats, log_w)
                        final_prot = self._gather_K(final_prot, idx)
                        final_rigid = self._gather_K(final_rigid, idx)

        if final_prot is None or final_rigid is None:
            raise RuntimeError("FKC biased rollout failed to produce final structures.")

        # Untile K particles → keep particle 0 only (FKC repo convention).
        if K_particles > 1:
            curr_feats = self._slice_feats(curr_feats, K_total, slice(0, orig_bsz))
            final_prot = final_prot[:orig_bsz]
            final_rigid = final_rigid[:orig_bsz]
        return final_prot, final_rigid, curr_feats


class HistoryGuidedSampler(BiasedActorSampler):
    """Sequential frame sampler using f_θ history correction instead of actor.

    For each extrapolation step, at each reverse diffusion step:
    1. Run frozen env model → S_F, x̂₀
    2. Cache S_F at (frame_id, step_idx)
    3. Read cached history scores at same step_idx
    4. f_θ produces per-residue correction δ
    5. Aggregate: Ŝ = S_F + Σ α · S_H
    6. Reverse diffuse with Ŝ
    After frame completes, cache final structure for RMSD/f_θ.
    """

    def __init__(
        self,
        *,
        env_model: network.DyneTrion,
        correction_net,
        diffuser: se3_diffuser.SE3Diffuser,
        runtime: RuntimeConfig,
        device: torch.device,
        model_conf: DictConfig,
    ):
        super().__init__(
            env_model=env_model, actor=None,
            diffuser=diffuser, runtime=runtime,
            model_conf=model_conf, device=device,
        )
        self.correction_net = correction_net

    def _single_rollout_guided(self, curr_feats, frame_id, score_cache, structure_cache):
        """Reverse diffusion for one frame with score caching and history blending."""
        from src.model.history_correction import per_residue_rmsd_weights

        bsz = curr_feats["aatype"].shape[0]
        t_placeholder = torch.ones((bsz, 1), device=self.device)
        reverse_steps = np.linspace(self.runtime.min_t, 1.0, self.runtime.num_t)[::-1]
        dt = 1.0 / self.runtime.num_t

        if self.model_conf.embed.embed_self_conditioning:
            with torch.no_grad():
                curr_feats = self._set_t_feats(curr_feats, float(reverse_steps[0]), t_placeholder)
                warm_out = self.env_model(curr_feats, is_training=False)
                curr_feats["sc_ca_t"] = warm_out["rigids"][..., 4:]

        final_prot, final_rigid = None, None

        for step_idx, t in enumerate(reverse_steps):
            curr_feats = self._set_t_feats(curr_feats, float(t), t_placeholder)
            is_last_step = step_idx == len(reverse_steps) - 1

            with torch.no_grad():
                model_out = self.env_model(curr_feats, is_training=False)
                rot_score = model_out["rot_score"]      # [B, F, N, 3]
                trans_score = model_out["trans_score"]
                rigid_pred = model_out["rigids"]         # [B, F, N, 7]
                x0_ca = rigid_pred[..., 4:]              # [B, F, N, 3]

                if self.model_conf.embed.embed_self_conditioning:
                    curr_feats["sc_ca_t"] = x0_ca

            # Cache current frame's env score at this step
            score_cache.store(frame_id, step_idx, rot_score, trans_score)

            # Read cached history scores at same step index
            S_H_rot, S_H_trans = score_cache.get_history_at_step(
                step_idx, exclude_frame=frame_id
            )

            lambda_blend = getattr(self.runtime, "lambda_blend", 0.0)

            if S_H_rot is not None:
                x0_hist = structure_cache.get_history()  # [B, H, N, 7]
                x0_hist_ca = x0_hist[..., 4:]            # [B, H, N, 3]
                res_mask = curr_feats["res_mask"][:, 0] if curr_feats["res_mask"].dim() == 3 else curr_feats["res_mask"]

                x0_ca_4d = x0_ca if x0_ca.dim() == 4 else x0_ca.unsqueeze(1)
                x0_rigids_4d = rigid_pred if rigid_pred.dim() == 4 else rigid_pred.unsqueeze(1)

                S_F_rot_4d = rot_score if rot_score.dim() == 4 else rot_score.unsqueeze(1)
                S_F_trans_4d = trans_score if trans_score.dim() == 4 else trans_score.unsqueeze(1)
                F_dim = S_F_rot_4d.shape[1]
                S_H_rot_exp = S_H_rot.unsqueeze(1).expand(-1, F_dim, -1, -1, -1) if S_H_rot.dim() == 4 else S_H_rot
                S_H_trans_exp = S_H_trans.unsqueeze(1).expand(-1, F_dim, -1, -1, -1) if S_H_trans.dim() == 4 else S_H_trans

                if lambda_blend > 0:
                    # Option A: fixed RMSD-weighted average of history scores
                    # No learned parameters — pure distance-based blending
                    rmsd_sigma = getattr(self.runtime, "history_distance_sigma", 1.0)
                    w = per_residue_rmsd_weights(
                        x0_ca_4d, x0_hist_ca, res_mask, sigma=rmsd_sigma,
                    )  # [B, F, H, N]
                    attn = torch.softmax(w, dim=2)  # normalize over H
                    attn = attn * res_mask[:, None, None, :]

                    V = torch.cat([S_H_rot_exp, S_H_trans_exp], dim=-1)  # [B, F, H, N, 6]
                    blend = torch.einsum("bfhn,bfhnd->bfnd", attn, V)  # [B, F, N, 6]
                    bias_rot = lambda_blend * blend[..., :3]
                    bias_trans = lambda_blend * blend[..., 3:]
                else:
                    # Original: learned correction network f_θ
                    t_tensor = torch.full((bsz,), float(t), device=self.device)
                    bias_rot, bias_trans, _ = self.correction_net(
                        x0_rigids_4d, x0_hist,
                        t_tensor, res_mask,
                        node_repr=curr_feats["node_repr"],
                        edge_repr=curr_feats["edge_repr"],
                        S_H_rot=S_H_rot_exp, S_H_trans=S_H_trans_exp,
                        x0_ca_current=x0_ca_4d, x0_ca_history=x0_hist_ca,
                        rmsd_sigma=getattr(self.runtime, "rmsd_sigma", 1.0),
                    )
                    bias_scale = getattr(self.runtime, "bias_scale", 1.0)
                    bias_rot = bias_scale * bias_rot
                    bias_trans = bias_scale * bias_trans

                S_hat_rot = S_F_rot_4d + bias_rot
                S_hat_trans = S_F_trans_4d + bias_trans

                rot_score = S_hat_rot.squeeze(1) if rot_score.dim() == 3 else S_hat_rot
                trans_score = S_hat_trans.squeeze(1) if trans_score.dim() == 3 else S_hat_trans

            mask = curr_feats["res_mask"][..., None]
            rot_score = rot_score * mask
            trans_score = trans_score * mask

            with torch.no_grad():
                if t > self.runtime.min_t:
                    diffuse_mask = (1.0 - curr_feats["fixed_mask"]) * curr_feats["res_mask"]
                    next_rigid, _, _, _ = self.diffuser.reverse_with_logprob(
                        rigid_t=Rigid.from_tensor_7(curr_feats["rigids_t"]),
                        rot_score=rot_score,
                        trans_score=trans_score,
                        t=float(t), dt=dt,
                        diffuse_mask=diffuse_mask,
                        noise_scale=self.runtime.noise_scale,
                        device=self.device,
                    )
                else:
                    next_rigid = Rigid.from_tensor_7(rigid_pred)

                curr_feats["rigids_t"] = next_rigid.to_tensor_7()

                if is_last_step:
                    final_rigid = rigid_pred
                    final_prot = all_atom.compute_backbone_atom37(
                        bb_rigids=next_rigid,
                        aatypes=curr_feats["aatype"],
                        torsions=model_out["angles"],
                    )[0]

        if final_prot is None or final_rigid is None:
            raise RuntimeError("History-guided rollout failed.")
        return final_prot, final_rigid, curr_feats

    def sample(self, batch_feats):
        """Sequential frame generation with score caching."""
        from src.model.history_correction import ScoreCache, StructureCache

        self.env_model.eval()
        self.correction_net.eval()

        curr_feats = self._prepare_init_feats(batch_feats)
        atom_steps, rigid_steps = [], []
        max_hist = getattr(self.runtime, "max_history", 17)
        score_cache = ScoreCache(max_history=max_hist)
        structure_cache = StructureCache(max_history=max_hist)

        pbar = tqdm(
            range(self.runtime.extrapolation_time),
            desc="History-Guided Extrapolation",
            leave=False,
        )
        for ext_idx in pbar:
            atom_pred, rigid_pred, curr_feats = self._single_rollout_guided(
                curr_feats, frame_id=ext_idx,
                score_cache=score_cache,
                structure_cache=structure_cache,
            )

            # Finalize caches for this frame
            score_cache.finalize_frame(ext_idx)
            structure_cache.push(frame_id=ext_idx, rigid=rigid_pred[:, -1])  # [B, N, 7]

            atom_steps.append(atom_pred)
            rigid_steps.append(rigid_pred)
            curr_feats = self._update_ref_with_prediction(curr_feats, atom_pred, rigid_pred)

        atom_traj = torch.stack(atom_steps, dim=1)  # [B, E, F, N, 37, 3]
        rigid_traj = torch.stack(rigid_steps, dim=1)  # [B, E, F, N, 7]
        return SampleResult(
            atom_traj=atom_traj,
            final_atom37=atom_traj[:, -1],
            final_rigids=rigid_traj[:, -1],
        )


class ActorEvaluator:
    def __init__(self, *, conf: DictConfig):
        self.conf = conf
        self.accelerator = Accelerator()
        self._log = logging.getLogger(__name__)

        logging.basicConfig(
            level=logging.INFO if self.accelerator.is_local_main_process else logging.ERROR
        )
        if conf.experiment.seed is not None:
            set_seed(conf.experiment.seed)

        self.runtime = self._build_runtime(conf)
        self.models = self._build_models(conf, self.runtime)
        self._test_loader = self._create_dataloader(conf)
        self.sampler = self._build_sampler(conf, self.runtime, self.models)

    def _build_runtime(self, conf: DictConfig) -> RuntimeConfig:
        eval_conf = conf.get("eval")
        mode = int(eval_conf.get("mode", 0)) if eval_conf is not None else 0

        base_save_dir = conf.experiment.get("save_dir", "./eval_outputs")
        if eval_conf is not None:
            base_save_dir = eval_conf.get("output_dir", base_save_dir)
        inline_refine_t = float(_conf_get(conf.experiment, "inline_refine_t", 0.0) or 0.0)
        mode_dir = f"mode_{mode}"
        if inline_refine_t > 0:
            mode_dir = f"{mode_dir}_inline_refined_t{inline_refine_t:g}"
        save_dir = os.path.join(base_save_dir, mode_dir)

        actor_blend_alpha = _conf_get(conf.experiment, "actor_blend_alpha", 1.0)
        if actor_blend_alpha is None:
            actor_blend_alpha = 1.0

        return RuntimeConfig(
            mode=mode,
            save_dir=save_dir,
            num_t=int(conf.data.get("num_t", 100)),
            min_t=float(conf.data.get("min_t", 0.01)),
            noise_scale=float(conf.experiment.get("noise_scale", 1.0)),
            max_history=int(conf.experiment.get("max_history", 17)),
            history_decay_lambda=float(conf.experiment.get("history_decay_lambda", 1.0)),
            history_decay_t_scale=float(conf.experiment.get("history_decay_t_scale", 0.0)),
            history_decay_mode=str(conf.experiment.get("history_decay_mode", "time")),
            history_distance_sigma=float(conf.experiment.get("history_distance_sigma", 1.0)),
            history_distance_variant=str(
                conf.experiment.get("history_distance_variant", "unnormalized")
            ),
            bias_gamma=float(_conf_get(conf.experiment, "bias_gamma", 0.05) or 0.05),
            center=bool(conf.experiment.get("center", True)),
            extrapolation_time=int(
                eval_conf.get(
                    "extrapolation_time", conf.experiment.get("extrapolation_time", 1)
                )
                if eval_conf is not None
                else conf.experiment.get("extrapolation_time", 1)
            ),
            actor_blend_alpha=float(actor_blend_alpha),
            overwrite=bool(eval_conf.get("overwrite", False)) if eval_conf else False,
            record_last_k=int(eval_conf.get("record_last_k", 0)) if eval_conf else 0,
            actor_condition_mode=str(conf.experiment.get("actor_condition_mode", "none")),
            bias_direction=str(conf.experiment.get("bias_direction", "away")),
            inline_refine_t=inline_refine_t,
            omega_method=str(conf.experiment.get("omega_method", "rmsd_kabsch")),
            fkc_enabled=bool(conf.experiment.get("fkc_enabled", False)),
            fkc_resample_k=int(conf.experiment.get("fkc_resample_k", 10)),
            fkc_beta=float(conf.experiment.get("fkc_beta", 1.0)),
            fkc_ess_frac=float(conf.experiment.get("fkc_ess_frac", 0.5)),
            fkc_final_reweight=bool(conf.experiment.get("fkc_final_reweight", True)),
            fkc_action_mode=str(conf.experiment.get("fkc_action_mode", "option_c")),
            fkc_num_particles=int(conf.experiment.get("fkc_num_particles", 1)),
            fkc_particle_chunk=int(conf.experiment.get("fkc_particle_chunk", 0)),
        )

    def _build_models(self, conf: DictConfig, runtime: RuntimeConfig) -> ModelBundle:
        diffuser = se3_diffuser.SE3Diffuser(conf.diffuser)

        actor = None
        actor_weight = getattr(conf.actor, "weight", None)
        if actor_weight is not None:
            actor = MetadynamicsActor(
                c_s=conf.actor.node_dim,
                c_z=conf.actor.edge_dim,
                c_hidden=conf.actor.c_hidden,
                no_heads=conf.actor.no_heads,
                no_qk_points=conf.actor.no_qk_points,
                no_v_points=conf.actor.no_v_points,
                num_ipa_blocks=conf.actor.num_blocks,
                max_rope_len=conf.actor.max_rope_len,
                max_energy=conf.actor.max_energy,
                coordinate_scaling=conf.diffuser.r3.coordinate_scaling,
            )
            actor_weight_path = _resolve_weight_path(actor_weight)
            self._log.info("Loading actor weight from %s", actor_weight_path)
            state_dict = _strip_module_prefix(
                torch.load(actor_weight_path, map_location="cpu", weights_only=False)
            )
            lora_meta = _maybe_prepare_actor_for_state_dict(
                actor, state_dict, getattr(conf, "experiment", None)
            )
            if lora_meta is not None:
                self._log.info(
                    "Detected LoRA actor checkpoint: rank=%d alpha=%.4f modules=%d",
                    lora_meta["rank"],
                    lora_meta["alpha"],
                    lora_meta["modules"],
                )
            actor.load_state_dict(state_dict, strict=True)
            actor.to(self.accelerator.device)
            actor.eval()
        else:
            self._log.warning("No actor weight provided. Actor-dependent modes will fail.")

        env_model = None
        if runtime.mode in (1, 2):
            env_weight = getattr(conf.model, "pretrained_weights", None)
            if env_weight is None:
                raise ValueError("`model.pretrained_weights` is required for mode 1/2.")
            env_model = network.DyneTrion(conf.model, diffuser)
            self._log.info("Loading env model from %s", env_weight)
            ckpt = torch.load(env_weight, map_location="cpu", weights_only=False)
            model_state = ckpt["model"] if "model" in ckpt else ckpt
            model_state = _strip_module_prefix(model_state)
            env_model.load_state_dict(model_state, strict=False)
            env_model.to(self.accelerator.device)
            env_model.eval()
            for p in env_model.parameters():
                p.requires_grad = False

        return ModelBundle(diffuser=diffuser, actor=actor, env_model=env_model)

    def _create_dataloader(self, conf: DictConfig):
        # Manual sharding: `data.rank=i data.grouped=N` via Hydra override.
        # Each rank processes `csv[rank::grouped]` → launch N independent
        # processes with different --rank + CUDA_VISIBLE_DEVICES for multi-GPU.
        dataset = ProteinSamplingDataset(
            data_conf=conf.data,
            diffuser=self.models.diffuser,
            rank=int(conf.data.get("rank", 0)),
            grouped=int(conf.data.get("grouped", 1)),
        )
        batch_size = int(conf.experiment.get("batch_size", 1))
        eval_conf = conf.get("eval")
        if eval_conf is not None:
            batch_size = int(eval_conf.get("eval_batch_size", batch_size))
        return data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(conf.data.get("num_loader_workers", 0)),
            pin_memory=True,
        )

    def _build_sampler(
        self,
        conf: DictConfig,
        runtime: RuntimeConfig,
        models: ModelBundle,
    ):
        if runtime.mode == 0:
            if models.actor is None:
                raise ValueError("Mode 0 requires `actor.weight`.")
            return ReverseSampler(
                actor=models.actor,
                diffuser=models.diffuser,
                runtime=runtime,
                device=self.accelerator.device,
            )

        if runtime.mode in (1, 2):
            if models.env_model is None:
                raise ValueError("Mode 1/2 requires an env model.")
            extrap_time = 1 if runtime.mode == 1 else max(1, runtime.extrapolation_time)
            runtime.extrapolation_time = extrap_time
            sampler_cls = FKCBiasedSampler if runtime.fkc_enabled else BiasedActorSampler
            if runtime.fkc_enabled:
                self._log.info(
                    "FKC enabled: resample_k=%d beta=%.4f", runtime.fkc_resample_k, runtime.fkc_beta
                )
            return sampler_cls(
                env_model=models.env_model,
                actor=models.actor,
                diffuser=models.diffuser,
                runtime=runtime,
                device=self.accelerator.device,
                model_conf=conf.model,
            )

        raise ValueError(f"Unsupported eval mode: {runtime.mode}")

    def _save_batch(
        self,
        atom_traj: torch.Tensor,
        batch_feats: Dict[str, torch.Tensor],
        batch_pdb_names,
    ):
        os.makedirs(self.runtime.save_dir, exist_ok=True)
        bsz = atom_traj.shape[0]
        saved_paths = []

        for i in range(bsz):
            pdb_id = str(batch_pdb_names[i])
            pdb_path = os.path.join(self.runtime.save_dir, f"{pdb_id}.pdb")
            if os.path.exists(pdb_path) and not self.runtime.overwrite:
                continue

            atom_pos = rearrange(atom_traj[i], "e f n a c -> (e f) n a c")
            aatype = du.move_to_np(batch_feats["aatype"][i, 0]).astype(int)
            sample_length = aatype.shape[0]
            b_factors = np.full((sample_length, 37), 100.0, dtype=np.float32)

            au.write_prot_to_pdb(
                prot_pos=du.move_to_np(atom_pos),
                file_path=pdb_path,
                aatype=aatype,
                no_indexing=True,
                b_factors=b_factors,
            )
            saved_paths.append(pdb_path)
        return saved_paths

    def start_evaluation(self):
        self._log.info("Starting evaluation with mode=%d", self.runtime.mode)
        self._log.info("Actor blend alpha=%.4f", self.runtime.actor_blend_alpha)
        self._log.info(
            "History distance variant=%s (mode=%s, sigma=%.4f)",
            self.runtime.history_distance_variant,
            self.runtime.history_decay_mode,
            self.runtime.history_distance_sigma,
        )
        self._log.info("Saving outputs to %s", self.runtime.save_dir)

        all_saved = []
        pbar = tqdm(
            self._test_loader,
            disable=not self.accelerator.is_local_main_process,
        )

        for i, batch in enumerate(pbar):
            batch_feats, batch_pdb_names = batch

            if not self.runtime.overwrite:
                existing = [
                    os.path.exists(os.path.join(self.runtime.save_dir, f"{n}.pdb"))
                    for n in batch_pdb_names
                ]
                if all(existing):
                    pbar.set_description(f"Eval batch {i} | skipped (all exist)")
                    continue

            batch_feats = _move_to_device(batch_feats, self.accelerator.device)

            with torch.no_grad():
                result = self.sampler.sample(batch_feats)
            saved_paths = self._save_batch(result.atom_traj, batch_feats, batch_pdb_names)
            all_saved.extend(saved_paths)
            pbar.set_description(f"Eval batch {i} | saved {len(all_saved)}")

        self._log.info(
            "Evaluation finished. Saved %d PDB files to %s",
            len(all_saved),
            self.runtime.save_dir,
        )
        return {"num_saved": len(all_saved), "save_dir": self.runtime.save_dir}


@hydra.main(version_base=None, config_path="config", config_name="eval_actor")
def run(conf: DictConfig) -> None:
    evaluator = ActorEvaluator(conf=conf)
    evaluator.start_evaluation()


if __name__ == "__main__":
    run()
