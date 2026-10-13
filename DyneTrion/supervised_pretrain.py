import os
import logging
import numpy as np
import torch
import hydra
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf
from torch.utils import data
from tqdm import tqdm
from einops import rearrange, repeat
from src.data.actor_pretrain_loader import (
    ActorPretrainDataset,
    strip_batch_dim_collate,
)
from src.data.history_supervised_loader import (
    HistoryAwareSupervisedDataset,
    history_supervised_collate,
)
from src.data.sampling import BucketedDistributedSampler
from src.model.actor import MetadynamicsActor
from src.data import se3_diffuser
import swanlab
import tree
from openfold.utils.rigid_utils import Rigid
from openfold.utils.loss import torsion_angle_loss
from src.analysis import utils as au
from src.analysis.geometry import rmsdalign
from src.data import utils as du
from src.data import all_atom
from src.model import network
import contextlib
from openfold.utils import rigid_utils

def _kabsch_rmsd_weights(
    pred_ca_h: torch.Tensor,
    res_mask: torch.Tensor,
    sigma: float,
    variant: str = "unnormalized",
) -> torch.Tensor:
    """Distance-based history weighting via Kabsch-aligned RMSD.

    Args:
        pred_ca_h: [B, F, H, N, 3] predicted CA positions per history pair
        res_mask: [B, F, N] residue mask
        sigma: temperature for exp(-rmsd/sigma)
        variant: "normalized", "unnormalized", or "occupancy_scaled"

    Returns:
        [B, F, H] weights
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
        if hist_len == 1:
            return torch.ones((bsz, frame_t, 1), device=pred_ca_h.device, dtype=out_dtype)

        ca_ref = pred_ca_h[:, :, -1, :, :]
        ca_ref_flat = repeat(ca_ref, "b f n c -> (b f h) n c", h=hist_len)
        ca_h_flat = rearrange(pred_ca_h, "b f h n c -> (b f h) n c")
        mask_flat = repeat(res_mask, "b f n -> (b f h) n", h=hist_len).to(
            device=pred_ca_h.device, dtype=pred_ca_h.dtype
        )
        aligned = rmsdalign(ca_ref_flat, ca_h_flat, weights=mask_flat)
        sq = (aligned - ca_ref_flat).pow(2).sum(dim=-1)
        denom = mask_flat.sum(dim=-1).clamp_min(1.0)
        rmsd_flat = torch.sqrt((sq * mask_flat).sum(dim=-1) / denom)
        rmsd = rearrange(rmsd_flat, "(b f h) -> b f h", b=bsz, f=frame_t, h=hist_len)
        sigma = max(float(sigma), 1e-6)
        w_raw = torch.exp(-rmsd / sigma)

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
        raise ValueError(f"Unsupported variant={variant!r}")
    return weights.to(dtype=out_dtype)


class SupervisedTrainer:
    def __init__(self, conf: DictConfig):
        self.accelerator = Accelerator(log_with="swanlab")
        self.conf = conf
        self._diff_conf = conf.diffuser
        self._exp_conf = conf.experiment
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        set_seed(conf.experiment.seed)
        self._init_recoder(conf)

        self._diffuser = se3_diffuser.SE3Diffuser(self._diff_conf)
        self.global_step = 0
        # Model
        self.actor = MetadynamicsActor(
            c_s=conf.actor.node_dim,
            c_z=conf.actor.edge_dim,
            c_hidden=conf.actor.c_hidden,
            no_heads=conf.actor.no_heads,
            no_qk_points=conf.actor.no_qk_points,
            no_v_points=conf.actor.no_v_points,
            num_ipa_blocks=conf.actor.num_blocks,
            max_rope_len=conf.actor.max_rope_len,
            max_energy=conf.actor.max_energy,
            coordinate_scaling=self._diff_conf.r3.coordinate_scaling,
        )

        # Load pretrained actor weights if provided
        actor_weight = getattr(conf.actor, "weight", None)
        if actor_weight is not None:
            self.logger.info(f"Loading pretrained actor from {actor_weight}")
            ckpt = torch.load(actor_weight, map_location="cpu", weights_only=False)
            state = ckpt.get("model", ckpt)
            state = {k.replace("module.", ""): v for k, v in state.items()}
            self.actor.load_state_dict(state, strict=False)
            self.logger.info("Pretrained actor loaded.")

        self.optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=conf.experiment.learning_rate
        )

        self._build_dataloader(conf)

        # Env-support loss config
        self._use_env_support = bool(getattr(self._exp_conf, "use_env_support_loss", False))
        self._lambda_support = float(getattr(self._exp_conf, "lambda_support", 0.1))
        self._env_support_noise_scale = float(getattr(self._exp_conf, "env_support_noise_scale", 1.0))
        self._use_history_score_repel = bool(getattr(self._exp_conf, "use_history_score_repel", False))
        self._lambda_history_repel = float(getattr(self._exp_conf, "lambda_history_repel", 0.1))
        self._use_cn_adaptive_weight = bool(getattr(self._exp_conf, "use_cn_adaptive_weight", False))
        self._actor_outputs_bias = bool(getattr(self._exp_conf, "actor_outputs_bias", False))
        self._model_conf = conf.model

        self.actor, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.actor, self.optimizer, self.train_dataloader
        )

        # Frozen env teacher
        if self._use_env_support:
            self._env_model = network.DyneTrion(conf.model, self._diffuser)
            self._load_pretrained_env(conf.model.pretrained_weights)
            self._env_model.to(self.accelerator.device)
            self._env_model.eval()
            for param in self._env_model.parameters():
                param.requires_grad_(False)




    def _build_dataloader(self, conf):
        use_dynamic = getattr(conf.data, "use_dynamic_supervised_loader", False)
        if use_dynamic:
            self.train_dataset = HistoryAwareSupervisedDataset(
                csv_path=conf.data.train_csv_path,
                diffuser=self._diffuser,
                data_conf=conf.data,
                is_training=True,
                max_res_len=conf.data.max_res_len,
            )
            collate_fn = history_supervised_collate
        else:
            self.train_dataset = ActorPretrainDataset(
                csv_path=conf.data.train_csv_path,
                diffuser=self._diffuser,
                energy_sample_stride=conf.data.energy_sample_stride,
                frame_sample_step=conf.data.frame_sample_step,
                samples_per_traj=conf.data.samples_per_traj,
                min_t=conf.data.min_t,
                max_res_len=conf.data.max_res_len,
            )
            collate_fn = strip_batch_dim_collate

        bucket_size = int(self._exp_conf.get("bucket_size", 32))
        sampler = BucketedDistributedSampler(
            lengths=self.train_dataset.lengths,
            batch_size=self._exp_conf.batch_size,
            num_replicas=1,
            rank=0,
            shuffle=True,
            drop_last=True,
            bucket_size=max(1, bucket_size),
            seed=int(self._exp_conf.get("seed", 42)),
            sort_by_square=True,
        )
        num_workers = conf.experiment.num_loader_workers
        self.train_dataloader = data.DataLoader(
            self.train_dataset,
            batch_size=conf.experiment.batch_size,
            sampler=sampler,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )

        if use_dynamic:
            self.val_dataset = HistoryAwareSupervisedDataset(
                csv_path=conf.data.test_csv_path,
                diffuser=self._diffuser,
                data_conf=conf.data,
                is_training=False,
                max_res_len=conf.data.max_res_len,
            )
        else:
            self.val_dataset = ActorPretrainDataset(
                csv_path=conf.data.test_csv_path,
                diffuser=self._diffuser,
                energy_sample_stride=conf.data.energy_sample_stride,
                frame_sample_step=conf.data.frame_sample_step,
                samples_per_traj=conf.data.samples_per_traj,
                min_t=conf.data.min_t,
                max_res_len=conf.data.max_res_len,
            )



    # ── History-aware actor input helpers ──────────────────────────────

    def _history_decay_weights(
        self, t_tensor, frame_t, hist_len, pred_ca_h=None, res_mask=None,
        mode="distance", sigma=1.0, variant="unnormalized",
        decay_lambda=1.0, decay_t_scale=0.5,
    ):
        """Compute per-history-frame weights.

        Args:
            t_tensor: [B, 1] diffusion time
            frame_t: F (number of query frames)
            hist_len: H (number of history frames)
            pred_ca_h: [B, F, H, N, 3] predicted CA for distance mode
            res_mask: [B, F, N] residue mask for distance mode
            mode: "distance" or "time"

        Returns:
            [B, F, H] normalized weights
        """
        if hist_len <= 0:
            raise ValueError(f"hist_len must be > 0, got {hist_len}")
        if mode == "distance":
            if pred_ca_h is None or res_mask is None:
                raise ValueError("distance mode requires pred_ca_h and res_mask")
            return _kabsch_rmsd_weights(
                pred_ca_h=pred_ca_h, res_mask=res_mask,
                sigma=sigma, variant=variant,
            )
        # Time-based decay
        t_bf = repeat(t_tensor, "b 1 -> b f", f=frame_t).clamp(0.0, 1.0)
        lam = decay_lambda * (1.0 + decay_t_scale * (1.0 - t_bf))
        hist_idx = torch.arange(hist_len, device=t_tensor.device, dtype=t_tensor.dtype)
        hist_age = (hist_len - 1) - hist_idx
        weights = torch.exp(-lam[..., None] * hist_age[None, None, :])
        return weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

    def _load_pretrained_env(self, ckpt_path):
        self.logger.info(f"Loading frozen env teacher from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_state = ckpt["model"] if "model" in ckpt else ckpt
        model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
        self._env_model.load_state_dict(model_state, strict=False)
        self.logger.info(f"Env teacher loaded from: {ckpt_path}")

    def _compute_env_scores(self, batch):
        """Run frozen env model on original batch, return [B, F, N, 3] scores.

        Args:
            batch: original dataset output with [B, F, N, ...] tensors.

        Returns:
            (env_rot_score, env_trans_score) each [B, F, N, 3], detached.
        """
        feats = {
            "rigids_t": batch["rigids_t"],                   # [B, F, N, 7]
            "t": batch["t"].unsqueeze(1),                    # [B, 1]
            "node_repr": batch["node_repr"],                 # [B, N, D]
            "edge_repr": batch["edge_repr"],                 # [B, N, N, D]
            "ref_rigids_0": batch["ref_rigids_0"],           # [B, R, N, 7]
            "motion_rigids_0": batch["motion_rigids_0"],     # [B, M, N, 7]
            "res_mask": batch["res_mask"],                   # [B, F, N]
            "fixed_mask": batch["fixed_mask"],               # [B, F, N]
            "aatype": batch["aatype"],                       # [B, F, N]
            "seq_idx": batch["seq_idx"],                     # [B, F, N]
            "sc_ca_t": batch["sc_ca_t"],                     # [B, F, N, 3]
            "motion_node_mask": batch["motion_node_mask"],   # [B, M, N]
            "ref_node_mask": batch["ref_node_mask"],         # [B, R, N]
        }
        for k in ["torsion_angles_sin_cos", "alt_torsion_angles_sin_cos", "torsion_angles_mask"]:
            if k in batch:
                feats[k] = batch[k]
        for k in ["ref_torsion_angles_sin_cos", "ref_alt_torsion_angles_sin_cos"]:
            if k in batch:
                feats[k] = batch[k]

        with torch.no_grad():
            env_out = self._env_model(feats, is_training=False)
        return env_out["rot_score"], env_out["trans_score"]  # [B, F, N, 3]

    def _compute_guided_scores(self, formated_batch, pred_ctx, env_rot_score, env_trans_score, B, F, H):
        """Compute guided scores and intermediate values shared by multiple losses.

        Returns:
            dict with guided_rot, guided_trans [B,F,N,3] and rigids_t_bf [B,F,N,7].
        """
        alpha = float(getattr(self._exp_conf, "actor_blend_alpha", 1.0))

        actor_rot_h = rearrange(pred_ctx["pred_rot_guided"], "(b f h) n c -> b f h n c", b=B, f=F, h=H)
        actor_trans_h = rearrange(pred_ctx["pred_trans_guided"], "(b f h) n c -> b f h n c", b=B, f=F, h=H)

        pred_ca_h = rearrange(
            pred_ctx["pred_x_ref_rigids"].get_trans(), "(b f h) n c -> b f h n c", b=B, f=F, h=H
        ).detach()
        res_mask_bf = rearrange(
            formated_batch["res_mask"], "(b f h) n -> b f h n", b=B, f=F, h=H
        )[:, :, 0]  # [B, F, N]
        hist_w = self._history_decay_weights(
            formated_batch["t"][:1].unsqueeze(1), frame_t=F, hist_len=H,
            pred_ca_h=pred_ca_h, res_mask=res_mask_bf,
        ).detach()  # [B, F, H]

        bias_rot = (hist_w[..., None, None] * (env_rot_score.unsqueeze(2) - actor_rot_h)).sum(dim=2)
        bias_trans = (hist_w[..., None, None] * (env_trans_score.unsqueeze(2) - actor_trans_h)).sum(dim=2)

        guided_rot = env_rot_score + alpha * bias_rot
        guided_trans = env_trans_score + alpha * bias_trans

        rigids_t_bf = rearrange(
            formated_batch["rigids_t"], "(b f h) n c -> b f h n c", b=B, f=F, h=H
        )[:, :, 0]  # [B, F, N, 7]

        return {
            "guided_rot": guided_rot,
            "guided_trans": guided_trans,
            "rigids_t_bf": rigids_t_bf,
        }

    def _compute_env_support_loss(self, formated_batch, env_rot_score, env_trans_score, guided_cache, B, F, H):
        """Compute env-support loss using precomputed guided scores.

        Args:
            formated_batch: [B*F*H, N, ...] from _prepare_history_batch
            env_rot_score: [B, F, N, 3] env rotation scores (detached)
            env_trans_score: [B, F, N, 3] env translation scores (detached)
            guided_cache: dict from _compute_guided_scores
            B, F, H: batch, frame, history dimensions

        Returns:
            scalar env-support loss
        """
        noise_scale = self._env_support_noise_scale
        t_val = float(formated_batch["t"][0].item())
        dt = 1.0 / max(int(getattr(self.conf.data, "num_t", 100)), 1)

        guided_rot = guided_cache["guided_rot"]
        guided_trans = guided_cache["guided_trans"]
        rigids_t_bf = guided_cache["rigids_t_bf"]

        guided_rot_flat = rearrange(guided_rot, "b f n c -> (b f) n c").float()
        guided_trans_flat = rearrange(guided_trans, "b f n c -> (b f) n c").float()
        x_t = Rigid.from_tensor_7(rearrange(rigids_t_bf, "b f n c -> (b f) n c").float())

        x_prop = self._diffuser.reverse_torch(
            rigid_t=x_t,
            rot_score=guided_rot_flat,
            trans_score=guided_trans_flat,
            t=t_val, dt=dt, noise_scale=noise_scale,
        )
        raw_loss_type = getattr(self._exp_conf, "env_support_loss_type", "nll")
        if raw_loss_type is None:
            loss_type = "nll"
        else:
            loss_type = str(raw_loss_type).strip().lower()
            if not loss_type:
                loss_type = "nll"

        rot_ss = rearrange(
            formated_batch["rot_score_scaling"], "(b f h) -> b f h", b=B, f=F, h=H
        )[:, :, 0]
        trans_ss = rearrange(
            formated_batch["trans_score_scaling"], "(b f h) -> b f h", b=B, f=F, h=H
        )[:, :, 0]
        rot_ss = rearrange(rot_ss, "b f -> (b f)")
        trans_ss = rearrange(trans_ss, "b f -> (b f)")
        rot_w = 1.0 / (rot_ss ** 2 + 1e-8)
        trans_w = 1.0 / (trans_ss ** 2 + 1e-8)

        env_rot_flat = rearrange(env_rot_score, "b f n c -> (b f) n c").float()
        env_trans_flat = rearrange(env_trans_score, "b f n c -> (b f) n c").float()
        if loss_type == "nll":
            logp_rot, logp_trans, _ = self._diffuser.transition_logprob_given_next(
                rigid_t=x_t, rigid_t_1=x_prop,
                rot_score=env_rot_flat,
                trans_score=env_trans_flat,
                t=t_val, dt=dt, noise_scale=noise_scale,
            )
            loss = -(rot_w * logp_rot + trans_w * logp_trans).mean()
        elif loss_type == "kl_forward":
            logq_rot, logq_trans, _ = self._diffuser.transition_logprob_given_next(
                rigid_t=x_t, rigid_t_1=x_prop,
                rot_score=guided_rot_flat,
                trans_score=guided_trans_flat,
                t=t_val, dt=dt, noise_scale=noise_scale,
            )
            logp_rot, logp_trans, _ = self._diffuser.transition_logprob_given_next(
                rigid_t=x_t, rigid_t_1=x_prop,
                rot_score=env_rot_flat,
                trans_score=env_trans_flat,
                t=t_val, dt=dt, noise_scale=noise_scale,
            )
            loss = (rot_w * (logq_rot - logp_rot) + trans_w * (logq_trans - logp_trans)).mean()
        else:
            supported = ("nll", "kl_forward")
            raise ValueError(
                f"Unsupported env_support_loss_type={loss_type!r}; "
                f"supported values are {supported}."
            )
        return loss

    def _compute_history_score_repel_loss(self, batch, formated_batch, guided_cache, B, F, H):
        """Score-space repulsion: penalise guided scores pointing toward history.

        For each history frame x_0^h, the analytical denoising score from the
        current noised state x_t toward x_0^h is computed via calc_rot_score /
        calc_trans_score (no neural network forward).  The loss is:

            L = Σ_h exp( -||s_guided - s_toward_h||² / (||s_toward_h||² + ε) )

        High when guided score is close to any history direction; near-zero
        when sufficiently different.  No margin hyper-parameter needed.
        """
        eps = 1e-6
        guided_rot = guided_cache["guided_rot"]      # [B, F, N, 3]
        guided_trans = guided_cache["guided_trans"]    # [B, F, N, 3]
        rigids_t_bf = guided_cache["rigids_t_bf"]      # [B, F, N, 7]
        x_t = Rigid.from_tensor_7(rigids_t_bf.reshape(B * F, -1, 7).float())

        history_rigids = batch["history_rigids"]  # [B, H, N, 7]
        history_expanded = repeat(history_rigids, "b h n c -> (b f) h n c", f=F)

        t_vec = batch["t"]  # [B]
        mask = rearrange(
            formated_batch["res_mask"], "(b f h) n -> b f h n", b=B, f=F, h=H
        )[:, :, 0, :, None]  # [B, F, N, 1]

        total_repel = torch.tensor(0.0, device=guided_rot.device)
        for h_idx in range(H):
            x_0_h = Rigid.from_tensor_7(history_expanded[:, h_idx].float())
            s_rot_h = self._diffuser.calc_rot_score(
                x_t.get_rots(),
                x_0_h.get_rots(),
                t_vec,
            ).reshape(B, F, -1, 3)
            s_trans_h = self._diffuser.calc_trans_score(
                x_t.get_trans(),
                x_0_h.get_trans(),
                t_vec[:, None, None],
                use_torch=True,
            ).reshape(B, F, -1, 3)
            s_rot_h = s_rot_h * mask
            s_trans_h = s_trans_h * mask

            rot_diff_sq = ((guided_rot - s_rot_h) ** 2 * mask).sum(dim=(-1, -2))
            trans_diff_sq = ((guided_trans - s_trans_h) ** 2 * mask).sum(dim=(-1, -2))

            rot_norm_sq = (s_rot_h ** 2 * mask).sum(dim=(-1, -2)).clamp_min(eps)
            trans_norm_sq = (s_trans_h ** 2 * mask).sum(dim=(-1, -2)).clamp_min(eps)

            repel_rot = torch.exp(-rot_diff_sq / rot_norm_sq)
            repel_trans = torch.exp(-trans_diff_sq / trans_norm_sq)
            total_repel = total_repel + (repel_rot + repel_trans).mean()

        return total_repel

    @torch.no_grad()
    def _compute_pred_cn_score(self, pred_ctx, formated_batch, B, F, H):
        """Compute match_cn from predicted x_0 vs ground-truth x_0 (history frame).

        Returns:
            cn_score: scalar in [0, 1], averaged over batch/frames.
        """
        # Predicted CA positions from model output [B, F, N, 3]
        pred_ca = rearrange(
            pred_ctx["pred_x_ref_rigids"].get_trans(),
            "(b f h) n c -> b f h n c", b=B, f=F, h=H,
        )[:, :, 0].detach()  # [B, F, N, 3]

        # Ground-truth CA positions: ref_rigids_0 in formated_batch is history_rigids
        # which has shape [B*F*H, N, 7]; take h=0 to get the current target
        gt_ca = rearrange(
            formated_batch["ref_rigids_0"],
            "(b f h) n c -> b f h n c", b=B, f=F, h=H,
        )[:, :, 0, :, 4:].detach()  # [B, F, N, 3] (trans part of rigid7)

        res_mask = rearrange(
            formated_batch["res_mask"],
            "(b f h) n -> b f h n", b=B, f=F, h=H,
        )[:, :, 0]  # [B, F, N]

        # Sequential CA-CA distance as proxy for match_ca
        pred_d = (pred_ca[:, :, :-1] - pred_ca[:, :, 1:]).norm(dim=-1)  # [B,F,N-1]
        # Valid sequential pairs
        pair_mask = res_mask[:, :, :-1] * res_mask[:, :, 1:]  # [B,F,N-1]

        ca_ok = ((pred_d < 4.5).float() * pair_mask)
        cn_score = ca_ok.sum(dim=-1) / pair_mask.sum(dim=-1).clamp_min(1.0)  # [B,F]
        return cn_score.mean()  # scalar



















    def _init_recoder(self, conf):
        if self.accelerator.is_main_process and self._exp_conf.use_recoder:
            sw_conf = self._exp_conf.swanlab
            sub_dir = getattr(sw_conf, "log_subdir")
            final_log_dir = os.path.join(self._exp_conf.save_dir, sub_dir)
            os.makedirs(final_log_dir, exist_ok=True)
            if sw_conf.api_key is not None:
                swanlab.login(sw_conf.api_key)
            tracker_init_kwargs = {
                "swanlab": {
                    "mode": sw_conf.mode,
                    "logdir": final_log_dir,
                    "name": self._exp_conf.name,
                }
            }
            self.accelerator.init_trackers(
                project_name=sw_conf.project,
                config=OmegaConf.to_container(conf, resolve=True),
                init_kwargs=tracker_init_kwargs,
            )

    def inference_fn(
        self,
        data_init,
        num_t=None,
        min_t=None,
        center=True,
        aux_traj=False,
        noise_scale=1.0,
    ):
        self.actor.eval()
        sample_feats = {
            k: (v.clone() if torch.is_tensor(v) else v) for k, v in data_init.items()
        }
        device = sample_feats["rigids_t"].device
        batch_size = sample_feats["rigids_t"].shape[0]

        if num_t is None:
            num_t = int(getattr(self.conf.data, "num_t", 100))
        if min_t is None:
            min_t = float(getattr(self.conf.data, "min_t", 0.01))
        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = 1.0 / num_t

        rigid_traj = []
        prot_traj = []
        last_pred_dict = None

        with torch.no_grad():
            for t in reverse_steps:
                if t > min_t:
                    t_tensor = torch.full((batch_size, ), float(t), device=device)
                    sample_feats["t"] = t_tensor

                    pred_dict = self.actor(
                        **sample_feats,
                        memory_cache=None,
                        context_rigids=None,
                    )
                    last_pred_dict = pred_dict
                    pred_x0 = pred_dict["final_rigids"]
                    x_t = Rigid.from_tensor_7(sample_feats["rigids_t"])

                    rot_score = self._diffuser.calc_rot_score(
                        x_t.get_rots(),
                        pred_x0.get_rots(),
                        t_tensor,
                    )
                    if rot_score.ndim != 3 or rot_score.shape[0] != batch_size:
                        raise RuntimeError(
                            f"Invalid rot_score shape {tuple(rot_score.shape)}; expected [B, N, 3]. "
                            "Check t shape before calc_rot_score."
                        )
                    trans_score = self._diffuser.calc_trans_score(
                        x_t.get_trans(),
                        pred_x0.get_trans(),
                        t_tensor[:, None, None],
                        use_torch=True,
                    )

                    mask = sample_feats["res_mask"][..., None]
                    rot_score = rot_score * mask
                    trans_score = trans_score * mask

                    diffuse_mask = (1.0 - sample_feats["fixed_mask"]) * sample_feats[
                        "res_mask"
                    ]
                    next_rigid = self._diffuser.reverse(
                        rigid_t=x_t,
                        rot_score=rot_score.detach().cpu().numpy(),
                        trans_score=trans_score.detach().cpu().numpy(),
                        t=float(t),
                        dt=dt,
                        diffuse_mask=diffuse_mask.detach().cpu().numpy(),
                        center=center,
                        noise_scale=noise_scale,
                        device=device,
                    )
                else:
                    pred_dict = self.actor(
                        **sample_feats,
                        memory_cache=None,
                        context_rigids=None,
                    )
                    last_pred_dict = pred_dict
                    next_rigid = pred_dict["final_rigids"]

                sample_feats["rigids_t"] = next_rigid.to_tensor_7().to(device)

                if aux_traj:
                    atom37_t = all_atom.compute_backbone_atom37(
                        bb_rigids=next_rigid,
                        aatypes=sample_feats["aatype"].long(),
                        torsions=pred_dict["angles"],
                    )[0]
                    rigid_traj.append(sample_feats["rigids_t"].detach().cpu())
                    prot_traj.append(atom37_t.detach().cpu())

        final_rigid = Rigid.from_tensor_7(sample_feats["rigids_t"])
        final_atom37 = all_atom.compute_backbone_atom37(
            bb_rigids=final_rigid,
            aatypes=sample_feats["aatype"].long(),
            torsions=last_pred_dict["angles"],
        )[0]

        ret = {
            "final_rigids": sample_feats["rigids_t"],
            "final_atom37": final_atom37,
        }
        if aux_traj and len(rigid_traj) > 0:
            ret["rigid_traj"] = torch.stack(rigid_traj, dim=1)
            ret["prot_traj"] = torch.stack(prot_traj, dim=1)
        return ret

    def eval(self, ckpt_dir=None):
        actor_model = self.accelerator.unwrap_model(self.actor)

        if ckpt_dir is not None:
            weight_path = ckpt_dir
            if os.path.isdir(ckpt_dir):
                weight_path = os.path.join(ckpt_dir, "pytorch_model.bin")
            state_dict = torch.load(weight_path, map_location="cpu")
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            actor_model.load_state_dict(state_dict, strict=True)

        self.actor.eval()
        device = self.accelerator.device
        num_t = int(getattr(self.conf.data, "num_t", 100))
        min_t = float(getattr(self.conf.data, "min_t", 0.01))
        pdb_dir = os.path.join(self.conf.experiment.save_dir, "eval_pdb")
        os.makedirs(pdb_dir, exist_ok=True)
        saved_pdbs = []

        pbar = tqdm(
            range(len(self.val_dataset)),
            disable=not self.accelerator.is_local_main_process,
        )
        for i in pbar:
            batch = self.val_dataset._get_infer_data(i)
            clip_feat = tree.map_structure(
                lambda x: x.unsqueeze(0).to(device) if torch.is_tensor(x) else x,
                batch,
            )
            infer_out = self.inference_fn(
                clip_feat,
                num_t=num_t,
                min_t=min_t,
                center=True,
                aux_traj=False,
                noise_scale=1.0,
            )
            atom37_t = infer_out["final_atom37"]
            sample_length = clip_feat["aatype"].shape[-1]
            b_factors = np.full((sample_length, 37), 100.0, dtype=np.float32)
            pdb_id = str(self.val_dataset.csv.iloc[i].get("pdb_id", f"sample_{i:05d}"))
            pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")

            au.write_prot_to_pdb(
                prot_pos=du.move_to_np(atom37_t[0]),
                file_path=pdb_path,
                aatype=du.move_to_np(clip_feat["aatype"][0]).astype(int),
                no_indexing=True,
                b_factors=b_factors,
            )
            saved_pdbs.append(pdb_path)

        self.logger.info(
            f"Holdout reverse-process finished. Saved {len(saved_pdbs)} PDBs to {pdb_dir}"
        )
        return {"num_saved": len(saved_pdbs), "pdb_dir": pdb_dir}

    def _assert_shapes(self, batch, pred_dict, pred_rot, pred_trans):
        b, n = batch["res_mask"].shape
        assert pred_rot.shape == (b, n, 3), f"pred_rot shape mismatch: {pred_rot.shape}"
        assert pred_trans.shape == (b, n, 3), f"pred_trans shape mismatch: {pred_trans.shape}"
        assert pred_dict["angles"].shape[:2] == (b, n), (
            f"angles shape mismatch: {pred_dict['angles'].shape}"
        )

    def _forward_and_scores(self, batch):
        pred_dict = self.actor(
            **batch,
            memory_cache=None,
            context_rigids=None,
        )
        pred_x_ref_rigids = pred_dict["final_rigids"]

        x_t = Rigid.from_tensor_7(batch["rigids_t"])
        # Force [B] to avoid accidental broadcast to [B, B, N, 3] in SO3 score.
        t_vec = batch["t"].reshape(-1)
        pred_rot = self._diffuser.calc_rot_score(
            x_t.get_rots(),
            pred_x_ref_rigids.get_rots(),
            t_vec,
        )
        pred_trans = self._diffuser.calc_trans_score(
            x_t.get_trans(),
            pred_x_ref_rigids.get_trans(),
            t_vec[:, None, None],
            use_torch=True,
        )
        mask = batch["res_mask"][..., None]
        pred_rot = pred_rot * mask
        pred_trans = pred_trans * mask
        self._assert_shapes(batch, pred_dict, pred_rot, pred_trans)
        return {
            "pred_dict": pred_dict,
            "pred_x_ref_rigids": pred_x_ref_rigids,
            "pred_rot_guided": pred_rot,
            "pred_trans_guided": pred_trans,
            "mask": mask,
            "loss_mask": batch["res_mask"] * (1.0 - batch["fixed_mask"]),
        }

    def _compute_bb_atom_loss(self, batch, pred_x_ref_rigids, pred_dict, loss_mask):
        pred_atom37, pred_atom37_mask, _, _ = all_atom.compute_backbone_atom37(
            bb_rigids=pred_x_ref_rigids,
            aatypes=batch["aatype"].long(),
            torsions=pred_dict["angles"],
        )
        gt_atom37, gt_atom37_mask, _, _ = all_atom.compute_backbone_atom37(
            bb_rigids=Rigid.from_tensor_7(batch["ref_rigids_0"].type(torch.float32)),
            aatypes=batch["aatype"].long(),
            torsions=batch["ref_torsion_angles_sin_cos"],
        )
        pred_atom37 = pred_atom37[..., :5, :]
        gt_atom37 = gt_atom37[..., :5, :]
        atom37_mask = (pred_atom37_mask[..., :5] * gt_atom37_mask[..., :5]).float()

        bb_atom_loss_mask = atom37_mask * loss_mask[..., None]
        bb_atom_loss_vec = torch.sum(
            (pred_atom37 - gt_atom37) ** 2 * bb_atom_loss_mask[..., None],
            dim=(-1, -2, -3),
        ) / (bb_atom_loss_mask.sum(dim=(-1, -2)) + 1e-10)
        bb_atom_loss_vec = bb_atom_loss_vec * (
            getattr(self._exp_conf, "bb_atom_loss_weight", 1.0)
            * getattr(self._exp_conf, "aux_loss_weight", 0.25)
        )
        bb_atom_t_filter = (
            batch["t"].reshape(-1)
            < getattr(self._exp_conf, "bb_atom_loss_t_filter", 0.25)
        ).float()
        return (bb_atom_loss_vec * bb_atom_t_filter).mean()

    def _compute_dist_mat_loss(self, batch, pred_x_ref_rigids, pred_dict, loss_mask):
        pred_atom37, _, _, _ = all_atom.compute_backbone_atom37(
            bb_rigids=pred_x_ref_rigids,
            aatypes=batch["aatype"].long(),
            torsions=pred_dict["angles"],
        )
        gt_atom37, _, _, _ = all_atom.compute_backbone_atom37(
            bb_rigids=Rigid.from_tensor_7(batch["ref_rigids_0"].type(torch.float32)),
            aatypes=batch["aatype"].long(),
            torsions=batch["ref_torsion_angles_sin_cos"],
        )
        pred_atom37 = pred_atom37[..., :5, :]
        gt_atom37 = gt_atom37[..., :5, :]

        gt_flat_atoms = gt_atom37.reshape(gt_atom37.shape[0], -1, 3)
        pred_flat_atoms = pred_atom37.reshape(pred_atom37.shape[0], -1, 3)
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1
        )
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1
        )

        flat_loss_mask = loss_mask.repeat_interleave(5, dim=-1)
        flat_res_mask = batch["res_mask"].repeat_interleave(5, dim=-1)
        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]
        pair_dist_mask = pair_dist_mask * (gt_pair_dists < 6.0)
        m_atoms = flat_loss_mask.shape[-1]
        diag_mask = 1.0 - torch.eye(
            m_atoms, device=pair_dist_mask.device, dtype=pair_dist_mask.dtype
        )
        pair_dist_mask = pair_dist_mask * diag_mask[None, :, :]

        dist_error = torch.sum(
            (gt_pair_dists - pred_pair_dists) ** 2 * pair_dist_mask,
            dim=(-1, -2),
        )
        denom = torch.sum(pair_dist_mask, dim=(-1, -2))
        dist_mat_loss_vec = dist_error / (denom + 1e-10)
        dist_mat_loss_vec = dist_mat_loss_vec * (
            getattr(self._exp_conf, "dist_mat_loss_weight", 1.0)
            * getattr(self._exp_conf, "aux_loss_weight", 0.25)
        )
        dist_t_filter = (
            batch["t"].reshape(-1)
            < getattr(self._exp_conf, "dist_mat_loss_t_filter", 0.25)
        ).float()
        return (dist_mat_loss_vec * dist_t_filter).mean()

    def _compute_losses(self, batch, ctx):
        pred_dict = ctx["pred_dict"]
        pred_x_ref_rigids = ctx["pred_x_ref_rigids"]
        pred_rot_guided = ctx["pred_rot_guided"]
        pred_trans_guided = ctx["pred_trans_guided"]
        mask = ctx["mask"]
        loss_mask = ctx["loss_mask"]

        gt_rot_guided = batch["score_rot_guided"] * mask
        gt_trans_guided = batch["score_trans_guided"] * mask
        rot_w = 1.0 / (batch["rot_score_scaling"][:, None, None] ** 2 + 1e-8)
        tr_w = 1.0 / (batch["trans_score_scaling"][:, None, None] ** 2 + 1e-8)

        loss_rot = (
            ((pred_rot_guided - gt_rot_guided) ** 2 * rot_w * mask).sum()
            / (mask.sum() * 3 + 1e-8)
        )
        loss_tr = (
            ((pred_trans_guided - gt_trans_guided) ** 2 * tr_w * mask).sum()
            / (mask.sum() * 3 + 1e-8)
        )
        trans_loss = torch.sum(
            (
                # batch["ref_rigids_0"][..., 4:]
                # batch["rigids_0"][..., 4:]
                batch["ref_rigids_0"][..., 4:]
                - pred_x_ref_rigids.to_tensor_7()[..., 4:]
            ).abs()
            * mask,
        ) / (mask.sum() + 1e-10)
        torsion_loss = (
            torsion_angle_loss(
                a=pred_dict["angles"],
                # a_gt=batch["torsion_angles_sin_cos"],
                # a_alt_gt=batch["alt_torsion_angles_sin_cos"],
                a_gt=batch["ref_torsion_angles_sin_cos"],
                a_alt_gt=batch["ref_alt_torsion_angles_sin_cos"],
                mask=batch["torsion_angles_mask"],
            ).mean(dim=-1)
            * self._exp_conf.torsion_loss_weight
        )
        bb_atom_loss = self._compute_bb_atom_loss(
            batch, pred_x_ref_rigids, pred_dict, loss_mask
        )
        dist_mat_loss = self._compute_dist_mat_loss(
            batch, pred_x_ref_rigids, pred_dict, loss_mask
        )

        # Keep current objective semantics; this refactor only reorganizes structure.
        total_loss = (
            trans_loss
            + loss_rot
            + torsion_loss
            + bb_atom_loss
            + dist_mat_loss
        )
        loss_dict = {
            "loss_rot": loss_rot,
            "loss_tr": loss_tr,
            "trans_loss": trans_loss,
            "torsion_loss": torsion_loss,
            "bb_atom_loss": bb_atom_loss,
            "dist_mat_loss": dist_mat_loss,
        }
        return total_loss, loss_dict

    def _build_step_output(self, total_loss, loss_dict, grad_norm):
        step_out = {"loss": float(total_loss.detach().item())}
        step_out.update({k: float(v.detach().item()) for k, v in loss_dict.items()})
        step_out["grad_norm"] = (
            float(grad_norm.item()) if grad_norm is not None else None
        )
        return step_out

    def _backward_and_step(self, total_loss, loss_dict):
        """Shared backward pass, grad clip, optimizer step."""
        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"Non-finite loss at step {self.global_step}: {loss_dict}"
            )
        self.accelerator.backward(total_loss)
        grad_norm = None
        if self.accelerator.sync_gradients:
            grad_norm = self.accelerator.clip_grad_norm_(self.actor.parameters(), 1.0)

        self.optimizer.step()
        self.global_step += 1
        return self._build_step_output(total_loss, loss_dict, grad_norm)

    def _prepare_history_batch(self, batch):
        """Flatten [B, F, H, N] history-aware batch to [B*F*H, N] for actor.

        Returns a dict compatible with _forward_and_scores / _compute_losses.
        """
        B, F, H = batch["score_rot_guided"].shape[:3]

        # per-frame [B,F,N,...] → repeat H → [B*F*H, N, ...]
        rigids_t = repeat(batch["rigids_t"], "b f n c -> (b f h) n c", h=H)
        res_mask = repeat(batch["res_mask"], "b f n -> (b f h) n", h=H)
        fixed_mask = repeat(batch["fixed_mask"], "b f n -> (b f h) n", h=H)
        aatype = repeat(batch["aatype"], "b f n -> (b f h) n", h=H)
        seq_idx = repeat(batch["seq_idx"], "b f n -> (b f h) n", h=H)

        # per-(F,H) [B,F,H,N,...] → flatten → [B*F*H, N, ...]
        score_rot_guided = rearrange(batch["score_rot_guided"], "b f h n c -> (b f h) n c")
        score_trans_guided = rearrange(batch["score_trans_guided"], "b f h n c -> (b f h) n c")

        # shared [B,N,...] → repeat F*H → [B*F*H, N, ...]
        node_repr = repeat(batch["node_repr"], "b n d -> (b f h) n d", f=F, h=H)
        edge_repr = repeat(batch["edge_repr"], "b i j d -> (b f h) i j d", f=F, h=H)

        # history [B,H,N,7] → repeat F → [B*F*H, N, 7] as ref_rigids_0
        ref_rigids_0 = repeat(batch["history_rigids"], "b h n c -> (b f h) n c", f=F)

        # scalar [B] → repeat F*H → [B*F*H]
        t = repeat(batch["t"], "b -> (b f h)", f=F, h=H)
        rot_score_scaling = repeat(batch["rot_score_scaling"], "b -> (b f h)", f=F, h=H)
        trans_score_scaling = repeat(batch["trans_score_scaling"], "b -> (b f h)", f=F, h=H)

        # torsion [B,F,N,...] → repeat H → [B*F*H, N, ...]
        torsion_angles_sin_cos = repeat(batch["torsion_angles_sin_cos"], "b f n t c -> (b f h) n t c", h=H)
        alt_torsion_angles_sin_cos = repeat(batch["alt_torsion_angles_sin_cos"], "b f n t c -> (b f h) n t c", h=H)
        torsion_angles_mask = repeat(batch["torsion_angles_mask"], "b f n t -> (b f h) n t", h=H)

        # history torsion [B,H,N,...] → repeat F → [B*F*H, N, ...]
        # Keep output key names as ref_* because downstream losses interpret
        # these as the active reference target for the current history pair.
        ref_torsion = repeat(
            batch["history_torsion_angles_sin_cos"], "b h n t c -> (b f h) n t c", f=F
        )
        ref_alt_torsion = repeat(
            batch["history_alt_torsion_angles_sin_cos"], "b h n t c -> (b f h) n t c", f=F
        )

        # motion/ref fields [B,M,N,...] / [B,R,N,...] → repeat F*H → [B*F*H, M/R, N, ...]
        motion_rigids_0 = repeat(batch["motion_rigids_0"], "b m n c -> (b f h) m n c", f=F, h=H)
        motion_node_mask = repeat(batch["motion_node_mask"], "b m n -> (b f h) m n", f=F, h=H)
        ref_node_mask = repeat(batch["ref_node_mask"], "b r n -> (b f h) r n", f=F, h=H)

        return {
            "rigids_t": rigids_t,
            "t": t,
            "node_repr": node_repr,
            "edge_repr": edge_repr,
            "ref_rigids_0": ref_rigids_0,
            "res_mask": res_mask,
            "fixed_mask": fixed_mask,
            "aatype": aatype,
            "seq_idx": seq_idx,
            "score_rot_guided": score_rot_guided,
            "score_trans_guided": score_trans_guided,
            "rot_score_scaling": rot_score_scaling,
            "trans_score_scaling": trans_score_scaling,
            "torsion_angles_sin_cos": torsion_angles_sin_cos,
            "alt_torsion_angles_sin_cos": alt_torsion_angles_sin_cos,
            "torsion_angles_mask": torsion_angles_mask,
            "ref_torsion_angles_sin_cos": ref_torsion,
            "ref_alt_torsion_angles_sin_cos": ref_alt_torsion,
            "motion_rigids_0": motion_rigids_0,
            "motion_node_mask": motion_node_mask,
            "ref_node_mask": ref_node_mask,
        }

    def _train_step(self, batch):
        """Base training step: guided-score matching loss only."""
        self.optimizer.zero_grad(set_to_none=True)
        pred_ctx = self._forward_and_scores(batch)
        total_loss, loss_dict = self._compute_losses(batch, pred_ctx)
        return self._backward_and_step(total_loss, loss_dict)

    def _train_step_with_history(self, batch):
        """History-aware training step: score matching + env-support + history repel."""
        self.optimizer.zero_grad(set_to_none=True)
        B, F, H = batch["score_rot_guided"].shape[:3]
        formated_batch = self._prepare_history_batch(batch)
        pred_ctx = self._forward_and_scores(formated_batch)

        total_loss, loss_dict = self._compute_losses(formated_batch, pred_ctx)

        # Env scores from original batch [B, F, N, 3]
        env_rot, env_trans = self._compute_env_scores(batch)

        # Compute guided scores once, reuse for both losses
        guided_cache = self._compute_guided_scores(
            formated_batch, pred_ctx, env_rot, env_trans, B, F, H,
        )

        env_support_loss = self._compute_env_support_loss(
            formated_batch, env_rot, env_trans, guided_cache, B, F, H,
        )

        # CN-adaptive loss weighting
        if self._use_cn_adaptive_weight:
            cn_score = self._compute_pred_cn_score(pred_ctx, formated_batch, B, F, H)
            # CN high → structure good → normal weights
            # CN low  → structure bad  → increase KL, decrease repel
            kl_weight = self._lambda_support * (2.0 - cn_score)
            repel_weight = self._lambda_history_repel * cn_score
            loss_dict["pred_cn_score"] = cn_score
            loss_dict["cn_adaptive_kl_weight"] = kl_weight
            loss_dict["cn_adaptive_repel_weight"] = repel_weight
        else:
            kl_weight = self._lambda_support
            repel_weight = self._lambda_history_repel

        total_loss = total_loss + kl_weight * env_support_loss
        loss_dict["env_support_loss"] = env_support_loss

        # Score-space history repulsion
        if self._use_history_score_repel:
            history_repel_loss = self._compute_history_score_repel_loss(
                batch, formated_batch, guided_cache, B, F, H,
            )
            total_loss = total_loss + repel_weight * history_repel_loss
            loss_dict["history_score_repel_loss"] = history_repel_loss

        return self._backward_and_step(total_loss, loss_dict)


    def _dispatch_train_step(self, batch):
        if self._use_env_support and "history_rigids" in batch:
            return self._train_step_with_history(batch)
        return self._train_step(batch)

    def _maybe_log_and_ckpt(self, step_out):
        if not self.accelerator.is_main_process:
            return
        log_dict = {
            "Train/Loss": step_out["loss"],
            "Train/loss_rot": step_out["loss_rot"],
            "Train/loss_tr": step_out["loss_tr"],
            "Train/loss_angles": step_out["torsion_loss"],
            "Train/trans_loss": step_out["trans_loss"],
            "Train/loss_bb_atom": step_out["bb_atom_loss"],
            "Train/loss_dist_mat": step_out["dist_mat_loss"],
        }
        if "env_support_loss" in step_out:
            log_dict["Train/env_support_loss"] = step_out["env_support_loss"]
        if "history_repel_loss" in step_out:
            log_dict["Train/history_repel_loss"] = step_out["history_repel_loss"]
        if "history_score_repel_loss" in step_out:
            log_dict["Train/history_score_repel_loss"] = step_out["history_score_repel_loss"]
        if "pred_cn_score" in step_out:
            log_dict["Train/pred_cn_score"] = step_out["pred_cn_score"]
        if "cn_adaptive_kl_weight" in step_out:
            log_dict["Train/cn_adaptive_kl_weight"] = step_out["cn_adaptive_kl_weight"]
        if "cn_adaptive_repel_weight" in step_out:
            log_dict["Train/cn_adaptive_repel_weight"] = step_out["cn_adaptive_repel_weight"]
        if step_out["grad_norm"] is not None:
            log_dict["Train/grad_norm"] = step_out["grad_norm"]
        if self.global_step % self._exp_conf.ckpt_freq == 0:
            save_path = os.path.join(
                self.conf.experiment.save_dir, f"supervised_step_{self.global_step}"
            )
            os.makedirs(save_path, exist_ok=True)
            self.accelerator.save_state(save_path, safe_serialization=False)
        self.accelerator.log(log_dict, step=self.global_step)

    def _init_epoch_meter(self):
        return {"loss_sum": 0.0, "n": 0}

    def _update_epoch_meter(self, meter, step_out):
        meter["loss_sum"] += step_out["loss"]
        meter["n"] += 1

    def _finish_epoch(self, epoch, meter):
        avg = meter["loss_sum"] / max(1, meter["n"])
        self.logger.info(f"Epoch {epoch} Average Loss: {avg:.4f}")

    def train(self):
        self.logger.info("Starting supervised pre-training...")
        epochs = self.conf.experiment.epochs
        for epoch in range(epochs):
            self.actor.train()
            epoch_meter = self._init_epoch_meter()
            pbar = tqdm(
                self.train_dataloader,
                disable=not self.accelerator.is_local_main_process,
            )
            for batch in pbar:
                step_out = self._dispatch_train_step(batch)
                self._update_epoch_meter(epoch_meter, step_out)
                self._maybe_log_and_ckpt(step_out)
                sup_str = f" | Sup: {step_out['env_support_loss']:.4f}" if "env_support_loss" in step_out else ""
                pbar.set_description(f"Epoch {epoch} | Loss: {step_out['loss']:.4f}{sup_str}")
            self._finish_epoch(epoch, epoch_meter)

@hydra.main(version_base=None, config_path="config", config_name="actor_train")
def run(conf: DictConfig) -> None:
    os.makedirs(conf.experiment.save_dir, exist_ok=True)
    trainer = SupervisedTrainer(conf)
    trainer.train()

if __name__ == "__main__":
    run()
