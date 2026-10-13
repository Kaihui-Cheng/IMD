"""SE(3) diffusion methods."""
import numpy as np
from src.data import so3_diffuser
from src.data import r3_diffuser
from src.data import so3_utils
from scipy.spatial.transform import Rotation
from openfold.utils import rigid_utils as ru
from src.data import utils as du
import torch
import logging
import math

# 修改 1: Torch 版本 (支持 Batch, 改为 Sum)
def gaussian_log_prob_torch(sample, mean, std, mask=None):
    """
    Diagonal Gaussian log-prob.
    sample, mean: (..., N, D)  e.g., (Batch, Residues, 3)
    std: scalar or broadcastable
    mask: (..., N) optional mask for residues
    return: (...,) e.g., (Batch,)
    """
    # 1. 计算逐元素的 LogP
    # Log Normal = -0.5 * ( (x-u)^2/s^2 + 2*log(s) + log(2pi) )
    logp_elem = -0.5 * (
        ((sample - mean) ** 2) / (std**2) + 2 * torch.log(std) + math.log(2 * math.pi)
    )

    # 2. 应用 Mask (处理 padding 或不需要计算的残基)
    if mask is not None:
        # mask: [B, N] -> [B, N, 1] 以匹配 [B, N, 3]
        logp_elem = logp_elem * mask[..., None]

    # 3. 求和 (Sum) 而不是平均 (Mean)
    # 对最后两个维度 (N, 3) 求和，保留前面的 Batch 维度
    return logp_elem.sum(dim=(-2, -1))


# 修改 2: Numpy 版本 (保持一致)
def gaussian_log_prob(sample, mean, std, mask=None):
    """
    Numpy version matching the Torch version.
    """
    logp_elem = -0.5 * (
        ((sample - mean) ** 2) / (std**2) + 2 * np.log(std) + math.log(2 * math.pi)
    )

    if mask is not None:
        logp_elem = logp_elem * mask[..., None]

    # 对最后两个维度求和 [B, N, 3]
    return logp_elem.sum(axis=(-2, -1))

def _extract_trans_rots(rigid: ru.Rigid):
    rot = rigid.get_rots().get_rot_mats().cpu().numpy()
    rot_shape = rot.shape
    num_rots = np.cumprod(rot_shape[:-2])[-1]
    rot = rot.reshape((num_rots, 3, 3))
    rot = Rotation.from_matrix(rot).as_rotvec().reshape(rot_shape[:-2] +(3,))
    tran = rigid.get_trans().cpu().numpy()
    return tran, rot

def _extract_trans_rots_torch(rigid: ru.Rigid):
    rots = rigid.get_rots()
    # If rotation matrices are available, use Log map directly
    # to avoid numerically fragile eigh in rot_mat → quaternion conversion.
    if rots._rot_mats is not None:
        rot_vec = so3_utils.Log(rots.get_rot_mats())
    else:
        rot_vec = du.quat_to_rotvec(rots.get_quats())
    tran = rigid.get_trans()
    return tran, rot_vec

def _assemble_rigid(rotvec, trans,device=torch.device('cpu')):
    rotvec_shape = rotvec.shape
    num_rotvecs = np.cumprod(rotvec_shape[:-1])[-1]
    rotvec = rotvec.reshape((num_rotvecs, 3))
    rotmat = Rotation.from_rotvec(rotvec).as_matrix().reshape(
        rotvec_shape[:-1] + (3, 3))
    return ru.Rigid(
            rots=ru.Rotation(
                rot_mats=torch.Tensor(rotmat).to(device)),
            trans=torch.tensor(trans).to(device))

def _assemble_rigid_torch(rotvec, trans, device=torch.device('cpu')):
    if not torch.is_tensor(rotvec):
        rotvec = torch.as_tensor(rotvec, device=device, dtype=torch.float32)
    else:
        rotvec = rotvec.to(device=device)
    if not torch.is_tensor(trans):
        trans = torch.as_tensor(trans, device=device, dtype=rotvec.dtype)
    else:
        trans = trans.to(device=device, dtype=rotvec.dtype)
    rot_mats = so3_utils.Exp(rotvec)
    return ru.Rigid(
            rots=ru.Rotation(rot_mats=rot_mats),
            trans=trans)


def _center_translations_for_reverse_kernel(
    trans: torch.Tensor,
    diffuse_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Match the centering convention used by reverse_with_logprob rollout sampling.

    This intentionally uses the same projection operator on both the executed
    sample and the Gaussian mean when evaluating transition log-probability.
    """
    denom = torch.sum(diffuse_mask, dim=-1, keepdim=True).clamp_min(1e-8)
    com = torch.sum(trans * diffuse_mask[..., None], dim=-2) / denom
    return trans - com[..., None, :]

class SE3Diffuser:

    def __init__(self, se3_conf):
        self._log = logging.getLogger(__name__)
        self._se3_conf = se3_conf

        self._diffuse_rot = se3_conf.diffuse_rot
        self._so3_diffuser = so3_diffuser.SO3Diffuser(self._se3_conf.so3)

        self._diffuse_trans = se3_conf.diffuse_trans
        self._r3_diffuser = r3_diffuser.R3Diffuser(self._se3_conf.r3)

    def forward_marginal(
            self,
            rigids_0: ru.Rigid,
            t: float,
            diffuse_mask: np.ndarray = None,
            as_tensor_7: bool=True,
        ):
        """
        Args:
            rigids_0: [..., N] openfold Rigid objects
            t: continuous time in [0, 1].

        Returns:
            rigids_t: [..., N] noised rigid. [..., N, 7] if as_tensor_7 is true.
            trans_score: [..., N, 3] translation score
            rot_score: [..., N, 3] rotation score
            trans_score_norm: [...] translation score norm
            rot_score_norm: [...] rotation score norm
        """
        trans_0, rot_0 = _extract_trans_rots(rigids_0)

        if not self._diffuse_rot:
            rot_t, rot_score, rot_score_scaling = (
                rot_0,
                np.zeros_like(rot_0),
                np.ones_like(t)
            )
        else:
            rot_t, rot_score = self._so3_diffuser.forward_marginal(
                rot_0, t)
            rot_score_scaling = self._so3_diffuser.score_scaling(t)

        if not self._diffuse_trans:
            trans_t, trans_score, trans_score_scaling = (
                trans_0,
                np.zeros_like(trans_0),
                np.ones_like(t)
            )
        else:
            trans_t, trans_score = self._r3_diffuser.forward_marginal(
                trans_0, t)
            trans_score_scaling = self._r3_diffuser.score_scaling(t)

        if diffuse_mask is not None:
            # diffuse_mask = torch.tensor(diffuse_mask).to(rot_t.device)
            rot_t = self._apply_mask(
                rot_t, rot_0, diffuse_mask[..., None])
            trans_t = self._apply_mask(
                trans_t, trans_0, diffuse_mask[..., None])

            trans_score = self._apply_mask(
                trans_score,
                np.zeros_like(trans_score),
                diffuse_mask[..., None])
            rot_score = self._apply_mask(
                rot_score,
                np.zeros_like(rot_score),
                diffuse_mask[..., None])
        rigids_t = _assemble_rigid(rot_t, trans_t)
        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()
        return {
            'rigids_t': rigids_t,
            'trans_score': trans_score,
            'rot_score': rot_score,
            'trans_score_scaling': trans_score_scaling,
            'rot_score_scaling': rot_score_scaling,
        }

    def calc_trans_0(self, trans_score, trans_t, t):
        return self._r3_diffuser.calc_trans_0(trans_score, trans_t, t)

    def calc_trans_score(self, trans_t, trans_0, t, use_torch=False, scale=True):
        return self._r3_diffuser.score(
            trans_t, trans_0, t, use_torch=use_torch, scale=scale)

    def calc_rot_score(self, rots_t, rots_0, t):
        rots_0_inv = rots_0.invert()
        quats_0_inv = rots_0_inv.get_quats()
        quats_t = rots_t.get_quats()
        quats_0t = ru.quat_multiply(quats_0_inv, quats_t)
        rotvec_0t = du.quat_to_rotvec(quats_0t)
        return self._so3_diffuser.torch_score(rotvec_0t, t)

    def _apply_mask(self, x_diff, x_fixed, diff_mask):
        return diff_mask * x_diff + (1 - diff_mask) * x_fixed

    def trans_parameters(self, trans_t, score_t, t, dt, mask):
        return self._r3_diffuser.distribution(
            trans_t, score_t, t, dt, mask)

    def score(
            self,
            rigid_0: ru.Rigid,
            rigid_t: ru.Rigid,
            t: float):
        tran_0, rot_0 = _extract_trans_rots(rigid_0)
        tran_t, rot_t = _extract_trans_rots(rigid_t)

        if not self._diffuse_rot:
            rot_score = np.zeros_like(rot_0)
        else:
            rot_score = self._so3_diffuser.score(
                rot_t, t)

        if not self._diffuse_trans:
            trans_score = np.zeros_like(tran_0)
        else:
            trans_score = self._r3_diffuser.score(tran_t, tran_0, t)

        return trans_score, rot_score

    def score_scaling(self, t):
        rot_score_scaling = self._so3_diffuser.score_scaling(t)
        trans_score_scaling = self._r3_diffuser.score_scaling(t)
        return rot_score_scaling, trans_score_scaling

    def reverse(
            self,
            rigid_t: ru.Rigid,
            rot_score: np.ndarray,
            trans_score: np.ndarray,
            t: float,
            dt: float,
            diffuse_mask: np.ndarray = None,
            center: bool=True,
            noise_scale: float=1.0,
            device: torch.device = torch.device('cpu')
        ):
        """Reverse sampling function from (t) to (t-1).

        Args:
            rigid_t: [..., N] protein rigid objects at time t.
            rot_score: [..., N, 3] rotation score.
            trans_score: [..., N, 3] translation score.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: [..., N] which residues to update.
            center: true to set center of mass to zero after step

        Returns:
            rigid_t_1: [..., N] protein rigid objects at time t-1.
        """
        trans_t, rot_t = _extract_trans_rots(rigid_t)
        if not self._diffuse_rot:
            rot_t_1 = rot_t
        else:
            rot_t_1 = self._so3_diffuser.reverse(
                rot_t=rot_t,
                score_t=rot_score,
                t=t,
                dt=dt,
                noise_scale=noise_scale,
                )
        if not self._diffuse_trans:
            trans_t_1 = trans_t
        else:
            trans_t_1 = self._r3_diffuser.reverse(
                x_t=trans_t,
                score_t=trans_score,
                t=t,
                dt=dt,
                center=center,
                noise_scale=noise_scale
                )

        if diffuse_mask is not None:
            trans_t_1 = self._apply_mask(
                trans_t_1, trans_t, diffuse_mask[..., None])
            rot_t_1 = self._apply_mask(
                rot_t_1, rot_t, diffuse_mask[..., None])

        return _assemble_rigid(rot_t_1, trans_t_1,device)

    def reverse_torch(
            self,
            *,
            rigid_t: ru.Rigid,
            rot_score: torch.Tensor,
            trans_score: torch.Tensor,
            t: float,
            dt: float,
            diffuse_mask: torch.Tensor | None = None,
            center: bool = True,
            noise_scale: float = 1.0,
            device: torch.device | None = None,
            z_rot: torch.Tensor | None = None,
            z_trans: torch.Tensor | None = None,
        ) -> ru.Rigid:
        """Torch-only reverse sampling function from time ``t`` to ``t-dt``."""
        trans_t, rot_t = _extract_trans_rots_torch(rigid_t)
        if device is None:
            device = trans_t.device
        dtype = trans_t.dtype

        if diffuse_mask is not None:
            diffuse_mask = diffuse_mask.to(device=device, dtype=dtype)

        if not self._diffuse_rot:
            rot_t_1 = rot_t
        else:
            g_t = torch.as_tensor(
                self._so3_diffuser.diffusion_coef(t), device=device, dtype=rot_t.dtype
            )
            rot_score = rot_score.to(device=device, dtype=rot_t.dtype)
            rot_noise = torch.randn_like(rot_score) if z_rot is None else z_rot
            rot_noise = noise_scale * rot_noise.to(device=device, dtype=rot_t.dtype)
            perturb = (g_t ** 2) * rot_score * dt + g_t * math.sqrt(dt) * rot_noise
            if diffuse_mask is not None:
                perturb = perturb * diffuse_mask[..., None]
            rot_t_1_mat = torch.matmul(so3_utils.Exp(rot_t), so3_utils.Exp(perturb))
            rot_t_1 = so3_utils.Log(rot_t_1_mat)

        if not self._diffuse_trans:
            trans_t_1 = trans_t
        else:
            trans_t_scaled = self._r3_diffuser._scale(trans_t)
            g_t = torch.as_tensor(
                self._r3_diffuser.diffusion_coef(t), device=device, dtype=trans_t_scaled.dtype
            )
            b_t = torch.as_tensor(
                self._r3_diffuser.b_t(t), device=device, dtype=trans_t_scaled.dtype
            )
            f_t = -0.5 * b_t * trans_t_scaled
            trans_score = trans_score.to(device=device, dtype=trans_t_scaled.dtype)
            trans_noise = torch.randn_like(trans_score) if z_trans is None else z_trans
            trans_noise = noise_scale * trans_noise.to(device=device, dtype=trans_t_scaled.dtype)
            perturb = (f_t - g_t**2 * trans_score) * dt + g_t * math.sqrt(dt) * trans_noise
            if diffuse_mask is not None:
                perturb = perturb * diffuse_mask[..., None]
            trans_t_1 = trans_t_scaled - perturb
            if center:
                if diffuse_mask is None:
                    center_mask = torch.ones(
                        trans_t.shape[:-1], device=device, dtype=trans_t_scaled.dtype
                    )
                else:
                    center_mask = diffuse_mask.to(device=device, dtype=trans_t_scaled.dtype)
                trans_t_1 = _center_translations_for_reverse_kernel(trans_t_1, center_mask)
            trans_t_1 = self._r3_diffuser._unscale(trans_t_1)

        if diffuse_mask is not None:
            trans_t_1 = self._apply_mask(trans_t_1, trans_t, diffuse_mask[..., None])
            rot_t_1 = self._apply_mask(rot_t_1, rot_t, diffuse_mask[..., None])

        return _assemble_rigid_torch(rot_t_1, trans_t_1, device)

    def sample_ref(
            self,
            n_samples: int,
            impute: ru.Rigid=None,
            diffuse_mask: np.ndarray=None,
            as_tensor_7: bool=False,
            dynamics: bool=None,
            device: torch.device | None = None,
            dtype: torch.dtype = torch.float32,
        ):
        """Samples rigids from reference distribution.

        Args:
            n_samples: Number of samples.
            impute: Rigid objects to use as imputation values if either
                translations or rotations are not diffused.
        """
        use_dynamics = self._se3_conf.dynamics if dynamics is None else dynamics

        if device is not None:
            if impute is not None:
                trans_impute, rot_impute = _extract_trans_rots_torch(impute)
                trans_impute = trans_impute.reshape((n_samples, 3)).to(device=device, dtype=dtype)
                rot_impute = rot_impute.reshape((n_samples, 3)).to(device=device, dtype=dtype)
                trans_impute = self._r3_diffuser._scale(trans_impute)

            if diffuse_mask is not None and impute is None:
                raise ValueError('Must provide imputation values.')

            if (not self._diffuse_rot) and impute is None:
                raise ValueError('Must provide imputation values.')

            if (not self._diffuse_trans) and impute is None:
                raise ValueError('Must provide imputation values.')

            if self._diffuse_rot:
                rot_ref = self._so3_diffuser.sample_ref_torch(
                    n_samples=n_samples,
                    device=device,
                    dtype=dtype,
                )
            else:
                rot_ref = rot_impute

            if self._diffuse_trans:
                trans_ref = self._r3_diffuser.sample_ref_torch(
                    n_samples=n_samples,
                    device=device,
                    dtype=dtype,
                )
            else:
                trans_ref = trans_impute

            if diffuse_mask is not None:
                diffuse_mask = torch.as_tensor(diffuse_mask, device=device, dtype=dtype)
                rot_ref = self._apply_mask(rot_ref, rot_impute, diffuse_mask[..., None])
                trans_ref = self._apply_mask(trans_ref, trans_impute, diffuse_mask[..., None])
            trans_ref = self._r3_diffuser._unscale(trans_ref)

            if not use_dynamics:
                rigids_t = _assemble_rigid_torch(rot_ref, trans_ref, device)
                if as_tensor_7:
                    rigids_t = rigids_t.to_tensor_7()
            else:
                frame_time = self._se3_conf.frame_time
                rot_ref = rot_ref.reshape(frame_time, -1, 3)
                trans_ref = trans_ref.reshape(frame_time, -1, 3)
                rigids_t = torch.stack(
                    [
                        _assemble_rigid_torch(rot_ref[i], trans_ref[i], device).to_tensor_7()
                        for i in range(frame_time)
                    ],
                    dim=0,
                )
            return {'rigids_t': rigids_t}

        if impute is not None:
            assert impute.shape[0] == n_samples
            trans_impute, rot_impute = _extract_trans_rots(impute)
            trans_impute = trans_impute.reshape((n_samples, 3))
            rot_impute = rot_impute.reshape((n_samples, 3))
            trans_impute = self._r3_diffuser._scale(trans_impute)

        if diffuse_mask is not None and impute is None:
            raise ValueError('Must provide imputation values.')

        if (not self._diffuse_rot) and impute is None:
            raise ValueError('Must provide imputation values.')

        if (not self._diffuse_trans) and impute is None:
            raise ValueError('Must provide imputation values.')

        if self._diffuse_rot:
            rot_ref = self._so3_diffuser.sample_ref(
                n_samples=n_samples)
        else:
            rot_ref = rot_impute

        if self._diffuse_trans:
            trans_ref = self._r3_diffuser.sample_ref(
                n_samples=n_samples
            )
        else:
            trans_ref = trans_impute

        if diffuse_mask is not None:
            rot_ref = self._apply_mask(
                rot_ref, rot_impute, diffuse_mask[..., None])
            trans_ref = self._apply_mask(
                trans_ref, trans_impute, diffuse_mask[..., None])
        trans_ref = self._r3_diffuser._unscale(trans_ref)

        if not use_dynamics:
            rigids_t = _assemble_rigid(rot_ref, trans_ref)
            if as_tensor_7:
                rigids_t = rigids_t.to_tensor_7()
        else:
            frame_time = self._se3_conf.frame_time
            rot_ref = rot_ref.reshape(frame_time,-1,3)
            trans_ref = trans_ref.reshape(frame_time,-1,3)
            rigids_t=[]
            for i in range(frame_time):
                rigids_t.append(_assemble_rigid(rot_ref[i], trans_ref[i]).to_tensor_7())
            rigids_t = torch.stack(rigids_t,dim=0)
        return {'rigids_t': rigids_t}


    def reverse_with_logprob(
        self,
        *,
        rigid_t: ru.Rigid,
        rot_score: torch.Tensor,  # (..., N, 3)
        trans_score: torch.Tensor,  # (..., N, 3)
        t: float,
        dt: float,
        diffuse_mask: torch.Tensor | None = None,
        center: bool = True,
        noise_scale: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        """
        One reverse step with log-prob computation using Torch.

        Returns:
            rigid_t_1: ru.Rigid
            logp_rot:  (batch,)
            logp_trans:(batch,)
            rot_z:     (..., N, 3)
        """

        # === extract ===
        trans_t, rot_t = _extract_trans_rots_torch(rigid_t)
        dtype = trans_t.dtype

        # ======================
        # translation (R^3)
        # ======================
        if self._diffuse_trans:
            trans_t = self._r3_diffuser._scale(trans_t)
            g_t = torch.tensor(self._r3_diffuser.diffusion_coef(t), device=device, dtype=trans_t.dtype)
            b_t = torch.tensor(self._r3_diffuser.b_t(t), device=device, dtype=trans_t.dtype)
            f_t = -0.5 * b_t * trans_t

            trans_score = trans_score.to(trans_t.dtype)

            z = noise_scale * torch.randn_like(trans_score)
            perturb = (f_t - g_t**2 * trans_score) * dt + g_t * math.sqrt(dt) * z

            if diffuse_mask is not None:
                perturb = perturb * diffuse_mask[..., None]
            else:
                diffuse_mask = torch.ones(trans_t.shape[:-1], device=device)

            trans_t_1 = trans_t - perturb

            if center:
                trans_t_1 = _center_translations_for_reverse_kernel(
                    trans_t_1,
                    diffuse_mask.to(device=device, dtype=trans_t.dtype),
                )
            trans_t_1 = self._r3_diffuser._unscale(trans_t_1)
        else:
            trans_t_1 = trans_t

        # ======================
        # rotation (SO(3) via rotvec)
        # ======================
        if self._diffuse_rot:
            g_t = torch.tensor(self._so3_diffuser.diffusion_coef(t), device=device, dtype=rot_t.dtype)
            rot_score = rot_score.to(rot_t.dtype)

            z = noise_scale * torch.randn_like(rot_score)
            rot_z = z

            perturb = (g_t ** 2) * rot_score * dt + g_t * math.sqrt(dt) * z
            if diffuse_mask is not None:
                perturb = perturb * diffuse_mask[..., None]

            rot_t_1_mat = torch.matmul(so3_utils.Exp(rot_t), so3_utils.Exp(perturb))
            rot_t_1 = so3_utils.Log(rot_t_1_mat)
        else:
            rot_t_1 = rot_t
            rot_z = torch.zeros_like(rot_score)

        rigid_t_1 = _assemble_rigid_torch(rot_t_1, trans_t_1, device)
        if diffuse_mask is None:
            diffuse_mask = torch.ones(trans_t.shape[:-1], device=device, dtype=dtype)
        else:
            diffuse_mask = diffuse_mask.to(device=device, dtype=dtype)
        logp_rot, logp_trans, _ = self.transition_logprob_given_next(
            rigid_t=rigid_t,
            rigid_t_1=rigid_t_1,
            rot_score=rot_score,
            trans_score=trans_score,
            t=t,
            dt=dt,
            diffuse_mask=diffuse_mask,
            center=center,
            noise_scale=noise_scale,
        )

        return (
            rigid_t_1,
            logp_rot,
            logp_trans,
            rot_z,
        )

    # ──────────────────────────────────────────────────────────────────
    # Girsanov-aware reverse step (post-hoc reweighting demo).
    # New function — does NOT replace `reverse_with_logprob`. Adds:
    #   • returns trans_z too (alongside rot_z),
    #   • when delta_rot / delta_trans (= env_score - actor_score) are
    #     supplied, accumulates per-step Girsanov increments
    #         I1_inc = g_t · ⟨delta, sqrt(dt)·z⟩      (vector dot prod)
    #         I2_inc = g_t² · ‖delta‖² · dt
    #     summed over (..., N, 3), giving shape (batch,).
    # See docs/girsanov_mbar_design.md for derivation.
    # ──────────────────────────────────────────────────────────────────
    def reverse_with_logprob_girsanov(
        self,
        *,
        rigid_t: ru.Rigid,
        rot_score: torch.Tensor,    # modified score = (1+α)·env - α·actor
        trans_score: torch.Tensor,  # modified score (same convention)
        t: float,
        dt: float,
        delta_rot: torch.Tensor | None = None,    # = env_rot   - actor_rot   (no α scaling)
        delta_trans: torch.Tensor | None = None,  # = env_trans - actor_trans
        diffuse_mask: torch.Tensor | None = None,
        center: bool = True,
        noise_scale: float = 1.0,
        device: torch.device = torch.device("cpu"),
        girsanov_dtype: torch.dtype | None = None,
    ):
        """One reverse step + Girsanov per-step accounting.

        Returns:
            rigid_t_1   : ru.Rigid
            logp_rot    : (batch,)
            logp_trans  : (batch,)
            rot_z       : (..., N, 3)
            trans_z     : (..., N, 3)            [None if not _diffuse_trans]
            girsanov    : dict with batch-shape tensors:
                            I1_trans, I2_trans, I1_rot, I2_rot
                          each = scalar per batch element; zeros if delta is None
        """
        trans_t, rot_t = _extract_trans_rots_torch(rigid_t)
        dtype = trans_t.dtype
        accum_dtype = girsanov_dtype or dtype
        batch_shape = trans_t.shape[:-2]   # (...,)
        I1_trans = torch.zeros(batch_shape, device=device, dtype=accum_dtype)
        I2_trans = torch.zeros(batch_shape, device=device, dtype=accum_dtype)
        I1_rot   = torch.zeros(batch_shape, device=device, dtype=accum_dtype)
        I2_rot   = torch.zeros(batch_shape, device=device, dtype=accum_dtype)
        trans_z_out = None

        # ───────── translation ─────────
        if self._diffuse_trans:
            trans_t = self._r3_diffuser._scale(trans_t)
            g_t = torch.tensor(self._r3_diffuser.diffusion_coef(t), device=device, dtype=trans_t.dtype)
            b_t = torch.tensor(self._r3_diffuser.b_t(t), device=device, dtype=trans_t.dtype)
            f_t = -0.5 * b_t * trans_t

            trans_score = trans_score.to(trans_t.dtype)
            z = noise_scale * torch.randn_like(trans_score)
            trans_z_out = z
            perturb = (f_t - g_t**2 * trans_score) * dt + g_t * math.sqrt(dt) * z

            if diffuse_mask is not None:
                perturb = perturb * diffuse_mask[..., None]
            else:
                diffuse_mask = torch.ones(trans_t.shape[:-1], device=device)

            trans_t_1 = trans_t - perturb
            if center:
                trans_t_1 = _center_translations_for_reverse_kernel(
                    trans_t_1,
                    diffuse_mask.to(device=device, dtype=trans_t.dtype),
                )
            trans_t_1 = self._r3_diffuser._unscale(trans_t_1)

            # Girsanov accounting (translation): delta is env - actor (no α)
            if delta_trans is not None:
                d = delta_trans.to(z.dtype)
                active = diffuse_mask[..., None].to(dtype=z.dtype)
                d = d * active
                z_effective = z * active
                if center:
                    # The executed translation kernel removes the center of
                    # mass after applying drift and noise.  Its likelihood
                    # ratio therefore lives in that same projected subspace.
                    d = _center_translations_for_reverse_kernel(
                        d,
                        diffuse_mask.to(device=device, dtype=z.dtype),
                    )
                    z_effective = _center_translations_for_reverse_kernel(
                        z_effective,
                        diffuse_mask.to(device=device, dtype=z.dtype),
                    )
                    d = d * active
                    z_effective = z_effective * active
                d_acc = d.to(accum_dtype)
                z_acc = z_effective.to(accum_dtype)
                g_acc = g_t.to(accum_dtype)
                # I1_inc = g_t * sum over (N, 3) of (d * sqrt(dt) * z)
                # I2_inc = g_t**2 * sum over (N, 3) of d**2 * dt
                I1_trans = g_acc * (
                    d_acc * math.sqrt(dt) * z_acc
                ).sum(dim=(-1, -2))
                I2_trans = (
                    (g_acc ** 2) * (d_acc * d_acc).sum(dim=(-1, -2)) * dt
                )
        else:
            trans_t_1 = trans_t

        # ───────── rotation ─────────
        if self._diffuse_rot:
            g_t = torch.tensor(self._so3_diffuser.diffusion_coef(t), device=device, dtype=rot_t.dtype)
            rot_score = rot_score.to(rot_t.dtype)
            z = noise_scale * torch.randn_like(rot_score)
            rot_z = z
            perturb = (g_t ** 2) * rot_score * dt + g_t * math.sqrt(dt) * z
            if diffuse_mask is not None:
                perturb = perturb * diffuse_mask[..., None]
            rot_t_1_mat = torch.matmul(so3_utils.Exp(rot_t), so3_utils.Exp(perturb))
            rot_t_1 = so3_utils.Log(rot_t_1_mat)

            if delta_rot is not None:
                d = delta_rot.to(z.dtype)
                if diffuse_mask is not None:
                    d = d * diffuse_mask[..., None]
                d_acc = d.to(accum_dtype)
                z_acc = z.to(accum_dtype)
                g_acc = g_t.to(accum_dtype)
                I1_rot = g_acc * (
                    d_acc * math.sqrt(dt) * z_acc
                ).sum(dim=(-1, -2))
                I2_rot = (
                    (g_acc ** 2) * (d_acc * d_acc).sum(dim=(-1, -2)) * dt
                )
        else:
            rot_t_1 = rot_t
            rot_z = torch.zeros_like(rot_score)

        rigid_t_1 = _assemble_rigid_torch(rot_t_1, trans_t_1, device)
        if diffuse_mask is None:
            diffuse_mask = torch.ones(trans_t.shape[:-1], device=device, dtype=dtype)
        else:
            diffuse_mask = diffuse_mask.to(device=device, dtype=dtype)
        logp_rot, logp_trans, _ = self.transition_logprob_given_next(
            rigid_t=rigid_t,
            rigid_t_1=rigid_t_1,
            rot_score=rot_score,
            trans_score=trans_score,
            t=t,
            dt=dt,
            diffuse_mask=diffuse_mask,
            center=center,
            noise_scale=noise_scale,
        )

        girsanov = {
            "I1_trans": I1_trans,
            "I2_trans": I2_trans,
            "I1_rot":   I1_rot,
            "I2_rot":   I2_rot,
        }
        return (
            rigid_t_1,
            logp_rot,
            logp_trans,
            rot_z,
            trans_z_out,
            girsanov,
        )

    def reverse_with_logprob_torch(
        self,
        *,
        rigid_t: ru.Rigid,
        rot_score,  # (..., N, 3)
        trans_score,  # (..., N, 3)
        t: float,
        dt: float,
        noise_scale: float = 1.0,
        diffuse_mask: None,
        center: bool = True,
        rigid_t_1: ru.Rigid | None = None,
    ):
        """
        One reverse step with log-prob computation.

        Returns:
            rigid_t_1: ru.Rigid
            logp_rot:  (batch,)
            logp_trans:(batch,)
        """
        if rigid_t_1 is None:
            raise ValueError("rigid_t_1 must be provided for reverse_with_logprob_torch.")
        logp_rot, logp_trans, _ = self.transition_logprob_given_next(
            rigid_t=rigid_t,
            rigid_t_1=rigid_t_1,
            rot_score=rot_score,
            trans_score=trans_score,
            t=t,
            dt=dt,
            diffuse_mask=diffuse_mask,
            center=center,
            noise_scale=noise_scale,
        )
        return logp_rot, logp_trans

    def transition_logprob_given_next(
        self,
        *,
        rigid_t: ru.Rigid,
        rigid_t_1: ru.Rigid,
        rot_score: torch.Tensor,  # (..., N, 3)
        trans_score: torch.Tensor,  # (..., N, 3)
        t: float,
        dt: float,
        diffuse_mask: torch.Tensor | None = None,
        center: bool = True,
        noise_scale: float = 1.0,
    ):
        """
        Evaluate reverse-step conditional log-probability on a fixed executed transition.

        Both rollout-time old_logp and PPO-time new/ref logp should call the same
        evaluator so they live in the same centered-kernel semantics.
        """
        trans_t, rot_t = _extract_trans_rots_torch(rigid_t)
        trans_t_1, rot_t_1 = _extract_trans_rots_torch(rigid_t_1)
        device = trans_t.device
        dtype = trans_t.dtype

        if diffuse_mask is None:
            diffuse_mask = torch.ones(
                trans_t.shape[:-1], device=device, dtype=dtype
            )
        else:
            diffuse_mask = diffuse_mask.to(device=device, dtype=dtype)

        if self._diffuse_trans:
            trans_t_scaled = self._r3_diffuser._scale(trans_t)
            trans_t_1_scaled = self._r3_diffuser._scale(trans_t_1)
            trans_score = trans_score.to(device=device, dtype=trans_t_scaled.dtype)
            g_t = torch.as_tensor(
                self._r3_diffuser.diffusion_coef(t),
                device=device,
                dtype=trans_t_scaled.dtype,
            )
            b_t = torch.as_tensor(
                self._r3_diffuser.b_t(t),
                device=device,
                dtype=trans_t_scaled.dtype,
            )
            if g_t.ndim > 0:
                while g_t.ndim < trans_t_scaled.ndim:
                    g_t = g_t.unsqueeze(-1)
                while b_t.ndim < trans_t_scaled.ndim:
                    b_t = b_t.unsqueeze(-1)
            f_t = -0.5 * b_t * trans_t_scaled
            mean_trans = trans_t_scaled - (f_t - g_t**2 * trans_score) * dt
            if center:
                trans_t_1_scaled = _center_translations_for_reverse_kernel(
                    trans_t_1_scaled,
                    diffuse_mask.to(device=device, dtype=trans_t_scaled.dtype),
                )
                mean_trans = _center_translations_for_reverse_kernel(
                    mean_trans,
                    diffuse_mask.to(device=device, dtype=trans_t_scaled.dtype),
                )
            std_trans = g_t * math.sqrt(dt) * noise_scale
            logp_trans = gaussian_log_prob_torch(
                sample=trans_t_1_scaled,
                mean=mean_trans,
                std=std_trans,
                mask=diffuse_mask,
            )
        else:
            logp_trans = torch.zeros(trans_t.shape[:-2], device=device, dtype=dtype)

        if self._diffuse_rot:
            rot_score = rot_score.to(device=device, dtype=rot_t.dtype)
            g_t = torch.as_tensor(
                self._so3_diffuser.diffusion_coef(t), device=device, dtype=rot_t.dtype
            )
            if g_t.ndim > 0:
                while g_t.ndim < rot_t.ndim:
                    g_t = g_t.unsqueeze(-1)
            rot_t_mat = so3_utils.Exp(rot_t)
            rot_t_1_mat = so3_utils.Exp(rot_t_1)
            perturb_obs = so3_utils.Log(
                torch.matmul(rot_t_mat.transpose(-1, -2), rot_t_1_mat)
            )
            perturb_obs = perturb_obs * diffuse_mask[..., None].to(device=device, dtype=rot_t.dtype)
            mean_rot = (g_t**2) * rot_score * dt
            std_rot = g_t * math.sqrt(dt) * noise_scale
            logp_rot = gaussian_log_prob_torch(
                sample=perturb_obs,
                mean=mean_rot,
                std=std_rot,
                mask=diffuse_mask.to(device=device, dtype=rot_t.dtype),
            )
        else:
            logp_rot = torch.zeros(rot_t.shape[:-2], device=device, dtype=rot_t.dtype)

        return logp_rot, logp_trans, logp_rot + logp_trans
