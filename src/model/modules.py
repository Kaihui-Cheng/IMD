"""Fork of Openfold's IPA."""

import numpy as np
import torch
import math
from scipy.stats import truncnorm
import torch.nn as nn
from typing import Optional, Callable, List, Sequence
from openfold.utils.rigid_utils import Rigid
from openfold.model.structure_module import AngleResnet
from openfold.utils.tensor_utils import chunk_layer
from src.data import all_atom
import torch.nn.functional as F
import inspect
import sys
from src.model.utils import rope_params, rope_apply_1d, RMSNorm
from src.model.attention import attention
import torch.utils.checkpoint as cp
from src.model.causal_modules import (
    CausalDyneTrionSelfAttention,
    prepare_blockwise_causal_attn_mask,
)
from src.model.utils import gpu_mem
from einops import repeat, rearrange

def _run_ipa_block(module, all_node, all_edge_embed, all_curr_rigids, all_node_mask):
    return module(all_node, all_edge_embed, all_curr_rigids, all_node_mask)

def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)

def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f

def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def compute_angles(ca_pos, pts):
    batch_size, num_res, num_heads, num_pts, _ = pts.shape
    calpha_vecs = (ca_pos[:, :, None, :] - ca_pos[:, None, :, :]) + 1e-10
    calpha_vecs = torch.tile(calpha_vecs[:, :, :, None, None, :], (1, 1, 1, num_heads, num_pts, 1))
    ipa_pts = pts[:, :, None, :, :, :] - torch.tile(ca_pos[:, :, None, None, None, :], (1, 1, num_res, num_heads, num_pts, 1))
    phi_angles = all_atom.calculate_neighbor_angles(
        calpha_vecs.reshape(-1, 3),
        ipa_pts.reshape(-1, 3)
    ).reshape(batch_size, num_res, num_res, num_heads, num_pts)
    return  phi_angles


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                lecun_normal_init_(self.weight)
            elif init == "relu":
                he_normal_init_(self.weight)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")


class StructureModuleTransition(nn.Module):
    def __init__(self, c):
        super(StructureModuleTransition, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(self.c)

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        s = s + s_initial
        s = self.ln(s)

        return s


class EdgeTransition(nn.Module):
    def __init__(
            self,
            *,
            node_embed_size,
            edge_embed_in,
            edge_embed_out,
            num_layers=2,
            node_dilation=2
        ):
        super(EdgeTransition, self).__init__()

        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = Linear(
            node_embed_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, node_embed, edge_embed):
        node_embed = self.initial_embed(node_embed)
        batch_size, num_res, _ = node_embed.shape
        edge_bias = torch.cat([
            torch.tile(node_embed[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(node_embed[:, None, :, :], (1, num_res, 1, 1)),
        ], axis=-1)
        edge_embed = torch.cat(
            [edge_embed, edge_bias], axis=-1).reshape(
                batch_size * num_res**2, -1)
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        edge_embed = edge_embed.reshape(
            batch_size, num_res, num_res, -1
        )
        return edge_embed

class EdgeTransitionOutProductMean(nn.Module):
    def __init__(self, c_node: int, c_edge: int, c_hidden: int, eps: float = 1e-6):
        super().__init__()
        self.c_node = c_node
        self.c_edge = c_edge
        self.c_hidden = c_hidden
        self.eps = eps

        self.node_norm = nn.LayerNorm(c_node)
        self.linear_a = Linear(c_node, c_hidden)
        self.linear_b = Linear(c_node, c_hidden)

        self.linear_out = Linear(c_hidden**2, c_edge, init="final")
        self.edge_norm = nn.LayerNorm(c_edge)

    def _op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        a: [B, N, C]
        b: [B, N, C]

        return:
            [B, N, N, C_edge]
        """
        # [B, N, N, C, C]
        outer = torch.einsum("bic,bjd->bijcd", a, b)

        # [B, N, N, C*C]
        outer = outer.reshape(
            outer.shape[:-2] + (self.c_hidden * self.c_hidden,)
        )

        # [B, N, N, C_edge]
        return self.linear_out(outer)

    @torch.jit.ignore
    def _chunk(self,
            a: torch.Tensor,
            b: torch.Tensor,
            chunk_size: int,

        ) -> torch.Tensor:
        # Since the "batch dim" in this case is not a true batch dimension
        # (in that the shape of the output depends on it), we need to
        # iterate over it ourselves
        outs = []
        N = a.shape[1]
        for i in range(0, N, chunk_size):
            a_i = a[:, i : i + chunk_size]  # [B, chunk, C]
            # [B, chunk, N, C, C]
            outer = torch.einsum("bic,bjd->bijcd", a_i, b)
            outer = outer.reshape(outer.shape[:-2] + (self.c_hidden * self.c_hidden,))
            outs.append(self.linear_out(outer))

        # concat over residue-i dimension
        delta = torch.cat(outs, dim=1)
        return delta



    def forward(
        self, node: torch.Tensor, edge: torch.Tensor, chunk_size: int | None = None
    )-> torch.Tensor:
        """
        Args:
            node: [B, N, C_node]
            edge: [B, N, N, C_edge]
            chunk_size: chunk over N dimension (recommended: 16 / 32)

        Returns:
            updated edge: [B, N, N, C_edge]
        """
        B, N, _ = node.shape
        node = self.node_norm(node)
        a = self.linear_a(node)  # [B, N, C]
        b = self.linear_b(node)  # [B, N, C]

        if chunk_size is None:
            delta = self._op(a, b)
        else:
            delta = self._chunk(a, b, chunk_size)
        # symemtric forced
        delta = 0.5 * (delta + delta.transpose(1, 2))
        edge = edge + delta
        return self.edge_norm(edge)


class InvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """
    def __init__(
        self,
        ipa_conf,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(InvariantPointAttention, self).__init__()
        self._ipa_conf = ipa_conf

        self.c_s = ipa_conf.c_s
        self.c_z = ipa_conf.c_z
        self.c_hidden = ipa_conf.c_hidden
        self.no_heads = ipa_conf.no_heads
        self.no_qk_points = ipa_conf.no_qk_points
        self.no_v_points = ipa_conf.no_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        self.linear_b = Linear(self.c_z, self.no_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        self.head_weights = nn.Parameter(torch.zeros((ipa_conf.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        # concat_out_dim =  (
        #     self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        # )
        concat_out_dim =  (
            self.c_z // 4 + self.c_hidden + self.no_v_points * 8
        )
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init="final")
        # self.linear_out = IPA_MLP(self.no_heads * concat_out_dim, self.c_s)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()
        # TODO: Remove after published checkpoint is updated without these weights.
        self.linear_rbf = Linear(20, 1)

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        r: Rigid,
        mask: torch.Tensor,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])

        if(_offload_inference):
            z[0] = z[0].cpu()

        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        # [*, N_res, N_res, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        # print(pt_displacement.shape[:])

        pt_att = pt_displacement ** 2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        a = a + pt_att

        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)
        # torch.save(a.cpu(),'a.pth')
        # exit()

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3).to(dtype=a.dtype)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v]
        o_pt = torch.sum(
            (
                a[..., None, :, :, None]
                * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt_out_ti = o_pt

        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(
            o_pt_dists, 2)
        o_pt_dists_out_ti = torch.sqrt(torch.sum(o_pt_out_ti ** 2, dim=-1) + self.eps)
        o_pt_norm_feats_out_ti = flatten_final_dims(o_pt_dists_out_ti, 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)
        o_pt_out_ti = o_pt_out_ti.reshape(*o_pt_out_ti.shape[:-3], -1, 3)

        if(_offload_inference):
            z[0] = z[0].to(o_pt.device)

        # [*, N_res, H, C_z // 4]
        pair_z = self.down_z(z[0]).to(dtype=a.dtype)
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)

        # [*, N_res, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair, *torch.unbind(o_pt_out_ti, dim=-1), o_pt_norm_feats_out_ti]
        # tmp = torch.cat(o_feats, dim=-1).to(dtype=z[0].dtype) # torch.Size([1+Frame, 135, 1344])

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat(
                o_feats, dim=-1
            ).to(dtype=z[0].dtype)
        )

        return s


class AF2InvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        inf: float = 1e5,
        eps: float = 1e-8,
        init="final",
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(AF2InvariantPointAttention, self).__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        self.linear_b = Linear(self.c_z, self.no_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim =  (
            self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init=init)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        r: Rigid,
        mask: torch.Tensor,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])

        if(_offload_inference):
            z[0] = z[0].cpu()

        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        # [*, N_res, N_res, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        # print(pt_displacement.shape[:])

        pt_att = pt_displacement ** 2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        a = a + pt_att

        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)
        # torch.save(a.cpu(),'a.pth')
        # exit()

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3).to(dtype=a.dtype)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v]
        o_pt = torch.sum(
            (
                a[..., None, :, :, None]
                * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))

        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(
            o_pt_dists, 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if(_offload_inference):
            z[0] = z[0].to(o_pt.device)

        # [*, N_res, H, C_z // 4]
        pair_z = self.down_z(z[0]).to(dtype=a.dtype)
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)

        # [*, N_res, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]
        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat(
                o_feats, dim=-1
            ).to(dtype=z[0].dtype)
        )

        return s


class IPA_MLP(nn.Module):
    def __init__(self, input_dim,c):
        super(IPA_MLP, self).__init__()

        self.c = c
        self.input_dim=input_dim
        self.linear_1 = Linear(self.input_dim, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="final")
        self.relu = nn.ReLU()

    def forward(self, s):
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)

        return s

class TorsionAngles(nn.Module):
    def __init__(self, c, num_torsions, eps=1e-8):
        super(TorsionAngles, self).__init__()

        self.c = c
        self.eps = eps
        self.num_torsions = num_torsions

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        # TODO: Remove after published checkpoint is updated without these weights.
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.linear_final = Linear(
            self.c, self.num_torsions * 2, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)

        s = s + s_initial
        unnormalized_s = self.linear_final(s)
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(unnormalized_s ** 2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        normalized_s = unnormalized_s / norm_denom

        return unnormalized_s, normalized_s


class AdaptiveLayerNorm(nn.Module):

    def __init__(
        self,
        *,
        dim,
        dim_cond
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm_cond = nn.LayerNorm(dim_cond)

        self.to_gamma = nn.Sequential(
            nn.Linear(dim_cond, dim),
            nn.Sigmoid()
        )

        self.to_beta = nn.Linear(dim_cond, dim,bias=False)

    def forward(self,x,cond):
        # x ['b n d']
        # cond['b n dc']
        normed = self.norm(x)
        normed_cond = self.norm_cond(cond)

        gamma = self.to_gamma(normed_cond)
        beta = self.to_beta(normed_cond)
        return normed * gamma + beta

class ScoreLayer(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out):
        super(ScoreLayer, self).__init__()

        self.linear_1 = Linear(dim_in, dim_hid, init="relu")
        self.linear_2 = Linear(dim_hid, dim_hid)
        self.linear_3 = Linear(dim_hid, dim_out, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = s + s_initial
        s = self.linear_3(s)
        return s


class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s

        self.linear = Linear(self.c_s, 6, init="final")

    def forward(self, s: torch.Tensor):
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector
        """
        # [*, 6]
        update = self.linear(s)

        return update


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=40):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)# MAX_L,1
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        # print(pe.shape,#torch.Size([1, 40, 33])
        #       pe[0, :, 0::2].shape,#torch.Size([40, 17])
        #       pe[0, :, 1::2].shape,#torch.Size([40, 16])
        #       position.shape,# torch.Size([40, 1])
        #       div_term.shape#torch.Size([17]
        #       )
        # len_sin = pe[0, :, 0::2].shape[-1]
        # len_cos = pe[0, :, 1::2].shape[-1]

        pe[0, :, 0::2] = torch.sin(position * div_term)#[:len_sin]
        pe[0, :, 1::2] = torch.cos(position * div_term)#[:len_cos]

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class DyneTrionSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, out_dim=None):
        assert dim % num_heads == 0
        super().__init__()
        self.out_dim = out_dim
        if out_dim is None:
            self.out_dim = dim

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, self.out_dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads * C / num_heads]
            seq_lens(Tensor): Shape [B]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        # if seq_lens.shape[0] != b:
        #     seq_lens = seq_lens.expand(b)
        assert seq_lens.shape[0] == b
        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v
        q, k, v = qkv_fn(x)
        x = attention(
            q=rope_apply_1d(q, freqs),
            k=rope_apply_1d(k, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
        )
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

class DyneTrionAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        node_dim,
        skip_dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        use_temporal=True,
        use_causal_temporal=False,
        local_attn_size=-1,
        sink_size=0,
        sapatial_out_dim=None
    ):
        super().__init__()
        self.dim = dim
        self.skip_dim = skip_dim
        self.node_dim = node_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.use_temporal = use_temporal
        self.use_causal_temporal = use_causal_temporal

        # spatial
        self.spatial_norm = nn.LayerNorm(2 * (dim + skip_dim), eps=eps)
        # , elementwise_affine=False
        self.spatial_attention = DyneTrionSelfAttention(
            dim=2 * (dim + skip_dim),  # times 2 as the ref is concated
            num_heads=num_heads,
            window_size=window_size,
            qk_norm=qk_norm,
            eps=eps,
            out_dim=sapatial_out_dim,
        )
        self.spatial_skip_embed = Linear(node_dim, skip_dim, init="final")
        self.spatial_post_embed = Linear(dim + skip_dim, dim, init="relu")

        # temporal
        if use_temporal and use_causal_temporal:
            raise ValueError("use_temporal and use_causal_temporal are mutually exclusive")

        if use_temporal or use_causal_temporal:
            self.temporal_skip_embed = Linear(node_dim, skip_dim, init="final")
            self.temporal_norm = nn.LayerNorm(dim + skip_dim, eps=eps)
            self.temporal_post_embed = Linear(dim + skip_dim, dim, init="relu")
            if use_temporal:
                self.temporal_attention = DyneTrionSelfAttention(
                    dim=dim + skip_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    qk_norm=qk_norm,
                    eps=eps,
                )
            else:
                self.temporal_attention = CausalDyneTrionSelfAttention(
                    dim=dim + skip_dim,
                    num_heads=num_heads,
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                    qk_norm=qk_norm,
                    eps=eps,
                )
                # raise NotImplementedError("Causal temporal attention is not implemented yet")

        # ffn
        self.ffn_norm = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

    def forward(
        self,
        ref_node,
        motion_node,
        frame_node,
        ref_node_init,
        motion_node_init,
        frame_node_init,
        ref_number,
        motion_number,
        # seq_lens,
        spatial_freqs,
        temporal_freqs,
        is_training=True, #TODO
        block_mask=None
    ):
        spatial_node_with_ref = torch.cat([ref_node, frame_node], dim=1)
        spatial_node_init_with_ref = torch.cat([ref_node_init, frame_node_init], dim=1)
        frame_num, res_num, _ = frame_node.shape[1:]
        seq_tfmr_in = torch.cat(
            [
                spatial_node_with_ref,
                self.spatial_skip_embed(spatial_node_init_with_ref),
            ],
            dim=-1,
        )
        x = torch.cat(
            (seq_tfmr_in[:,1:],
             repeat(seq_tfmr_in[:,:1],"b 1 n d -> b t n d", t=frame_num)
             ), dim=-1
        )
        x = rearrange(x, "b t n d -> (b t) n d")
        seq_lens = torch.tensor([u.size(0) for u in x], dtype=torch.long)
        y = self.spatial_attention(self.spatial_norm(x), seq_lens, spatial_freqs)
        x = x + y
        spatial_out = x[..., : (self.dim + self.skip_dim)] #NOTE ???
        spatial_out = rearrange(
            spatial_out, "(b t) n d-> b t n d", t=frame_num
        )
        # spatial_out = spatial_out[:, ref_number:]
        frame_node = frame_node + self.spatial_post_embed(spatial_out)

        if self.use_temporal or self.use_causal_temporal:
            temporal_node = torch.concat([motion_node, frame_node], dim=1)
            temporal_node_init_with_motion = torch.cat([motion_node_init,frame_node_init],dim=1)

            x = torch.cat(
                [
                    temporal_node,
                    self.temporal_skip_embed(temporal_node_init_with_motion),
                ],
                dim=-1,
            ) #B, T,N,D
            x = rearrange(x, 'b f n d -> (b n) f d')
            seq_lens = torch.tensor([u.size(0) for u in x], dtype=torch.long)
            if self.use_temporal:
                y = self.temporal_attention(self.temporal_norm(x), seq_lens, temporal_freqs)
            else:
                y = self.temporal_attention(self.temporal_norm(x), seq_lens, temporal_freqs, block_mask=block_mask)
            x = x + y
            temporal_out = rearrange(x, "(b n) f d -> b f n d", n=res_num)
            temporal_node = temporal_node + self.temporal_post_embed(temporal_out)
            frame_node = temporal_node[:, motion_number:]

        y =  self.ffn(self.ffn_norm(frame_node))
        frame_node = frame_node + y
        return frame_node

    def forward_on_spatial(
        self,
        ref_node,
        frame_node,
        ref_node_init,
        frame_node_init,
        # seq_lens,
        spatial_freqs,

    ):
        stacked_nodes = torch.cat([frame_node, ref_node], dim=0)
        stacked_init_nodes = torch.cat([frame_node_init, ref_node_init], dim=0)
        embedded_nodes = torch.cat(
            [
                stacked_nodes,
                self.spatial_skip_embed(stacked_init_nodes),
            ],
            dim=-1,
        )
        frame_feat, ref_feat = torch.chunk(embedded_nodes, chunks=2, dim=0)
        fusion_input = torch.cat([frame_feat, ref_feat], dim=-1)
        B, N, _ = frame_feat.shape
        seq_lens = torch.full((B,), N, dtype=torch.long, device=fusion_input.device)

        attn_residual = self.spatial_attention(
            self.spatial_norm(fusion_input), seq_lens, spatial_freqs
        )
        attn_out_frame = frame_feat + attn_residual
        frame_node = frame_node + self.spatial_post_embed(attn_out_frame)
        ffn_residual = self.ffn(self.ffn_norm(frame_node))
        out_frame_node = frame_node + ffn_residual

        return out_frame_node


class ScoreModel(nn.Module):

    def __init__(self, model_conf, diffuser):
        super(ScoreModel, self).__init__()
        self._model_conf = model_conf
        ipa_conf = model_conf.ipa
        self._ipa_conf = ipa_conf
        self.diffuser = diffuser

        self.scale_pos = lambda x: x * ipa_conf.coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / ipa_conf.coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)

        self.frame_time = self._model_conf.frame_time
        self.motion_number = self._model_conf.motion_number
        self.ref_number = self._model_conf.ref_number
        self.total_time = self.frame_time + self.motion_number + self.ref_number
        self.block_mask = None
        self.trunk = nn.ModuleDict()
        self.register_buffer(
            "temporal_causal_mask",
            nn.Transformer.generate_square_subsequent_mask(
                self.motion_number + self.frame_time
            ),
            persistent=False,
        )
        self.register_buffer(
            "spatial_freqs",
            rope_params(
                max_seq_len=self._model_conf.max_rope_len,
                dim=2
                * (self._model_conf.node_embed_size + self._model_conf.skip_dim)
                // self._model_conf.attention_num_heads,
            ),
        )
        self.register_buffer(
            "temporal_freqs",
            rope_params(
                max_seq_len=self._model_conf.max_rope_len,
                dim=(self._model_conf.node_embed_size + self._model_conf.skip_dim)
                // self._model_conf.attention_num_heads,
            ),
        )
        self.local_attn_size = model_conf.local_attn_size
        self.num_frame_per_block = model_conf.num_frame_per_block
        for b in range(int(ipa_conf.num_blocks)):
            self.trunk[f'ipa_{b}'] = InvariantPointAttention(ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(ipa_conf.c_s)
            self.trunk[f"DyneTrionAttn_{b}"] = DyneTrionAttentionBlock(
                dim=self._model_conf.node_embed_size,
                node_dim=self._model_conf.node_embed_size,
                skip_dim=self._model_conf.skip_dim,
                ffn_dim=4 * self._model_conf.node_embed_size,
                num_heads=self._model_conf.attention_num_heads,
                use_causal_temporal=model_conf.use_causal_temporal,
                use_temporal=model_conf.use_temporal,
                local_attn_size=model_conf.local_attn_size,
                sink_size=model_conf.sink_size,
            )
            self.trunk[f'node_transition_{b}'] = StructureModuleTransition(c=ipa_conf.c_s)
            # module for rigids update and egde update
            self.trunk[f'bb_update_{b}'] = BackboneUpdate(ipa_conf.c_s)

            # if b < ipa_conf.num_blocks - 1:  #
            #     # self.trunk[f'edge_transition_{b}'] = EdgeTransition(
            #     #     node_embed_size=ipa_conf.c_s,
            #     #     edge_embed_in=edge_in,
            #     #     edge_embed_out=self._model_conf.edge_embed_size,
            #     # )
            #     self.trunk[f"edge_transition_{b}"] = EdgeTransitionOutProductMean(
            #         c_node=self._model_conf.node_embed_size,
            #         c_edge=self._model_conf.edge_embed_size,
            #         c_hidden=self._model_conf.egde_hidden_dim,
            #     )
        # module for angle prediction
        self.angle_resnet = AngleResnet(
            c_in=ipa_conf.c_s,
            c_hidden=ipa_conf.c_s,
            no_blocks=2,
            no_angles=7,
            epsilon=1e-12,
        )


    def forward(self, node_embed, edge_embed, input_feats, is_training=True, drop_ref=False):  #
        '''
        init_node_embed: [F,N,D] node features from embeder with diffuser t embeddings
        edge_embed: edge features from embeder without diffuser t embeddings, since we update egde features with node features
        input_feats: input infomation from dataloader
        '''
        # # position embedding add in the beginning
        # initialize the input tensors
        # torch.cuda.reset_peak_memory_stats()
        # gpu_mem("start forward")

        diffuser_time_t = input_feats['t']

        node_mask = input_feats['res_mask'].type(torch.float32)[:, :1] # [B, 1, Res]
        diffuse_mask = (1 - input_feats['fixed_mask'][:,:1].type(torch.float32)) * node_mask  # [B, 1, Res]
        edge_mask = node_mask[..., None] * node_mask[..., None, :]  # [B, 1, Res, Res]
        # gpu_mem("after input prep")
        init_frames = input_feats['rigids_t'].type(torch.float32)
        curr_rigids = Rigid.from_tensor_7(torch.clone(init_frames))
        init_rigids = Rigid.from_tensor_7(init_frames)

        curr_rigids = self.scale_rigids(curr_rigids) # trans with muliply with 0.1 scale from unit ans to unit nano meter

        # processing ref and motion rigids
        ref_rigids = self.scale_rigids(Rigid.from_tensor_7(input_feats['ref_rigids_0']))
        motion_rigids = self.scale_rigids(Rigid.from_tensor_7(input_feats['motion_rigids_0']))
        # gpu_mem("after rigid init")
        # processing ref and motion node
        ref_node = input_feats["ref_node_repr"] * node_mask[..., None]
        ref_edge = input_feats["ref_edge_repr"] * edge_mask[..., None]
        # motion node and egde is same as the reference since no time embedings are introduced
        motion_node = repeat(ref_node, "b 1 n d -> b t n d", t=self.motion_number)
        motion_edge = repeat(ref_edge, "b 1 i j d -> b t i j d", t=self.motion_number)

        frame_edge = edge_embed
        frame_node = node_embed
        # define init node features
        frame_node_init = frame_node
        ref_node_init = ref_node
        motion_node_init = motion_node
        # gpu_mem("after node/edge init")

        # initial causal temporal mask
        if self._model_conf.use_causal_temporal and (self.block_mask is None): #and is_training:
            self.block_mask = prepare_blockwise_causal_attn_mask(
                device=frame_node.device,
                num_frames=self.motion_number + self.frame_time,
                frame_seqlen=1,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size,
            )

        # loop of diffuser block
        for b in range(self._ipa_conf.num_blocks):
            # run referencenet, spatial alignment
            # gpu_mem(f"block {b} start")
            all_curr_rigids = torch.cat(
                [
                    ref_rigids.to_tensor_7(),
                    motion_rigids.to_tensor_7(),
                    curr_rigids.to_tensor_7(),
                ],
                dim=1,
            )
            all_node = torch.concat([ref_node, motion_node, frame_node], dim=1)
            all_node_mask = repeat(node_mask, "b 1 n -> b t n", t=self.total_time)
            all_node = all_node * all_node_mask[..., None]
            all_edge_embed = (
                torch.concat([ref_edge, motion_edge, edge_embed], dim=1)
                * edge_mask[..., None]
            )
            all_curr_rigids = Rigid.from_tensor_7(all_curr_rigids)
            all_ipa_embed = self.trunk[f'ipa_{b}'](
                    all_node,
                    all_edge_embed,
                    all_curr_rigids,
                    all_node_mask)
            # gpu_mem(f"block {b} after ipa")
            all_node = self.trunk[f'ipa_ln_{b}'](all_node + all_ipa_embed)
            ref_node, motion_node, frame_node = torch.split(
                all_node, [self.ref_number, self.motion_number, self.frame_time], dim=1
            )
            # seq_lens = torch.tensor([u.size(0) for u in ref_node], dtype=torch.long)
            frame_node = self.trunk[f"DyneTrionAttn_{b}"](
                ref_node,
                motion_node,
                frame_node,
                ref_node_init,
                motion_node_init,
                frame_node_init,
                # seq_lens=seq_lens,
                spatial_freqs=self.spatial_freqs,
                temporal_freqs=self.temporal_freqs,
                ref_number=self.ref_number,
                motion_number=self.motion_number,
                is_training=is_training,
                block_mask=self.block_mask,
            )
            # gpu_mem(f"block {b} after attention")
            # apply MLP for node features
            all_node = self.trunk[f"node_transition_{b}"](
                torch.cat([ref_node, motion_node, frame_node], dim=1)
            )
            # gpu_mem(f"block {b} after node_transition")
            ref_node, motion_node, frame_node = torch.split(
                all_node, [self.ref_number, self.motion_number, self.frame_time], dim=1
            )
            rigid_update = self.trunk[f'bb_update_{b}'](frame_node * diffuse_mask[..., None])

            curr_rigids = curr_rigids.compose_q_update_vec(rigid_update, diffuse_mask[..., None])

            # # update edges with edge features
            # if b < self._ipa_conf.num_blocks - 1:
            #     # gpu_mem(f"block {b} before edge_transition")
            #     # all_edge = self.trunk[f'edge_transition_{b}'](all_node,torch.cat([motion_edge,ref_edge,frame_edge],dim=0))
            #     all_edge = self.trunk[f"edge_transition_{b}"](
            #         all_node,
            #         torch.cat([ref_edge, motion_edge, frame_edge], dim=0),
            #         chunk_size=self._model_conf.egde_chunk_size,
            #     )
            #     # gpu_mem(f"block {b} after edge_transition")
            #     edge_embed *= edge_mask[..., None]
            #      # update ref edge
            #     ref_edge, motion_edge,frame_edge = torch.split(all_edge, [self.ref_number, self.motion_number, self.frame_time],dim=0)

        # rot_score = self.diffuser.calc_rot_score(
        #     init_rigids.get_rots(),
        #     curr_rigids.get_rots(),
        #     diffuser_time_t
        # )
        fn = lambda t: rearrange(t, "b f ... -> (b f) ...")
        time_expanded = repeat(diffuser_time_t, "b 1 -> b f", f=self.frame_time)
        rot_score = self.diffuser.calc_rot_score(
            init_rigids.get_rots().map_tensor_fn(fn),
            curr_rigids.get_rots().map_tensor_fn(fn),
            fn(time_expanded)  # 这里使用了 fn
        )
        rot_score = rearrange(rot_score, "(b f) n d -> b f n d", f=self.frame_time)
        rot_score = rot_score * node_mask[..., None]
        curr_rigids = self.unscale_rigids(curr_rigids) # scale to angstrom
        trans_score = self.diffuser.calc_trans_score(
            init_rigids.get_trans(),
            curr_rigids.get_trans(),
            diffuser_time_t[:, None, None],
            use_torch=True,
        )
        trans_score = trans_score * node_mask[..., None]
        # directly predict the angles for side chains with the last layer  node features
        # gpu_mem("after output heads")
        unorm_angles, angles = self.angle_resnet(frame_node, frame_node_init)
        # merge the outputs
        model_out = {
            'angles': angles,
            'unorm_angles':unorm_angles,
            'rot_score': rot_score,
            'trans_score': trans_score,
            'final_rigids': curr_rigids,
            'rigid_update':rigid_update,
            'frame_node': frame_node,
        }
        return model_out
