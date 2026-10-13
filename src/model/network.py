"""Score network module."""
import torch
import math
from torch import nn
from torch.nn import functional as F
from openfold.utils import feats
from src.data import utils as du
from src.data import all_atom
from src.model import modules
import functools as fn
from openfold.utils.tensor_utils import batched_gather
from typing import Dict, Text, Tuple
from einops import rearrange, repeat
import torch
from src.model.utils import get_timestep_embedding
from openfold.np import residue_constants as rc
Tensor = torch.Tensor

class SequenceEmbedder(nn.Module):

    def __init__(self, model_conf):
        super(SequenceEmbedder, self).__init__()
        self._model_conf = model_conf
        self._embed_conf = model_conf.embed
        # Time step embedding
        diffuser_time_embed_size = self._embed_conf.index_embed_size
        node_embed_size = self._model_conf.node_embed_size
        edge_embed_size = self._model_conf.edge_embed_size
        time_embed_size = node_embed_size
        self.timestep_embed = fn.partial(
            get_timestep_embedding,
            embedding_dim=time_embed_size,
        )

        self.node_timestep_proj= nn.Sequential(
                    nn.Linear(time_embed_size,node_embed_size//2),
                    nn.SiLU(),
                    nn.Linear(node_embed_size//2,node_embed_size),
                )
        self.node_ln = nn.LayerNorm(node_embed_size)


        self.edge_timestep_proj = nn.Sequential(
                    nn.Linear(time_embed_size,edge_embed_size//2),
                    nn.SiLU(),
                    nn.Linear(edge_embed_size//2,edge_embed_size),
        )
        self.edge_ln = nn.LayerNorm(edge_embed_size)


    def forward(self,node_repr,edge_repr,seq_idx,t):
        """Embeds a
        Args:
            node_repr: [B, Frame, Res, D_node] node features from FOLD model like GeoForm(from OmegaFold)
            edge_repr: [B, Frame, Res, Res, D_edge] edge features from FOLD model like GeoForm(from OmegaFold)
            seq_idx: [..., N] Positional sequence index for each residue.
            t: Sampled t in [0, 1].

        Returns:
            node_embed: [B, N, D_node]
            edge_embed: [B, N, N, D_edge]
        """
        num_batch, num_frames, num_res = seq_idx.shape
        t_embed = self.timestep_embed(t)
        # processing node
        node_t_step_embedings = self.node_timestep_proj(t_embed)
        node_t_step_embedings = torch.tile(
            node_t_step_embedings[:, :, None, :], (1, num_frames, num_res, 1)
        )  # (B,Frame, Res,32)

        node_embed = self.node_ln(node_repr) #
        ref_node_repr = node_embed[:, :1]  # self.ln(node_repr[0])
        node_embed = node_embed + node_t_step_embedings

        # processing edge
        # edge_embed = edge_repr.reshape(num_batch, num_res * num_res, -1)
        edge_t_step_embedings = self.edge_timestep_proj(t_embed)
        edge_t_step_embedings = torch.tile(
            edge_t_step_embedings[:, :, None, None, :],
            (1, num_frames, num_res, num_res, 1),
        )  # (B,Frame, Res*Res,32)

        edge_embed = self.edge_ln(edge_repr)
        ref_edge_repr = edge_embed[:, :1]
        edge_embed = edge_embed + edge_t_step_embedings

        return node_embed, edge_embed ,ref_node_repr,ref_edge_repr,t_embed

class DyneTrion(nn.Module):

    def __init__(self, model_conf, diffuser):
        super(DyneTrion, self).__init__()
        self._model_conf = model_conf
        self.embedding_layer = SequenceEmbedder(model_conf)

        self.diffuser = diffuser
        self.score_model = modules.ScoreModel(model_conf, diffuser)
        self.expand_node = nn.Linear(model_conf.node_input_embed_size, model_conf.node_embed_size)
        self.expand_edge = nn.Linear(128, model_conf.edge_embed_size)
        # cache for inference

        # self.node_repr = None
        # self.edge_repr = None

    def _apply_mask(self, aatype_diff, aatype_0, diff_mask):
        return diff_mask * aatype_diff + (1 - diff_mask) * aatype_0

    def _process_repr(self,input_feats):

        num_res = input_feats['node_repr'].shape[1]  # [B, Res, N_dim]

        # print(input_feats['node_repr'].shape[:], input_feats['edge_repr'].shape[:])
        input_feats['expand_node_repr'] = self.expand_node(input_feats['node_repr'])
        egde_repr = self.expand_edge(rearrange(input_feats["edge_repr"], "b i j d -> b (i j) d"))
        input_feats["expand_edge_repr"] = rearrange(egde_repr, "b (i j) d -> b i j d", i=num_res)
        # Initial embeddings of positonal and relative indices.
        # TODO here we change just give one node and repeat after proj
        init_node_embed, init_edge_embed, ref_node_repr, ref_edge_repr, t_embed = self.embedding_layer(
            node_repr=repeat(input_feats['expand_node_repr'], 'b n d -> b t n d', t=1),
            edge_repr=repeat(input_feats['expand_edge_repr'], 'b i j d -> b t i j d', t=1),#
            seq_idx=input_feats['seq_idx'][:, :1],
            t=input_feats['t'],
        )
        init_node_embed = repeat(
            init_node_embed, "b 1 n d -> b t n d", t=self._model_conf.frame_time
        )
        init_edge_embed = repeat(
            init_edge_embed, "b 1 i j d -> b t i j d", t=self._model_conf.frame_time
        )
        return init_node_embed, init_edge_embed, ref_node_repr, ref_edge_repr, t_embed


    def forward(self, input_feats, drop_ref=False, is_training=True, return_trunk_features=False):
        """Forward computes the reverse diffusion conditionals p(X^t|X^{t+1})
        for each item in the batch

        Args:
            X: the noised samples from the noising process, of shape [Batch, N, D].
                Where the T time steps are t=1,...,T (i.e. not including the un-noised X^0)

        Returns:
            model_out: dictionary of model outputs.
        """
        fixed_mask = input_feats["fixed_mask"].type(torch.float32)  # [B, Frame, Res]
        # edge_mask = bb_mask[..., None] * bb_mask[..., None, :]

        init_node_embed, init_edge_embed, ref_node_repr, ref_edge_repr, t_embed = (
            self._process_repr(input_feats)
        )
        input_feats.update(
            {
                "ref_node_repr": ref_node_repr,
                "ref_edge_repr": ref_edge_repr,
                "t_embed": t_embed,
            }
        )
        # Run main network
        model_out = self.score_model(init_node_embed, init_edge_embed, input_feats,drop_ref=drop_ref,is_training=is_training)
        gt_angles = input_feats['torsion_angles_sin_cos'] #in SE3 [..., 2, :] # only angle psi is used，here use the all 7 angles
        angles_pred = self._apply_mask(model_out['angles'], gt_angles, 1 - fixed_mask[..., None, None] ) # could del since fixed_masks always equal to  1
        unorm_angles = self._apply_mask(model_out['unorm_angles'], gt_angles, 1 - fixed_mask[..., None, None]) # could del since fixed_masks always equal to  1
        pred_out = {
            'angles': angles_pred,
            'unorm_angles': unorm_angles,
            'rot_score': model_out['rot_score'],
            'trans_score': model_out['trans_score'],
        }
        rigids_pred = model_out['final_rigids']


        pred_out['rigids'] = rigids_pred.to_tensor_7()
        if return_trunk_features:
            pred_out["trunk_frame_node"] = model_out["frame_node"]
            pred_out["t_embed"] = input_feats["t_embed"]
        if self._model_conf.return_atom37:
            # convert rigids and angles to frames
            all_frames = feats.torsion_angles_to_frames(rigids_pred,angles_pred,
                                                        input_feats['aatype'],
                                                        all_atom.DEFAULT_FRAMES.to(angles_pred.device))
            # convert frame to atom14 with idealize amino acid
            atom14_pos = all_atom.frames_to_atom14_pos(all_frames,input_feats['aatype'])
            # change atom14 to atom37
            atom37_pos, atom37_mask = atom14_to_atom37(
                atom14_pos, input_feats["aatype"]
            )
            pred_out['atom37'] = atom37_pos.to(rigids_pred.device)
        return pred_out

    def biased_sample(self, src, tgt, bias_fn, beta=0.1):
        """
        bias_fn: 传入 actor.get_bias_score
        beta: 偏置力权重
        """
        x = src.clone()
        log_probs = []

        # 模拟去噪循环 (此处简化为单步，实际应循环 T 步)
        for t in range(self.T):
            # 1. 计算原始得分
            score = self.forward(x, tgt)

            # 2. 获取实时偏置力
            bias_score = bias_fn(x)

            # 3. 总合力驱动
            total_score = score + beta * bias_score

            # 4. 更新状态并记录 log_prob (用于 RL 更新)
            x_next = x + total_score + torch.randn_like(x) * 0.01
            log_p = -0.5 * torch.sum((x_next - x - total_score) ** 2)  # 简化概率密度

            x = x_next
            log_probs.append(log_p)

        return x, torch.stack(log_probs).mean()


def get_rc_tensor(rc_np, aatype):
    return torch.tensor(rc_np, device=aatype.device)[aatype]

def atom14_to_atom37(
    atom14_data: torch.Tensor,  # (*, N, 14, ...)
    aatype: torch.Tensor # (*, N)
) -> Tuple:    # (*, N, 37, ...)
    """Convert atom14 to atom37 representation."""
    idx_atom37_to_atom14 = get_rc_tensor(rc.RESTYPE_ATOM37_TO_ATOM14, aatype).long()
    no_batch_dims = len(aatype.shape) - 1
    atom37_data = batched_gather(
        atom14_data,
        idx_atom37_to_atom14,
        dim=no_batch_dims + 1,
        no_batch_dims=no_batch_dims + 1
    )
    atom37_mask = get_rc_tensor(rc.RESTYPE_ATOM37_MASK, aatype)
    if len(atom14_data.shape) == no_batch_dims + 2:
        atom37_data *= atom37_mask
    elif len(atom14_data.shape) == no_batch_dims + 3:
        atom37_data *= atom37_mask[..., None].to(dtype=atom37_data.dtype)
    else:
        raise ValueError("Incorrectly shaped data")
    return atom37_data, atom37_mask
