import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import math
from src.model.modules import (
    AF2InvariantPointAttention,
    StructureModuleTransition,
    Linear, BackboneUpdate, DyneTrionAttentionBlock
)
from openfold.model.structure_module import AngleResnet
from typing import Optional, List, Union
from openfold.utils.rigid_utils import Rigid
from einops import rearrange, repeat
from src.model.causal_modules import CausalDyneTrionSelfAttention
from src.model.utils import rope_params
from icecream import ic
from src.model.utils import get_timestep_embedding
import functools as fn


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear layers."""

    def __init__(self, orig_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.orig_linear = orig_linear
        self.rank = rank
        self.scaling = alpha / rank
        in_features = orig_linear.in_features
        out_features = orig_linear.out_features
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * (1.0 / rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.orig_linear(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out

# class SinusoidalPosEmb(nn.Module):
#     """标准的正弦时间编码"""

#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, x):
#         # x: [B, 1]
#         device = x.device
#         half_dim = self.dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
#         emb = x * emb[None, :]
#         emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
#         return emb


# class TrajectoryMemoryAttention(nn.Module):
#     """
#     [Context as Memory] 核心模块
#     利用 Cross-Attention 将当前构象特征(Query)与历史轨迹特征(Key/Value)进行交互。
#     物理意义：感知"我是否回到了之前去过的地方"。
#     """

#     def __init__(self, c_s, num_heads=4):
#         super().__init__()
#         self.c_s = c_s
#         self.context_attn = CausalDyneTrionSelfAttention(
#             dim=c_s,
#             num_heads=num_heads,
#         )
#         self.context_norm = nn.LayerNorm(c_s, eps=1e-6)
#         self.layer_norm = nn.LayerNorm(c_s)
#         self.gate = nn.Sequential(Linear(c_s * 2, c_s, init="gating"), nn.Sigmoid())


#     def forward(self, s_curr, memory_cache, freqs, block_mask):
#         """
#         Args:
#             s_curr: [B, F, N, C] 当前帧的几何特征 (Query)
#             memory_cache: [B, N_extra, F, N, C] 历史 K 帧的特征 (Key/Value)
#             freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
#             block_mask (BlockMask): chunk attention mask
#         Returns:
#             s_enhanced: [B, N, C] 注入历史信息的特征
#         """
#         if memory_cache is None:
#             print("skipping! context")
#             return s_curr

#         B, F, N, C = s_curr.shape
#         # memory_cache 是 List[Tensor]
#         if isinstance(memory_cache, list):
#             if len(memory_cache) == 0:
#                 return s_curr
#             # Stack: [B, E, F, N, C]
#             hist_tensor = torch.stack(memory_cache, dim=1)
#             # [B, E*F, N, C]
#             hist_tensor = rearrange(hist_tensor, "b e f n c -> b (e f) n c")
#         else:
#             hist_tensor = memory_cache
#         # [B, E*F+F, N, C]
#         combined = torch.cat([hist_tensor, s_curr], dim=1)
#         total_frames = combined.shape[1]

#         x = rearrange(combined, "b f n d -> (b n) f d")
#         seq_lens = torch.full((x.shape[0],), total_frames, dtype=torch.long, device=x.device)
#         y = self.context_attn(
#             self.context_norm(x), seq_lens, freqs, block_mask=block_mask
#         )
#         y = rearrange(y, "(b n) f d -> b f n d", b=B, n=N)
#         y_curr = y[:, -F:]
#         gate_input = torch.cat([s_curr, y_curr], dim=-1)
#         g = self.gate(gate_input)
#         s_enhanced = s_curr + g * y_curr

#         s_enhanced = self.layer_norm(s_enhanced)

#         # if mask is not None:
#         #     s_enhanced = s_enhanced * mask.unsqueeze(-1)

#         return s_enhanced

# class EnergyHead(nn.Module):
#     """
#     SE(3)-invariant Energy Head.
#     Predicts a scalar potential from node-level scalar features.
#     Uses Max and Mean pooling concatenation to capture both local extreme interactions
#     and global conformational states.
#     Uses SiLU for smooth derivatives (forces) to ensure physical validity.
#     """

#     def __init__(self, c_s, hidden_dim=256):
#         super().__init__()
#         # 拼接 Max 和 Mean 特征，因此输入维度扩展为 c_s * 2
#         self.mlp = nn.Sequential(
#             nn.Linear(c_s * 2, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, 1),  # Bare logits: outputs range (-inf, inf)
#         )

#     def forward(self, s, mask):
#         """
#         Args:
#             s: [B, F, N, C_s] Node-level scalar features (SE(3) invariant)
#             mask: [B, F, N] Residue mask
#         Returns:
#             potential: [B, F, 1] Scalar potential
#         """
#         # 扩展 mask 维度以匹配特征维度 [B, F, N, 1]
#         m = mask.unsqueeze(-1)

#         # 1. 均值池化 (Mean Pooling): 捕获全局构象状态
#         masked_feat_sum = (s * m).sum(dim=-2)
#         num_residues = m.sum(dim=-2).clamp(min=1e-6)
#         mean_feat = masked_feat_sum / num_residues  # [B, F, C_s]

#         # 2. 最大池化 (Max Pooling): 捕获局部极值信号（如空间位阻）
#         # 必须使用 masked_fill 将 mask 为 0 的 padding 区域填充为极小值，
#         # 防止 0 成为某些全负特征通道的最大值。
#         masked_s = s.masked_fill(m == 0, -1e9)
#         max_feat, _ = masked_s.max(dim=-2)  # [B, F, C_s]

#         # 3. 特征拼接: [B, F, C_s * 2]
#         global_feat = torch.cat([max_feat, mean_feat], dim=-1)

#         # 能量映射
#         return self.mlp(global_feat)

class TimeEmbeder(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super(TimeEmbeder, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.timestep_embed = fn.partial(
            get_timestep_embedding,
            embedding_dim=node_dim,
        )
        self.node_timestep_proj = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.node_ln = nn.LayerNorm(node_dim)

        self.edge_timestep_proj = nn.Sequential(
            nn.Linear(node_dim, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
        )
        self.edge_ln = nn.LayerNorm(edge_dim)

    def forward(self, node_repr, edge_repr, t):
        # 1. 计算基础时间嵌入

        t_embed = self.timestep_embed(t)

        # 2. 处理节点表征 (利用 Broadcasting 替代 Tile)
        node_t_step_embedings = self.node_timestep_proj(t_embed)
        # 将形状从 [B, node_dim] 调整为 [B, 1, node_dim]
        node_t_step_embedings = node_t_step_embedings.unsqueeze(1)

        node_embed = self.node_ln(node_repr)
        # 张量相加时，[B, 1, node_dim] 会自动广播以匹配 [B, N, node_dim]
        node_embed = node_embed + node_t_step_embedings

        # 3. 处理边表征 (利用 Broadcasting 替代 Tile)
        edge_t_step_embedings = self.edge_timestep_proj(t_embed)
        # 将形状从 [B, edge_dim] 调整为 [B, 1, 1, edge_dim]
        edge_t_step_embedings = edge_t_step_embedings.unsqueeze(1).unsqueeze(2)

        edge_embed = self.edge_ln(edge_repr)
        # 张量相加时，[B, 1, 1, edge_dim] 会自动广播以匹配 [B, N, N, edge_dim]
        edge_embed = edge_embed + edge_t_step_embedings

        return node_embed, edge_embed, t_embed


class MetadynamicsActor(nn.Module):
    def __init__(
        self,
        c_s=128,
        c_z=64,
        c_hidden=128,
        no_heads=4,
        no_qk_points=4,
        no_v_points=4,
        num_ipa_blocks=2,
        max_rope_len=2048,
        skip_dim=128,
        max_energy=20,
        coordinate_scaling=0.1
    ):
        super().__init__()
        # 1. Embedders
        self.time_embedder = TimeEmbeder(node_dim=c_s, edge_dim=c_z)

        # 2. RoPE Freqs (用于 Memory Attention)
        # 确保与数据配置对齐
        self.register_buffer(
            "context_freqs",
            rope_params(
                max_seq_len=max_rope_len,
                dim=c_s // no_heads, # // 2,
            ),
        )
        self.scale_pos = lambda x: x * coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)
        self.register_buffer(
            "spatial_freqs",
            rope_params(max_seq_len=1024, dim=2 * (c_s + skip_dim) // no_heads),
        )
        # 3. Backbone: IPA Blocks
        # 使用 ModuleList 方便迭代，并统一结构
        self.ipa_layers = nn.ModuleList()
        for _ in range(num_ipa_blocks):
            self.ipa_layers.append(
                nn.ModuleDict(
                    {
                        "ipa": AF2InvariantPointAttention(
                            c_s=c_s,
                            c_z=c_z,
                            c_hidden=c_hidden,
                            no_heads=no_heads,
                            no_qk_points=no_qk_points,
                            no_v_points=no_v_points,
                            init="relu",
                        ),
                        "ipa_ln": nn.LayerNorm(c_s),
                        "DyneTrionAttn": DyneTrionAttentionBlock(
                            dim=c_s,
                            node_dim=c_s,
                            skip_dim=skip_dim,
                            ffn_dim=4 * c_s,
                            num_heads=no_heads,
                            sapatial_out_dim=c_s + skip_dim,
                            use_causal_temporal=False,
                            use_temporal=False,
                        ),
                        "node_transition": StructureModuleTransition(c=c_s),
                        "bb_update": BackboneUpdate(c_s=c_s),
                    }
                )
            )
        self.angle_resnet = AngleResnet(
            c_in=c_s,
            c_hidden=c_s,
            no_blocks=2,
            no_angles=7,
            epsilon=1e-12,
        )
        # 4. Memory & Output
        # self.memory_attn = TrajectoryMemoryAttention(c_s, num_heads=no_heads)
        # self.layer_norm_final = nn.LayerNorm(c_s)
        # self.head = EnergyHead(c_s=c_s)  # , max_energy=max_energy

    def inject_lora(self, rank: int = 8, alpha: float = 16.0,
                    target_modules: Optional[List[str]] = None):
        """Inject LoRA adapters into IPA linear layers.

        Args:
            rank: LoRA rank.
            alpha: LoRA scaling factor.
            target_modules: names of Linear attrs inside each IPA to wrap.
                Defaults to q/kv projections and output projection.
        """
        if target_modules is None:
            target_modules = [
                "linear_q", "linear_kv",
                "linear_q_points", "linear_kv_points",
                "linear_out",
            ]
        count = 0
        for layer_dict in self.ipa_layers:
            ipa = layer_dict["ipa"]
            for attr_name in target_modules:
                orig = getattr(ipa, attr_name, None)
                if orig is not None and isinstance(orig, nn.Linear):
                    setattr(ipa, attr_name, LoRALinear(orig, rank=rank, alpha=alpha))
                    count += 1
        return count

    def freeze_backbone(self):
        """Freeze all parameters, then unfreeze only LoRA params."""
        for p in self.parameters():
            p.requires_grad = False
        lora_params = 0
        for m in self.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.requires_grad = True
                m.lora_B.requires_grad = True
                lora_params += m.lora_A.numel() + m.lora_B.numel()
        total_params = sum(p.numel() for p in self.parameters())
        return lora_params, total_params

    def forward(
        self,
        rigids_t: torch.Tensor,  # [B,N,7]
        t: torch.Tensor,  # [B]
        node_repr: torch.Tensor,  # [B,N,D]
        edge_repr: torch.Tensor,  # [B,N,N,D]
        ref_rigids_0: torch.Tensor,  # [B,N,7]
        res_mask: torch.Tensor,  # [B,N]
        memory_cache: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        context_rigids: Optional[torch.Tensor] = None,
        # block_mask=None,
        **kwargs,
    ):
        B, N, _ = rigids_t.shape
        # #TODO
        curr_rigids = self.scale_rigids(Rigid.from_tensor_7(rigids_t))
        ref_rigids = self.scale_rigids(Rigid.from_tensor_7(ref_rigids_0))

        # B. 初始特征映射
        ref_node = node_repr
        ref_edge = edge_repr
        if res_mask is None:
            res_mask = torch.ones([B, N], device=node_repr.device)

        # C. 注入时间信息
        frame_node, frame_edge, t_emb = self.time_embedder(node_repr, edge_repr, t)

        frame_node_init = frame_node
        ref_node_init = ref_node
        for layer_dict in self.ipa_layers:
            def _ipa_layer_fn(layer_dict, ref_node, frame_node, ref_edge, frame_edge,
                              ref_rigids_t7, curr_rigids_t7, res_mask,
                              ref_node_init, frame_node_init, spatial_freqs):
                all_curr_rigids = Rigid.from_tensor_7(
                    torch.cat([ref_rigids_t7, curr_rigids_t7], dim=0)
                )
                all_node = torch.cat([ref_node, frame_node], dim=0)
                all_egde = torch.cat([ref_edge, frame_edge], dim=0)
                all_mask = torch.cat([res_mask, res_mask], dim=0)
                all_node = all_node + layer_dict["ipa"](
                    s=all_node, z=all_egde, r=all_curr_rigids, mask=all_mask
                )
                all_node = layer_dict["ipa_ln"](all_node)
                ref_node, frame_node = torch.chunk(all_node, chunks=2, dim=0)
                frame_node = layer_dict["DyneTrionAttn"].forward_on_spatial(
                    ref_node, frame_node, ref_node_init, frame_node_init,
                    spatial_freqs=spatial_freqs,
                )
                all_node = layer_dict["node_transition"](
                    torch.cat([ref_node, frame_node], dim=0)
                )
                ref_node, frame_node = torch.chunk(all_node, chunks=2, dim=0)
                rigid_update = layer_dict["bb_update"](frame_node * res_mask[..., None])
                return ref_node, frame_node, rigid_update

            ref_rigids_t7 = ref_rigids.to_tensor_7()
            curr_rigids_t7 = curr_rigids.to_tensor_7()
            if self.training and getattr(self, '_use_grad_checkpoint', False):
                ref_node, frame_node, rigid_update = cp.checkpoint(
                    _ipa_layer_fn, layer_dict,
                    ref_node, frame_node, ref_edge, frame_edge,
                    ref_rigids_t7, curr_rigids_t7, res_mask,
                    ref_node_init, frame_node_init, self.spatial_freqs,
                    use_reentrant=False,
                )
            else:
                ref_node, frame_node, rigid_update = _ipa_layer_fn(
                    layer_dict, ref_node, frame_node, ref_edge, frame_edge,
                    ref_rigids_t7, curr_rigids_t7, res_mask,
                    ref_node_init, frame_node_init, self.spatial_freqs,
                )
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, res_mask[..., None]
            )
        curr_rigids = self.unscale_rigids(curr_rigids)  # scale to angstrom
        unorm_angles, angles = self.angle_resnet(frame_node, frame_node_init)
        model_out = {
            "final_rigids": curr_rigids,
            "rigid_update": rigid_update,
            "angles": angles,
            "unorm_angles": unorm_angles,
        }
        return model_out
        # # E. Contextual Memory (关键：感知历史轨迹)
        # # 将 s 升维以匹配 TrajectoryMemoryAttention 的 [B, F, N, C]
        # if process_context_rigids:
        #     s_context, s = s.split([context_len, F], dim=1)
        #     mask_context, mask = mask.split([context_len, F], dim=1)
        #     memory_cache = [s_context]
        # # 注意：这里需要传入 RoPE 频率和 BlockMask
        # s_enhanced = self.memory_attn(
        #     s_curr=s,
        #     memory_cache=memory_cache,
        #     freqs=self.context_freqs,
        #     block_mask=block_mask,
        # )
        # # s_enhanced = s_enhanced.squeeze(1)  # 回到 [B, N, C]

        # F. 输出势能标量
        # potential shape: [B, 1]
        # potential = self.head(s, mask)

        # return pred_dict


# if __name__ == "__main__":
#     print("=" * 60)
#     print("正在启动 Rigid-Aware Metadynamics Actor 调试程序...")
#     print("=" * 60)

#     # 1. 环境与超参数设置
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"运行设备: {device}")

#     B = 2  # Batch Size
#     N_F = 16
#     N = 5  # Residue Number (模拟一个小蛋白)
#     C_s = 32  # Node Feature Dim (为了测试方便设小一点)
#     C_z = 16  # Edge Feature Dim

#     # 2. 实例化模型
#     print("\n[Step 1] 初始化模型...")
#     try:
#         actor = MetadynamicsActor(
#             c_s=C_s,
#             c_z=C_z,
#             c_hidden=16,  # IPA hidden
#             no_heads=2,  # IPA heads
#             num_ipa_blocks=1,
#             # n_experts=4,
#             max_energy=20.0,
#         ).to(device)
#         print("✅ 模型初始化成功")
#     except Exception as e:
#         print(f"❌ 模型初始化失败: {e}")
#         exit()

#     # 3. 构造模拟数据 (Mock Data)
#     print("\n[Step 2] 构造模拟输入数据...")

#     # A. 刚体 (Rigids): [B, N, 7] (RotQuat + Trans)
#     # 必须开启 requires_grad=True 以便测试求力
#     rigids_t = torch.randn(B, N, 7, device=device)
#     rigids_t.requires_grad_(True)

#     # B. 时间 (Time): [B, 1]
#     t = torch.rand(B, 1, device=device)

#     # C. 静态参考 (Reference - Apo): 模拟从 Dataset 加载的数据
#     s_apo = torch.randn(B, N, C_s, device=device)
#     z_apo = torch.randn(B, N, N, C_z, device=device)

#     # D. Mask
#     mask = torch.ones(B, N, device=device)

#     print(f"  - Rigids shape: {rigids_t.shape}")
#     print(f"  - Reference s shape: {s_apo.shape}")

#     # 4. 测试场景一：冷启动 (无历史记忆)
#     print("\n[Step 3] 测试 Forward Pass (无 Memory)...")
#     try:
#         potential, curr_feat = actor(
#             rigids_t=rigids_t,
#             t=t,
#             s_apo=s_apo,
#             z_apo=z_apo,
#             mask=mask,
#             memory_cache=None,  # 空记忆
#         )
#         print(f"✅ Forward 成功")
#         print(f"  - 输出 Potential shape: {potential.shape} (预期 [B, 1])")
#         print(f"  - 输出 Feature shape: {curr_feat.shape} (预期 [B, N, C_s])")
#         print(f"  - Potential 值示例: {potential.detach().cpu().numpy().flatten()}")
#     except Exception as e:
#         print(f"❌ Forward 失败: {e}")
#         import traceback

#         traceback.print_exc()

#     # 5. 测试场景二：带历史记忆 (Context Injection)
#     print("\n[Step 4] 测试 Forward Pass (带 Memory Cache)...")
#     # 模拟 Memory: 假设我们在过去生成了 5 帧，取其中 2 帧作为 Context
#     # 必须 detach，因为历史不需要梯度
#     fake_history_1 = torch.randn(B, N, C_s, device=device)
#     fake_history_2 = torch.randn(B, N, C_s, device=device)
#     memory_cache = [fake_history_1, fake_history_2]

#     try:
#         potential_with_mem, _ = actor(
#             rigids_t=rigids_t,
#             t=t,
#             s_apo=s_apo,
#             z_apo=z_apo,
#             mask=mask,
#             memory_cache=memory_cache,
#         )
#         print(f"✅ 带记忆 Forward 成功")

#         # 简单验证记忆是否起了作用 (输出应该不同)
#         diff = (potential_with_mem - potential).abs().mean().item()
#         print(f"  - 记忆注入对势能的影响差异 (Diff): {diff:.6f}")
#         if diff > 1e-6:
#             print("  - 结论: Memory 模块正在工作 (Cross-Attention 产生了影响)")
#         else:
#             print("  - 警告: Memory 似乎没有改变输出 (可能是参数初始化问题或门控全关)")

#     except Exception as e:
#         print(f"❌ 带记忆 Forward 失败: {e}")

#     # 6. 测试场景三：梯度流与求力 (Force Calculation)
#     print("\n[Step 5] 测试梯度回传 (Gradient Check)...")
#     try:
#         # 我们需要对 potential 求和或取平均来backward
#         loss = potential_with_mem.sum()

#         # 清空梯度 (如果是循环中)
#         if rigids_t.grad is not None:
#             rigids_t.grad.zero_()

#         # 自动求导: 计算 Force
#         # 这是 Metadynamics 最关键的一步：F = -grad(V)
#         grads = torch.autograd.grad(
#             outputs=loss,
#             inputs=rigids_t,
#             grad_outputs=torch.ones_like(loss),
#             create_graph=True,  # 如果需要二阶导(如Hessian)则为True，RL通常只需一阶
#             retain_graph=True,
#         )[0]

#         print(f"✅ 梯度计算成功")
#         print(f"  - Gradients (Force) shape: {grads.shape} (预期 [B, N, 7])")

#         # 检查梯度是否全为 0 (那意味着 Rigids 没有参与计算，IPA 断开了)
#         grad_norm = grads.norm().item()
#         print(f"  - Gradient Norm: {grad_norm:.6f}")
#         if grad_norm > 0:
#             print("  - 结论: 梯度流正常，IPA Backbone 与 Rigids 连接正确")
#         else:
#             print("  - ❌ 错误: 梯度为 0！Rigids 输入没有参与计算图，请检查 IPA 输入")

#     except Exception as e:
#         print(f"❌ 梯度计算失败: {e}")

#     # 7. 测试场景四：约束检查 (Constraint Check)
#     print("\n[Step 6] 测试势能约束 (Tanh Clipping)...")
#     # 为了测试，我们故意把输入搞得很大，看输出是否爆炸
#     with torch.no_grad():
#         big_s = s_apo * 100.0
#         # 临时替换一下 forward 中的输入流(这里只能简单看输出范围)
#         pot_check, _ = actor(rigids_t, t, big_s, z_apo, mask, None)
#         max_val = pot_check.abs().max().item()
#         print(f"  - 极端输入下的最大势能绝对值: {max_val:.4f}")
#         print(f"  - 预设 Max Energy: {actor.head.max_energy}")
#         if max_val <= actor.moe_head.max_energy + 1e-3:
#             print("✅ 约束生效: 输出未超过设定范围")
#         else:
#             print("❌ 约束失效: 输出超过了设定范围")

#     print("\n" + "=" * 60)
#     print("Debug 流程结束。如果以上步骤全为 ✅，则 Actor 模块功能完备。")
#     print("=" * 60)
