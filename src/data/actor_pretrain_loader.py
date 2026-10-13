import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import logging
import torch.nn.functional as F
from openfold.data import data_transforms
from openfold.utils import rigid_utils
import tree
import random
from openfold.np import protein
from openfold.data.data_pipeline import make_protein_features
from openfold.np import residue_constants
from src.data.DyneTrion_data_loader_dynamic import parse_dynamics_chain_feats

def parse_chain_feats(chain_feats, frame_num=2, scale_factor=1.0):
    ca_idx = residue_constants.atom_order["CA"]

    # all_atom_positions: [S*F, N, 37, 3]
    pos = np.asarray(chain_feats["all_atom_positions"], dtype=np.float32)
    if pos.ndim != 4:
        raise ValueError(f"all_atom_positions must be 4D, got {pos.shape}")

    sf, n_res, n_atom, _ = pos.shape
    if sf % frame_num != 0:
        raise ValueError(f"num_frames={sf} not divisible by frame_num={frame_num}")
    s = sf // frame_num

    # all_atom_mask: [S*F, N, 37] or [N, 37]
    atom_mask = np.asarray(chain_feats["all_atom_mask"], dtype=np.float32)
    if atom_mask.ndim == 2:
        atom_mask = np.broadcast_to(
            atom_mask[None, ...], (sf, n_res, atom_mask.shape[-1])
        ).copy()
    elif atom_mask.ndim == 3 and atom_mask.shape[0] == 1:
        atom_mask = np.broadcast_to(
            atom_mask, (sf, n_res, atom_mask.shape[-1])
        ).copy()
    elif atom_mask.ndim != 3:
        raise ValueError(f"all_atom_mask must be 2D/3D, got {atom_mask.shape}")

    # Backbone CA mask: [S*F, N]
    bb_mask = atom_mask[:, :, ca_idx]
    chain_feats["bb_mask"] = bb_mask

    # Reshape without einops.
    pos = pos.reshape(s, frame_num, n_res, n_atom, 3)
    bb_mask_reshape = bb_mask.reshape(s, frame_num, n_res)

    # Center by frame-0 CA centroid per sample.
    ca_ref = pos[:, 0, :, ca_idx, :]  # [S, N, 3]
    m_ref = bb_mask_reshape[:, 0, :]  # [S, N]
    denom = np.clip(m_ref.sum(axis=1, keepdims=True), 1e-5, None)  # [S, 1]
    center = (ca_ref * m_ref[..., None]).sum(axis=1) / denom  # [S, 3]

    pos -= center[:, None, None, None, :]
    pos /= float(scale_factor)

    # Back to [S*F, N, 37, 3], apply mask.
    pos = pos.reshape(sf, n_res, n_atom, 3)
    pos *= atom_mask[..., None]

    # Keep contiguous float32 for downstream torch conversion.
    pos = np.ascontiguousarray(pos, dtype=np.float32)
    atom_mask = np.ascontiguousarray(atom_mask, dtype=np.float32)

    chain_feats["all_atom_positions"] = pos
    chain_feats["all_atom_mask"] = atom_mask
    chain_feats["atom_positions"] = pos  # compatibility alias
    return chain_feats

class ActorPretrainDataset(Dataset):
    def __init__(
        self,
        csv_path,
        diffuser,
        energy_sample_stride=10,
        samples_per_traj=8,  # 每个轨迹随机采样的独立帧数
        min_t=0.01,
        max_res_len=512,
        frame_sample_step=40,
        **kwargs # 吸收配置文件中可能残留的其他参数 (如 context_window) 避免报错
    ):
        super().__init__()
        self._log = logging.getLogger(__name__)
        self.csv = pd.read_csv(csv_path)
        self.csv = self.csv[self.csv["seq_len"] < max_res_len]
        self.energy_sample_stride = energy_sample_stride
        self.samples_per_traj = samples_per_traj
        self.min_t = min_t
        self.diffuser = diffuser
        self.frame_sample_step = frame_sample_step
        self.frame_num = 2
        self._log.info(
            f"Minimalist Actor Pretrain Dataset initialized with {len(self.csv)} examples. "
            f"Sampling {self.samples_per_traj} independent frames per trajectory."
        )
        self._lengths = self.csv["seq_len"].astype(int).clip(lower=1).tolist()

    def __len__(self):
        return len(self.csv)

    @property
    def lengths(self):
        return self._lengths

    def _load_raw_data(self, position_file, energy_path):
        feats = dict(np.load(position_file, allow_pickle=True))
        # energy_data = np.load(energy_path).astype(np.float32)
        # energy_data = energy_data[:: self.energy_sample_stride]
        # assert len(feats["all_atom_positions"]) == len(energy_data)
        return feats, len(feats["all_atom_positions"])

    def _read_repr(self, feature_path):
        assert os.path.exists(feature_path), f"Feature file not found: {feature_path}"
        attr_dict = dict(np.load(feature_path, allow_pickle=True))
        node_data = attr_dict.get("node_repr", attr_dict.get("single"))
        edge_data = attr_dict.get("edge_repr", attr_dict.get("pair"))
        return {"node_repr": torch.tensor(node_data), "edge_repr": torch.tensor(edge_data)}

    def generate_spaced_integers(self, L, n, s):
        """
        Generate n random integers within the range [0, L-1] such that
        the minimum distance between any two integers is at least s.

        Parameters:
        ----------
        L : int
            Total length of the sequence (index range from 0 to L-1).
        n : int
            Number of random integers to sample.
        s : int
            Minimum stride (distance) required between any two points,
            i.e., |x_i - x_j| >= s.

        Returns:
        -------
        numpy.ndarray
            An array of n sorted integers satisfying the stride constraint.
        """

        if (n - 1) * s + 1 > L:
            raise ValueError(
                f"Constraint Error: Cannot fit {n} points with stride {s} "
                f"within a range of length {L}."
            )
        effective_L = L - (n - 1) * (s - 1)
        y = np.random.choice(effective_L, n, replace=False)
        y.sort()
        offsets = np.arange(n) * (s - 1)
        x = y + offsets
        return x.astype(int)

    def _batched_forward_marginal(self, rigids_0_batched, S):
        t_batch = torch.empty(S).uniform_(self.min_t, 1.0)
        diffused_dicts_list = []

        for b in range(S):
            rigids_0_single = rigids_0_batched[b]
            t_single = float(t_batch[b].item())

            diffused_single = self.diffuser.forward_marginal(
                rigids_0=rigids_0_single, t=t_single, as_tensor_7=True
            )
            diffused_dicts_list.append(diffused_single)

        diffused_batched_dict = tree.map_structure(
            lambda *xs: torch.stack([torch.as_tensor(x) for x in xs], dim=0),
            *diffused_dicts_list,
        )

        return diffused_batched_dict, t_batch

    def _process_batched_frames(self, raw_feats, pair_indices):
        S = self.samples_per_traj
        F = self.frame_num  # 2
        # 压平索引，一次性切出 S*F 帧
        flat_indices = pair_indices.flatten()  # [S*2]

        feats = raw_feats.copy()
        # [S*2, N, 37, 3] 巨型切片
        feats["all_atom_positions"] = feats["all_atom_positions"][flat_indices]
        feats = parse_chain_feats(feats, frame_num=F)

        aatype_static = feats["aatype"]
        if aatype_static.ndim == 2:
            aatype_static = np.argmax(aatype_static, axis=-1)

        aatype_tensor = torch.tensor(aatype_static).long()
        # Use repeat (not expand) to avoid overlapping-storage tensors that
        # can fail in DataLoader pin_memory.
        aatype_batched = aatype_tensor.unsqueeze(0).repeat(S * F, 1)
        mask_tensor = torch.tensor(feats["all_atom_mask"]) # [S*2, N, 37]
        pos_tensor = torch.tensor(feats["all_atom_positions"]) # [S*2, N, 37, 3]

        chain_feats = {
            "aatype": aatype_batched,
            "all_atom_positions": pos_tensor,
            "all_atom_mask": mask_tensor,
        }

        # 💡 OpenFold 变换原生支持 Batch 维度，这里一次性处理 S*2 帧，极快！
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)

        # ==================== 特征拆包与重组 ====================
        # [S*2, N] -> [S, 2, N]
        def extract_i(tensor):
            return tensor.view(S, F, *tensor.shape[1:])[:, 1].contiguous()
        def extract_ref(tensor):
            return tensor.view(S, F, *tensor.shape[1:])[:, 0].contiguous()

        gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(chain_feats["rigidgroups_gt_frames"])[:, :, 0]
        gt_bb_rigid_7 = gt_bb_rigid.to_tensor_7().view(S, F, -1, 7) # [S, 2, N, 7]

        rigids_ref = rigid_utils.Rigid.from_tensor_7(gt_bb_rigid_7[:, 0]) # [S, N]
        rigids_i = rigid_utils.Rigid.from_tensor_7(gt_bb_rigid_7[:, 1])   # [S, N]

        # feats["bb_mask"] is [S*F, N]. Keep only current-frame mask -> [S, N].
        res_mask = torch.tensor(feats["bb_mask"]).view(S, F, -1)[:, 1].contiguous()

        final_feats = {
            "aatype": extract_i(chain_feats["aatype"]), # [S, N]
            "seq_idx": torch.tensor(raw_feats["residue_index"]).unsqueeze(0).repeat(S, 1),
            "torsion_angles_sin_cos": extract_i(chain_feats["torsion_angles_sin_cos"]),
            "alt_torsion_angles_sin_cos": extract_i(chain_feats["alt_torsion_angles_sin_cos"]),
            "torsion_angles_mask": extract_i(chain_feats["torsion_angles_mask"]),
            "res_mask": res_mask,
            "fixed_mask": 1.0 - res_mask,
            "rigids_0": rigids_i.to_tensor_7(),
            "ref_rigids_0": rigids_ref.to_tensor_7(),
            "ref_torsion_angles_sin_cos": extract_ref(chain_feats["torsion_angles_sin_cos"]),
            "ref_alt_torsion_angles_sin_cos": extract_ref(chain_feats["alt_torsion_angles_sin_cos"]),
            "pair_index": pair_indices, # [S, 2]
        }

        # ==================== 一次性加噪与分数计算 ====================
        # 生成 S 个独立的噪声时间步 [S]
        t_batch = torch.empty(S).uniform_(self.min_t, 1.0)
        diffused_batched_dict, t_batch = self._batched_forward_marginal(rigids_i, S)

        final_feats.update(diffused_batched_dict)
        final_feats["t"] = t_batch
        # ========================================================
        # 👑 核心升级：计算流形上的残差分数 (Bias Score Target)
        # score' = score(x_{i,t} -> x_{ref}) - score(x_{i,t} -> x_i)
        # ========================================================
        # 将生成的 7 维 tensor 重新转为 Rigid 对象以提取平移和旋转
        rigids_i_t = rigid_utils.Rigid.from_tensor_7(diffused_batched_dict["rigids_t"])

        # --- A. 平移分数 (Translation Score) ---
        trans_i_t = rigids_i_t.get_trans()  # [N, 3]
        trans_i = rigids_i.get_trans()  # [N, 3]
        trans_ref = rigids_ref.get_trans()  # [N, 3]

        # 注意：不同扩散框架对 t 的 shape 要求可能不同，如果报错请将 t 扩展为 [N, 1]
        score_trans_base = self.diffuser.calc_trans_score(
            trans_i_t, trans_i, final_feats["t"][:, None, None]
        )
        score_trans_guided = self.diffuser.calc_trans_score(
            trans_i_t, trans_ref, final_feats["t"][:, None, None]
        )
        target_trans_score = score_trans_guided - score_trans_base  # [N, 3]

        # --- B. 旋转分数 (Rotation Score) ---
        rots_i_t = rigids_i_t.get_rots()  # [N, 3, 3]
        rots_i = rigids_i.get_rots()
        rots_ref = rigids_ref.get_rots()

        score_rot_base = self.diffuser.calc_rot_score(
            rots_i_t, rots_i, final_feats["t"]
        )
        score_rot_guided = self.diffuser.calc_rot_score(
            rots_i_t, rots_ref, final_feats["t"]
        )
        target_rot_score = score_rot_guided - score_rot_base  # [N, 3] (切空间向量)

        # 将这两个 Ground Truth 存入 final_feats
        final_feats["target_trans_score"] = target_trans_score.detach()
        final_feats["target_rot_score"] = target_rot_score.detach()
        final_feats["score_trans_guided"] = score_trans_guided.detach()
        final_feats["score_rot_guided"] = score_rot_guided.detach()
        final_feats["score_trans_base"] = score_trans_base.detach()
        final_feats["score_rot_base"] = score_rot_base.detach()
        # Ensure no overlapping-storage tensors enter DataLoader pin-memory.
        final_feats = tree.map_structure(
            lambda x: x.contiguous().clone() if torch.is_tensor(x) else x,
            final_feats,
        )
        return final_feats

    def _process_pair_frame(self, raw_feats, energy_data, pair_index):
        """核心：提取纯粹的单帧数据，丢弃所有时间(F)维度"""
        feats = raw_feats.copy()


        feats["all_atom_positions"] = feats["all_atom_positions"][pair_index]
        # target_energy = energy_data[pair_index]

        feats = parse_dynamics_chain_feats(feats)

        aatype_static = feats["aatype"]
        if aatype_static.ndim == 2:
            aatype_static = np.argmax(aatype_static, axis=-1)

        aatype_tensor = torch.tensor(aatype_static).long() # [N]
        mask_tensor = torch.tensor(feats["all_atom_mask"]) # [N, 37]
        pos_tensor = torch.tensor(feats["all_atom_positions"]) # [N, 37, 3]

        chain_feats = {
            "aatype": aatype_tensor.unsqueeze(0).expand(self.frame_num, -1),
            "all_atom_positions": pos_tensor,
            "all_atom_mask": mask_tensor.unsqueeze(0).expand(self.frame_num, -1, -1),
        }

        # 3. OpenFold Transforms (处理无 F 维度的单帧结构)
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)

        # 4. 获取 Rigids (去除任何 F 维度)
        gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(
            chain_feats["rigidgroups_gt_frames"]
        )[:, :, 0]
        res_mask = torch.tensor(feats["bb_mask"]).unsqueeze(0).expand(self.frame_num, -1) # [N]
        # 分离出参考帧 x_{ref} 和当前帧 x_i
        rigids_ref = gt_bb_rigid[0]  # [N]
        rigids_i = gt_bb_rigid[1]  # [N]

        final_feats = {
            "aatype": chain_feats["aatype"][1],
            "seq_idx": torch.tensor(raw_feats["residue_index"]),
            # "target_energy": torch.tensor(target_energy).float()[1],  # 纯标量
            "torsion_angles_sin_cos": chain_feats["torsion_angles_sin_cos"][1],
            "alt_torsion_angles_sin_cos": chain_feats["alt_torsion_angles_sin_cos"][1],
            "torsion_angles_mask": chain_feats["torsion_angles_mask"][1],
            "res_mask": res_mask[1],
            "fixed_mask": 1.0 - res_mask[1],
            "rigids_0": rigids_i.to_tensor_7(),  # [N, 7]
            "ref_rigids_0": rigids_ref.to_tensor_7(),  # [N, 7]
            "ref_torsion_angles_sin_cos": chain_feats["torsion_angles_sin_cos"][0],
            "ref_alt_torsion_angles_sin_cos": chain_feats["alt_torsion_angles_sin_cos"][0],
            "pair_index": pair_index,
        }
        t = np.random.uniform(self.min_t, 1.0)
        diffused_feat_dict = self.diffuser.forward_marginal(
            rigids_0=gt_bb_rigid[1], t=t, as_tensor_7=True
        )
        final_feats.update(diffused_feat_dict)
        final_feats["t"] = torch.tensor(t)

        # ========================================================
        # 👑 核心升级：计算流形上的残差分数 (Bias Score Target)
        # score' = score(x_{i,t} -> x_{ref}) - score(x_{i,t} -> x_i)
        # ========================================================
        # 将生成的 7 维 tensor 重新转为 Rigid 对象以提取平移和旋转
        rigids_i_t = rigid_utils.Rigid.from_tensor_7(diffused_feat_dict["rigids_t"])

        # --- A. 平移分数 (Translation Score) ---
        trans_i_t = rigids_i_t.get_trans()  # [N, 3]
        trans_i = rigids_i.get_trans()  # [N, 3]
        trans_ref = rigids_ref.get_trans()  # [N, 3]

        # 注意：不同扩散框架对 t 的 shape 要求可能不同，如果报错请将 t 扩展为 [N, 1]
        score_trans_base = self.diffuser.calc_trans_score(
            trans_i_t, trans_i, final_feats["t"][None]
        )
        score_trans_guided = self.diffuser.calc_trans_score(
            trans_i_t, trans_ref, final_feats["t"][None]
        )
        target_trans_score = score_trans_guided - score_trans_base  # [N, 3]

        # --- B. 旋转分数 (Rotation Score) ---
        rots_i_t = rigids_i_t.get_rots()  # [N, 3, 3]
        rots_i = rigids_i.get_rots()
        rots_ref = rigids_ref.get_rots()

        score_rot_base = self.diffuser.calc_rot_score(
            rots_i_t, rots_i, final_feats["t"][None]
        )[0]
        score_rot_guided = self.diffuser.calc_rot_score(
            rots_i_t, rots_ref, final_feats["t"][None]
        )[0]
        target_rot_score = score_rot_guided - score_rot_base  # [N, 3] (切空间向量)

        # 将这两个 Ground Truth 存入 final_feats
        final_feats["target_trans_score"] = target_trans_score.detach()
        final_feats["target_rot_score"] = target_rot_score.detach()
        final_feats["score_trans_guided"] = score_trans_guided.detach()
        final_feats["score_rot_guided"] = score_rot_guided.detach()
        final_feats["score_trans_base"] = score_trans_base.detach()
        final_feats["score_rot_base"] = score_rot_base.detach()
        return final_feats

    def __getitem__(self, idx):
        """
        核心逻辑：每次被 DataLoader 调用，只返回 1 个蛋白质的 1 帧。
        """
        csv_row = self.csv.iloc[idx]
        repr_dict = self._read_repr(csv_row["embed_path"])
        raw_feats, num_frames = self._load_raw_data(csv_row["pos_path"], None)

        pair_indices = np.array([
            self.generate_spaced_integers(L=num_frames, n=self.frame_num, s=self.frame_sample_step)
            for _ in range(self.samples_per_traj)
        ])
        stacked_sample = self._process_batched_frames(raw_feats, pair_indices)
        # Match feature batch dimension [S, ...] for model forward.
        s = int(self.samples_per_traj)
        stacked_sample["node_repr"] = repr_dict["node_repr"].unsqueeze(0).repeat(s, 1, 1).contiguous()
        stacked_sample["edge_repr"] = repr_dict["edge_repr"].unsqueeze(0).repeat(s, 1, 1, 1).contiguous()
        # stacked_sample = tree.map_structure(
        #     lambda *xs: torch.stack(xs, dim=0), *stacked_sample
        # )
        return stacked_sample
        # return final_sample

    def _get_infer_data(self, idx):
        csv_row = self.csv.iloc[idx]
        repr_dict = self._read_repr(csv_row["embed_path"])
        pdb_path = csv_row["pdb_path"]
        pdb_name = csv_row["pdb_id"]
        with open(pdb_path, "r") as f:
            prot = protein.from_pdb_string(f.read())
            pdb_feats = make_protein_features(prot, pdb_name)
        chain_feats = {
            'aatype': torch.tensor(np.argmax(pdb_feats['aatype'], axis=-1)).long(),
            'all_atom_positions': torch.tensor(pdb_feats['all_atom_positions']),
            'all_atom_mask': torch.tensor(pdb_feats['all_atom_mask']),
        }
        n_res = chain_feats["aatype"].shape[0]
        # 3. OpenFold Transforms (处理无 F 维度的单帧结构)
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)

        gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(
            chain_feats["rigidgroups_gt_frames"]
        )[:,  0]
        # read the C-alpha mask
        res_mask = torch.tensor(pdb_feats["all_atom_mask"][:, 1])

        rigids_ref = gt_bb_rigid  # [N]

        final_feats = {
            "aatype": chain_feats["aatype"],
            "seq_idx": torch.tensor(pdb_feats["residue_index"]),
            "torsion_angles_sin_cos": chain_feats["torsion_angles_sin_cos"],
            "res_mask": res_mask,
            "fixed_mask": 1.0 - res_mask,
            "ref_rigids_0": rigids_ref.to_tensor_7(),  # [N, 7]
        }
        t = 1.0
        diffused_feat_dict = self.diffuser.sample_ref(
            n_samples=n_res,
            diffuse_mask=None,
            as_tensor_7=True,
        )
        # [0] remove the batch
        final_feats.update({"rigids_t": diffused_feat_dict["rigids_t"]})
        final_feats["t"] = t
        final_feats.update(repr_dict)
        return tree.map_structure(lambda x: x if torch.is_tensor(x) else torch.tensor(x), final_feats)

def strip_batch_dim_collate(batch_list):
    # 因为 batch_size=1，所以 batch_list 长度必定为 1
    assert len(batch_list) == 1
    # 直接返回 __getitem__ 给出的那个字典，不进行 Stack
    return batch_list[0]

if __name__ == "__main__":
    from src.data import se3_diffuser
    from omegaconf import DictConfig, OmegaConf
    diff_conf = OmegaConf.load("DyneTrion/config/train.yaml")
    diffuser = se3_diffuser.SE3Diffuser(diff_conf.diffuser)
    dataset = ActorPretrainDataset(
        csv_path="./data/mmseqs_clus_debug_energy.csv",
        energy_sample_stride=10,
        samples_per_traj=8,
        diffuser=diffuser,
        frame_sample_step=40,
    )
    # a = dataset[0]
    x = dataset._get_infer_data(0)
    for k, v in x.items():
        print(k,v.shape)
    exit()

    # for k,v in a.items():
    #     print(k,v.shape)
    # exit()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        collate_fn=strip_batch_dim_collate,
    )
    for batch in dataloader:
        for k,v in batch.items():
            print(k, v.shape)
        exit()

    # custom_collate = SameProteinDiffusionCollate(
    #     diffuser=diffuser, samples_per_batch=16, min_t=0.01
    # )

    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=1, shuffle=True, collate_fn=custom_collate, num_workers=4
    # )
    # for batch in dataloader:
    #     print("Batch t shape:", batch["t"].shape)            # 预期: [16]
    #     print("Batch rigids_t shape:", batch["rigids_t"].shape) # 预期: [16, N, 7]
    #     print("Batch aatype shape:", batch["aatype"].shape)     # 预期: [16, N]
    #     # a = batch['rigids_t'][...,4:]-batch['rigids_0'][...,4:]
    #     # for t,x in zip(batch['t'], a.abs().sum(dim=(1,2))):
    #     #     print(f"Noise t:{t:.4f} | pertubation: {x:.4f}")
    #     # Noise t:0.7991 | pertubation: 1478.4878
    #     # Noise t:0.7048 | pertubation: 1437.8566
    #     # Noise t:0.7228 | pertubation: 1446.5404
    #     # Noise t:0.4949 | pertubation: 1183.0525
    #     # Noise t:0.1582 | pertubation: 525.6987
    #     # Noise t:0.9230 | pertubation: 1428.1750
    #     # Noise t:0.3050 | pertubation: 1069.3717
    #     # Noise t:0.2061 | pertubation: 857.5479
    #     # Noise t:0.4997 | pertubation: 1264.7023
    #     # Noise t:0.1964 | pertubation: 748.1078
    #     # Noise t:0.2859 | pertubation: 1020.7949
    #     # Noise t:0.9594 | pertubation: 1458.0072
    #     # Noise t:0.8748 | pertubation: 1484.0341
    #     # Noise t:0.6082 | pertubation: 1464.5175
    #     # Noise t:0.3776 | pertubation: 1091.8749
    #     # Noise t:0.4144 | pertubation: 1129.0267
    #     break

    # # exit()
    # # dataloader = torch.utils.data.DataLoader(
    # #     dataset,
    # #     batch_size=2, # 现在可以安全地调大 batch_size 了！
    # #     collate_fn=pretrain_collate_fn,
    # #     num_workers=4,
    # #     shuffle=True,
    # # )

    # # for batch in dataloader:
    # #     for k, v in batch.items():
    # #         if hasattr(v, "shape"):
    # #             print(f"{k:<25} | Shape: {v.shape}")
    # #         else:
    # #             print(f"{k:<25} | Len: {len(v)}")
    # #     break
