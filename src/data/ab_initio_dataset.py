import os
import torch
import numpy as np
import pandas as pd
import tree
import logging
from torch.utils import data

from openfold.np import protein
from openfold.data.data_pipeline import make_protein_features
from openfold.data import data_transforms
from openfold.utils import rigid_utils

class AbInitioInferenceDataset(data.Dataset):
    def __init__(
            self,
            *,
            data_conf,
            diffuser,
            is_training=False,
            rank=0,
            grouped=1
        ):
        self._log = logging.getLogger(__name__)
        self._data_conf = data_conf
        self._diffuser = diffuser
        self._is_training = is_training

        # Load CSV and filter based on rank/group for distributed inference
        pdb_csv = pd.read_csv(self._data_conf.test_csv_path)
        pdb_csv = pdb_csv[pdb_csv["seq_len"] < self._data_conf.max_res_len]
        pdb_csv = pdb_csv[rank::grouped]
        max_protein_num = getattr(self._data_conf, "max_protein_num", None)
        if max_protein_num is not None and int(max_protein_num) > 0:
            self.csv = pdb_csv.head(int(max_protein_num))
        else:
            self.csv = pdb_csv

        self._log.info(
            f"AbInitio Inference Dataset initialized with {len(self.csv)} examples "
            f"(max_protein_num={max_protein_num})"
        )

    def __len__(self):
        return len(self.csv)

    def _compute_reference_info(self, pdb_path):
        import mdtraj
        from src.data.foldness import compute_contacts
        reference_traj = mdtraj.load(pdb_path)
        reference_traj = reference_traj.atom_slice(
            reference_traj.topology.select("name CA")
        )
        reference_info = compute_contacts(traj=reference_traj)
        return reference_info

    def __getitem__(self, idx):
        csv_row = self.csv.iloc[idx]
        pdb_name = csv_row['pdb_id']
        pdb_path = csv_row['pdb_path']

        # Extract features from the ground truth PDB file
        with open(pdb_path, 'r') as f:
            prot = protein.from_pdb_string(f.read())
            pdb_feats = make_protein_features(prot, pdb_name)

        frame_time = self._data_conf.frame_time
        ref_frame = self._data_conf.ref_number
        motion_frame = self._data_conf.motion_number
        total_frames = ref_frame + motion_frame + frame_time
        n_res = pdb_feats['aatype'].shape[0]

        # Expand single-frame PDB features to match the total required frames
        all_feats = {
            'aatype': torch.tensor(np.argmax(pdb_feats['aatype'], axis=-1)).long().unsqueeze(0).expand(total_frames, -1),
            'all_atom_positions': torch.tensor(pdb_feats['all_atom_positions']).double().unsqueeze(0).expand(total_frames, -1, -1, -1),
            'all_atom_mask': torch.tensor(pdb_feats['all_atom_mask']).double().unsqueeze(0).expand(total_frames, -1, -1),
        }

        # Apply standard OpenFold transformations
        all_feats = data_transforms.atom37_to_frames(all_feats)
        all_feats = data_transforms.make_atom14_masks(all_feats)
        all_feats = data_transforms.make_atom14_positions(all_feats)
        all_feats = data_transforms.atom37_to_torsion_angles()(all_feats)

        # Get rigid transformations in 7-dim format
        gt_rigids = rigid_utils.Rigid.from_tensor_4x4(all_feats['rigidgroups_gt_frames'])[:, :, 0].to_tensor_7()

        # Build prediction target features (noisy frames section)
        final_feats = {
            'aatype': all_feats['aatype'][ref_frame + motion_frame:],
            'seq_idx': torch.tensor(pdb_feats['residue_index']).unsqueeze(0).expand(frame_time, -1),
            'residx_atom14_to_atom37': all_feats['residx_atom14_to_atom37'][ref_frame + motion_frame:],
            'residue_index': torch.tensor(pdb_feats['residue_index']).unsqueeze(0).expand(frame_time, -1),
            'res_mask': torch.tensor(pdb_feats['all_atom_mask'][:, 1]).unsqueeze(0).expand(frame_time, -1),
            'atom37_pos': all_feats['all_atom_positions'][ref_frame + motion_frame:],
            'atom37_mask': all_feats['all_atom_mask'][ref_frame + motion_frame:],
            'rigidgroups_0': all_feats['rigidgroups_gt_frames'][ref_frame + motion_frame:],
            'torsion_angles_sin_cos': all_feats['torsion_angles_sin_cos'][ref_frame + motion_frame:],
            'alt_torsion_angles_sin_cos': all_feats['alt_torsion_angles_sin_cos'][ref_frame + motion_frame:],
            'torsion_angles_mask': all_feats['torsion_angles_mask'][ref_frame + motion_frame:],
        }

        # Populate reference and motion features from ground truth PDB frames
        final_feats.update({
            # Expand ground truth rigid [1, N, 7] -> [ref_frame, N, 7]
            'ref_rigids_0': gt_rigids[:1].expand(ref_frame, -1, -1),
            # Expand CA mask [1, N] -> [ref_frame, N]
            'ref_node_mask': final_feats['res_mask'][:1].expand(ref_frame, -1),
            # Expand atom positions [1, N, 37, 3] -> [ref_frame, N, 37, 3]
            'ref_atom37_pos': all_feats['all_atom_positions'][:1].expand(ref_frame, -1, -1, -1),

            # Expand ground truth rigid [1, N, 7] -> [motion_frame, N, 7]
            'motion_rigids_0': gt_rigids[:1].expand(motion_frame, -1, -1),
            # Expand CA mask [1, N] -> [motion_frame, N]
            'motion_node_mask': final_feats['res_mask'][:1].expand(motion_frame, -1),
            # Expand atom positions [1, N, 37, 3] -> [motion_frame, N, 37, 3]
            'motion_atom37_pos': all_feats['all_atom_positions'][:1].expand(motion_frame, -1, -1, -1),
        })

        # Load pre-computed embeddings if available
        node_edge_feature_path = csv_row['embed_path']
        if os.path.exists(node_edge_feature_path):
            attr_dict = dict(np.load(node_edge_feature_path))
            final_feats.update({
                'node_repr': torch.tensor(attr_dict.get('node_repr', attr_dict.get('single'))),
                'edge_repr': torch.tensor(attr_dict.get('edge_repr', attr_dict.get('pair')))
            })

        # Diffusion initialization (Starting from complete noise for Ab Initio inference)
        t = 1.0
        diff_feats_t = self._diffuser.sample_ref(
            n_samples=frame_time * n_res,
            diffuse_mask=None,
            as_tensor_7=True,
            dynamics=self._diffuser._se3_conf.dynamics,
        )

        final_feats.update(diff_feats_t)
        final_feats["t"] = torch.tensor([t])
        final_feats['fixed_mask'] = torch.zeros((frame_time, n_res))
        final_feats['sc_ca_t'] = torch.zeros((frame_time, n_res, 3))
        final_feats['rigids_0'] = final_feats['rigids_t']

        # Include contact information for evaluation metrics
        reference_info = self._compute_reference_info(pdb_path)
        final_feats.update({
            "ref_contact_indices": reference_info.contact_indices,
            "ref_contact_distances_angstrom": reference_info.contact_distances_angstrom,
        })

        return tree.map_structure(lambda x: x if torch.is_tensor(x) else torch.tensor(x), final_feats), pdb_name

    def _get_row(self, idx):
        return self.__getitem__(idx)
