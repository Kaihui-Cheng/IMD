"""PDB dataset loader."""
import sys
import os
import math
from typing import Optional
from src.experiments import utils as eu
import torch
import torch.distributed as dist

import tree
import numpy as np
import torch
import pandas as pd
import logging
import random
import functools as fn
from src.data import se3_diffuser
from torch.utils import data
from src.data import utils as du
from src.data.foldness import compute_contacts
from openfold.data import data_transforms
from openfold.np import residue_constants
from openfold.utils import rigid_utils
# from data import pdb_data_loader
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import mdtraj
from icecream import ic
from einops import rearrange, repeat


def parse_dynamics_chain_feats(chain_feats, scale_factor=1.):
    ca_idx = residue_constants.atom_order['CA']
    chain_feats['bb_mask'] = chain_feats['all_atom_mask'][:, ca_idx] # [N,37]
    bb_pos = chain_feats['all_atom_positions'][0,:, ca_idx] # [F,N,37,3]->[N,3] select first protein as anchor
    bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats['bb_mask']) + 1e-5) # [3]
    centered_pos = chain_feats['all_atom_positions'] - bb_center[None, None, None, :] # [F,N,37,3]
    scaled_pos = centered_pos / scale_factor
    chain_feats['all_atom_positions'] = scaled_pos * (chain_feats['all_atom_mask'][..., None][np.newaxis, ...])
    chain_feats['bb_positions'] = chain_feats['all_atom_positions'][:,:,ca_idx]# [F,N,3]
    return chain_feats


class PdbDataset(data.Dataset):
    def __init__(
            self,
            *,
            data_conf,
            diffuser,
            is_training,
            is_testing=False,
            is_random_test=False,
            rank=0, # only used for multi node evaluation
            grouped=1 # only used for multi node evaluation
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._is_test = is_testing
        self._is_random_test=is_random_test
        self._data_conf = data_conf
        self._init_metadata(rank, grouped)
        self._diffuser = diffuser
        self.offset =  {idx: 0 for idx in range(len(self.csv))}

    @property
    def is_training(self):
        return self._is_training

    @property
    def diffuser(self):
        return self._diffuser

    @property
    def data_conf(self):
        return self._data_conf

    def _init_metadata(self, rank=0, grouped=1):
        """Initialize metadata."""
        filter_conf = self.data_conf.filtering
        if self._is_training:
            pdb_csv = pd.read_csv(self.data_conf.csv_path)
            before = len(pdb_csv)
            pdb_csv = pdb_csv.dropna()
            after = len(pdb_csv)
            print(f"[INFO] Dropped {before - after} rows containing None/NaN values (kept {after} rows).")
            pdb_csv = pdb_csv[pdb_csv.seq_len <= filter_conf.max_len]
            fliter_len = len(pdb_csv)
            print(f"[INFO] Dropped {after - fliter_len} rows exceeding len (kept {fliter_len} rows).")

        elif self._is_test:
            pdb_csv = pd.read_csv(self.data_conf.test_csv_path)
            pdb_csv = pdb_csv[rank::grouped]
            print(f"Grouped by {grouped} | Rank: {rank}")
            # pdb_csv = pdb_csv[pdb_csv.seq_len <= filter_conf.max_len]
            pdb_csv = pdb_csv.head(self.data_conf.max_protein_num)
        else:
            pdb_csv = pd.read_csv(self.data_conf.val_csv_path)
            # pdb_csv = pdb_csv[pdb_csv.seq_len <= filter_conf.max_len]
            # pdb_csv = pdb_csv.head(6)# keep 6 for validation

        print(pdb_csv)
        self._create_split(pdb_csv)

    def _create_split(self, pdb_csv):
        # Training or validation specific logic.
        if self.is_training:
            self.csv = pdb_csv#[pdb_csv.split == 'train']
            self._log.info(f'Training: {len(self.csv)} examples')
        else:
            self.csv = pdb_csv#[pdb_csv.split == 'val']
            self._log.info(f'Validation: {len(self.csv)} examples')

    def select_random_samples(self,arr, t, k):
        n = arr.shape[0]  # Obtain the size of the first dimension, the number of samples
        if t > n:
            raise ValueError("t cannot be greater than the number of samples")
        start_index = np.random.randint(0, n - (t)*k + 1)  # randomly select the start indexnp.random.randint(0, n - t*(k-1))
        end_index = start_index + (t)*k # the end index
        selected_samples = arr[start_index:end_index:k]  # select with step k
        return selected_samples,start_index


    def select_first_samples(self, arr, number, k):
        n = arr.shape[0]  # length of trajectory
        if number > n:
            raise ValueError("t cannot be greater than the number of samples")

        start_index = 0 #np.random.randint(0, n - (t)*k + 1)  # Randomly select the starting index.
        end_index = start_index + (number) * k # endding index
        selected_samples = arr[start_index:end_index:k]  # Select t consecutive samples with a step of k.
        return selected_samples


    def select_with_motion_continue(self, arr, number, k, ref_frame):
        arr = np.asarray(arr)
        n = len(arr)
        max_start_index = n - number * k - 1
        if max_start_index < 0:
            raise ValueError(
                "The array is too small to select t elements with the given interval s."
            )
        start_index = np.random.randint(0, max_start_index + 1)
        motion_part = arr[start_index : start_index + number * k : k]
        ref_indices = np.random.choice(n, size=ref_frame, replace=False)
        ref_part = arr[ref_indices]
        combined = np.concatenate([ref_part, motion_part])
        return combined

    def _process_csv_row(self, processed_file_path):
        processed_feats = dict(np.load(processed_file_path,allow_pickle=True))

        motion_frame = self.data_conf.motion_number
        ref_frame = self.data_conf.ref_number
        frame_time = self.data_conf.frame_time
        # here to sample frame_time continuous positions.
        frame_time_ref_motion = ref_frame + motion_frame + frame_time
        motion_frame_number = motion_frame + frame_time
        if self._is_training:
            if self.data_conf.keep_first is not None:
                first_data = processed_feats['all_atom_positions'][:self.data_conf.keep_first]
                # print("="*100)
                tmp = self.select_with_motion_continue(
                    first_data,
                    motion_frame_number,
                    self.data_conf.frame_sample_step,
                    ref_frame,
                )
            else:
                tmp = self.select_with_motion_continue(
                    processed_feats["all_atom_positions"],
                    motion_frame_number,
                    self.data_conf.frame_sample_step,
                    ref_frame,
                )

        else:
            tmp = self.select_first_samples(
                processed_feats["all_atom_positions"],
                number=frame_time_ref_motion,
                k=self.data_conf.frame_sample_step,
            )
            start_index = 0

        processed_feats['all_atom_positions'] = tmp
        processed_feats = parse_dynamics_chain_feats(processed_feats)
        chain_feats = {
            'aatype': torch.tensor(np.argmax(processed_feats['aatype'],axis=-1)).long().unsqueeze(0).expand(frame_time_ref_motion, -1),
            'all_atom_positions': torch.tensor(processed_feats['all_atom_positions']).double(),
            'all_atom_mask': torch.tensor(processed_feats['all_atom_mask']).double().unsqueeze(0).expand(frame_time_ref_motion, -1, -1)
        }

        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)
        #NOTE motion ref frame
        motion_feats = {
            'motion_rigids_0': rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'][ref_frame: ref_frame+motion_frame])[:,:, 0].to_tensor_7(),
            'motion_node_mask': torch.tensor(processed_feats['bb_mask']).unsqueeze(0).expand(motion_frame, -1),
            'motion_atom37_pos': chain_feats['all_atom_positions'][ref_frame: ref_frame+motion_frame] ,
        }
        ref_feats = {
            'ref_rigids_0': rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'][:ref_frame])[:,:, 0].to_tensor_7(),
            'ref_node_mask': torch.tensor(processed_feats['bb_mask']).unsqueeze(0).expand(ref_frame, -1),
            'ref_atom37_pos': chain_feats['all_atom_positions'][:ref_frame] ,
        }


        final_feats = {
            'aatype': chain_feats['aatype'][ref_frame+motion_frame:],
            'seq_idx':  torch.tensor(processed_feats['residue_index']).unsqueeze(0).expand(frame_time, -1),
            # 'chain_idx': new_chain_idx,
            'residx_atom14_to_atom37': chain_feats['residx_atom14_to_atom37'][ref_frame+motion_frame:],
            'residue_index': torch.tensor(processed_feats['residue_index']).unsqueeze(0).expand(frame_time, -1),
            'res_mask': torch.tensor(processed_feats['bb_mask']).unsqueeze(0).expand(frame_time, -1),
            'atom37_pos': chain_feats['all_atom_positions'][ref_frame+motion_frame:],
            'atom37_mask': chain_feats['all_atom_mask'][ref_frame+motion_frame:],
            # 'atom14_pos': chain_feats['atom14_gt_positions'],
            'rigidgroups_0': chain_feats['rigidgroups_gt_frames'][ref_frame+motion_frame:],
            'torsion_angles_sin_cos': chain_feats['torsion_angles_sin_cos'][ref_frame+motion_frame:],
            'alt_torsion_angles_sin_cos':chain_feats['alt_torsion_angles_sin_cos'][ref_frame+motion_frame:],
            'torsion_angles_mask':chain_feats['torsion_angles_mask'][ref_frame+motion_frame:],
        }

        final_feats.update(ref_feats)
        final_feats.update(motion_feats)

        if not self._is_training:
            final_feats.update({'start_index':start_index})

        return final_feats

    def _create_diffused_masks(self, atom37_pos, rng, row):
        bb_pos = atom37_pos[:, residue_constants.atom_order['CA']]
        dist2d = np.linalg.norm(bb_pos[:, None, :] - bb_pos[None, :, :], axis=-1)

        # Randomly select residue then sample a distance cutoff
        # TODO: Use a more robust diffuse mask sampling method.
        diff_mask = np.zeros_like(bb_pos)
        attempts = 0
        while np.sum(diff_mask) < 1:
            crop_seed = rng.integers(dist2d.shape[0])
            seed_dists = dist2d[crop_seed]
            max_scaffold_size = min(
                self._data_conf.scaffold_size_max,
                seed_dists.shape[0] - self._data_conf.motif_size_min
            )
            scaffold_size = rng.integers(
                low=self._data_conf.scaffold_size_min,
                high=max_scaffold_size
            )
            dist_cutoff = np.sort(seed_dists)[scaffold_size]
            diff_mask = (seed_dists < dist_cutoff).astype(float)
            attempts += 1
            if attempts > 100:
                raise ValueError(
                    f'Unable to generate diffusion mask for {row}')
        return diff_mask

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        # Sample data example.
        example_idx = idx
        csv_row = self.csv.iloc[example_idx]
        if "pdb_id" in csv_row:
            pdb_name = csv_row['pdb_id']
        else:
            raise ValueError('Need chain identifier.')
        processed_file_path = csv_row["pos_path"]
        chain_feats = self._process_csv_row(processed_file_path)


        frame_time = chain_feats['aatype'].shape[0]
        node_edge_feature_path = csv_row['embed_path']  # here
        assert os.path.exists(node_edge_feature_path)
        attr_dict = dict(np.load(node_edge_feature_path))

        # NOTE
        try:
            chain_feats.update({'node_repr':torch.tensor(attr_dict['node_repr'])})
            chain_feats.update({'edge_repr':torch.tensor(attr_dict['edge_repr'])})
        except:
            chain_feats.update({'node_repr':torch.tensor(attr_dict['single'])})
            chain_feats.update({'edge_repr':torch.tensor(attr_dict['pair'])})
        #TODO prob to be zero reference pos information
        # if random.random() < self._data_conf.cfg_drop_rate:
        #     chain_feats['ref_rigids_0'] = torch.zeros_like(chain_feats['ref_rigids_0'])
        #     chain_feats['ref_atom37_pos'] = torch.zeros_like(chain_feats['ref_atom37_pos'])

        if random.random() < 0.2:
            chain_feats["motion_rigids_0"] = repeat(
                chain_feats["ref_rigids_0"],
                "1 n d -> t n d",
                t=self.data_conf.motion_number,
            )
            chain_feats["motion_node_mask"] = repeat(
                chain_feats["ref_node_mask"],
                "1 n -> t n",
                t=self.data_conf.motion_number,
            )
            chain_feats["motion_atom37_pos"] = repeat(
                chain_feats["ref_atom37_pos"],
                "1 n r d -> t n r d",
                t=self.data_conf.motion_number,
            )

        # Use a fixed seed for evaluation.
        if self.is_training:
            rng = np.random.default_rng(None)
        else:
            rng = np.random.default_rng(idx)

        gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_0'])[:,:, 0]

        diffused_mask = np.ones_like(chain_feats['res_mask'])
        if np.sum(diffused_mask) < 1:
            raise ValueError('Must be diffused')
        fixed_mask = 1 - diffused_mask
        chain_feats['fixed_mask'] = fixed_mask
        chain_feats['rigids_0'] = gt_bb_rigid.to_tensor_7()
        chain_feats['sc_ca_t'] = torch.zeros_like(gt_bb_rigid.get_trans())

        # Sample t and diffuse.
        if self.is_training:
            # TODO here add different t
            t = rng.uniform(self._data_conf.min_t, 1.0)
            diff_feats_t = self._diffuser.forward_marginal(
                rigids_0=gt_bb_rigid,
                t=t,
                diffuse_mask=None
            )
            ref_diff_feats = self._diffuser.forward_marginal(
                rigids_0=rigid_utils.Rigid.from_tensor_7(chain_feats['ref_rigids_0']),
                t=t,
                diffuse_mask=None
            )
            chain_feats['ref_rot_score'] = ref_diff_feats['rot_score']
        else:
            t = 1.0
            # todo here
            if self._data_conf.dynamics:
                diff_feats_t = self.diffuser.sample_ref(
                    n_samples=gt_bb_rigid.shape[0]*gt_bb_rigid.shape[1],
                    diffuse_mask=None,
                    as_tensor_7=True,
                )
            else:
                diff_feats_t = self.diffuser.sample_ref(
                    n_samples=gt_bb_rigid.shape[0],
                    impute=gt_bb_rigid,
                    diffuse_mask=None,
                    as_tensor_7=True,
                )

        chain_feats.update(diff_feats_t)

        chain_feats["t"] = torch.tensor([t])
        if not self.is_training:
            start_index = chain_feats.pop('start_index')
            # print(start_index)
        # Convert all features to tensors.
        final_feats = tree.map_structure(lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats)

        if self.is_training:
            return final_feats
        else:
            return final_feats, pdb_name,start_index


    def _get_row(self, idx):
        # Sample data example.
        csv_row = self.csv.iloc[idx]
        # if "pdb_id" in csv_row:
        #     pdb_name = csv_row["pdb_id"]
        # else:
        #     raise ValueError("Need chain identifier.")

        processed_file_path = csv_row["pos_path"]
        pdb_name = os.path.basename(processed_file_path).split(".")[0]
        # chain_feats = self._process_csv_row(processed_file_path)

        processed_feats = dict(np.load(processed_file_path,allow_pickle=True))
        # process feats
        motion_frame = self.data_conf.motion_number
        ref_frame = self.data_conf.ref_number
        frame_time = self.data_conf.frame_time
        # here to sample frame_time continuous positions.
        # frame_time_ref_motion = ref_frame + motion_frame + frame_time
        frame_time_ref_motion = ref_frame + motion_frame
        tmp = self.select_first_samples(
            processed_feats["all_atom_positions"] if not self.data_conf.last else processed_feats["all_atom_positions"][-frame_time_ref_motion:],
            number=frame_time_ref_motion,
            k=self.data_conf.frame_sample_step,
        )

        processed_feats['all_atom_positions'] = tmp
        processed_feats = parse_dynamics_chain_feats(processed_feats)
        chain_feats = {
            'aatype': torch.tensor(np.argmax(processed_feats['aatype'],axis=-1)).long().unsqueeze(0).expand(frame_time_ref_motion, -1),
            'all_atom_positions': torch.tensor(processed_feats['all_atom_positions']).double(),
            'all_atom_mask': torch.tensor(processed_feats['all_atom_mask']).double().unsqueeze(0).expand(frame_time_ref_motion, -1, -1)
        }

        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)
        # motion ref frame
        motion_feats = {
            'motion_rigids_0': rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'][ref_frame: ref_frame+motion_frame])[:,:, 0].to_tensor_7(),
            'motion_node_mask': torch.tensor(processed_feats['bb_mask']).unsqueeze(0).expand(motion_frame, -1),
            'motion_atom37_pos': chain_feats['all_atom_positions'][ref_frame: ref_frame+motion_frame] ,
        }
        ref_feats = {
            'ref_rigids_0': rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'][:ref_frame])[:,:, 0].to_tensor_7(),
            'ref_node_mask': torch.tensor(processed_feats['bb_mask']).unsqueeze(0).expand(ref_frame, -1),
            'ref_atom37_pos': chain_feats['all_atom_positions'][:ref_frame] ,
        }



        final_feats = {
            'aatype': chain_feats['aatype'][:1].expand(frame_time, -1),
            'seq_idx':  torch.tensor(processed_feats['residue_index']).unsqueeze(0).expand(frame_time, -1),
            # 'chain_idx': new_chain_idx,
            'residx_atom14_to_atom37': chain_feats['residx_atom14_to_atom37'][:1].expand(frame_time, -1, -1),
            'residue_index': torch.tensor(processed_feats['residue_index']).unsqueeze(0).expand(frame_time, -1),
            'res_mask': torch.tensor(processed_feats['bb_mask']).unsqueeze(0).expand(frame_time, -1),
            'atom37_pos': chain_feats['all_atom_positions'][:1].expand(frame_time, -1, -1, -1),
            'atom37_mask': chain_feats['all_atom_mask'][:1].expand(frame_time, -1, -1),
            # 'atom14_pos': chain_feats['atom14_gt_positions'],
            'rigidgroups_0': chain_feats['rigidgroups_gt_frames'][:1].expand(frame_time, -1,-1,-1,-1),
            'torsion_angles_sin_cos': chain_feats['torsion_angles_sin_cos'][:1].expand(frame_time, -1,-1,-1),
            'alt_torsion_angles_sin_cos':chain_feats['alt_torsion_angles_sin_cos'][:1].expand(frame_time, -1,-1,-1),
            'torsion_angles_mask':chain_feats['torsion_angles_mask'][:1].expand(frame_time, -1,-1),
        }

        final_feats.update(ref_feats)
        final_feats.update(motion_feats)
        chain_feats =  final_feats
        # process feats



        node_edge_feature_path = csv_row["embed_path"]  # here
        assert os.path.exists(node_edge_feature_path)
        attr_dict = dict(np.load(node_edge_feature_path))

        # NOTE
        try:
            chain_feats.update({"node_repr": torch.tensor(attr_dict["node_repr"])})
            chain_feats.update({"edge_repr": torch.tensor(attr_dict["edge_repr"])})
        except:
            chain_feats.update({"node_repr": torch.tensor(attr_dict["single"])})
            chain_feats.update({"edge_repr": torch.tensor(attr_dict["pair"])})

        gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(chain_feats["rigidgroups_0"])[:, :, 0]
        diffused_mask = np.ones_like(chain_feats["res_mask"])
        if np.sum(diffused_mask) < 1:
            raise ValueError("Must be diffused")
        fixed_mask = 1 - diffused_mask
        chain_feats["fixed_mask"] = fixed_mask
        chain_feats["rigids_0"] = gt_bb_rigid.to_tensor_7()
        chain_feats["sc_ca_t"] = torch.zeros_like(gt_bb_rigid.get_trans())

        t = 1.0
        # todo here
        if self._data_conf.dynamics:
            diff_feats_t = self.diffuser.sample_ref(
                n_samples=frame_time * gt_bb_rigid.shape[1],
                diffuse_mask=None,
                as_tensor_7=True,
            )
        else:
            diff_feats_t = self.diffuser.sample_ref(
                n_samples=frame_time,
                impute=gt_bb_rigid,
                diffuse_mask=None,
                as_tensor_7=True,
            )

        chain_feats.update(diff_feats_t)

        chain_feats["t"] = t
        # Convert all features to tensors.
        final_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats
        )
        return final_feats, pdb_name


class RLPdbDataset(PdbDataset):

    def _compute_reference_info(self, pdb_path):
        reference_traj = mdtraj.load(pdb_path)
        reference_traj = reference_traj.atom_slice(
            reference_traj.topology.select("name CA")
        )
        reference_info = compute_contacts(traj=reference_traj)
        return reference_info

    def __getitem__(self, idx):
        # Sample data example.
        csv_row = self.csv.iloc[idx]
        processed_file_path = csv_row["pos_path"]

        pdb_name = os.path.basename(processed_file_path).split(".")[0]
        # chain_feats = self._process_csv_row(processed_file_path)
        processed_feats = dict(np.load(processed_file_path,allow_pickle=True))
        # process feats
        motion_frame = self.data_conf.motion_number
        ref_frame = self.data_conf.ref_number
        frame_time = self.data_conf.frame_time
        # here to sample frame_time continuous positions.
        # frame_time_ref_motion = ref_frame + motion_frame + frame_time
        frame_time_ref_motion = ref_frame + motion_frame
        if self._is_training:
            tmp, start_index = self.select_random_samples(
                processed_feats["all_atom_positions"],
                frame_time_ref_motion,
                k=self.data_conf.frame_sample_step,
            )
        else:
            ic("sampling test data from 1st frame!")
            tmp = self.select_first_samples(
                processed_feats["all_atom_positions"],
                number=frame_time_ref_motion,
                k=self.data_conf.frame_sample_step,
            )

        processed_feats['all_atom_positions'] = tmp
        processed_feats = parse_dynamics_chain_feats(processed_feats)
        chain_feats = {
            'aatype': torch.tensor(np.argmax(processed_feats['aatype'],axis=-1)).long().unsqueeze(0).expand(frame_time_ref_motion, -1),
            'all_atom_positions': torch.tensor(processed_feats['all_atom_positions']).double(),
            'all_atom_mask': torch.tensor(processed_feats['all_atom_mask']).double().unsqueeze(0).expand(frame_time_ref_motion, -1, -1)
        }

        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)
        # motion ref frame
        motion_feats = {
            'motion_rigids_0': rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'][ref_frame: ref_frame+motion_frame])[:,:, 0].to_tensor_7(),
            'motion_node_mask': torch.tensor(processed_feats['bb_mask']).unsqueeze(0).expand(motion_frame, -1),
            'motion_atom37_pos': chain_feats['all_atom_positions'][ref_frame: ref_frame+motion_frame] ,
        }
        ref_feats = {
            'ref_rigids_0': rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'][:ref_frame])[:,:, 0].to_tensor_7(),
            'ref_node_mask': torch.tensor(processed_feats['bb_mask']).unsqueeze(0).expand(ref_frame, -1),
            'ref_atom37_pos': chain_feats['all_atom_positions'][:ref_frame] ,
        }



        final_feats = {
            'aatype': chain_feats['aatype'][:1].expand(frame_time, -1),
            'seq_idx':  torch.tensor(processed_feats['residue_index']).unsqueeze(0).expand(frame_time, -1),
            # 'chain_idx': new_chain_idx,
            'residx_atom14_to_atom37': chain_feats['residx_atom14_to_atom37'][:1].expand(frame_time, -1, -1),
            'residue_index': torch.tensor(processed_feats['residue_index']).unsqueeze(0).expand(frame_time, -1),
            'res_mask': torch.tensor(processed_feats['bb_mask']).unsqueeze(0).expand(frame_time, -1),
            'atom37_pos': chain_feats['all_atom_positions'][:1].expand(frame_time, -1, -1, -1),
            'atom37_mask': chain_feats['all_atom_mask'][:1].expand(frame_time, -1, -1),
            # 'atom14_pos': chain_feats['atom14_gt_positions'],
            'rigidgroups_0': chain_feats['rigidgroups_gt_frames'][:1].expand(frame_time, -1,-1,-1,-1),
            'torsion_angles_sin_cos': chain_feats['torsion_angles_sin_cos'][:1].expand(frame_time, -1,-1,-1),
            'alt_torsion_angles_sin_cos':chain_feats['alt_torsion_angles_sin_cos'][:1].expand(frame_time, -1,-1,-1),
            'torsion_angles_mask':chain_feats['torsion_angles_mask'][:1].expand(frame_time, -1,-1),
        }

        final_feats.update(ref_feats)
        final_feats.update(motion_feats)
        chain_feats =  final_feats
        # process feats



        node_edge_feature_path = csv_row["embed_path"]  # here
        assert os.path.exists(node_edge_feature_path)
        attr_dict = dict(np.load(node_edge_feature_path))

        # NOTE
        try:
            chain_feats.update({"node_repr": torch.tensor(attr_dict["node_repr"])})
            chain_feats.update({"edge_repr": torch.tensor(attr_dict["edge_repr"])})
        except:
            chain_feats.update({"node_repr": torch.tensor(attr_dict["single"])})
            chain_feats.update({"edge_repr": torch.tensor(attr_dict["pair"])})

        # Use a fixed seed for evaluation.

        gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(chain_feats["rigidgroups_0"])[:, :, 0]
        diffused_mask = np.ones_like(chain_feats["res_mask"])
        if np.sum(diffused_mask) < 1:
            raise ValueError("Must be diffused")
        fixed_mask = 1 - diffused_mask
        chain_feats["fixed_mask"] = fixed_mask
        chain_feats["rigids_0"] = gt_bb_rigid.to_tensor_7()
        chain_feats["sc_ca_t"] = torch.zeros_like(gt_bb_rigid.get_trans())

        t = 1.0
        # todo here
        if self._data_conf.dynamics:
            diff_feats_t = self.diffuser.sample_ref(
                n_samples=frame_time * gt_bb_rigid.shape[1],
                diffuse_mask=None,
                as_tensor_7=True,
            )
        else:
            diff_feats_t = self.diffuser.sample_ref(
                n_samples=frame_time,
                impute=gt_bb_rigid,
                diffuse_mask=None,
                as_tensor_7=True,
            )

        chain_feats.update(diff_feats_t)

        chain_feats["t"] = t
        reference_info = self._compute_reference_info(csv_row["pdb_path"])
        final_feats.update(
            {
                "ref_contact_indices": reference_info.contact_indices,
                "ref_contact_distances_angstrom": reference_info.contact_distances_angstrom,
            }
        )

        # Convert all features to tensors.
        final_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats
        )
        #TODO add ref conformation for c-alpha-c-alpha
        return final_feats, pdb_name
