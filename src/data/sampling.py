import os
import math
import torch
import numpy as np
import pandas as pd
import tree
import logging
from torch.utils import data
from torch.utils.data import Sampler
import torch.distributed as dist

from openfold.np import protein
from openfold.data.data_pipeline import make_protein_features
from openfold.data import data_transforms
from openfold.utils import rigid_utils
from src.data.ab_initio_dataset import AbInitioInferenceDataset

class ProteinSamplingDataset(AbInitioInferenceDataset):
    """
    Dataset for Ab Initio inference with optional target ground truth loading.
    Inherits from AbInitioInferenceDataset.
    """
    def __init__(
        self,
        *,
        data_conf,
        diffuser,
        is_training=False,
        rank=0,
        grouped=1,
    ):
        super().__init__(
            data_conf=data_conf,
            diffuser=diffuser,
            is_training=is_training,
            rank=rank,
            grouped=grouped,
        )
        if "seq_len" not in self.csv.columns:
            raise ValueError(
                "ProteinSamplingDataset requires 'seq_len' column in CSV for length bucketing."
            )
        if self.csv["seq_len"].isna().any():
            raise ValueError("Found NaN values in required 'seq_len' column.")
        self._lengths = self.csv["seq_len"].astype(int).clip(lower=1).tolist()

    @property
    def lengths(self):
        return self._lengths

    def __getitem__(self, idx):
        """
        Retrieves features for a given index, adding target positions if available.
        """
        # Call the parent class __getitem__ to get the base features and pdb_name
        final_feats, pdb_name = super().__getitem__(idx)

        # Get the current row from the CSV metadata
        csv_row = self.csv.iloc[idx]
        frame_time = self._data_conf.frame_time

        # Check if target_pdb_path / holo_pdb_path exists in the CSV and is not empty
        target_col = None
        for col in ('target_pdb_path', 'holo_pdb_path'):
            if col in csv_row and pd.notna(csv_row[col]):
                target_col = col
                break
        if target_col is not None:
            target_path = csv_row[target_col]

            if os.path.exists(target_path):
                try:
                    # Extract ground truth target features from the target PDB
                    with open(target_path, 'r') as f:
                        target_prot = protein.from_pdb_string(f.read())
                        # Note: use pdb_name or target_path identifier
                        target_pdb_feats = make_protein_features(target_prot, pdb_name)

                    # Convert target positions to tensor: [N, 37, 3]
                    target_pos = torch.tensor(target_pdb_feats['all_atom_positions']).double()

                    # Expand to match prediction frame dimensions: [frame_time, N, 37, 3]
                    final_feats['target_atom37_pos'] = target_pos.unsqueeze(0).expand(frame_time, -1, -1, -1)

                except Exception as e:
                    self._log.warning(f"Could not process target PDB at {target_path}: {e}")

        # Ensure all added structures are proper tensors
        final_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x),
            final_feats
        )

        return final_feats, pdb_name

    def _get_row(self, idx):
        """
        Helper method to retrieve a processed row, matching the PdbDataset interface.
        """
        return self.__getitem__(idx)


class BucketedDistributedSampler(Sampler):
    """Length-aware distributed sampler for variable-length protein rollout."""

    def __init__(
        self,
        *,
        lengths,
        batch_size: int,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        drop_last: bool = True,
        bucket_size: int = 256,
        seed: int = 42,
        sort_by_square: bool = True,
    ):
        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        self.lengths = [int(x) for x in lengths]
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.bucket_size = int(bucket_size)
        self.seed = int(seed)
        self.sort_by_square = bool(sort_by_square)
        self.epoch = 0

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.bucket_size <= 0:
            raise ValueError(f"bucket_size must be positive, got {self.bucket_size}")

        self.global_batch = self.batch_size * self.num_replicas
        if self.drop_last:
            n_global_batches = len(self.lengths) // self.global_batch
        else:
            n_global_batches = math.ceil(len(self.lengths) / self.global_batch)
        self.num_samples = n_global_batches * self.batch_size

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        n = len(self.lengths)
        if n == 0:
            return iter([])

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.sort_by_square:
            key = [x * x for x in self.lengths]
        else:
            key = self.lengths

        sorted_idx = sorted(range(n), key=lambda i: key[i])

        full_buckets = []
        tail_bucket = None
        for start in range(0, n, self.bucket_size):
            bucket = sorted_idx[start : start + self.bucket_size]
            # Keep the last incomplete bucket at the end to avoid sharp length jumps.
            if len(bucket) < self.bucket_size:
                tail_bucket = bucket
                continue
            if self.shuffle and len(bucket) > 1:
                perm = torch.randperm(len(bucket), generator=g).tolist()
                bucket = [bucket[i] for i in perm]
            full_buckets.append(bucket)

        if self.shuffle and len(full_buckets) > 1:
            bucket_perm = torch.randperm(len(full_buckets), generator=g).tolist()
            full_buckets = [full_buckets[i] for i in bucket_perm]

        if tail_bucket is not None:
            if self.shuffle and len(tail_bucket) > 1:
                perm = torch.randperm(len(tail_bucket), generator=g).tolist()
                tail_bucket = [tail_bucket[i] for i in perm]
            full_buckets.append(tail_bucket)

        buckets = full_buckets

        indices = [idx for bucket in buckets for idx in bucket]
        remainder = len(indices) % self.global_batch
        if remainder != 0:
            if self.drop_last:
                indices = indices[: len(indices) - remainder]
            else:
                pad = self.global_batch - remainder
                indices = indices + indices[:pad]

        if len(indices) == 0:
            return iter([])

        global_batches = torch.tensor(indices, dtype=torch.long).view(-1, self.global_batch)
        # For global_batch==1, shuffling global batches is equivalent to a full random
        # permutation and destroys the local length ordering built by bucketing.
        if self.shuffle and self.global_batch > 1 and global_batches.size(0) > 1:
            batch_perm = torch.randperm(global_batches.size(0), generator=g)
            global_batches = global_batches[batch_perm]

        start = self.rank * self.batch_size
        end = (self.rank + 1) * self.batch_size
        local_indices = global_batches[:, start:end].reshape(-1).tolist()
        return iter(local_indices)
