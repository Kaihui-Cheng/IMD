# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from dataclasses import dataclass
from itertools import combinations

import mdtraj
import numpy as np
import torch
from Bio import pairwise2
from Bio.pairwise2 import Alignment

# from bioemu.chemgraph import ChemGraph

# Constants for contact score computation
CONTACT_BETA: float = 5.0
CONTACT_DELTA: float = 0.0
CONTACT_LAMBDA: float = 1.2

# Constants for defining contact indices
SEQUENCE_SEPARATION = 3
CONTACT_CUTOFF_ANGSTROM = 10.0


@dataclass
class ReferenceInfo:
    """Describes contacts between residues in a reference structure."""

    contact_indices: np.ndarray  # Indices of residues which contact each other. Shape (2, n_contacts)
    contact_distances_angstrom: np.ndarray  # Distances between residues which contact each other. Shape (n_contacts,)
    sequence: str


@dataclass
class TargetInfo:
    """Parameters to compute foldedness from fraction of native contacts."""

    p_fold_thr: float  # This FNC has foldedness 0.5
    steepness: float  # Steepness of foldedness w.r.t. FNC
    p_fold_target: float  # Target mean foldedness value


def _get_aligned_indices(seq_alignment_1: str, seq_alignment_2: str):
    """
    Return the indices of the aligned residues in sequence 1 without gaps in the alignment.
    E.g. seq1=ABCDE, seq2=GABDF, then seq_alignment_1='-ABCDE-', seq_alignment_2='GAB-D-F'.
    The dashes are gaps andnot counted when incrementing the index.
    """
    aligned_indices = []
    n = 0
    for i, s in enumerate(seq_alignment_1):
        if s != "-":
            if seq_alignment_2[i] != "-":
                aligned_indices.append(n)
            n += 1
    return aligned_indices


def foldedness_from_fnc(fnc: torch.Tensor, p_fold_thr: float, steepness: float) -> torch.Tensor:
    """
    Compute foldedness from fraction of native contacts (FNC).

    Args:
        fnc: Fraction of native contacts.
        p_fold_thr: FNC that has foldedness 0.5.
        steepness: Steepness of the sigmoid function.

    Returns:
        Foldedness values.
    """
    return torch.sigmoid(2 * steepness * (fnc - p_fold_thr))


def compute_fnc_for_list(
    batch: list[torch.Tensor],
    contact_indices: torch.Tensor,
    contact_distances_angstrom: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the collective variable for a batch of samples.
    It is assumed that the batch contains samples from the same system.

    The reference sequence and batch sequence should be same. Only the contacts in the
    alignment are used for computing the fraction of native contacts.

    Args:
        batch: Batch of samples, containing CA positions in (angs) (B, Time_Len, Seq, 3).
        reference_info (ReferenceInfo): Reference information containing the contact indices and
            distances for the reference structure.

    Returns:
        torch tensor of collective variables.
    """
    all_positions = torch.stack(
        [x for x in batch], dim=0
    )  # Shape: (batch_size, n_residues, 3)

    sample_contact_distances = torch.linalg.norm(
        all_positions[..., contact_indices[0, :], :]
        - all_positions[..., contact_indices[1, :], :],
        dim=-1,
    )  # angstrom

    contact_scores = _compute_contact_score(
        sample_contact_distances=sample_contact_distances,
        reference_contact_distances=contact_distances_angstrom.unsqueeze(0),
    )

    # Average contact scores over contacts for each sample
    fnc = torch.mean(contact_scores, dim=-1)
    return fnc


def _compute_contact_score(
    *,
    sample_contact_distances: torch.Tensor,
    reference_contact_distances: torch.Tensor,
) -> torch.Tensor:
    """
    Compute contact scores for all pairs of contacts.

    Args:
        sample_contact_distances: Distances computed for contacts in batch
            structures, in angstrom.
        reference_contact_distances: Reference contact distances in angstrom.

    Returns:
        torch.Tensor: Contact scores for all pairwise interactions, same shape as the input tensors.
    """
    q_ij = torch.special.expit(
        -CONTACT_BETA
        * (
            sample_contact_distances
            - CONTACT_LAMBDA * (reference_contact_distances + CONTACT_DELTA)
        )
    )

    return q_ij


def compute_contacts(traj: mdtraj.Trajectory) -> ReferenceInfo:
    """
    Find CA-CA contacts in a single-frame trajectory.

    Args:
        reference_conformation (mdtraj.Trajectory): Reference conformation.  Coordinates in this Trajectory object are in nm.

    Returns:
        ReferenceInfo with sequence, contact indices and distances.
    """
    traj = traj.atom_slice(traj.topology.select("name CA"))
    heavy_atoms = traj.topology.select("name CA")
    sequence = traj.topology.to_fasta()[0]
    assert np.all(heavy_atoms == np.arange(len(sequence)))  # checking if we could do this instead.
    heavy_pairs = []
    for i, j in combinations(heavy_atoms, 2):
        res_position_i = traj.topology.atom(i).residue.index
        res_position_j = traj.topology.atom(j).residue.index

        # Check if residues are sufficiently far apart in the sequence:
        if abs(res_position_i - res_position_j) > SEQUENCE_SEPARATION:
            heavy_pairs.append((i, j))

    # Compute the distances between the valid heavy pairs.
    heavy_pairs = np.array(heavy_pairs)
    heavy_pairs_distances_angstrom = mdtraj.compute_distances(traj, heavy_pairs)[0] * 10.0

    # Filter according to cutoff
    heavy_pairs = heavy_pairs[heavy_pairs_distances_angstrom <= CONTACT_CUTOFF_ANGSTROM]
    heavy_pairs_distances_angstrom = heavy_pairs_distances_angstrom[
        heavy_pairs_distances_angstrom <= CONTACT_CUTOFF_ANGSTROM
    ]
    assert isinstance(heavy_pairs, np.ndarray)  # shut up mypy
    return ReferenceInfo(
        contact_indices=heavy_pairs.T,
        contact_distances_angstrom=heavy_pairs_distances_angstrom,
        sequence=sequence,
    )


def compute_collective_variable_for_list(
    batch: list[torch.Tensor], contact_indices: torch.Tensor
) -> torch.Tensor:
    """
    Compute the collective variable for a batch of samples.
    It is assumed that the batch contains samples from the same system.

    The reference sequence and batch sequence should be same. Only the contacts in the
    alignment are used for computing the fraction of native contacts.

    Args:
        batch: Batch of samples, containing CA positions in (angs) (B, Time_Len, Seq, 3).
        reference_info (ReferenceInfo): Reference information containing the contact indices and
            distances for the reference structure.

    Returns:
        torch tensor of collective variables.
    """

    # Map from aligned_indices_ref to aligned_indices_sample to get the overlapped contacts in the sample

    all_positions = torch.stack([x for x in batch], dim=0)  # Shape: (batch_size, n_residues, 3)


    sample_contact_distances = (
        torch.linalg.norm(
            all_positions[... , contact_indices[0, :], :]
            - all_positions[... , contact_indices[1, :], :],
            dim=-1,
        )
    )  # angstrom

    # contact_scores = _compute_contact_score(
    #     sample_contact_distances=aligned_sample_contact_distances,
    #     reference_contact_distances=aligned_ref_contact_distances.unsqueeze(0),
    # )

    # # Average contact scores over contacts for each sample
    # fnc = torch.mean(contact_scores, dim=1)
    # assert fnc.shape == (len(batch),)
    return sample_contact_distances




def compute_metadynamics_bias_potential(
    S_seq: torch.Tensor,  # (B, T, d)
    sigma: torch.Tensor | float = 1.0,
    omega: float = 1.0,
):
    """
    Compute metadynamics bias potential.

    Args:
        S_seq: CV trajectory tensor of shape (batch, time, d)
        sigma: Gaussian width in CV space. Scalar or tensor of shape (d,)
        omega: Height of deposited Gaussian

    Returns:
        V_G: Bias potential of shape (batch, time)
    """
    assert S_seq.dim() == 3, "S_seq must have shape (B, T, d)"

    B, T, d = S_seq.shape
    device = S_seq.device
    dtype = S_seq.dtype

    # --- handle sigma ---
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, device=device, dtype=dtype)

    if sigma.dim() == 0:
        sigma = sigma.repeat(d)          # scalar -> (d,)
    elif sigma.shape != (d,):
        raise ValueError(f"sigma must be scalar or shape (d,), got {sigma.shape}")

    sigma = sigma.view(1, 1, 1, d)

    # --- compute pairwise CV differences ---
    S_t = S_seq.unsqueeze(2)    # (B, T, 1, d)
    S_tau = S_seq.unsqueeze(1)  # (B, 1, T, d)

    diff = S_t - S_tau

    exponent = -((diff ** 2) / (2 * sigma ** 2)).sum(dim=-1)  # (B, T, T)

    # --- causal mask: only t' < t ---
    mask = torch.tril(
        torch.ones(T, T, device=device, dtype=dtype),
        diagonal=-1
    )

    V_G = omega * torch.exp(exponent) * mask
    V_G = V_G.sum(dim=-1)  # sum over t'

    return V_G  # (B, T)



if __name__ == "__main__":
    from icecream import ic
    sample_reference_pdb_path = "datasets/inference/pdb_file/3nke_B.pdb"
    reference_traj = mdtraj.load(sample_reference_pdb_path)
    reference_traj = reference_traj.atom_slice(reference_traj.topology.select("name CA"))
    reference_info = compute_contacts(traj=reference_traj)
    ic(reference_info)
    gen_position = torch.load('tmp.pt')
    ca_gen_position = gen_position[...,1,:]
    ca_gen_position = ca_gen_position.unsqueeze(0)
    # compute CVs
    sample_contact_distances = compute_collective_variable_for_list(
        batch=ca_gen_position, contact_indices=reference_info.contact_indices
    )
    ic(sample_contact_distances.shape)
    compute_fnc_for_list(
        batch=ca_gen_position,
        contact_indices=reference_info.contact_indices,
        contact_distances_angstrom=reference_info.contact_distances_angstrom,
    )

    exit()
    # compute reward
    V_G = compute_metadynamics_bias_potential(sample_contact_distances, sigma=0.5)
    ic(V_G)
    V_G = compute_metadynamics_bias_potential(sample_contact_distances, sigma=1.0)
    ic(V_G)
    V_G = compute_metadynamics_bias_potential(sample_contact_distances, sigma=2.0)
    ic(V_G)
    V_G = compute_metadynamics_bias_potential(sample_contact_distances, sigma=5.0)
    ic(V_G)
    ic(V_G.shape)
