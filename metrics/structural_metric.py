import os
import mdtraj as md

import pandas as pd
import tempfile
import numpy as np
from tqdm import tqdm
import torch
import argparse
from multiprocessing import Pool
import pickle
import sys
from icecream import ic
from mdtraj import Trajectory

from itertools import combinations
from openfold.np import protein
from openfold.data.data_pipeline import make_protein_features
from openfold.utils.loss import lddt, torsion_angle_loss  # lddt_ca


def save_pickle(obj, path):
    """Save a Python object to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"✅ Saved to {path}")

def load_pickle(path):
    """Load a Python object from a pickle file."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"✅ Loaded from {path}")
    return obj

def parse_args():
    parser = argparse.ArgumentParser(description="Process input parameters for trajectory analysis.")
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Input CSV file path",
    )
    parser.add_argument(
        "--gen_root",
        type=str,
        required=True,
        help="Root directory for generated structures",
    )
    parser.add_argument("--gen_frame_cut_off", type=int, default=-1)
    parser.add_argument("--frame_cut_off", type=int, default=-1)
    parser.add_argument('--gen_sample_stride', type=int, default=1, help='Stride for sampling generated structures')
    parser.add_argument('--md_stride', type=int, default=10, help='Stride for MD trajectory sampling 10ps here since traj saved with 1ps')
    parser.add_argument("--rank_idx", type=int, default=0, help="Batch index to process, you may pass machine rank as it like $RANK")
    parser.add_argument("--world_size", type=int, default=1, help="num of batches to create like $WORLD_SIZE")
    parser.add_argument('--device', type=str, default="cuda", choices=["cpu", "cuda"], help='Device to use for computations (cpu or cuda)')
    parser.add_argument('--batch_size', type=int ,default=-1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--boost_idx", type=int, default=1)
    parser.add_argument("--jobs_name", type=str, required=True)
    return parser.parse_args()




def _compute_filtered_contacts(
    trajectory: md.Trajectory,
    ca_clash_cutoff: float,
    sequence_separation: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute close contacts between atoms in residues. Compared to `mdtraj.compute_contacts`, this
    routine first filters residues based on their Ca distances. Only residues with their Ca within
    a cutoff will be considered for the final contact computation, reducing the number of contacts
    considered significantly. This should speed up contact computation by a factor of ten.

    NOTE: This filtering assumes that all atoms in a residue will be distorted together with the Ca.
    It will e.g. not detect a single N from a residue colliding with something far away if the
    remaining atoms behave normally. However, this situation should not happen as we use a rigid
    frame based representation and MD data should be fine as well due to the harmonic potentials.

    Args:
        trajectory: Trajectory to compute contacts for.
        ca_clash_cutoff: Cutoff used for Ca based prefiltering. Units are Angstrom.
        sequence_separation: Only consider contacts between atoms at least this many residues apart
          in the sequence.

    Returns:
        Contact distances [n_frames x contacts] and indices of pairs [n_contacts x 2]. These will be
        different from the `mdtraj.compute_contacts` due to the filtering, but will lead to the same
        masks (if `sequence_separation` is set to 2).
    """
    # Get indices of Ca atoms and build an array for mapping Ca atom indices to their respective
    # residue indices.
    idx_ca: list[int] = []
    map_ca_to_resid: np.ndarray = np.full((trajectory.top.n_atoms,), -1)
    for atom in trajectory.top.atoms:
        if atom.name == "CA":
            idx_ca.append(atom.index)
            map_ca_to_resid[atom.index] = atom.residue.index

    # Generate all possible Ca index pairs (`combinations` excludes self pairs) and compute Ca
    # distances for preliminary filtering.
    idx_ca_pairs: np.ndarray = np.array(list(combinations(idx_ca, 2)))
    distances_ca = md.compute_distances(trajectory, atom_pairs=idx_ca_pairs, periodic=False)

    # Convert to Angstrom and filter. The Ca distances still have the trajectory dimension
    # (n_frames x n_distances), for the filter we consider any contact that is closer than the
    # cutoff at one point in the trajectory.
    within_ca_cutoff = np.any(
        md.utils.in_units_of(distances_ca, "nanometers", "angstrom") < ca_clash_cutoff,
        axis=0,
    )

    # Select contacts within cutoff and convert atom indices to residue indices for contact
    # computation.
    idx_contact_residues = map_ca_to_resid[idx_ca_pairs[within_ca_cutoff]]

    # Native `compute_contacts` only considers residues at least two amino acids apart in the
    # sequence for contact computation. We apply a similar filter here (`sequence_separation=2`)
    # will reproduce the `compute_contacts` behavior.
    mask_sequence_separation = (
        np.abs(idx_contact_residues[:, 0] - idx_contact_residues[:, 1]) > sequence_separation
    )
    idx_contact_residues = idx_contact_residues[mask_sequence_separation]

    # MDTraj cannot compute contacts for an empty set of residue pairs.
    if len(idx_contact_residues) == 0:
        return np.empty((trajectory.n_frames, 0), dtype=np.float32), idx_contact_residues

    # Compute contacts on the filtered residue pairs.
    contact_distances, contact_idx = md.compute_contacts(
        trajectory, contacts=idx_contact_residues, periodic=False
    )
    return contact_distances, contact_idx


def rmsdalign(a, b, weights=None):
    """Aligns b to a using Kabsch algorithm and computes RMSD"""
    # Assuming a and b have shape [*, N, 3]
    B = a.shape[:-2]  # Handle batch dimensions
    N = a.shape[-2]

    if weights is None:
        weights = a.new_ones(*B, N)  # Default weights if none are provided
    weights = weights.unsqueeze(-1)

    # Calculate the weighted mean of a and b
    a_mean = (a * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    a = a - a_mean
    b_mean = (b * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    b = b - b_mean

    # Compute the cross-covariance matrix B
    B = torch.einsum('...ji,...jk->...ik', weights * a, b)

    # SVD decomposition
    u, s, vh = torch.linalg.svd(B)

    # Ensure a proper rotation (Kabsch algorithm)
    sgn = torch.sign(torch.linalg.det(u @ vh))
    s[...,-1] *= sgn
    u[...,:,-1] *= sgn.unsqueeze(-1)

    C = u @ vh  # Rotation matrix to align B to A
    aligned_b = b @ C.mT + a_mean  # Apply rotation to b

    return aligned_b

def read_ca(traj):
    ca_idx = traj.topology.select('name CA')
    traj = traj.atom_slice(ca_idx)
    return traj.xyz

def compute_lddt_ca(
    all_atom_pred_pos,
    all_atom_positions,
    all_atom_mask=None,
    cutoff=15.0,
    eps=1e-10,
    per_residue=True,
):
    if all_atom_mask is None:
        all_atom_mask = torch.ones(all_atom_pred_pos.shape[:-1])
        all_atom_mask = all_atom_mask[..., None].to(all_atom_pred_pos.device)

    lddt_ca = lddt(
        all_atom_pred_pos,
        all_atom_positions,
        all_atom_mask,
        cutoff=cutoff,
        eps=eps,
        per_residue=per_residue,
    )
    return lddt_ca

def compute_diversity(pred_positions: torch.Tensor, per_residue=True):
    """
    diversity—the average dissimilarity (1 - lDDT_Cα) between pairs of predicted structures.
    """
    n = pred_positions.shape[0]  # Number of predicted structures
    diversity_score = 0.0
    count = 0
    lddt_score_list = []
    # 计算每对预测结构的 lDDT
    for i in tqdm(range(n), desc='Compute Diversity'):
        # lddt_score = compute_lddt_ca(pred_positions[i : i + 1], pred_positions[i:])
        # lddt_score_list.append(lddt_score)
        for j in range(i + 1, n):  # 只计算上三角部分
            lddt_score = compute_lddt_ca(
                pred_positions[i : i + 1],
                pred_positions[j : j + 1],
                per_residue=per_residue,
            )
            diversity_score += 1 - lddt_score  # 计算 1 - lDDT_Cα
            count += 1
    # 计算平均值
    return diversity_score / count if count > 0 else 0

def compute_precision_recall(
    pred_positions: torch.Tensor,
    crystal_positions: torch.Tensor,
    per_residue=False,
    batch_size=-1,
):
    """
    precision—the average lDDT_Cα from each prediction to the closest crystal structure;
    recall—the average lDDT_Cα from each crystal structure to the closest prediction.
    """
    n_pred = pred_positions.shape[0]  # Number of predicted structures
    n_crystal = crystal_positions.shape[0]  # Number of crystal structures
    lddt_matrix_precision = torch.zeros((n_pred, n_crystal)).to(pred_positions.device)

    for i in tqdm(range(n_pred), desc="Compute lDDT matrix (Precision)"):
        if batch_size != -1:
            for j in range(0, n_crystal, batch_size):
                j_end = min(j + batch_size, n_crystal)
                lddt_matrix_precision[i, j:j_end] = compute_lddt_ca(
                    pred_positions[i : i + 1],
                    crystal_positions[j:j_end],
                    per_residue=per_residue,
                )
        else:
            lddt_matrix_precision[i, :] = compute_lddt_ca(
                pred_positions[i : i + 1],
                crystal_positions,
                per_residue=per_residue,
            )

    precision = lddt_matrix_precision.max(dim=1).values.mean()


    lddt_matrix_recall = torch.zeros((n_crystal, n_pred), device=pred_positions.device)
    for i in tqdm(range(n_crystal), desc="Compute lDDT matrix (Recall)"):
        lddt_matrix_recall[i, :] = compute_lddt_ca(
            crystal_positions[i : i + 1],  # 单个晶体结构
            pred_positions,                 # 所有预测结构
            per_residue=per_residue,
        )

    recall = lddt_matrix_recall.max(dim=1).values.mean()
    return precision, recall

def process_protein(
    pdb_path,
    traj_path,
    gen_path,
    batch_size=-1,
    gen_sample_stride=1,
    md_stride=10,
    device="cpu",
    frame_cut_off=-1,
    gen_frame_cut_off=-1
):
    # 读取gen 轨迹和 traj 轨迹， 要求计算gen的每一帧和traj轨迹之间的rmsd，以及traj 轨迹之间两两距离的最大值
    gen_traj = md.load(gen_path)
    md_traj = md.load(traj_path, top=pdb_path, stride=md_stride) #interval 10ps


    gen_traj = gen_traj[::gen_sample_stride]

    if frame_cut_off != -1:
        md_traj = md_traj[:frame_cut_off]
    if gen_frame_cut_off != -1:
        gen_traj = gen_traj[: gen_frame_cut_off]

    ic(len(md_traj), len(gen_traj),frame_cut_off,gen_frame_cut_off)

    gen_traj_ca = torch.tensor(read_ca(gen_traj) * 10).to(device)  # nm to angs
    md_traj_ca = torch.tensor(read_ca(md_traj) * 10 ).to(device) # nm to angs
    #NOTE 需要 align ？ LDDT 不需要align
    # gen_traj_ca = gen_traj_ca[:10]
    # md_traj_ca = md_traj_ca[:10]

    diversity_score = compute_diversity(pred_positions=gen_traj_ca, per_residue=False)

    precision, recall = compute_precision_recall(
        pred_positions=gen_traj_ca,
        crystal_positions=md_traj_ca,
        per_residue=False,
        batch_size=batch_size,
    )
    return diversity_score.item(), precision.item(), recall.item()

def build_pdb_to_path(gen_dir):
    out_mapper = {}
    for file_name in os.listdir(gen_dir):
        if not file_name.endswith(".pdb"):
            continue
        pdb_id = file_name.split("_time_")[0]
        pdb_id = pdb_id.split(".pdb")[0]
        out_mapper[pdb_id] = os.path.join(gen_dir, file_name)
    return out_mapper


def filter_unphysical_traj_masks(
    traj: Trajectory,
    max_ca_seq_distance: float = 4.5,
    max_cn_seq_distance: float = 2.0,
    clash_distance: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    See `filter_unphysical_traj` for more details.
    """
    # CA-CA residue distance between sequential neighbouring pairs.
    seq_contiguous_resid_pairs = np.array(
        [(r.index, r.index + 1) for r in list(traj.topology.residues)[:-1]]
    )

    ca_seq_distances, _ = md.compute_contacts(
        traj, scheme="ca", contacts=seq_contiguous_resid_pairs, periodic=False
    )
    ca_seq_distances = md.utils.in_units_of(ca_seq_distances, "nanometers", "angstrom")
    frames_match_ca_seq_distance = np.all(ca_seq_distances < max_ca_seq_distance, axis=1)

    # C-N distance between sequential neighbouring pairs.
    cn_atom_pair_indices: list[tuple[int, int]] = []

    for resid_i, resid_j in seq_contiguous_resid_pairs:
        residue_i, residue_j = (
            traj.topology.residue(resid_i),
            traj.topology.residue(resid_j),
        )
        c_i, n_j = (
            list(residue_i.atoms_by_name("C")),
            list(residue_j.atoms_by_name("N")),
        )
        assert len(c_i) == len(n_j) == 1
        cn_atom_pair_indices.append((c_i[0].index, n_j[0].index))

    assert cn_atom_pair_indices

    cn_seq_distances = md.compute_distances(traj, cn_atom_pair_indices, periodic=False)
    cn_seq_distances = md.utils.in_units_of(cn_seq_distances, "nanometers", "angstrom")

    frames_match_cn_seq_distance = np.all(cn_seq_distances < max_cn_seq_distance, axis=1)

    # Clashes between any two atoms from different residues
    # Maximum cutoff we need to look at is maximal distance of a frame atom from Ca (O with Ca-O
    # ~ 2.4 Angstrom, 2.5 with small buffer for safety) times two plus the clash limit we set.
    maximal_ca_with_clash = 2 * 2.5 + clash_distance
    rest_distances, _ = _compute_filtered_contacts(
        traj, ca_clash_cutoff=maximal_ca_with_clash, sequence_separation=2
    )
    rest_distances = md.utils.in_units_of(rest_distances, "nanometers", "angstrom")
    frames_non_clash = np.all(
        rest_distances > clash_distance,
        axis=1,
    )
    # return frames_match_ca_seq_distance, frames_match_cn_seq_distance, frames_non_clash
    return (
        ca_seq_distances < max_ca_seq_distance,
        cn_seq_distances < max_cn_seq_distance,
        rest_distances > clash_distance,
    )


def compute_per_frame_validity_mask(
    traj: Trajectory,
    tau: float = 0.9,
    mode: str = "all_and",
) -> np.ndarray:
    """Per-frame validity mask under a threshold τ.

    mode="ca_only": frame valid if (frac CA pairs OK) ≥ τ. Legacy gate used by
                   `eval_tica._per_frame_q` and `diversity_quality_curve.py`.
    mode="all_and": frame valid only if all three per-residue compliance rates
                   (CA, CN, Clash) are ≥ τ. The unified paper gate.
    """
    frames_ca, frames_cn, frames_clash = filter_unphysical_traj_masks(traj)
    p_ca = frames_ca.mean(axis=1)
    if mode == "ca_only":
        return p_ca >= tau
    if mode == "all_and":
        p_cn = frames_cn.mean(axis=1)
        p_clash = frames_clash.mean(axis=1)
        return (p_ca >= tau) & (p_cn >= tau) & (p_clash >= tau)
    raise ValueError(f"unknown gate mode: {mode!r}; expected 'ca_only' or 'all_and'")


def do_job(params):
    args, row, gen_path = params
    pdb_id = row['pdb_id']
    traj_path = row['traj_path']
    pdb_path = row['pdb_path']
    task_path = os.path.join(
        args.save_dir, args.jobs_name, pdb_id + ".pkl"
    )
    if os.path.exists(task_path):
        result = load_pickle(task_path)
    else:
        if (gen_path is None) or (not os.path.exists(gen_path)):
            return
        diversity_score, precision, recall = process_protein(
            pdb_path,
            traj_path,
            gen_path,
            gen_sample_stride=args.gen_sample_stride,
            md_stride=args.md_stride,
            batch_size=args.batch_size,
            device=args.device,
            frame_cut_off=args.frame_cut_off,
            gen_frame_cut_off=args.gen_frame_cut_off
        )

        result = {
            "pdb_id": pdb_id,
            "diversity_score": diversity_score,
            "precision": precision,
            "recall": recall,
        }
        traj = md.load(gen_path)
        if args.gen_frame_cut_off != -1:
            traj = traj[: args.gen_frame_cut_off]
        ic(len(traj))
        frames_ca, frames_cn, frames_clash = filter_unphysical_traj_masks(
            traj=traj,
            max_ca_seq_distance=4.5,
            max_cn_seq_distance=2.0,
            clash_distance=1.0,
        )

        # ============ per-frame pass rate (within protein) ============
        validated_ca = frames_ca#.mean(axis=-1)  # [F，N]
        validated_cn = frames_cn#.mean(axis=-1)
        validated_clash = frames_clash#.mean(axis=-1)

        result["validated_ca"] = validated_ca
        result["validated_cn"] = validated_cn
        result["validated_clash"] = validated_clash
        save_pickle(result, task_path)
    return result


def post_process_df(df):
    cols = ["validated_ca", "validated_cn", "validated_clash"]
    print()
    for col in cols:
        def summarize(arr):
            if arr is None:
                return np.nan
            if np.size(arr) == 0:
                # An empty clash candidate set means no detected clashes.
                return 1.0 if col == "validated_clash" else np.nan
            return np.mean(arr)
        df[col] = df[col].apply(summarize)
    return df

def main(args):
    df = pd.read_csv(args.input_csv)
    df = df[args.rank_idx::args.world_size]
    print(df)
    pdb_to_path_mapper = build_pdb_to_path(args.gen_root)
    results = []
    jobs = [
        (args, row, pdb_to_path_mapper.get(row["pdb_id"], None))
        for _, row in df.iterrows()
    ]

    jobs_dir = os.path.join(args.save_dir, args.jobs_name)
    os.makedirs(jobs_dir, exist_ok=True)
    if args.num_workers>1:
        p = Pool(args.num_workers)
        p.__enter__()
        __map__ = p.imap
    else:
        __map__ = map


    for result in tqdm(__map__(do_job, jobs), total=len(jobs)):
        if result is None:
            continue
        results.append(result)

    if args.num_workers > 1:
        p.__exit__(None, None, None)

    if args.world_size==1:
        save_path = os.path.join(args.save_dir, args.jobs_name + ".pkl")
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        results_df = pd.DataFrame(results)
        save_path = os.path.join(args.save_dir, args.jobs_name + ".csv")
        results_df.to_csv(save_path, index=False)
        results_df = post_process_df(results_df)
        print(f"Results saved to {save_path}")
        mean_df = results_df.select_dtypes(include="number").mean().to_frame().T
        mean_df.insert(0, "name", args.jobs_name)
        mean_df.to_csv(os.path.join(args.save_dir, args.jobs_name + "_mean.csv"),index=False,)
        print(mean_df)
    else:
        print("Please run again & set world size=1 to get the final metric")

if __name__ == "__main__":
    args = parse_args()
    main(args)
