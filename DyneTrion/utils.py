
from openfold.utils.validation_metrics import drmsd
import torch
import numpy as np
import gc
import matplotlib.pyplot as plt
from openfold.np import residue_constants#
from openfold.utils.superimposition import superimpose
import MDAnalysis as mda
from MDAnalysis.analysis import rms,align
from matplotlib.ticker import FuncFormatter
from einops import rearrange
import os

def format_func(value, tick_number):
    return f"{value:.1f}"
formatter = FuncFormatter(format_func)


@torch.no_grad()
def compute_validation_metrics_all(
    gt_coords,  # [B, F, N, 37, 3]
    pred_coords,  # [B, F, N, 37, 3]
    all_atom_mask,  # [B, F, N, 37]
):
    metrics = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_tensor(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x.to(device)

    gt_coords = to_tensor(gt_coords)
    pred_coords = to_tensor(pred_coords)
    all_atom_mask = to_tensor(all_atom_mask)

    ca_pos = residue_constants.atom_order["CA"]
    gt_ca = gt_coords[..., ca_pos, :]
    pred_ca = pred_coords[..., ca_pos, :]
    mask_ca = all_atom_mask[..., ca_pos]

    metrics["drmsd_ca"] = drmsd(pred_ca, gt_ca, mask=mask_ca).mean(dim=-1)

    # [B, F, N, 37, 3]
    diff_all_sq = (gt_coords - pred_coords) ** 2
    sum_sq_all = torch.sum(diff_all_sq * all_atom_mask[..., None], dim=(-1, -2, -3))
    count_all = torch.sum(all_atom_mask, dim=(-1, -2)) + 1e-10
    metrics["rmse_all"] = torch.sqrt(sum_sq_all / count_all).mean(dim=-1)


    diff_ca_sq = (gt_ca - pred_ca) ** 2
    sum_sq_ca = torch.sum(diff_ca_sq * mask_ca[..., None], dim=(-1, -2))
    count_ca = torch.sum(mask_ca, dim=-1) + 1e-10
    metrics["rmse_ca"] = torch.sqrt(sum_sq_ca / count_ca).mean(dim=-1)

    superimposed_pred, alignment_rmsd_ca = superimpose(
        gt_ca,
        pred_ca,
    )
    metrics["rmsd_ca_aligned"] = alignment_rmsd_ca.mean(dim=-1)

    return metrics

@torch.no_grad()
def plot_curve_merged(metric_merged, save_path, row_num=None, col_num=2, suffix=None):
    first_key = list(metric_merged.keys())[0]
    num_samples = len(metric_merged[first_key])
    if row_num is None:
        row_num = num_samples

    fig_width = col_num * 5
    fig_height = row_num * 4
    fig, axes = plt.subplots(
        row_num, col_num, figsize=(fig_width, fig_height), dpi=200, squeeze=False
    )

    plot_cols = ["rmse_all", "rmse_ca"]
    titles = ["RMSE (All)", "RMSE (CA)"]

    for label, df in metric_merged.items():
        for idx, (_, row) in enumerate(df.iterrows()):
            if idx >= row_num:
                break

            pdb_name = row.get("pdb_id", row.get("pdb_name", f"Sample {idx}"))

            for j in range(min(col_num, len(plot_cols))):
                ax = axes[idx, j]
                data_to_plot = row[plot_cols[j]]
                ax.plot(
                    data_to_plot,
                    label=label,
                    marker="o",
                    markersize=4,
                    linestyle="-",
                    alpha=0.8,
                )

                if label == first_key:
                    ax.set_title(f"{pdb_name} | {titles[j]}", fontsize=10)
                    ax.set_xlabel("Time Step / Frame")
                    ax.grid(True, linestyle="--", alpha=0.6)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=1)

    plt.suptitle("RMSD/RMSE Trajectory Comparison", fontsize=16, y=1.02)
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    file_name = f"rmse_rmsd_{suffix}.png" if suffix else "rmse_rmsd.png"
    save_full_path = os.path.join(save_path, file_name)

    plt.savefig(save_full_path, bbox_inches="tight")
    plt.close()

    return save_full_path

@torch.no_grad()
def plot_rot_trans_curve(error_dict, save_path, formatter=None):
    names = error_dict["name"]
    cols = len(names)
    rows = 2

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 4, rows * 4), dpi=300, squeeze=False
    )

    keys_to_plot = [
        ("ave_rot", "Rotation/°", "ARC_rot"),
        ("ave_trans", "Translation/Å", "ARC_trans_MSE"),
    ]

    for idx, name in enumerate(names):
        for r, (data_key, ylabel, traj_key) in enumerate(keys_to_plot):
            ax = axes[r, idx]
            pred_data = error_dict[data_key][idx].mean(0)
            x = np.arange(len(pred_data))
            ax.plot(x, pred_data, label="Pred", marker="o", linestyle="-", markersize=4)
            if r == 0:
                ax.set_title(name, fontsize=12)
            if idx == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            if r == 1 and formatter is not None:
                ax.yaxis.set_major_formatter(formatter)

            ax.grid(True, linestyle=":", alpha=0.6)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    save_full_path = os.path.join(save_path, "rot_trans_error.png")
    plt.savefig(save_full_path, bbox_inches="tight")
    plt.close()

    return save_full_path


@torch.no_grad()
def calculate_rmsf(pdb_file, reference_select="protein and name CA"):
    u = mda.Universe(pdb_file)
    atoms = u.select_atoms(reference_select)
    aligner = align.AlignTraj(u, atoms, select=reference_select, in_memory=True).run()
    atoms = u.select_atoms(reference_select)
    rmsf_analysis = rms.RMSF(atoms).run()
    return rmsf_analysis.rmsf


@torch.no_grad()
def average_quaternion_distances(q1, q2):
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)

    dot_product = np.einsum("...i,...i", q1, q2)
    dot_product = np.clip(np.abs(dot_product), -1.0, 1.0)

    angles = 2 * np.arccos(dot_product)
    angles_deg = np.degrees(angles)

    return np.mean(angles_deg, axis=-1)


def average_translation_distances(translations1, translations2, measurement="MSE"):
    if measurement == "MSE":
        average_distances = np.mean((translations1 - translations2) ** 2, axis=(-1, -2))
    else:
        average_distances = np.mean(
            np.abs(translations1 - translations2), axis=(-1, -2)
        )
    return average_distances
