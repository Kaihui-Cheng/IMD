"""Animate unbiased and biased trajectories in one MD-fitted TICA space.

Inputs: .npy arrays (frames, >=2), or .npz files containing ``proj`` and
an optional boolean ``valid`` mask. All inputs must use the same TICA model.
Optional multi-model PDBs add synchronized C-alpha backbone views.
"""

import argparse
from pathlib import Path
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


def load_projection(path):
    data = np.load(path, allow_pickle=False, mmap_mode="r")
    if isinstance(data, np.lib.npyio.NpzFile):
        xy = data["proj"]
        valid = data["valid"].astype(bool) if "valid" in data else np.ones(len(xy), bool)
        data.close()
    else:
        xy, valid = data, np.ones(len(data), bool)
    if xy.ndim != 2 or xy.shape[1] < 2 or valid.shape != (len(xy),):
        raise ValueError(f"Expected a (frames, >=2) projection and a matching validity mask: {path}")
    xy = np.asarray(xy[:, :2], dtype=float)
    valid &= np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    if valid.sum() < 2:
        raise ValueError(f"At least two finite, valid samples are required: {path}")
    return xy, valid


def load_ca_backbone(path, n_frames):
    """Read the first original PDB models as ordered C-alpha coordinates in Å."""
    frames, coordinates, identities = [], [], []
    expected = None
    with path.open() as handle:
        for line in handle:
            if line.startswith("ATOM  ") and line[12:16].strip() == "CA" and line[16] in " A":
                coordinates.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                identities.append(line[17:27])
            elif line.startswith("ENDMDL"):
                if not coordinates:
                    raise ValueError(f"PDB model contains no C-alpha atoms: {path}")
                if expected is None:
                    expected = tuple(identities)
                if tuple(identities) != expected:
                    raise ValueError(f"C-alpha residue order changes between PDB models: {path}")
                frames.append(coordinates)
                coordinates, identities = [], []
                if len(frames) == n_frames:
                    break
    if len(frames) != n_frames:
        raise ValueError(f"Expected at least {n_frames} original PDB models: {path}")
    xyz = np.asarray(frames, dtype=float)
    if not np.isfinite(xyz).all():
        raise ValueError(f"Non-finite C-alpha coordinates: {path}")
    return xyz, expected


def align_backbones(backbones):
    """Align both runs to the first baseline frame, using proper rotations only."""
    reference = backbones[0][0] - backbones[0][0].mean(axis=0)
    _, _, basis = np.linalg.svd(reference, full_matrices=False)
    basis[-1] *= np.linalg.det(basis)
    reference = reference @ basis.T
    aligned = []
    for xyz in backbones:
        centered = xyz - xyz.mean(axis=1, keepdims=True)
        u, _, vt = np.linalg.svd(centered.transpose(0, 2, 1) @ reference)
        u[:, :, -1] *= np.linalg.det(u @ vt)[:, None]
        aligned.append(centered @ (u @ vt))
    return aligned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="MD reference TICA projection")
    parser.add_argument("--baseline", type=Path, required=True, help="Unbiased TICA projection")
    parser.add_argument("--biased", type=Path, required=True, help="Biased TICA projection")
    parser.add_argument("--baseline-pdb", type=Path, help="Unbiased multi-model PDB, in projection frame order")
    parser.add_argument("--biased-pdb", type=Path, help="Biased multi-model PDB, in projection frame order")
    parser.add_argument("--protein", default="Protein")
    parser.add_argument("--alpha", type=float, default=0.1, help="Bias strength label")
    parser.add_argument("--refinement-start-t", type=float, help="Label pre-refined inputs with their noise level; does not run refinement")
    parser.add_argument("--max-frames", type=int, default=2048, help="Maximum original generated frames to show (default: 2048)")
    parser.add_argument("--stride", type=int, default=32, help="Original frames revealed per video frame")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo/tica_demo.mp4"))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--preview", action="store_true", help="Save the final distribution as PNG only")
    args = parser.parse_args()
    if args.stride < 1 or args.fps < 1 or args.max_frames < 2:
        parser.error("Stride and FPS must be positive; max-frames must be at least 2.")
    if args.refinement_start_t is not None and not 0 < args.refinement_start_t <= 1:
        parser.error("The refinement noise-level label must be in (0, 1].")
    if (args.baseline_pdb is None) != (args.biased_pdb is None):
        parser.error("Provide both --baseline-pdb and --biased-pdb for synchronized structures.")
    reference, ref_valid = load_projection(args.reference)
    if not ref_valid.all():
        reference = reference[ref_valid]
    trajectories = [load_projection(args.baseline), load_projection(args.biased)]
    n_frames = len(trajectories[0][0])
    if len(trajectories[1][0]) != n_frames:
        parser.error("Baseline and biased projections must have the same frame count.")
    n_frames = min(n_frames, args.max_frames)
    trajectories = [(xy[:n_frames], valid[:n_frames]) for xy, valid in trajectories]
    structures = None
    if args.baseline_pdb is not None:
        loaded = [load_ca_backbone(path, n_frames) for path in [args.baseline_pdb, args.biased_pdb]]
        if loaded[0][1] != loaded[1][1]:
            parser.error("Baseline and biased PDBs must have the same ordered C-alpha residues.")
        structures = align_backbones([item[0] for item in loaded])

    # Both panels use the same MD density, fixed axes, and generated-frame clock.
    samples = [reference] + [xy[valid] for xy, valid in trajectories]
    low = np.array([min(xy[:, i].min() for xy in samples) for i in range(2)])
    high = np.array([max(xy[:, i].max() for xy in samples) for i in range(2)])
    pad = np.maximum((high - low) * 0.05, 1e-6)
    low, high = low - pad, high + pad
    edges = [np.linspace(low[i], high[i], 81) for i in range(2)]
    counts = np.zeros((80, 80))
    for start in range(0, len(reference), 8192):
        chunk = reference[start:start + 8192]
        counts += np.histogram2d(chunk[:, 0], chunk[:, 1], bins=edges)[0]
    density = np.log1p(gaussian_filter(counts.T, sigma=1))
    centers = [(edge[:-1] + edge[1:]) / 2 for edge in edges]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    if structures is None:
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), dpi=150, sharex=True, sharey=True)
        fig.subplots_adjust(left=0.075, right=0.91, bottom=0.20, top=0.80, wspace=0.15)
    else:
        fig = plt.figure(figsize=(12.8, 7.2), dpi=150)
        # Large square TICA panels lead each comparison column; structures sit below.
        axes = [fig.add_axes([left, 0.36, 0.29, 0.29 * 16 / 9]) for left in (0.12, 0.55)]
    title = "TICA trajectory evolution" if structures is None else "TICA and conformation"
    fig.suptitle(f"{args.protein} · {title}", y=0.95, fontsize=21)
    colors = ["#596579", "#d94d3c"]
    artists = []
    for ax, title, color in zip(axes, ["Without bias (α = 0)", f"With bias (α = {args.alpha:g})"], colors):
        background = ax.contourf(*centers, density, levels=np.linspace(0, density.max(), 12), cmap="Blues", alpha=0.65)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel="TIC 1", title=title)
        ax.set_box_aspect(1)
        if structures is not None:
            ax.set_ylabel("TIC 2")
        ax.spines[["top", "right"]].set_visible(False)
        visits = ax.scatter([], [], s=7, color=color, alpha=0.25, edgecolors="none", zorder=3)
        trail, = ax.plot([], [], color=color, lw=1.4, alpha=0.8, zorder=4)
        current = ax.scatter([], [], s=70, color=color, edgecolors="white", linewidths=1.2, zorder=5)
        artists.append((visits, trail, current))
    axes[0].set_ylabel("TIC 2")
    colorbar_bounds = [0.93, 0.27, 0.012, 0.40] if structures is None else [0.91, 0.41, 0.012, 0.42]
    cax = fig.add_axes(colorbar_bounds)
    fig.colorbar(background, cax=cax, label="MD log density", format="%.1f")
    counter = fig.text(0.5, 0.055 if structures is not None else 0.13, "", ha="center", fontsize=12)
    footer = "MD background · Dots: visited samples · Line: recent path · "
    footer += ("Protein: same frame as the large TICA marker" if structures is not None
               else "Large dot: latest valid sample")
    fig.text(0.5, 0.018 if structures is not None else 0.075, footer, ha="center", fontsize=9)
    protein_artists = []
    if structures is not None:
        radius = 1.05 * max(np.abs(xyz[valid]).max() for xyz, (_, valid) in zip(structures, trajectories))
        for row, color in enumerate(colors):
            ax = fig.add_axes([0.12 + row * 0.43, 0.09, 0.29, 0.23], projection="3d")
            ax.set(xlim=(-radius, radius), ylim=(-radius, radius), zlim=(-radius, radius))
            ax.set_box_aspect((1, 1, 1), zoom=1.45)
            ax.set_proj_type("ortho")
            ax.view_init(elev=20, azim=-60)
            ax.set_axis_off()
            ax.patch.set_alpha(0)
            backbone, = ax.plot([], [], [], color=color, lw=3, solid_capstyle="round")
            label = ax.text2D(0.5, 0.02, "", transform=ax.transAxes, ha="center", fontsize=10)
            protein_artists.append((backbone, label))

    def update(stop):
        refinement_label = (f"Post-refined (t = {args.refinement_start_t:g}) · "
                            if args.refinement_start_t is not None else "")
        counter.set_text(f"{refinement_label}Generated frames: {stop:,} / {n_frames:,}")
        for row, ((xy, valid), (visits, trail, current)) in enumerate(zip(trajectories, artists)):
            indices = np.flatnonzero(valid[:stop])
            visited = xy[indices]
            visits.set_offsets(visited)
            recent = xy[max(0, stop - 128):stop].copy()
            recent[~valid[max(0, stop - 128):stop]] = np.nan
            trail.set_data(recent[:, 0], recent[:, 1])
            current.set_offsets(visited[-1:] if len(visited) else np.empty((0, 2)))
            if structures is not None:
                backbone, label = protein_artists[row]
                if len(indices):
                    index = indices[-1]
                    backbone.set_data_3d(*structures[row][index].T)
                    label.set_text(f"Cα backbone · frame {index + 1:,}")
                else:
                    backbone.set_data_3d([], [], [])
                    label.set_text("No valid frame yet")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    update(n_frames)
    fig.savefig(args.output.with_suffix(".png"))
    if not args.preview:
        stops = list(range(args.stride, n_frames, args.stride)) + [n_frames]
        stops = [1] * args.fps + stops + [n_frames] * args.fps
        with tempfile.TemporaryDirectory(prefix="imd-tica-") as directory:
            for index, stop in enumerate(stops):
                update(stop)
                fig.savefig(Path(directory) / f"{index:07d}.png", dpi=150)
            subprocess.run([
                args.ffmpeg, "-y", "-v", "error", "-nostdin", "-cpuflags", "0",
                "-threads", "1", "-filter_threads", "1", "-filter_complex_threads", "1",
                "-framerate", str(args.fps), "-i", str(Path(directory) / "%07d.png"),
                "-frames:v", str(len(stops)), "-c:v", "libx264", "-preset", "ultrafast",
                "-x264-params", "asm=0", "-threads", "1", "-crf", "18", "-tune", "zerolatency",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
            ], check=True, stdin=subprocess.DEVNULL)
        print(f"Saved {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
