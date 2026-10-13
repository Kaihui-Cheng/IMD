# Figures and demo media

## Figure 1

`figure1.png` is the unmodified framework figure from:

Kaihui Cheng, Zhiqiang Cai, Wenkai Xiang, Zhihang Hu, Siyu Zhu, Tzuhsiung Yang,
and Yuan Qi. *Learning Implicit Bias in Generative Spaces for Accelerating Protein
Dynamics Emulation*. arXiv:2606.01833v2, 2026.

- [Paper and Figure 1](https://arxiv.org/html/2606.01833v2#S2.F1)
- [Original PNG](https://arxiv.org/html/2606.01833v2/pipeline.png)
- Copyright: the paper authors, 2026. License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The figure's license is separate from the repository's Apache-2.0 code license.

## TICA demo

The demos compare **A3D, WW domain (GTT), and Protein G (NuG2)**, each without
bias (α=0) and with bias (α=0.1). For each protein, both runs use the same
two-dimensional MD-fitted TICA plane. Each animation shows the first 2,048
generated frames, with unnormalized history weights (σ=1, H=3). **Both runs are
post-refined at t=0.2** using the frozen environment model before visualization.
The TICA projections are freshly computed from these refined structures.

- Background: the same MD reference distribution and fixed axis limits.
- Small dots: all valid samples visited so far.
- Line: the most recent 128 original frames, with gaps at invalid frames.
- Large dot: the latest valid sample.
- Protein below each TICA panel: the Cα backbone of that same original frame.

The left column shows the unbiased run; the right column shows the biased run.
Each TICA plot uses a square plotting box, with its synchronized backbone below.
Both backbones are centered and aligned by proper Kabsch rotations to the first
unbiased structure. Its principal axes define the initial orientation. The
camera and coordinate scale stay fixed and shared between both columns, so overall
rotation and translation do not obscure conformational changes.

Each MP4 is 1920 × 1080 at 20 fps (H.264, CRF 18). Each animation update reveals
32 original frames; one-second holds at the start and end give a 5.2-second video.
Each GIF is a 1280-pixel-wide preview; download the MP4 for full resolution.
Playback follows generated-frame index and does not show a physical timescale
or a runtime speedup.

## Run the script

Use the IMD Python environment and FFmpeg with `libx264`:

```bash
python scripts/animate_tica.py \
  --reference path/to/md_tica.npy \
  --baseline path/to/no_bias_refined_tica.npz \
  --biased path/to/bias_refined_tica.npz \
  --baseline-pdb path/to/no_bias_refined.pdb \
  --biased-pdb path/to/bias_refined.pdb \
  --protein A3D --alpha 0.1 --max-frames 2048 --refinement-start-t 0.2 \
  --output outputs/demo/tica_demo.mp4
```

Each input is a `.npy` array with shape `(frames, >=2)`, or a `.npz` file with
`proj` of that shape and an optional boolean `valid` mask of shape `(frames,)`.
The two generated trajectories must have the same frame count.
**Project all three inputs using the same TICA model fitted to the MD reference.**
The script reads projections; it does not fit separate TICA models to the runs.
For refined inputs, project the **refined coordinates**, retaining original frame
order and using the same MD-fitted model. Do not pair refined PDBs with raw
trajectory projections.

The optional PDB inputs must be multi-model trajectories with `ENDMDL` records,
in the **original projection frame order**, including frames marked invalid.
Both must contain the same ordered Cα residues and at least the displayed number
of frames. The script reads only that prefix. Invalid frames are skipped in both
views; the protein and large TICA marker retain the latest valid frame together.
Omit both PDB arguments for a TICA-only animation.

`--max-frames` limits playback to the first original generated frames (default: 2048).
`--preview` writes only the final distribution PNG. `--stride` controls how many
original frames each animation step reveals; `--fps` controls playback rate.
Use `--ffmpeg /path/to/ffmpeg` to select an encoder installation.
`--refinement-start-t 0.2` labels inputs that have already been post-refined;
this option does not perform refinement. Omit it for unrefined inputs.

## Data and rendering settings

The demos use post-refined rollouts for both α=0 and α=0.1. Each 16-frame
segment receives frozen-environment noise-then-denoise refinement at start_t=0.2,
preserving the original frame order. This is post-processing of existing rollouts;
the displayed frames do not feed back into later sampling.

Each run uses its protein-specific MD-fitted TICA model, with the lags below.
Fresh projections use Cα pair distances in nm for residue separations ≥8.
Validity masks come from the matching refined caches and require CA, CN, and
clash-free rates ≥0.9.

| Protein (dataset ID) | MD TICA lag |
| --- | --- |
| A3D | 2500 |
| WW domain (GTT) | 4000 |
| Protein G (NuG2) | 4000 |

Masking preserves original frame indices; all valid samples are displayed,
without density-based filtering or coordinate interpolation. For these demos,
all 12,288 displayed-prefix refined structure frames were reprojected through
their original MD-fitted TICA models and checked against the matching refined
caches. Rigid alignment preserves Cα distances.

The background is `log(1 + count)` on an 80 × 80 MD histogram, smoothed with a
Gaussian of one bin for display. The colorbar represents log density, not
calibrated free energy; its tick labels retain one decimal place. These are
qualitative comparisons of existing runs; paired noise was not verified.

Model weights, raw trajectories and projection arrays are not distributed here.
The generated demo media are released under Apache-2.0.
