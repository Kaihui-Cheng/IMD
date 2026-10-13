# Initial release scope

This is an independently initialized public repository extracted from the current
DyneTrion working tree. It contains the AlphaFold/OpenFold-based environment,
supervised IMD actor, sampling, data preparation, and structural evaluation.

Excluded: weights, embeddings, trajectories, experiment results, local logs,
notebooks and scratch files, manuscript drafts, private agent settings, abandoned
GRPO/PPO training, the experimental DMD training branch, alternative backends,
and study-specific TICA/FES analysis and post-processing refinement scripts.
Figure 1, a rendered TICA demo, and a small script for animating precomputed
TICA projections are included; the underlying experiment trajectories and model
weights are not included.

Publication changes:

- Extracted supervised training from the shared experimental trainer and removed
  the study-specific endpoint-guidance environment hook from sampling.
- Renamed `rl_actor.py` to `actor.py`, `rl_pair_loader.py` to `sampling.py`,
  `RLPairDataset` to `ProteinSamplingDataset`, and actor configs to
  `actor_base.yaml` / `actor_train.yaml`.
- Retained the existing network parameter names and numerical sampling logic.
- Extracted shared weighted Kabsch alignment from the former RL rewards file.
- Set default history weighting to unnormalized, sigma 1, history length 3.
- Replaced machine-specific defaults with documented relative paths.
- Fixed coordinate preprocessing to read `--datacsv` consistently.
- Handled empty clash-contact candidate sets in structural evaluation.
- Made vendored OpenFold's DeepSpeed import optional; documented the changes.
- Added Python packaging, dependency pins, documentation, examples, licenses,
  contribution guidance, CI, and checkpoint-free tests.

The code release does not claim reproduction of paper metrics. Real-data training
and pretrained-checkpoint sampling require separately supplied artifacts.
