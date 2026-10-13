# Third-party code

Original file-level notices are retained. IMD modifications are covered by the
project license; upstream code retains its original copyright and license.

| Source | Included or adapted files | License |
| --- | --- | --- |
| [OpenFold](https://github.com/aqlaboratory/openfold) | Vendored subset under `openfold/`; IPA-related components in `src/model/modules.py` | [Apache-2.0](LICENSES/openfold.txt) |
| [AlphaFold](https://github.com/google-deepmind/alphafold) | AlphaFold-derived files in `openfold/`, `src/data/protein.py`, and `src/data/residue_constants.py` | [Apache-2.0](LICENSES/alphafold.txt) |
| [Wan2.2](https://github.com/Wan-Video/Wan2.2) | `src/model/attention.py` and adaptations in `src/model/causal_modules.py` | [Apache-2.0](LICENSE) |
| [Self-Forcing](https://github.com/guandeh17/Self-Forcing) | Adaptations in `src/model/causal_modules.py` | [Apache-2.0](LICENSE) |
| [BioEmu](https://github.com/microsoft/bioemu) | `src/data/foldness.py`; contact-filtering logic adapted in `metrics/structural_metric.py` | [MIT](LICENSES/bioemu.txt) |

Wan and Self-Forcing supply attention utilities only; their video models and
pretrained weights are not included. These utilities use the Apache-2.0 license
provided in [LICENSE](LICENSE), with upstream source and copyright notices retained.

The exact historical upstream revisions of the inherited source snapshot were
not recorded. The links identify the upstream projects, rather than asserting
that this snapshot matches their current main branches.

Vendored OpenFold modifications in this release make DeepSpeed optional, retaining
PyTorch activation checkpointing and precision handling when it is unavailable.
The vendored subset supplies structural utilities for IMD; it is not a complete
OpenFold training or Evoformer inference distribution. Install upstream OpenFold
separately to generate sequence representations.

Model weights, training data, and external databases are distributed separately
and are not licensed by this repository's software license.
