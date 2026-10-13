# Data preparation

Run commands from the repository root after installing IMD. Paths in CSV cells
are resolved against the process working directory; use absolute paths if you
will launch from another directory. A CSV row describes one protein. Ensure all
structures and embeddings have the same residue order and sequence length.

## Input CSVs

| Purpose | Required columns | Template |
| --- | --- | --- |
| Sequence/trajectory preprocessing | `pdb_id,seq_len,sequence,pdb_path,traj_path` | `examples/raw.csv` |
| Supervised training and validation | `pdb_id,seq_len,sequence,embed_path,pos_path` | `examples/train.csv` |
| Sampling | `pdb_id,seq_len,pdb_path,embed_path` | `examples/inference.csv` |
| Structural evaluation | `pdb_id,pdb_path,traj_path` | `examples/raw.csv` |

Do not use the placeholder rows as real data. Supply a single protein chain with
complete backbone atoms. Sequence length filtering is strict (`seq_len <
max_res_len`); raise the configured limit to include longer proteins.

## AlphaFold/OpenFold representations

Generate FASTA files:

```bash
python -m datasets.toolbox.build_fasta \
  --input_csv data/raw.csv --outdir data/fasta
```

Use an independent installation of [upstream OpenFold](https://github.com/aqlaboratory/openfold)
to obtain Evoformer single and pair representations, using the `finetuning_ptm_2`
setup used by the original DyneTrion workflow. Save each protein's output as:

```python
import numpy as np

# out is the output dictionary from upstream OpenFold inference.
np.savez("data/embeddings/protein_id.npz",
         single=out["single"].detach().cpu().numpy(),
         pair=out["pair"].detach().cpu().numpy())
```

Remove any batch/recycle dimensions before saving. Expected shapes are
`single: [N,384]`, `pair: [N,N,128]` for the supplied IMD model configuration.
The aliases `node_repr` and `edge_repr` are also accepted by the loaders. This
release does not include AlphaFold/OpenFold model weights or sequence databases.

## Coordinates

```bash
python -m datasets.toolbox.process_coordination \
  --datacsv data/raw.csv \
  --outdir data/coordinates \
  --num_workers 1 --groups 1 --group_idx 0 --stride 10
```

This converts trajectories to AlphaFold atom37 features. Each `.npz` contains
`all_atom_positions: [T,N,37,3]` in Angstroms, `all_atom_mask: [N,37]`, and
sequence/residue features from `make_protein_features`.

The actor loader combines reference frames from the start of the trajectory with
a sampled window ordered as `[H history | M motion | F query]`. It requires at
least `1 + (H+M+F-1)*frame_sample_step` saved trajectory frames. With the default
H=3, M=2, F=16, stride=40 this is 801 saved frames. Preprocessing stride and
training stride multiply, so choose them according to the available trajectory.

## Final training CSV

```bash
python -m datasets.toolbox.build_final_csv \
  --input_csv data/raw.csv --output_csv data/train.csv \
  --coordination_npz_dir data/coordinates --embed_dir data/embeddings
```

Prepare `data/val.csv` separately without leaking validation proteins into the
training set. This utility reports missing feature files; resolve missing paths
before training. Sampling starts from a PDB and requires no MD trajectory file.
