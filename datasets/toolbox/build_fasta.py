import os
import pandas as pd
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_csv',
        type=str,
        default='datasets/train/train_data.csv',
        help='Input CSV file containing protein sequences'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default='datasets/fasta',
        help='Output directory to save FASTA files'
    )
    return parser.parse_args()


def write_fasta(path, pdb_id, sequence):
    """
    Write a single protein sequence to a FASTA file.
    """
    with open(path, 'w') as f:
        f.write(f">{pdb_id}\n")
        f.write(f"{sequence}\n")


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    for idx, row in df.iterrows():
        pdb_id = row['pdb_id']
        sequence = row['sequence']

        save_path = os.path.join(args.outdir, f"{pdb_id}.fasta")
        write_fasta(save_path, pdb_id, sequence)
