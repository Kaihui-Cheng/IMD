import os
import numpy as np
import pandas as pd
import argparse


def add_coodination_path(row, source_coordination_dir):
    coodination_path = os.path.abspath(
        os.path.join(source_coordination_dir, row["pdb_id"] + ".npz")
    )
    if not os.path.exists(coodination_path):
        print(f"lack coor:{coodination_path}")
        return None
    return coodination_path


def add_embedding_path(row, source_embeddingd_dir):
    embedding_path = os.path.abspath(
        os.path.join(source_embeddingd_dir, row["pdb_id"] + ".npz")
    )
    if not os.path.exists(embedding_path):
        print("lack embeds ", row["pdb_id"])
        return None
    return embedding_path


def main(args):
    df = pd.read_csv(args.input_csv)
    df["embed_path"] = df.apply(lambda row: add_embedding_path(row, args.embed_dir), axis=1)
    df["pos_path"] = df.apply(lambda row: add_coodination_path(row, args.coordination_npz_dir), axis=1)
    df.to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    parsers = argparse.ArgumentParser()
    parsers.add_argument("--input_csv", type=str, default="datasets/train/train_data.csv")
    parsers.add_argument("--output_csv", type=str, default="datasets/train/train_data_final.csv")
    parsers.add_argument("--coordination_npz_dir",type=str, required=True)
    parsers.add_argument("--embed_dir",type=str, required=True)
    args = parsers.parse_args()

    main(args)
