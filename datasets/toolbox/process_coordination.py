import argparse
import sys
import mdtraj, os, tempfile, tqdm
import pandas as pd
from multiprocessing import Pool,cpu_count
import numpy as np


import numpy as np
from openfold.np import protein
from openfold.data.data_pipeline import make_protein_features


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datacsv', type=str, default='datasets/train/check_train_data.csv')
    parser.add_argument('--outdir', type=str, default='datasets/train/coordination')
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    parser.add_argument('--groups', type=int, default=1)
    parser.add_argument('--group_idx', type=int, default=0)
    parser.add_argument('--stride', type=int, default=10)
    args = parser.parse_args()
    return args



def do_job(job_tuple):
    name, pdb_path, traj_path, save_path,stride = job_tuple
    if os.path.exists(save_path):
        return
    traj = mdtraj.load(traj_path,top=pdb_path)
    f, temp_path = tempfile.mkstemp(); os.close(f)
    positions_stacked = []
    for i in tqdm.trange(0, len(traj), stride):
        traj[i].save_pdb(temp_path)
        with open(temp_path) as f:
            prot = protein.from_pdb_string(f.read())
            pdb_feats = make_protein_features(prot, name)
            positions_stacked.append(pdb_feats['all_atom_positions'])
    pdb_feats['all_atom_positions'] = np.stack(positions_stacked)
    np.savez(save_path, **pdb_feats)
    os.unlink(temp_path)


def main(args, df):
    jobs = []
    for idx, row in df.iterrows():
        name = row['pdb_id']
        jobs.append((name,row['pdb_path'],row['traj_path'], os.path.join(args.outdir, f"{name}.npz"),args.stride))

    if args.num_workers > 1:
        p = Pool(args.num_workers)
        p.__enter__()
        __map__ = p.imap
    else:
        __map__ = map
    for _ in tqdm.tqdm(__map__(do_job, jobs), total=len(jobs)):
        pass
    if args.num_workers > 1:
        p.__exit__(None, None, None)


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.datacsv)
    df = df[args.group_idx::args.groups]
    main(args, df)
