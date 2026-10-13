import sys
import os
import copy
import gc
import logging
import pickle
import random
import time
from collections import defaultdict
from datetime import datetime

import GPUtil

import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import tree
from Bio.SVDSuperimposer import SVDSuperimposer
from icecream import ic
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils import data
import mdtraj as md

import swanlab
from tqdm import tqdm

from DyneTrion.utils import (
    compute_validation_metrics_all,
    residue_constants,
    plot_curve_merged,
    plot_rot_trans_curve,
    average_quaternion_distances,
    average_translation_distances,
)
from openfold.utils import rigid_utils as ru
from openfold.utils.loss import lddt_ca, torsion_angle_loss
from src.data import DyneTrion_data_loader_dynamic
from src.analysis import utils as au
from src.data import se3_diffuser, all_atom
from src.data import utils as du
from src.experiments import utils as eu
from src.model import network

from einops import rearrange, repeat

class Experiment:

    def __init__(
            self,
            *,
            conf: DictConfig,
        ):
        """Initialize experiment.

        Args:
            exp_cfg: Experiment configuration.
        """
        self._log = logging.getLogger(__name__)
        self._available_gpus = ''.join([str(x) for x in GPUtil.getAvailable(order='memory', limit = 8)])

        # Configs
        self._conf = conf
        self._exp_conf = conf.experiment
        if HydraConfig.initialized() and 'num' in HydraConfig.get().job:
            self._exp_conf.name = (f'{self._exp_conf.name}_{HydraConfig.get().job.num}')
        self._diff_conf = conf.diffuser
        self._model_conf = conf.model
        self._data_conf = conf.data
        self._use_recoder = self._exp_conf.use_recoder
        self._use_ddp = self._exp_conf.use_ddp
        self.dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
        # 1. initialize ddp info if in ddp mode
        if self._use_ddp :
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            self.ddp_info = eu.get_ddp_info()
            if self.ddp_info['rank'] not in [0,-1]:
                self._log.addHandler(logging.NullHandler())
                self._log.setLevel("ERROR")
                self._use_recoder = False
                self._exp_conf.ckpt_dir = None

        self.trained_epochs = 0
        self.trained_steps = 0

        # Initialize experiment objects
        self._diffuser = se3_diffuser.SE3Diffuser(self._diff_conf)
        self._model = network.DyneTrion(self._model_conf, self.diffuser)

        num_parameters = sum(p.numel() for p in self._model.parameters())

        if self._conf.model.use_temporal and self._conf.model.frozen_spatial:
            self._log.info('Frozen model and only train temporal module')
            # only train motion module
            for param in self._model.parameters():
                param.requires_grad = False
            for name, param in self._model.named_parameters():
                if 'temporal' in name: # 'frame'
                    param.requires_grad = True

        trainable_num_parameters = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        self._exp_conf.num_parameters = num_parameters
        self._exp_conf.trainable_num_parameters  = num_parameters
        self._log.info(f'Number of model parameters {num_parameters}, trainable parameters:{trainable_num_parameters}')
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._exp_conf.learning_rate,amsgrad=True)
        if conf.experiment.warm_start:
            ckpt_path = conf.experiment.warm_start
            _, optimizer_his, epoch, step = self.load_pretrianed_model(ckpt_path=ckpt_path)

            if conf.experiment.reuse_step:
                print('Reuse Step')
                self.trained_epochs = epoch
                self.trained_steps = step
                print(f"loading model from: {ckpt_path}")
            # self._optimizer.load_state_dict(optimizer_his)

        self._init_log()
        self._init_best_eval()
        if not self.conf.experiment.training:
            seed = 0
        else:
            seed = dist.get_rank()
        self._set_seed(seed)

    def _init_best_eval(self):
        self.best_trained_steps = 0
        self.best_trained_epoch = 0
        self.best_rmse_ca = 10000
        self.best_rmse_all = 10000
        self.best_drmsd = 10000
        self.best_rot_error=1000
        self.best_trans_error = 1000
        self.best_ref_rot_error = 1000
        self.best_ref_trans_error = 1000

    def _init_log(self):

        if self._exp_conf.ckpt_dir is not None:
            # Set-up checkpoint location
            ckpt_dir = os.path.join(
                self._exp_conf.ckpt_dir,
                # self._exp_conf.name,
                self.dt_string )
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir, exist_ok=True)
            self._exp_conf.ckpt_dir = ckpt_dir
            self._log.info(f'Checkpoints saved to: {ckpt_dir}')
        else:
            self._log.info('Checkpoint not being saved.')

        if self._exp_conf.eval_dir is not None :
            eval_dir = os.path.join(
                self._exp_conf.eval_dir,
                self._exp_conf.name,
                self.dt_string )
            self._exp_conf.eval_dir = eval_dir
            self._log.info(f'Evaluation saved to: {eval_dir}')
        else:
            self._exp_conf.eval_dir = os.devnull
            self._log.info(f'Evaluation will not be saved.')

    def load_pretrianed_model(self, ckpt_path):
        try:
            self._log.info(f'Loading checkpoint from {ckpt_path}')
            ckpt_pkl = torch.load(ckpt_path, map_location='cpu')

            if ckpt_pkl is not None and 'model' in ckpt_pkl:
                ckpt_model = ckpt_pkl['model']

                if ckpt_model is not None:
                    ckpt_model = {k.replace('module.', ''): v for k, v in ckpt_model.items()}
                    model_state_dict = self._model.state_dict()
                    # pretrained_dict = {k: v for k, v in ckpt_model.items() if k in model_state_dict}
                    pretrained_dict = {k: v for k, v in ckpt_model.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
                    model_state_dict.update(pretrained_dict)
                    self._model.load_state_dict(model_state_dict)
                    self._log.info(f'Warm starting from: {ckpt_path}')
                    # del ckpt_pkl,ckpt_model,pretrained_dict,model_state_dict
                    # gc.collect()
                    return ckpt_pkl['conf'], ckpt_pkl['optimizer'], ckpt_pkl['epoch'], ckpt_pkl['step']
                else:
                    self._log.error("Checkpoint model is None.")
                    return False
            else:
                self._log.error("Checkpoint or model not found in checkpoint file.")
                return False
        except Exception as e:
            self._log.error(f"Error loading checkpoint: {e}")
            return False


    @property
    def diffuser(self):
        return self._diffuser

    @property
    def model(self):
        return self._model

    @property
    def conf(self):
        return self._conf

    def create_dataset(self):

        # Datasets
        train_dataset = DyneTrion_data_loader_dynamic.PdbDataset(
            data_conf=self._data_conf,
            diffuser=self._diffuser,
            is_training=True
        )

        valid_dataset = DyneTrion_data_loader_dynamic.PdbDataset(
            data_conf=self._data_conf,
            diffuser=self._diffuser,
            is_training=False
        )
        # Loaders
        num_workers = self._exp_conf.num_loader_workers

        persistent_workers = True if num_workers > 0 else False

        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        train_loader = data.DataLoader(
                train_dataset,
                batch_size=self._exp_conf.batch_size if not self._exp_conf.use_ddp else self._exp_conf.batch_size // self.ddp_info['world_size'],
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=2 if num_workers > 0 else None,
                persistent_workers=persistent_workers,
                drop_last=False,
                sampler=sampler,
                multiprocessing_context='fork' if num_workers != 0 else None,
        )
        valid_loader = data.DataLoader(
                valid_dataset,
                batch_size=self._exp_conf.eval_batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=2 if num_workers > 0 else None,
                persistent_workers=persistent_workers,
                drop_last=False,
                multiprocessing_context='fork' if num_workers != 0 else None,
        )



        return train_loader, valid_loader

    def init_swanlab_logger(self):
        self._log.info("Initializing SwanLab Recoder.")
        conf_dict = OmegaConf.to_container(self._conf, resolve=True)
        swanlab_mode = conf_dict["experiment"]["recoder"]["mode"]
        if swanlab_mode == "cloud":
            swanlab.login(api_key=conf_dict["experiment"]["recoder"]["api_key"])
        self.swanlab_logger = swanlab.init(
            project=conf_dict["experiment"]["project"],
            experiment_name=conf_dict["experiment"]["name"],
            config=conf_dict,
            mode=swanlab_mode,
            logdir=conf_dict["experiment"]["recoder"]["save_path"],
        )

    def start_training(self, return_logs=False):
        # Set environment variables for which GPUs to use.
        if HydraConfig.initialized() and 'num' in HydraConfig.get().job:
            replica_id = int(HydraConfig.get().job.num)
        else:
            replica_id = 0
        if self._use_recoder and replica_id == 0:
            self.init_swanlab_logger()
        assert(not self._exp_conf.use_ddp or self._exp_conf.use_gpu)
        # GPU mode
        if torch.cuda.is_available() and self._exp_conf.use_gpu:
            if self._use_ddp:
                local_rank = self.ddp_info['local_rank']
                device = torch.device("cuda", local_rank)
                self._model = self.model.to(device)
                if self.ddp_info['world_size'] > 1:
                    self._model = DDP(
                        self._model, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=True,
                    )
                self._log.info(f"Using device: {device}, DDP local rank: {local_rank}")
            elif self._exp_conf.num_gpus == 1:
                gpu_id = self._available_gpus[replica_id]
                device = f"cuda:{gpu_id}"
                self._model = self.model.to(device)
                self._log.info(f"Using device: {device}")
            elif self._exp_conf.num_gpus > 1:
                device_ids = [f"cuda:{i}" for i in self._available_gpus[:self._exp_conf.num_gpus]]
                if len(self._available_gpus) < self._exp_conf.num_gpus:
                    raise ValueError(f"require {self._exp_conf.num_gpus} GPUs, but only {len(self._available_gpus)} GPUs available ")
                self._log.info(f"Multi-GPU training in DP mode: {device_ids}")
                gpu_id = self._available_gpus[replica_id]
                device = f"cuda:{gpu_id}"
                self._model = DP(self._model, device_ids=device_ids)
                self._model = self.model.to(device)
        else:
            device = 'cpu'
            self._model = self.model.to(device)
            self._log.info(f"Using device: {device}")

        # if self.conf.experiment.warm_start:
        #     for state in self._optimizer.state.values():
        #         for k, v in state.items():
        #             if torch.is_tensor(v):
        #                 state[k] = v.to(device)

        self._model.train()

        (train_loader,valid_loader) = self.create_dataset()

        logs = []
        # torch.cuda.empty_cache()
        for epoch in range(self.trained_epochs, self._exp_conf.num_epoch):
            self.trained_epochs = epoch
            train_loader.sampler.set_epoch(epoch)
            epoch_log = self.train_epoch(
                train_loader,
                valid_loader,
                device,
                return_logs=return_logs
            )
            # self._schedule.step()

            if return_logs:
                logs.append(epoch_log)
        if self._exp_conf.ckpt_dir is not None:
            ckpt_path = os.path.join(self._exp_conf.ckpt_dir, f'last_step_{self.trained_steps}.pth')
            du.write_checkpoint(
                ckpt_path,
                copy.deepcopy(self.model.state_dict()),
                self._conf,
                copy.deepcopy(self._optimizer.state_dict()),
                self.trained_epochs,
                self.trained_steps,
                logger=self._log,
                use_torch=True
            )
        self._log.info('Done')
        return logs

    def update_fn(self, data):
        """Updates the state using some data and returns metrics."""
        self._optimizer.zero_grad()
        loss, aux_data = self.loss_fn(data)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self._optimizer.step()

        return loss, aux_data

    def train_epoch(self, train_loader, valid_loader, device,return_logs=False):
        log_lossses = defaultdict(list)
        global_logs = []
        log_time = time.time()
        step_time = time.time()
        # Run evaluation

        for train_feats in train_loader:
            self.model.train()
            train_feats = tree.map_structure(lambda x: x.to(device), train_feats)
            # eval_dir = os.path.join(self._exp_conf.eval_dir, f'step_{self.trained_steps}')
            # os.makedirs(eval_dir, exist_ok=True)
            # results = self.eval_fn(
            #     eval_dir, valid_loader, device, noise_scale=self._exp_conf.noise_scale
            # )
            loss, aux_data = self.update_fn(train_feats)

            if return_logs:
                global_logs.append(loss)
            for k,v in aux_data.items():
                log_lossses[k].append(du.move_to_np(v))
            self.trained_steps += 1

            if self.trained_steps == 1 or self.trained_steps % self._exp_conf.log_freq == 0:
                elapsed_time = time.time() - log_time
                log_time = time.time()
                step_per_sec = self._exp_conf.log_freq / elapsed_time
                rolling_losses = tree.map_structure(np.mean, log_lossses)
                loss_log = ' '.join([
                    f'{k}={v[0]:.4f}'
                    for k,v in rolling_losses.items() if 'batch' not in k
                ])
                self._log.info(f'Epoch[{self.trained_epochs}/{self._exp_conf.num_epoch}] trained_steps: [{self.trained_steps}]: {loss_log}, steps/sec={step_per_sec:.5f}')
                log_lossses = defaultdict(list)

            # Take checkpoint
            if self._exp_conf.ckpt_dir is not None and ((self.trained_steps % self._exp_conf.ckpt_freq) == 0 or (self._exp_conf.early_ckpt and self.trained_steps == 100)):
                ckpt_path = os.path.join(self._exp_conf.ckpt_dir, f'step_{self.trained_steps}.pth')
                du.write_checkpoint(
                    ckpt_path,
                    copy.deepcopy(self.model.state_dict()),
                    self._conf,
                    copy.deepcopy(self._optimizer.state_dict()),
                    self.trained_epochs,
                    self.trained_steps,
                    logger=self._log,
                    use_torch=True
                )

                if self._exp_conf.enable_validation:
                    # Run evaluation
                    self._log.info(f'Running evaluation of {ckpt_path}')
                    start_time = time.time()
                    eval_dir = os.path.join(self._exp_conf.eval_dir, f'step_{self.trained_steps}')
                    os.makedirs(eval_dir, exist_ok=True)
                    results = self.eval_fn(eval_dir, valid_loader, device,
                        noise_scale=self._exp_conf.noise_scale
                    )
                    eval_time = time.time() - start_time
                    eval_logs = {"Eval/Eval_time": float(eval_time)}

                    mean_metrics = results["metrics"].mean(numeric_only=True)
                    for metric_name, value in mean_metrics.items():
                        eval_logs[f"Eval/{metric_name}"] = float(value)
                    # 标量指标
                    eval_logs = {
                        "Eval/RigidError_rot_pred": float(results["rot_trans_error_mean"]["ave_rot"]),
                        "Eval/RigidError_trans_pred": float(results["rot_trans_error_mean"]["ave_trans"]),
                    }

                    # matplotlib Figure
                    eval_logs.update({
                        # "Eval/Fig_dis_curve": swanlab.Image(results["fig_dis_curve"]),
                        "Eval/Fig_error": swanlab.Image(results["fig_error"]),
                    })
                    swanlab.log(eval_logs, step=self.trained_steps)

                    info_dict = {
                        "info/best_trained_steps": self.best_trained_steps,
                        "info/best_trained_epoch": self.best_trained_epoch,
                        "info/best_rmse_all": float(self.best_rmse_all),
                        "info/relat_rmse_ca": float(self.best_rmse_ca),
                        "info/rot_error": float(self.best_rot_error),
                        "info/ref_rot_error": float(self.best_ref_rot_error),
                        "info/trans_error": float(self.best_trans_error),
                        "info/ref_trans_error": float(self.best_ref_trans_error),
                        "info/relat_drmsd": float(self.best_drmsd),
                    }
                    swanlab.log(info_dict, print_to_console=True)
                    self._log.info(f'Finished evaluation in {eval_time:.2f}s')

            # Remote log to tensorborad.
            if self._use_recoder:
                step_time = time.time() - step_time
                example_per_sec = self._exp_conf.batch_size / step_time
                step_time = time.time()
                # Logging basic metrics
                log_metrics = {
                    # Losses
                    "Train/Loss_total": loss,
                    "Train/Loss_rot": aux_data["rot_loss"],
                    "Train/Loss_trans": aux_data["trans_loss"],
                    "Train/Loss_torsion": aux_data["torsion_loss"],
                    "Train/Loss_bb_atom": aux_data["bb_atom_loss"],
                    "Train/Loss_dist_mat": aux_data["dist_mat_loss"],
                    # Rigid updates
                    # "Train/Rigid_rot0": aux_data["update_rots"][0],
                    # "Train/Rigid_rot1": aux_data["update_rots"][1],
                    # "Train/Rigid_rot2": aux_data["update_rots"][2],
                    # "Train/Rigid_trans0": aux_data["update_trans"][0],
                    # "Train/Rigid_trans1": aux_data["update_trans"][1],
                    # "Train/Rigid_trans2": aux_data["update_trans"][2],
                    # Speed
                    "Train/Speed_examples_per_sec": float(example_per_sec),
                }


                bb_grads = [p.grad for name, p in self.model.named_parameters() if p.grad is not None and "bb_update" in name]
                if bb_grads:
                    bb_grads_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2.0) for g in bb_grads]), 2.0).item()
                    log_metrics["Train/Grad_bb_update"] = bb_grads_norm

                # Logging checkpoint metrics if available
                if torch.isnan(loss):
                    swanlab.log({"Alerts": f"Encountered NaN loss after {self.trained_epochs} epochs, {self.trained_steps} steps"},
                                step=self.trained_steps)
                    raise Exception("NaN encountered")

                swanlab.log(log_metrics, step=self.trained_steps)

        if return_logs:
            return global_logs

    def eval_fn(self, eval_dir, valid_loader, device, min_t=None, num_t=None, noise_scale=1.0,is_training=True):
        # === 初始化路径与指标 ===
        dirs = self._prepare_eval_dirs(eval_dir, is_training)
        metrics = self._init_eval_metrics()

        # === 评估每个样本 ===
        for valid_feats, pdb_names, start_index in tqdm(valid_loader, desc="Evaluating", ncols=100):
            result = self._process_one_protein_for_eval(
                valid_feats, pdb_names, start_index,
                device=device,
                dirs=dirs,
                min_t=min_t,
                num_t=num_t,
                noise_scale=noise_scale,
                is_training=is_training
            )
            self._accumulate_metrics(metrics, result)

        # === 生成汇总结果 ===
        return self._finalize_eval_outputs(metrics, dirs, eval_dir)


    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def eval_extension(self, eval_dir, valid_loader, device, min_t=None, num_t=None, noise_scale=1.0,seed=42):
        self._set_seed(seed)
        # ergodic the validation
        pdb_base_path, ref_base_path = self._prepare_extension_eval_dirs(eval_dir)
        print(f"\n{'*' * 10} Protein number: {len(valid_loader)} {'*' * 10}")
        extrapolation_time = self._conf.eval.extrapolation_time

        for valid_feats, pdb_names, start_index  in valid_loader:
            self._process_one_protein_extrapolation(
                extrapolation_time,
                valid_feats,
                pdb_names,
                ref_base_path,
                pdb_base_path,
                device,
                min_t,
                num_t,
                noise_scale,
            )


    def _self_conditioning(self, batch,drop_ref=False):
        model_sc = self.model(batch,drop_ref=drop_ref,is_training = self._exp_conf.training)
        batch['sc_ca_t'] = model_sc['rigids'][..., 4:]
        return batch

    def loss_fn(self, batch):
        """Computes loss and auxiliary data.

        Args:
            batch: Batched data.
            model_out: Output of model ran on batch.

        Returns:
            loss: Final training loss scalar.
            aux_data: Additional logging data.
        """

        if self._model_conf.embed.embed_self_conditioning and random.random() > 0.5:
            with torch.no_grad():
                batch = self._self_conditioning(batch)
        model_out = self.model(batch, is_training=self._exp_conf.training)

        bb_mask = batch['res_mask']
        diffuse_mask = 1 - batch['fixed_mask']

        loss_mask = bb_mask * diffuse_mask

        torsion_loss = (
            torsion_angle_loss(
                a=model_out["angles"],
                a_gt=batch["torsion_angles_sin_cos"],
                a_alt_gt=batch["alt_torsion_angles_sin_cos"],
                mask=batch["torsion_angles_mask"],
            ).mean(dim=-1)
            * self._exp_conf.torsion_loss_weight
        )

        gt_rot_score = batch['rot_score']
        rot_score_scaling = batch['rot_score_scaling']

        batch_loss_mask = torch.any(bb_mask, dim=-1)

        pred_rot_score = model_out['rot_score'] * diffuse_mask[..., None]
        pred_trans_score = model_out['trans_score'] * diffuse_mask[..., None]

        # Translation x0 loss
        gt_trans_x0 = batch['rigids_0'][..., 4:]
        pred_trans_x0 = model_out['rigids'][..., 4:]


        # x0: [B, F, N, 3] mask: [B, F, N]
        trans_loss = torch.sum(
            (gt_trans_x0 - pred_trans_x0).abs() * loss_mask[..., None], dim=(-1, -2, -3)
        ) / (loss_mask.sum(dim=(-1, -2)) + 1e-10)
        trans_loss *= self._exp_conf.trans_loss_weight
        # x0: [B, F, N, 3] mask: [B, F, N]
        rot_mse = (gt_rot_score - pred_rot_score) ** 2 * loss_mask[..., None]
        rot_loss = torch.sum(rot_mse / rot_score_scaling**2, dim=(-1, -2, -3)) / (
            loss_mask.sum(dim=(-1, -2)) + 1e-10
        )
        t_mask = (batch['t'].squeeze(-1) > self._exp_conf.rot_loss_t_threshold).float()
        rot_loss = rot_loss * t_mask * int(self._diff_conf.diffuse_rot)


        # Backbone atom loss
        pred_atom37 = model_out["atom37"][..., :5, :]
        gt_rigids = ru.Rigid.from_tensor_7(batch['rigids_0'].type(torch.float32))
        gt_psi = batch['torsion_angles_sin_cos'][..., 2, :]  # psi
        gt_atom37, atom37_mask, _, _ = all_atom.compute_backbone(gt_rigids, gt_psi) # psi

        gt_atom37 = gt_atom37[..., :5, :]
        atom37_mask = atom37_mask[..., :5]

        gt_atom37 = gt_atom37.to(pred_atom37.device)
        atom37_mask = atom37_mask.to(pred_atom37.device)
        bb_atom_loss_mask = atom37_mask * loss_mask[..., None]
        bb_atom_loss = torch.sum(
            (pred_atom37 - gt_atom37) ** 2 * bb_atom_loss_mask[..., None],
            dim=(-1, -2, -3, -4),
        ) / (bb_atom_loss_mask.sum(dim=(-1, -2, -3)) + 1e-10)
        bb_atom_loss *= (
            self._exp_conf.bb_atom_loss_weight * self._exp_conf.aux_loss_weight
        )
        t_filter_mask = (
            batch["t"].squeeze(-1) < self._exp_conf.bb_atom_loss_t_filter
        ).float()
        bb_atom_loss = bb_atom_loss * t_filter_mask


        gt_flat_atoms = rearrange(gt_atom37, "b f n a d -> b f (n a) d")
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[..., :, None, :] - gt_flat_atoms[..., None, :, :], dim=-1
        )
        pred_flat_atoms = rearrange(pred_atom37, "b f n a d -> b f (n a) d")
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[..., :, None, :] - pred_flat_atoms[..., None, :, :], dim=-1
        )

        flat_loss_mask = repeat(loss_mask, "b f n -> b f (n 5)")
        flat_res_mask = repeat(bb_mask, "b f n -> b f (n 5)")

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[..., None, :]

        # No loss on anything >6A
        proximity_mask = gt_pair_dists < 6
        pair_dist_mask  = pair_dist_mask * proximity_mask
        m_atoms = flat_loss_mask.shape[-1]
        diag_mask = 1.0 - torch.eye(m_atoms, device=pair_dist_mask.device)
        pair_dist_mask = pair_dist_mask * diag_mask[None, None, :, :] # 考虑到 B, F 维度

        dist_error = torch.sum(
            (gt_pair_dists - pred_pair_dists) ** 2 * pair_dist_mask, dim=(-1, -2, -3)
        )
        denom = torch.sum(pair_dist_mask, dim=(-1, -2, -3))
        dist_mat_loss = dist_error / (denom + 1e-10)
        dist_mat_loss *= self._exp_conf.dist_mat_loss_weight * self._exp_conf.aux_loss_weight
        t_mask = (batch['t'].squeeze(-1) < self._exp_conf.dist_mat_loss_t_filter).float()
        dist_mat_loss = dist_mat_loss * t_mask

        final_loss = (
            rot_loss
            + trans_loss
            + bb_atom_loss
            + dist_mat_loss
            + torsion_loss
        )
        def normalize_loss(x):
            return x.mean()

        aux_data = {
            'total_loss': normalize_loss(final_loss).detach(),
            'rot_loss': normalize_loss(rot_loss).detach(),
            'trans_loss': normalize_loss(trans_loss).detach(),
            'bb_atom_loss': normalize_loss(bb_atom_loss).detach(),
            'dist_mat_loss': normalize_loss(dist_mat_loss).detach(),
            'torsion_loss':normalize_loss(torsion_loss).detach(),
        }
        return normalize_loss(final_loss), aux_data

    def _calc_trans_0(self, trans_score, trans_t, t):
        beta_t = self._diffuser._se3_diffuser._r3_diffuser.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        cond_var = 1 - torch.exp(-beta_t)
        return (trans_score * cond_var + trans_t) / torch.exp(-1/2*beta_t)

    def _set_t_feats(self, feats, t, t_placeholder):
        feats['t'] = t * t_placeholder
        rot_score_scaling, trans_score_scaling = self.diffuser.score_scaling(t)
        feats['rot_score_scaling'] = rot_score_scaling * t_placeholder
        feats['trans_score_scaling'] = trans_score_scaling * t_placeholder
        return feats

    def forward_traj(self, x_0, min_t, num_t):
        forward_steps = np.linspace(min_t, 1.0, num_t)[:-1]
        x_traj = [x_0]
        for t in forward_steps:
            x_t = self.diffuser.se3_diffuser._r3_diffuser.forward(
                x_traj[-1], t, num_t)
            x_traj.append(x_t)
        x_traj = torch.stack(x_traj, axis=0)
        return x_traj

    def inference_fn(
            self,
            data_init,
            num_t=None,
            min_t=None,
            center=True,
            aux_traj=False,
            self_condition=True,
            noise_scale=1.0,
        ):
        """Inference function.

        Args:
            data_init: Initial data values for sampling.
        """
        self._model.eval()
        # Run reverse process.
        sample_feats = data_init.copy()
        device = sample_feats["rigids_t"].device
        batch = sample_feats["aatype"].shape[0]
        t_placeholder = torch.ones((batch, 1)).to(device)
        current_rigid_obj = ru.Rigid.from_tensor_7(sample_feats["rigids_t"])


        if num_t is None:
            num_t = self._data_conf.num_t
        if min_t is None:
            min_t = self._data_conf.min_t
        # print("=" * 20, num_t)
        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = 1/num_t
        all_rigids = []
        all_bb_prots = []
        t_start = time.perf_counter()
        with torch.no_grad():
            # self-condition
            if self._model_conf.embed.embed_self_conditioning and self_condition:
                sample_feats = self._set_t_feats(sample_feats, reverse_steps[0], t_placeholder)
                sample_feats = self._self_conditioning(sample_feats)
            for step_idx, t in enumerate(reverse_steps):
                # model infer
                fixed_mask = sample_feats['fixed_mask'] * sample_feats['res_mask']
                diffuse_mask = (1 - sample_feats['fixed_mask']) * sample_feats['res_mask']
                if t > min_t:
                    sample_feats = self._set_t_feats(sample_feats, t, t_placeholder)
                    model_out = self.model(sample_feats, is_training = self._exp_conf.training)
                    rot_score = model_out['rot_score']
                    trans_score = model_out['trans_score']
                    rigid_pred = model_out['rigids']
                    # use CFG inference
                    if self._conf.model.cfg_drop_rate > 0.01:
                        model_out_uncond = self.model(sample_feats,drop_ref = True,is_training = self._exp_conf.training)
                        trans_score_unref = model_out_uncond["trans_score"]
                        rot_score_unref = model_out_uncond["rot_score"]
                        cfg_gamma = self._conf.model.cfg_gamma
                        trans_score = trans_score_unref + cfg_gamma*(trans_score-trans_score_unref)

                    if self._model_conf.embed.embed_self_conditioning:
                        sample_feats['sc_ca_t'] = rigid_pred[..., 4:]
                    current_rigid_obj = self.diffuser.reverse_torch(
                        rigid_t=current_rigid_obj,
                        rot_score=rot_score,
                        trans_score=trans_score,
                        diffuse_mask=diffuse_mask,
                        t=t,
                        dt=dt,
                        center=center,
                        noise_scale=noise_scale,
                        device=device
                    )
                else:
                    model_out = self.model(sample_feats,is_training = self._exp_conf.training)
                    rigid_pred = model_out['rigids']
                    current_rigid_obj = ru.Rigid.from_tensor_7(model_out['rigids'])
                sample_feats['rigids_t'] = current_rigid_obj.to_tensor_7().to(device)
                infer_end_time = time.time()
                # post process
                gt_trans_0 = sample_feats['rigids_t'][..., 4:]
                pred_trans_0 = rigid_pred[..., 4:]
                trans_pred_0 = diffuse_mask[..., None] * pred_trans_0 + fixed_mask[..., None] * gt_trans_0

                angles = model_out['angles']
                if step_idx == len(reverse_steps)-1:
                    atom37_t = all_atom.compute_backbone_atom37(
                        bb_rigids=current_rigid_obj,
                        aatypes=sample_feats['aatype'],
                        torsions = angles
                        )[0]
                    all_rigids.append(model_out["rigids"])
                    all_bb_prots.append(atom37_t)
        #
        inference_time = time.perf_counter() - t_start
        print(f"inference_time:{inference_time:.2f} | num_t:{num_t} | noise_scale:{noise_scale}")
        flip = lambda x: torch.flip(torch.stack(x, dim=1), dims=(1,))
        all_bb_prots = flip(all_bb_prots)
        all_rigids = flip(all_rigids)
        ret = {
            'prot_traj': all_bb_prots, # with reverse
            'rigid_traj': all_rigids
        }
        return ret


    def _calc_rot_trans_error(self,pred_rigids,gt_rigids):
        # print(pred_rigids.shape,gt_rigids.shape,ref_rigids.shape)
        # pred out
        average_quat_distances = average_quaternion_distances(
            gt_rigids[..., :4], pred_rigids[..., :4]
        )
        average_trans_distances = average_translation_distances(
            gt_rigids[..., 4:], pred_rigids[..., 4:], measurement="MAE"
        )
        # caculate relative motion
        # [B,T,N,4]
        # 1. Rotation Velocity
        # [B, 1:T, N, 4] v.s [B, 0:T-1, N, 4]
        time_rot_dif = average_quaternion_distances(
            gt_rigids[:, 1:, :, :4],
            gt_rigids[:, :-1, :, :4]
        )

        # 2. Translation Velocity
        time_trans_dif = average_translation_distances(
            gt_rigids[:, 1:, :, 4:],
            gt_rigids[:, :-1, :, 4:],
            measurement="MAE"
        )

        return (
            average_quat_distances,
            average_trans_distances,
            time_rot_dif,
            time_trans_dif,
        )

    def _prepare_eval_dirs(self, eval_dir, is_training):
        """Create and return all necessary evaluation directories."""
        dirs = {
            "sample": os.path.join(eval_dir, "sample"),
            "gt": os.path.join(eval_dir, "gt"),
            "pred_npz": os.path.join(eval_dir, "pred_npz") if not is_training else None,
        }
        for path in dirs.values():
            if path:
                os.makedirs(path, exist_ok=True)
        return dirs

    def _init_eval_metrics(self):
        """Initialize containers for evaluation metrics."""
        return {
            "metric_list": [],
            "metric_all_list": [],
            "first_frame_all_list": [],
            "save_name_list": [],
            "start_index_list": [],
            "rot_trans_error_dict": {
                "name": [], "ave_rot": [], "ave_trans": [],
                "time_rot_dif": [], "time_trans_dif": [],
            },
            "save_pdb_dict": {},
        }

    def _process_one_protein_for_eval(
        self,
        valid_feats,
        pdb_names,
        start_index,
        device,
        dirs,
        min_t=None,
        num_t=None,
        noise_scale=1.0,
        is_training=True,
    ):
        """Evaluate a single protein and compute metrics."""
        save_name = pdb_names[0].split(".")[0]
        frame_time = self._model_conf.frame_time
        sample_length = valid_feats["aatype"].shape[-1]
        diffuse_mask = np.ones(sample_length)
        b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))

        # === Step 1. prepare init feats ===
        init_feats = self._prepare_init_feats(valid_feats, device, frame_time, sample_length)

        # === Step 2. inference ===
        sample_out = self.inference_fn(init_feats, num_t=num_t, min_t=min_t, aux_traj=True, noise_scale=noise_scale)

        # === Step 3. compute metrics ===
        result_metrics = self._compute_metrics(valid_feats, sample_out)

        # === Step 4. save predictions ===
        pdb_paths = self._save_eval_outputs(
            save_name, valid_feats, sample_out, dirs, b_factors
        )

        # === Step 5. rotation / translation error ===
        rot_trans_err = self._calc_rot_trans_error(
            du.move_to_np(sample_out["rigid_traj"][:, 0]),
            gt_rigids=du.move_to_np(init_feats["rigids_0"]),
        )

        return {
            "save_name": save_name,
            "start_index": start_index,
            "metrics": result_metrics,
            "pdb_paths": pdb_paths,
            "rot_trans_err": rot_trans_err,
        }

    def _accumulate_metrics(self, metrics, result):
        """Accumulate results for later aggregation."""
        metrics["save_name_list"].append(result["save_name"])
        metrics["start_index_list"].append(result["start_index"])
        metrics["metric_list"].append(result["metrics"]["mean_eval"])
        metrics["metric_all_list"].append(result["metrics"]["all_eval"])
        metrics["save_pdb_dict"][result["save_name"]] = result["pdb_paths"]

        ave_quat, ave_trans,  time_rot_dif, time_trans_dif = result["rot_trans_err"]
        rdict = metrics["rot_trans_error_dict"]
        rdict["name"].append(result["save_name"])
        rdict["ave_rot"].append(ave_quat)
        rdict["ave_trans"].append(ave_trans)
        rdict["time_rot_dif"].append(time_rot_dif)
        rdict["time_trans_dif"].append(time_trans_dif)

    def _finalize_eval_outputs(self, metrics, dirs, eval_dir):
        """Aggregate metrics, plot results, and return evaluation summary."""
        # === change the DataFrame ===
        ckpt_eval_metrics = pd.DataFrame(metrics["metric_list"])
        ckpt_eval_metrics.insert(0, "pdb_name", metrics["save_name_list"])
        ckpt_eval_metrics.to_csv(os.path.join(eval_dir, "metrics.csv"), index=False)

        # === Comparison Curver ===
        # metric_merge_dict = {
        #     "Pred": ckpt_eval_metrics,
        # }
        # curve_fig = plot_curve_merged(metric_merge_dict, eval_dir, row_num=len(metrics["save_name_list"]), col_num=2)
        error_fig = plot_rot_trans_curve(
            metrics["rot_trans_error_dict"], save_path=eval_dir
        )

        # === Get the Average Score ===
        ckpt_eval_metrics = ckpt_eval_metrics.applymap(
            lambda x: float(x) if isinstance(x, np.ndarray) else x
        )
        mean_dict = ckpt_eval_metrics.drop(columns=["pdb_name"]).mean().to_dict()
        rot_trans_error_mean = {k: np.mean(v) for k, v in metrics["rot_trans_error_dict"].items() if k != "name"}

        model_ckpt_update = self._update_best_model(mean_dict, rot_trans_error_mean)

        self._log_eval_summary(mean_dict, rot_trans_error_mean)

        return {
            "metrics": ckpt_eval_metrics,
            # "fig_dis_curve": curve_fig,
            "fig_error": error_fig,
            "model_ckpt_update": model_ckpt_update,
            "rot_trans_error_mean": rot_trans_error_mean,
            "save_pdb_dict": metrics["save_pdb_dict"],
        }

    def _compute_metrics(self, valid_feats, sample_out):
        """
        Compute validation metrics for predicted, aligned, and reference trajectories.
        """
        gt_pos = valid_feats["atom37_pos"]  # [B, F, N, 37,3]
        gt_mask = valid_feats["atom37_mask"]  # [B, F, N, 37]

        # === Unaligned prediction metrics
        eval_dic = compute_validation_metrics_all(
            gt_coords=gt_pos,
            pred_coords=sample_out["prot_traj"][:, 0],  # [B, recorded_reverse, F, N, 37]
            all_atom_mask=gt_mask,
        )
        mean_eval_dic = {k: sum(v) / len(v) for k, v in eval_dic.items()}

        return {
            "mean_eval": {
                k: (v.cpu().numpy() if torch.is_tensor(v) else v)
                for k, v in mean_eval_dic.items()
            },
            "all_eval": {
                k: (v.cpu().numpy() if torch.is_tensor(v) else v)
                for k, v in eval_dic.items()
            }
        }

    def _save_eval_outputs(
        self,
        save_name,
        valid_feats,
        sample_out,
        dirs,
        b_factors,
    ):
        """
        Save GT, predicted, and aligned structures in PDB and NPZ format.
        """
        gt_path = os.path.join(dirs["gt"], f"{save_name}_gt.pdb")
        sample_path = os.path.join(dirs["sample"], f"{save_name}.pdb")
        sample_aligned_path = os.path.join(dirs["sample"], f"{save_name}_aligned.pdb")
        first_motion_path = os.path.join(dirs["sample"], f"{save_name}_first_motion.pdb")

        # === save GT
        aatype = du.move_to_np(valid_feats["aatype"][0, 0]).astype(np.int64)
        au.write_prot_to_pdb(
            prot_pos=du.move_to_np(valid_feats["atom37_pos"][0]),
            file_path=gt_path,
            aatype=aatype,
            no_indexing=True,
            b_factors=b_factors,
        )
        # === save unaligned prediction
        au.write_prot_to_pdb(
            prot_pos=du.move_to_np(sample_out["prot_traj"][0, 0]),
            file_path=sample_path,
            aatype=aatype,
            no_indexing=True,
            b_factors=b_factors,
        )
        # === save aligned prediction
        traj = md.load(sample_path)
        traj = traj.superpose(traj, frame=0)
        traj.save(sample_aligned_path)

        # === save first_motion
        au.write_prot_to_pdb(
            prot_pos=np.concatenate(
                (
                    du.move_to_np(valid_feats["motion_atom37_pos"][0]),
                    du.move_to_np(valid_feats["ref_atom37_pos"][0]),
                ),
                axis=0,
            ),
            file_path=first_motion_path,
            aatype=aatype,
            no_indexing=True,
            b_factors=b_factors,
        )
        return {"gt": gt_path, "gen": sample_path}

    def _prepare_extension_eval_dirs(self, eval_dir):
        pdb_base_path = os.path.join(eval_dir, "extension_pdb")
        ref_base_path = os.path.join(eval_dir, "reference_pdb")
        os.makedirs(pdb_base_path, exist_ok=True)
        os.makedirs(ref_base_path, exist_ok=True)
        return pdb_base_path, ref_base_path

    def _process_one_protein_extrapolation(
        self,
        extrapolation_time,
        valid_feats,
        pdb_names,
        ref_base_path,
        pdb_base_path,
        device,
        min_t=None,
        num_t=None,
        noise_scale=1.0,
    ):
        """Process one protein sequence and perform trajectory extrapolation."""

        # === 准备工作 ===
        protein_name = pdb_names[0]
        frame_time = self._model_conf.frame_time
        ref_number = self._model_conf.ref_number
        motion_number = self._model_conf.motion_number

        aatype = du.move_to_np(valid_feats["aatype"]).astype(np.int64)
        sample_length = aatype.shape[-1]
        b_factors = np.tile((np.ones(sample_length) * 100)[:, None], (1, 37))
        pdb_path = os.path.join(pdb_base_path, f"{protein_name}_time_{extrapolation_time}.pdb")
        if os.path.exists(pdb_path):
            print(f"✅ {protein_name} already existed in: {pdb_path}")
            return
        # Reference structure 保存
        ref_all_atom_positions = du.move_to_np(valid_feats["ref_atom37_pos"][0])
        au.write_prot_to_pdb(
            prot_pos=ref_all_atom_positions,
            file_path=os.path.join(ref_base_path, f"{protein_name}.pdb"),
            aatype=aatype[0, 0],
            no_indexing=True,
            b_factors=b_factors,
        )

        print(f"[Eval] Processing {protein_name}, length={extrapolation_time}")

        # === 初始化轨迹 ===
        atom_traj, rigid_traj = [], []

        # === 迭代推理 ===
        pbar = tqdm(range(extrapolation_time), desc=f"{protein_name}", ncols=80)
        # === 初始化输入 ===
        valid_feats = self._prepare_init_feats(valid_feats, device, frame_time, sample_length)

        for j in pbar:
            # === 执行推理 ===
            sample_out = self.inference_fn(
                valid_feats,
                num_t=num_t,
                min_t=min_t,
                aux_traj=True,
                noise_scale=noise_scale,
            )

            atom_pred = sample_out["prot_traj"][0]
            rigid_pred = sample_out["rigid_traj"][0]

            # 保存最后frame的结果
            atom_traj.append(atom_pred[-frame_time:])
            rigid_traj.append(rigid_pred[-frame_time:])

            # === 更新ref状态 ===
            valid_feats = self._update_ref_with_prediction(
                valid_feats,
                atom_pred,
                rigid_pred,
                ref_number,
                motion_number,
                device,
            )

        # === 拼接轨迹并保存 ===
        atom_traj = torch.concat(atom_traj, dim=1)
        # rigid_traj = np.concatenate(rigid_traj, axis=0)

        au.write_prot_to_pdb(
            prot_pos=du.move_to_np(atom_traj[0]),
            file_path=pdb_path,
            aatype=aatype[0, 0],
            no_indexing=True,
            b_factors=b_factors,
        )

    def _prepare_init_feats(self, valid_feats, device, frame_time, sample_length):
        B = valid_feats["aatype"].shape[0]
        F = frame_time
        N = sample_length

        # 1. Initialize core features (Dimensions: [B, F, N, ...])
        init_core = {
            "res_mask": torch.ones((B, F, N)),
            "fixed_mask": torch.zeros((B, F, N)),
            "res_idx": repeat(torch.arange(1, N + 1), "n -> b f n", b=B, f=F),
            "sc_ca_t": torch.zeros((B, F, N, 3)),
            "torsion_angles_sin_cos": torch.zeros((B, F, N, 7, 2)), # 建议加上 F 维保持一致
        }
        # 2. Sample initial noised rigids (Rotations and Translations)
        ref_sample = self.diffuser.sample_ref(
            n_samples=B * N * F,
            as_tensor_7=True,
            device=device,
        )
        ref_sample["rigids_t"] = rearrange(ref_sample["rigids_t"], "f (b n) c -> b f n c", b=B, f=F, n=N)
        # 3. Merge all feature dictionaries
        # Unpacking order: init_core and ref_sample will overwrite keys in valid_feats
        init_feats = {
            **valid_feats,
            **init_core,
            **ref_sample
        }
        # 4. Final validation and device transfer
        # Ensure all tensors match the expected Batch size and move them to the GPU/CPU.
        assert all(v.shape[0] == B for v in init_feats.values() if torch.is_tensor(v))
        return tree.map_structure(lambda x: x.to(device) if torch.is_tensor(x) else x, init_feats)


    def _update_ref_with_prediction(self, valid_feats, atom_pred, rigid_pred, ref_number, motion_number, device, change_ref=False):
        """Update reference features with newly predicted frames."""
        B, F, N, _ = rigid_pred.shape
        valid_feats["motion_rigids_0"] = rigid_pred[:, -motion_number:]
        valid_feats["motion_atom37_pos"] = atom_pred[:, -motion_number:]
        ref_sample = self.diffuser.sample_ref(
            n_samples=B * N * F,
            as_tensor_7=True,
            device=device,
        )
        ref_sample["rigids_t"] = rearrange(
            ref_sample["rigids_t"], "f (b n) c -> b f n c", b=B, f=F, n=N
        )
        valid_feats["rigids_t"] = ref_sample["rigids_t"].to(device)
        return valid_feats



    def _update_best_model(self, mean_dict: dict, rot_trans_error_mean: dict) -> bool:
        """
        Compare current evaluation metrics with previous best; update if improved.
        """
        model_ckpt_update = False

        # 判断是否更好（你可以按需调整阈值）
        better_rmse = mean_dict["rmse_all"] < self.best_rmse_all
        better_rot = rot_trans_error_mean["ave_rot"] < self.best_rot_error
        better_trans = rot_trans_error_mean["ave_trans"] < self.best_trans_error

        if better_rmse or better_rot or better_trans:
            self.best_rmse_all = mean_dict["rmse_all"]
            self.best_rmse_ca = mean_dict["rmse_ca"]
            self.best_drmsd = mean_dict["drmsd_ca"]

            self.best_rot_error = rot_trans_error_mean["ave_rot"]
            self.best_trans_error = rot_trans_error_mean["ave_trans"]

            self.best_trained_steps = self.trained_steps
            self.best_trained_epoch = self.trained_epochs
            model_ckpt_update = True

        return model_ckpt_update
    def _log_eval_summary(self, mean_dict: dict, rot_trans_error_mean: dict) -> None:
        """
        Print evaluation summary and current best metrics.
        """
        info = f"Step:{self.trained_steps} "
        for k, v in mean_dict.items():
            info += f"avg_{k}:{v:.4f} "
        for k, v in rot_trans_error_mean.items():
            if k != "name":
                info += f"avg_{k}:{v:.4f} "

        self._log.info("Evaluation Results: " + info)
        self._log.info(
            f"Best so far | steps/epoch: {self.best_trained_steps}/{self.best_trained_epoch} | "
            f"rmse_all: {self.best_rmse_all:.4f} | "
            f"rmse_ca: {self.best_rmse_ca:.4f} | "
            f"drmsd_ca: {self.best_drmsd:.4f} | "
            f"rot_error: {self.best_rot_error:.4f}/{self.best_ref_rot_error:.4f} | "
            f"trans_error: {self.best_trans_error:.4f}/{self.best_ref_trans_error:.4f}"
        )

@hydra.main(version_base=None, config_path="./config", config_name="train")
def run(conf: DictConfig) -> None:

    exp = Experiment(conf=conf)
    exp.start_training()


if __name__ == '__main__':
    run()
