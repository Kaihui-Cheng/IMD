# Usage

Launch commands from the repository root with the virtual environment activated.
All entry points accept Hydra overrides (`key=value`). Inspect a configuration
without loading data or weights:

```bash
python -m DyneTrion.supervised_pretrain --cfg job --resolve
python -m DyneTrion.eval_actor_pipeline --cfg job --resolve
python -m DyneTrion.train --cfg job --resolve
```

## Actor training

`DyneTrion/config/actor_train.yaml` composes `train.yaml` and `actor_base.yaml`.
It defaults to a history-aware loader, score matching and structural losses
toward history frames, and forward KL environment support (`lambda_support=0.1`).
Cloud logging is disabled; set `experiment.use_recoder=true` to enable local
SwanLab logging. Credentials must be supplied externally if enabling cloud mode.

Training saves Accelerate state under
`experiment.save_dir/supervised_step_<step>/`. `actor.weight=<file>` initializes
actor parameters; it is not a complete optimizer-state resume operation.

## Sampling

`DyneTrion/config/eval_actor.yaml` uses mode 2 (multiple extrapolation segments),
16 generated frames per segment, and 16 extrapolation segments. Adjust
`data.frame_time` and `eval.extrapolation_time` to control these values.
`eval.overwrite=true` replaces existing outputs.

```bash
bash scripts/sample.sh experiment.actor_blend_alpha=0.2 experiment.max_history=3
```

### Paper terminology and release defaults

The paper calls the bias strength **η** and the distance temperature **κ**.
Their configuration keys are `experiment.actor_blend_alpha` and
`experiment.history_distance_sigma`, respectively. The history-bank size is
`experiment.max_history` (H).

The sampler implements both `experiment.history_distance_variant=normalized`
(softmax weights, as written in the paper's Eq. 3) and `unnormalized` (distance
weights without division by their sum). The released configuration defaults to
`unnormalized`, σ=1, H=3, α=0.1; do not assume these defaults reproduce every
paper experiment. The demo uses α=0.1 and the same unnormalized weighting.

The paper's long-horizon refinement is a separate post-processing workflow and
is not included in this initial release. The retained `inline_refine_t` field
is not an implementation of that workflow; leave it at zero.

For an environment baseline, use
`experiment.actor_blend_alpha=0 actor.weight=null`. The actor checkpoint is then
unnecessary. Optional FKC and history-correction hooks remain in the sampler for
source compatibility; they are disabled in the supplied main configuration.
Their study-specific analysis pipelines are not part of this release.

## Environment training

The DyneTrion environment trainer is included for users preparing their own
pretrained environment. Its settings differ from supervised actor training:

```bash
NUM_PROCESSES=1 bash scripts/train_env.sh \
  data.csv_path=data/train.csv data.val_csv_path=data/val.csv \
  experiment.num_gpus=1 experiment.base_root=outputs/environment
```

This uses torchrun and the environment trainer's DDP path. Actual convergence and
GPU memory requirements depend on the data and configuration. The initial public
release is validated with CPU tests, rather than a complete GPU training run.
