# Checkpoints

Checkpoints are intentionally excluded from Git. No download URLs are available
in this initial code release. Once published, this page will list the environment
and supervised actor checkpoints, checksums, architecture settings, and terms.

| Artifact | Local convention | Required for |
| --- | --- | --- |
| DyneTrion environment | `checkpoints/env_model.pth` | Actor training and sampling |
| Supervised IMD actor | `checkpoints/actor/pytorch_model.bin` | Guided sampling |

The environment loader accepts a raw state dictionary or a dictionary containing
`model`; it strips `module.` prefixes. Sampling accepts the actor's raw state
dictionary in `pytorch_model.bin`, or its containing Accelerate checkpoint
directory. Actor training's optional `actor.weight` expects a checkpoint file.

The supplied configuration is for the noncausal environment with
`model.use_causal_temporal=false`, `model.use_temporal=true`, and
`model.num_frame_per_block=6`. These values must match the environment checkpoint.
For any causal checkpoint, use the architecture recorded when it was trained.

The initial public actor uses supervised environment-support training. Abandoned
RL/GRPO and DMD checkpoints are not compatible substitutes. AlphaFold/OpenFold
embedding-model weights are a separate upstream dependency.
