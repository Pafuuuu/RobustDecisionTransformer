# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

This is a **Windows-compatible fork** of the Robust Decision Transformer (RDT). The original repo requires Linux/Mac (mujoco-py, D4RL), but the training pipeline has been patched to run on Windows without those dependencies. Evaluation still requires Mac/Linux with mujoco-py.

**Windows dependencies (training only):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124  # CUDA build
pip install numpy tqdm pyrallis wandb gym h5py requests
```

**Mac/Linux dependencies (training + evaluation):**
```bash
pip install mujoco-py d4rl gym torch numpy tqdm pyrallis wandb h5py
```

## Key Commands

**Download dataset (Windows, run once):**
```bash
python download_dataset.py
```

**Generate downsampled dataset (10% of trajectories):**
```bash
cd utils && python ratio_dataset.py --env_name walker2d-medium-replay-v2 --ratio 0.1 --dataset_path ../datasets
```
Creates `datasets/original/walker2d-medium-replay-v2_ratio_0.1.pt`. The `datasets/` directory is gitignored.

**Train RDT (Windows, no evaluation):**
```bash
python -m RDT --seed 0 --env walker2d-medium-replay-v2 --corruption_mode none --dataset_path datasets --save_model true
```
Checkpoints saved every 50 epochs to `~/results/corruption/{group}/{env}/{run_id}/`. Full run: 100 epochs × 1000 steps, ~50 min on RTX 4080 Laptop.

**Train with corruption:**
```bash
# Random
python -m RDT --seed 0 --env walker2d-medium-replay-v2 --corruption_mode random --corruption_obs 1.0 --corruption_rate 0.3 --dataset_path datasets --save_model true

# Adversarial (uses pre-trained IQL weights in IQL_model/)
python -m RDT --seed 0 --env walker2d-medium-replay-v2 --corruption_agent IQL --corruption_mode adversarial --corruption_obs 1.0 --corruption_rate 0.3 --dataset_path datasets --save_model true
```

**Evaluate on Mac (loads 100.pt from checkpoint dir):**
```bash
python -m RDT --eval_only true --seed 0 --env walker2d-medium-replay-v2 --corruption_mode none --logdir ~/results/corruption
```

**Run a baseline algorithm:**
```bash
python -m algos.RIQL --seed 0 --env walker2d-medium-replay-v2 --corruption_mode none --dataset_path datasets
```

**Quick smoke test (verify training runs, ~10 seconds):**
```bash
python -m RDT --seed 0 --env walker2d-medium-replay-v2 --corruption_mode none --dataset_path datasets --num_epochs 1 --num_updates_on_epoch 10
```

## Architecture

### Data flow

```
HDF5 file
  → ratio_dataset.py        # downsample to .pt file (one-time)
  → SequenceDataset          # loads .pt, applies corruption, serves batches
  → attack_dataset()         # optionally corrupts obs/act/rew in-place
  → training loop in RDT.py
```

### Training loop (`RDT.py`)

The main novelty is the **self-correction loop** that runs inside `train()` after `correct_start` epochs:

1. Forward pass predicts both actions and rewards (dual-head transformer)
2. `compute_loss()` uses **weighted MSE** (`wmse`): samples where the model prediction diverges greatly from the data label are downweighted, making the model robust to corrupted labels
3. `correct_outliers()` computes z-scores of per-sample prediction errors against a running mean/std (`RunningMeanStd`). Samples exceeding `correct_threshold` sigma are flagged as corrupted and their dataset entries are overwritten with the model's own predictions
4. This self-correction happens every `correct_freq` steps after epoch `correct_start`

### Model (`utils/dt_functions.py` → `DecisionTransformer`)

Standard GPT-style causal transformer. Input sequence interleaves `(return-to-go, state, action)` triplets. Outputs:
- **Action head**: predicts next action from state token embeddings (`out_s_emb`)
- **Reward head**: predicts reward from action token embeddings (`out_a_emb`)

The reward head exists solely to enable self-correction — during training, large reward prediction errors identify corrupted reward labels.

### Corruption system (`utils/attack.py`)

Two modes, applied to the dataset before training:
- **Random**: adds uniform noise scaled by per-feature std
- **Adversarial**: uses a pre-trained IQL critic (`IQL_model/{env}/3000.pt`) to find perturbations that maximize Q-value shift, making corrupted samples as misleading as possible

Corrupted datasets are cached to `{dataset_path}/log_attack_data/{env}/` and reused on subsequent runs unless `--froce_attack true`.

### Logging

Each run creates a directory at `~/results/corruption/{group}/{env}/{alg}_{env}_{corrupt_tag}_{seed}_{timestamp}_{uuid}/` containing:
- `params.json` — full config (required by `--eval_only`)
- `progress.csv` — per-epoch metrics
- `log.txt` — human-readable training log
- `{50,100}.pt` — model checkpoints (if `--save_model true`)

### Windows compatibility patches

These changes were made to this fork to enable training on Windows:
- `import d4rl` is wrapped in `try/except` in `RDT.py` and `utils/attack.py`
- `ENV_DIMS` dict in `RDT.py` replaces `gym.make()` for extracting state/action dimensions
- `eval_fn()` calls removed from `train()` (evaluation requires mujoco-py)
- `utils/ratio_dataset.py` uses `h5py` directly instead of `gym.make().get_dataset()`
- `torch.load(..., weights_only=False)` in `utils/dt_functions.py` for PyTorch 2.6+ compatibility

### Config system

All scripts use `pyrallis` for config. Boolean flags require explicit values: `--save_model true` not `--save_model`. Per-environment hyperparameters (target returns, reward scale, wmse coefficients, correct thresholds) are set automatically in `TrainConfig.__post_init__()`.

### Adding a new environment

1. Add entry to `ENV_DIMS` in `RDT.py` with `state_dim`, `action_dim`, `max_action`
2. Add target returns and reward scale in `TrainConfig.__post_init__()`
3. Add RDT-specific params (wmse_coef, embedding_dropout, correct_threshold) in the same method

## GitHub workflow

Always commit changes with descriptive messages and push to `https://github.com/Pafuuuu/RobustDecisionTransformer` after every edit session.
