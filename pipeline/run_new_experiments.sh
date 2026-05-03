#!/usr/bin/env bash
# Runs 8 training jobs: hopper + halfcheetah × RDT + DT × random + adversarial
# Estimated runtime: ~6-7 hours total (sequential, single GPU)
# Results saved to ~/results/corruption/

set -e
DATASET="datasets"
SEED=0
RANDOM_FLAGS="--corruption_mode random --corruption_obs 1.0 --corruption_act 1.0 --corruption_rew 1.0 --corruption_rate 0.3"
ADV_FLAGS="--corruption_agent IQL --corruption_mode adversarial --corruption_obs 1.0 --corruption_rate 0.3"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── hopper ──────────────────────────────────────────────────────────────────
log "1/8  RDT hopper random"
python -m RDT --seed $SEED --env hopper-medium-replay-v2 \
  $RANDOM_FLAGS --dataset_path $DATASET --save_model true

log "2/8  DT  hopper random"
python -m algos.DT --seed $SEED --env hopper-medium-replay-v2 \
  $RANDOM_FLAGS --dataset_path $DATASET --save_model true

log "3/8  RDT hopper adversarial"
python -m RDT --seed $SEED --env hopper-medium-replay-v2 \
  $ADV_FLAGS --dataset_path $DATASET --save_model true

log "4/8  DT  hopper adversarial"
python -m algos.DT --seed $SEED --env hopper-medium-replay-v2 \
  $ADV_FLAGS --dataset_path $DATASET --save_model true

# ── halfcheetah ──────────────────────────────────────────────────────────────
log "5/8  RDT halfcheetah random"
python -m RDT --seed $SEED --env halfcheetah-medium-replay-v2 \
  $RANDOM_FLAGS --dataset_path $DATASET --save_model true

log "6/8  DT  halfcheetah random"
python -m algos.DT --seed $SEED --env halfcheetah-medium-replay-v2 \
  $RANDOM_FLAGS --dataset_path $DATASET --save_model true

log "7/8  RDT halfcheetah adversarial"
python -m RDT --seed $SEED --env halfcheetah-medium-replay-v2 \
  $ADV_FLAGS --dataset_path $DATASET --save_model true

log "8/8  DT  halfcheetah adversarial"
python -m algos.DT --seed $SEED --env halfcheetah-medium-replay-v2 \
  $ADV_FLAGS --dataset_path $DATASET --save_model true

log "All 8 training runs complete. Checkpoints in ~/results/corruption/"
