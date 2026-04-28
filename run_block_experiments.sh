#!/usr/bin/env bash
set -e

DATASET="datasets"
ENV="walker2d-medium-replay-v2"
SEED=0
RATE=0.3

echo "=== [1/2] RDT block mixed corruption ==="
python -m RDT \
  --seed $SEED --env $ENV \
  --corruption_mode block \
  --corruption_obs 1.0 --corruption_act 1.0 --corruption_rew 1.0 \
  --corruption_rate $RATE \
  --dataset_path $DATASET --save_model true
echo "RDT_BLOCK_DONE"

echo "=== [2/2] DT block mixed corruption ==="
python -m algos.DT \
  --seed $SEED --env $ENV \
  --corruption_mode block \
  --corruption_obs 1.0 --corruption_act 1.0 --corruption_rew 1.0 \
  --corruption_rate $RATE \
  --dataset_path $DATASET --save_model true
echo "DT_BLOCK_DONE"

echo "All block training runs complete"
