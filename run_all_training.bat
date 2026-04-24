@echo off
echo ============================================================
echo STEP 1/3: Training RDT (random attack, 100 epochs)
echo ============================================================
python -m RDT --seed 0 --env walker2d-medium-replay-v2 ^
  --corruption_mode random ^
  --corruption_obs 1.0 --corruption_act 1.0 --corruption_rew 1.0 ^
  --corruption_rate 0.3 ^
  --dataset_path datasets --save_model true --down_sample true
if %errorlevel% neq 0 ( echo RDT FAILED & exit /b 1 )
echo RDT done.

echo ============================================================
echo STEP 2/3: Training DT (random attack, 100 epochs)
echo ============================================================
python -m algos.DT --seed 0 --env walker2d-medium-replay-v2 ^
  --corruption_mode random ^
  --corruption_obs 1.0 --corruption_act 1.0 --corruption_rew 1.0 ^
  --corruption_rate 0.3 ^
  --dataset_path datasets --save_model true --down_sample true
if %errorlevel% neq 0 ( echo DT FAILED & exit /b 1 )
echo DT done.

echo ============================================================
echo STEP 3/3: Training RIQL (random attack, 1000 epochs)
echo ============================================================
python -m algos.RIQL --seed 0 --env walker2d-medium-replay-v2 ^
  --corruption_mode random ^
  --corruption_obs 1.0 --corruption_act 1.0 --corruption_rew 1.0 ^
  --corruption_rate 0.3 ^
  --dataset_path datasets --save_model true --down_sample true
if %errorlevel% neq 0 ( echo RIQL FAILED & exit /b 1 )
echo RIQL done.

echo ============================================================
echo ALL TRAINING COMPLETE
echo ============================================================
