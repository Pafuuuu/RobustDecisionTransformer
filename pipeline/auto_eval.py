"""Monitors run_new_experiments.log and evaluates each job as it completes."""
import subprocess, time, os, glob, sys

LOG_FILE = "run_new_experiments.log"
RESULTS_BASE = os.path.expanduser("~/results/corruption")
TODAY = "20260427"

# (job_num_that_signals_completion, alg_module, env, attack_glob, group)
# Job N's completion is signalled when job N+1 starts (or "All 8" for last job)
JOBS = [
    (5, "algos.DT", "hopper-medium-replay-v2",       "DT_hopper*adv_obs*",                "2023082100"),
    (6, "RDT",      "halfcheetah-medium-replay-v2",  "RDT_halfcheetah*rnd_obs_act_rew*",  "2024062305"),
    (7, "algos.DT", "halfcheetah-medium-replay-v2",  "DT_halfcheetah*rnd_obs_act_rew*",   "2023082100"),
    (8, "RDT",      "halfcheetah-medium-replay-v2",  "RDT_halfcheetah*adv_obs*",          "2024062305"),
]
FINAL_JOB = ("algos.DT", "halfcheetah-medium-replay-v2", "DT_halfcheetah*adv_obs*", "2023082100")

def log_read():
    with open(LOG_FILE) as f:
        return f.read()

def wait_for_signal(signal):
    print(f"  waiting for: {signal!r}", flush=True)
    while signal not in log_read():
        time.sleep(30)

def find_checkpoint(group, env, pattern):
    # Search all groups if needed (halfcheetah may use a new group)
    search_roots = [os.path.join(RESULTS_BASE, group, env)]
    for g in os.listdir(RESULTS_BASE):
        root = os.path.join(RESULTS_BASE, g, env)
        if root not in search_roots and os.path.isdir(root):
            search_roots.append(root)

    candidates = []
    for root in search_roots:
        candidates += glob.glob(os.path.join(root, pattern))

    # Filter to today's runs only
    today_runs = [d for d in candidates if TODAY in os.path.basename(d)]
    if not today_runs:
        today_runs = candidates  # fallback: all runs

    if not today_runs:
        return None
    # Pick most recently created
    return max(today_runs, key=os.path.getmtime)

def run_eval(alg_module, env, ckpt_dir):
    print(f"\n{'='*60}", flush=True)
    print(f"EVAL: {alg_module} | {env}", flush=True)
    print(f"DIR:  {os.path.basename(ckpt_dir)}", flush=True)
    print('='*60, flush=True)

    # Wait for 100.pt to appear (training writes it at the very end)
    ckpt_file = os.path.join(ckpt_dir, "100.pt")
    for _ in range(60):
        if os.path.exists(ckpt_file):
            break
        time.sleep(10)
    else:
        print(f"ERROR: 100.pt not found after 10 min in {ckpt_dir}", flush=True)
        return

    cmd = [
        sys.executable, "-m", alg_module,
        "--eval_only", "true",
        "--seed", "0",
        "--env", env,
        "--corruption_mode", "none",
        "--checkpoint_dir", ckpt_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Print last 60 lines (summary table + progress)
    lines = (result.stdout + result.stderr).splitlines()
    print("\n".join(lines[-60:]), flush=True)

# ── main ──────────────────────────────────────────────────────────────────────
print("Auto-eval monitor started. Watching run_new_experiments.log ...", flush=True)

for signal_job, alg_module, env, pat, group in JOBS:
    wait_for_signal(f"{signal_job}/8")
    print(f"\nJob {signal_job-1} done — finding checkpoint ...", flush=True)
    ckpt = find_checkpoint(group, env, pat)
    if ckpt:
        run_eval(alg_module, env, ckpt)
    else:
        print(f"ERROR: no checkpoint found for pattern {pat}", flush=True)

# Final job: wait for "All 8 complete"
print("\nWaiting for final job (8/8) to complete ...", flush=True)
wait_for_signal("All 8 training runs complete")
time.sleep(30)  # let the file system settle

alg_module, env, pat, group = FINAL_JOB
print(f"\nJob 8 done — evaluating {alg_module} {env} adversarial", flush=True)
ckpt = find_checkpoint(group, env, pat)
if ckpt:
    run_eval(alg_module, env, ckpt)
else:
    print(f"ERROR: no checkpoint found for pattern {pat}", flush=True)

print("\n\nAll evaluations complete!", flush=True)
