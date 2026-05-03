"""
Runs all 11 training + eval jobs sequentially without manual intervention.
Each entry: train → grep log for checkpoint path → eval → done.
"""
import subprocess, sys, os, re, glob

BASE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.expanduser("~/results/corruption")

# Groups used by each module's logger
GROUPS = {
    "RDT":      "2024062305",
    "algos.DT": "2023082100",
}

# Each experiment: (label, module, train_flags)
# Eval flags are always just --eval_only true --checkpoint_dir <found> --n_episodes 100
EXPERIMENTS = [
    # ── Part 1: individual corruption types (RDT + DT each) ──────────────────
    ("RDT_state",   "RDT", [
        "--corruption_mode","random","--corruption_obs","1.0",
        "--corruption_act","0.0","--corruption_rew","0.0","--corruption_rate","0.3",
    ]),
    ("DT_state",    "algos.DT", [
        "--corruption_mode","random","--corruption_obs","1.0",
        "--corruption_act","0.0","--corruption_rew","0.0","--corruption_rate","0.3",
    ]),
    ("RDT_action",  "RDT", [
        "--corruption_mode","random","--corruption_obs","0.0",
        "--corruption_act","1.0","--corruption_rew","0.0","--corruption_rate","0.3",
    ]),
    ("DT_action",   "algos.DT", [
        "--corruption_mode","random","--corruption_obs","0.0",
        "--corruption_act","1.0","--corruption_rew","0.0","--corruption_rate","0.3",
    ]),
    ("RDT_reward",  "RDT", [
        "--corruption_mode","random","--corruption_obs","0.0",
        "--corruption_act","0.0","--corruption_rew","1.0","--corruption_rate","0.3",
    ]),
    ("DT_reward",   "algos.DT", [
        "--corruption_mode","random","--corruption_obs","0.0",
        "--corruption_act","0.0","--corruption_rew","1.0","--corruption_rate","0.3",
    ]),
    # ── Part 2: ablations on RDT under relevant attack ────────────────────────
    ("RDT_abl_no_dropout", "RDT", [
        "--corruption_mode","random","--corruption_obs","1.0",
        "--corruption_act","0.0","--corruption_rew","0.0","--corruption_rate","0.3",
        "--embedding_dropout","0.0",
    ]),
    ("RDT_abl_no_wmse", "RDT", [
        "--corruption_mode","random","--corruption_obs","1.0",
        "--corruption_act","0.0","--corruption_rew","0.0","--corruption_rate","0.3",
        "--loss_fn","mse",
    ]),
    ("RDT_abl_no_correction", "RDT", [
        "--corruption_mode","random","--corruption_obs","0.0",
        "--corruption_act","1.0","--corruption_rew","0.0","--corruption_rate","0.3",
        "--correct_start","101",
    ]),
    # ── Part 3: higher severity (rate=0.5, state attack) ─────────────────────
    ("RDT_state_rate05", "RDT", [
        "--corruption_mode","random","--corruption_obs","1.0",
        "--corruption_act","0.0","--corruption_rew","0.0","--corruption_rate","0.5",
    ]),
    ("DT_state_rate05",  "algos.DT", [
        "--corruption_mode","random","--corruption_obs","1.0",
        "--corruption_act","0.0","--corruption_rew","0.0","--corruption_rate","0.5",
    ]),
]

ENV = "walker2d-medium-replay-v2"
SEED = "0"
DATASET = "datasets"


def run_cmd(cmd, label):
    print(f"\n{'='*64}\n  {label}\n{'='*64}")
    print("CMD:", " ".join(cmd))
    sys.stdout.flush()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE)
    combined = result.stdout + result.stderr
    # Print last 2000 chars to keep log readable
    print(combined[-2000:] if len(combined) > 2000 else combined)
    sys.stdout.flush()
    if result.returncode != 0:
        print(f"[FATAL] {label} exited with code {result.returncode}")
        sys.exit(result.returncode)
    return combined


def find_ckpt(output, module):
    m = re.search(r"Logging to\s+(\S+)", output)
    if m:
        return m.group(1).strip()
    # Fallback: newest directory in the group
    group = GROUPS.get(module, "2024062305")
    dirs = sorted(
        glob.glob(os.path.join(LOGDIR, group, ENV, "*")),
        key=os.path.getmtime,
    )
    return dirs[-1] if dirs else None


def main():
    # Skip experiments whose checkpoints already exist (resume support)
    # (we track via a simple done-file)
    done_file = os.path.join(BASE, "run_experiments_done.txt")
    done = set()
    if os.path.exists(done_file):
        with open(done_file) as f:
            done = {l.strip() for l in f}

    for label, module, train_flags in EXPERIMENTS:
        if label in done:
            print(f"[SKIP] {label} already completed.")
            continue

        # ── Train ─────────────────────────────────────────────────────────────
        train_cmd = [
            sys.executable, "-m", module,
            "--seed", SEED, "--env", ENV,
            "--dataset_path", DATASET,
            "--save_model", "true",
            "--down_sample", "true",
        ] + train_flags
        output = run_cmd(train_cmd, f"TRAIN  {label}")

        ckpt = find_ckpt(output, module)
        if not ckpt:
            print(f"[FATAL] Cannot find checkpoint for {label}")
            sys.exit(1)
        print(f"  --> checkpoint: {ckpt}")

        # ── Eval ──────────────────────────────────────────────────────────────
        # corruption params are re-loaded from params.json by __post_init__,
        # so we only need to pass eval-specific flags here.
        eval_cmd = [
            sys.executable, "-m", module,
            "--seed", SEED, "--env", ENV,
            "--dataset_path", DATASET,
            "--eval_only", "true",
            "--checkpoint_dir", ckpt,
            "--n_episodes", "100",
        ]
        run_cmd(eval_cmd, f"EVAL   {label}")

        # Mark done
        with open(done_file, "a") as f:
            f.write(label + "\n")
        print(f"  [done] {label} complete -> {ckpt}/eval_00/progress.csv")

    print("\n" + "="*64)
    print("  ALL EXPERIMENTS COMPLETE")
    print("="*64)


if __name__ == "__main__":
    main()
