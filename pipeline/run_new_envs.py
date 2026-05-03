"""
New-environment experiments: hopper + halfcheetah with random and adversarial corruption.
Same train -> find checkpoint -> eval pattern as run_experiments.py.
"""
import subprocess, sys, os, re, glob

BASE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.expanduser("~/results/corruption")

GROUPS = {
    "RDT":      "2024062305",
    "algos.DT": "2023082100",
}

EXPERIMENTS = [
    # ── Hopper: random state attack ─────────────────────────────────────────
    ("hopper_RDT_state_rnd", "RDT", "hopper-medium-replay-v2", [
        "--corruption_mode", "random",
        "--corruption_obs", "1.0", "--corruption_act", "0.0", "--corruption_rew", "0.0",
        "--corruption_rate", "0.3",
    ]),
    ("hopper_DT_state_rnd", "algos.DT", "hopper-medium-replay-v2", [
        "--corruption_mode", "random",
        "--corruption_obs", "1.0", "--corruption_act", "0.0", "--corruption_rew", "0.0",
        "--corruption_rate", "0.3",
    ]),
    # ── Hopper: adversarial state attack ────────────────────────────────────
    ("hopper_RDT_state_adv", "RDT", "hopper-medium-replay-v2", [
        "--corruption_mode", "adversarial",
        "--corruption_obs", "1.0", "--corruption_act", "0.0", "--corruption_rew", "0.0",
        "--corruption_rate", "0.3",
    ]),
    ("hopper_DT_state_adv", "algos.DT", "hopper-medium-replay-v2", [
        "--corruption_mode", "adversarial",
        "--corruption_obs", "1.0", "--corruption_act", "0.0", "--corruption_rew", "0.0",
        "--corruption_rate", "0.3",
    ]),
    # ── Halfcheetah: random state attack ────────────────────────────────────
    ("halfcheetah_RDT_state_rnd", "RDT", "halfcheetah-medium-replay-v2", [
        "--corruption_mode", "random",
        "--corruption_obs", "1.0", "--corruption_act", "0.0", "--corruption_rew", "0.0",
        "--corruption_rate", "0.3",
    ]),
    ("halfcheetah_DT_state_rnd", "algos.DT", "halfcheetah-medium-replay-v2", [
        "--corruption_mode", "random",
        "--corruption_obs", "1.0", "--corruption_act", "0.0", "--corruption_rew", "0.0",
        "--corruption_rate", "0.3",
    ]),
    # ── Halfcheetah: adversarial state attack ───────────────────────────────
    ("halfcheetah_RDT_state_adv", "RDT", "halfcheetah-medium-replay-v2", [
        "--corruption_mode", "adversarial",
        "--corruption_obs", "1.0", "--corruption_act", "0.0", "--corruption_rew", "0.0",
        "--corruption_rate", "0.3",
    ]),
    ("halfcheetah_DT_state_adv", "algos.DT", "halfcheetah-medium-replay-v2", [
        "--corruption_mode", "adversarial",
        "--corruption_obs", "1.0", "--corruption_act", "0.0", "--corruption_rew", "0.0",
        "--corruption_rate", "0.3",
    ]),
]

SEED = "0"
DATASET = "datasets"


def run_cmd(cmd, label):
    print(f"\n{'='*64}\n  {label}\n{'='*64}")
    print("CMD:", " ".join(cmd))
    sys.stdout.flush()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE)
    combined = result.stdout + result.stderr
    print(combined[-2000:] if len(combined) > 2000 else combined)
    sys.stdout.flush()
    if result.returncode != 0:
        print(f"[FATAL] {label} exited with code {result.returncode}")
        sys.exit(result.returncode)
    return combined


def find_ckpt(output, module, env):
    m = re.search(r"Logging to\s+(\S+)", output)
    if m:
        return m.group(1).strip()
    group = GROUPS.get(module, "2024062305")
    dirs = sorted(
        glob.glob(os.path.join(LOGDIR, group, env, "*")),
        key=os.path.getmtime,
    )
    return dirs[-1] if dirs else None


def main():
    done_file = os.path.join(BASE, "run_new_envs_done.txt")
    done = set()
    if os.path.exists(done_file):
        with open(done_file) as f:
            done = {l.strip() for l in f}

    for label, module, env, train_flags in EXPERIMENTS:
        if label in done:
            print(f"[SKIP] {label} already completed.")
            continue

        train_cmd = [
            sys.executable, "-m", module,
            "--seed", SEED, "--env", env,
            "--dataset_path", DATASET,
            "--save_model", "true",
            "--down_sample", "true",
        ] + train_flags
        output = run_cmd(train_cmd, f"TRAIN  {label}")

        ckpt = find_ckpt(output, module, env)
        if not ckpt:
            print(f"[FATAL] Cannot find checkpoint for {label}")
            sys.exit(1)
        print(f"  --> checkpoint: {ckpt}")

        eval_cmd = [
            sys.executable, "-m", module,
            "--seed", SEED, "--env", env,
            "--dataset_path", DATASET,
            "--eval_only", "true",
            "--checkpoint_dir", ckpt,
            "--n_episodes", "100",
        ]
        run_cmd(eval_cmd, f"EVAL   {label}")

        with open(done_file, "a") as f:
            f.write(label + "\n")
        print(f"  [done] {label} -> {ckpt}/eval_00/progress.csv")

    print("\n" + "=" * 64)
    print("  ALL NEW-ENV EXPERIMENTS COMPLETE")
    print("=" * 64)


if __name__ == "__main__":
    main()
