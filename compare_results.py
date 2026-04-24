"""
Read progress.csv from RDT, DT, and RIQL eval runs and print a comparison table.

Usage:
    python compare_results.py \
        --rdt  <rdt_eval_dir>   \
        --dt   <dt_eval_dir>    \
        --riql <riql_eval_dir>  \
        [--out results.csv]
"""
import argparse
import os
import csv
import numpy as np


def read_progress(eval_dir):
    """Return last row of progress.csv as a dict."""
    path = os.path.join(eval_dir, "eval_00", "progress.csv")
    if not os.path.exists(path):
        path = os.path.join(eval_dir, "progress.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No progress.csv found under {eval_dir}")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def extract_score(row, alg):
    """Pull normalized score mean/std from a progress.csv row."""
    candidates = [
        "eval/normalized_score_mean",
        "eval/5000_normalized_score_mean",
        "eval/2500_normalized_score_mean",
    ]
    mean_key = next((k for k in candidates if k in row), None)
    std_key = mean_key.replace("_mean", "_std") if mean_key else None
    reward_candidates = [
        "eval/reward_mean",
        "eval/5000_reward_mean",
        "eval/2500_reward_mean",
    ]
    reward_key = next((k for k in reward_candidates if k in row), None)

    score_mean = float(row[mean_key]) if mean_key and row.get(mean_key) else float("nan")
    score_std  = float(row[std_key])  if std_key  and row.get(std_key)  else float("nan")
    reward     = float(row[reward_key]) if reward_key and row.get(reward_key) else float("nan")
    return score_mean, score_std, reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdt",  required=True, help="RDT eval run directory")
    parser.add_argument("--dt",   required=True, help="DT  eval run directory")
    parser.add_argument("--riql", required=True, help="RIQL eval run directory")
    parser.add_argument("--out",  default="", help="Optional CSV output path")
    args = parser.parse_args()

    results = []
    for alg, d in [("RDT", args.rdt), ("DT", args.dt), ("RIQL", args.riql)]:
        row = read_progress(d)
        mean, std, reward = extract_score(row, alg)
        results.append({"algorithm": alg, "norm_score_mean": mean, "norm_score_std": std, "reward_mean": reward})

    header = f"{'Algorithm':<10} {'Norm Score':>12} {'±':>4} {'Std':>8}  {'Reward':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        print(f"{r['algorithm']:<10} {r['norm_score_mean']:>12.2f} {'+/-':>4} {r['norm_score_std']:>8.2f}  {r['reward_mean']:>10.1f}")
    print("=" * len(header) + "\n")

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["algorithm", "norm_score_mean", "norm_score_std", "reward_mean"])
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
