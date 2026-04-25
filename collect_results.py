"""
Collect all eval results across walker2d, hopper, halfcheetah experiments
and write a single summary CSV.
"""
import csv, os

HOME = os.path.expanduser("~")
R = os.path.join(HOME, "results", "corruption")

def read_csv(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}

def get(row, *keys):
    for k in keys:
        if k in row and row[k]:
            return float(row[k])
    return float("nan")

# Each entry: (label, env, algorithm, corruption_mode, corruption_type,
#              corruption_rate, note, eval_dir)
EXPERIMENTS = [
    # ── Walker2d – main attacks ────────────────────────────────────────────
    ("walker2d_RDT_state_rnd", "walker2d-medium-replay-v2", "RDT", "random", "state", 0.3, "",
     f"{R}/2024062305/walker2d-medium-replay-v2/RDT_walker2d-medium-replay-v2_rnd_obs_0_20260424034444_2dd4568f-c457-43a2-8fe3-cf3cf314dcee"),
    ("walker2d_DT_state_rnd",  "walker2d-medium-replay-v2", "DT",  "random", "state", 0.3, "",
     f"{R}/2023082100/walker2d-medium-replay-v2/DT_walker2d-medium-replay-v2_rnd_obs_0_20260424034447_be65e80b-4100-492d-bf8a-c8dccc02931e"),
    ("walker2d_RDT_action_rnd","walker2d-medium-replay-v2", "RDT", "random", "action", 0.3, "",
     f"{R}/2024062305/walker2d-medium-replay-v2/RDT_walker2d-medium-replay-v2_rnd_act_0_20260424125119_faf25c36-aade-4fb1-8713-5cd4a52b979e"),
    ("walker2d_DT_action_rnd", "walker2d-medium-replay-v2", "DT",  "random", "action", 0.3, "",
     f"{R}/2023082100/walker2d-medium-replay-v2/DT_walker2d-medium-replay-v2_rnd_act_0_20260424133550_3ed6a8db-c9c5-4af3-a87c-9c809519ec1f"),
    ("walker2d_RDT_reward_rnd","walker2d-medium-replay-v2", "RDT", "random", "reward", 0.3, "",
     f"{R}/2024062305/walker2d-medium-replay-v2/RDT_walker2d-medium-replay-v2_rnd_rew_0_20260424142930_75fd841f-44a5-4d9f-87df-3a0bcb30f783"),
    ("walker2d_DT_reward_rnd", "walker2d-medium-replay-v2", "DT",  "random", "reward", 0.3, "",
     f"{R}/2023082100/walker2d-medium-replay-v2/DT_walker2d-medium-replay-v2_rnd_rew_0_20260424153319_2a69e676-3994-44fe-83d0-dc0d17fff388"),
    # ── Walker2d – mixed (from earlier session) ────────────────────────────
    ("walker2d_RDT_mixed_rnd", "walker2d-medium-replay-v2", "RDT", "random", "mixed", 0.3, "",
     f"{R}/2024062305/walker2d-medium-replay-v2/RDT_walker2d-medium-replay-v2_rnd_obs_act_rew_0_20260423222002_07f1877a-7a68-4c54-a97d-251cbdcecc8b"),
    ("walker2d_DT_mixed_rnd",  "walker2d-medium-replay-v2", "DT",  "random", "mixed", 0.3, "",
     f"{R}/2023082100/walker2d-medium-replay-v2/DT_walker2d-medium-replay-v2_rnd_obs_act_rew_0_20260423232926_4bad0dd9-73e3-4e4d-9818-ec66ff3d753c"),
    # ── Walker2d – ablations ───────────────────────────────────────────────
    ("walker2d_RDT_state_no_dropout", "walker2d-medium-replay-v2", "RDT", "random", "state", 0.3, "no_dropout",
     f"{R}/2024062305/walker2d-medium-replay-v2/RDT_walker2d-medium-replay-v2_rnd_obs_0_20260424162854_816c85ed-46d9-4609-86e3-a778ba151259"),
    ("walker2d_RDT_state_no_wmse",    "walker2d-medium-replay-v2", "RDT", "random", "state", 0.3, "no_wmse",
     f"{R}/2024062305/walker2d-medium-replay-v2/RDT_walker2d-medium-replay-v2_rnd_obs_0_20260424172605_d6b84d60-4c0b-4c8d-b5ed-92384bf056b4"),
    ("walker2d_RDT_action_no_correction", "walker2d-medium-replay-v2", "RDT", "random", "action", 0.3, "no_correction",
     f"{R}/2024062305/walker2d-medium-replay-v2/RDT_walker2d-medium-replay-v2_rnd_act_0_20260424182918_886f1239-7f2e-4262-b1a3-be323ae090b3"),
    # ── Walker2d – stress test rate=0.5 ───────────────────────────────────
    ("walker2d_RDT_state_rate05", "walker2d-medium-replay-v2", "RDT", "random", "state", 0.5, "",
     f"{R}/2024062305/walker2d-medium-replay-v2/RDT_walker2d-medium-replay-v2_rnd_obs_0_20260424195603_8e5d6ede-72a2-46b0-9e55-023452c60806"),
    ("walker2d_DT_state_rate05",  "walker2d-medium-replay-v2", "DT",  "random", "state", 0.5, "",
     f"{R}/2023082100/walker2d-medium-replay-v2/DT_walker2d-medium-replay-v2_rnd_obs_0_20260424205656_b2a22eab-4bfb-4f64-b6f7-50e6200f317e"),
    # ── Hopper ────────────────────────────────────────────────────────────
    ("hopper_RDT_state_rnd", "hopper-medium-replay-v2", "RDT", "random",      "state", 0.3, "",
     f"{R}/2024062305/hopper-medium-replay-v2/RDT_hopper-medium-replay-v2_rnd_obs_0_20260424224247_4f027d24-039c-4a07-b83c-4685597372f5"),
    ("hopper_DT_state_rnd",  "hopper-medium-replay-v2", "DT",  "random",      "state", 0.3, "",
     f"{R}/2023082100/hopper-medium-replay-v2/DT_hopper-medium-replay-v2_rnd_obs_0_20260424235954_0ae8dcc3-d655-49b2-a55f-fe14a3cded64"),
    ("hopper_RDT_state_adv", "hopper-medium-replay-v2", "RDT", "adversarial", "state", 0.3, "",
     f"{R}/2024062305/hopper-medium-replay-v2/RDT_hopper-medium-replay-v2_adv_obs_0_20260425010418_0f8a8661-4fd8-44a0-8d0e-ecd224344986"),
    ("hopper_DT_state_adv",  "hopper-medium-replay-v2", "DT",  "adversarial", "state", 0.3, "",
     f"{R}/2023082100/hopper-medium-replay-v2/DT_hopper-medium-replay-v2_adv_obs_0_20260425020349_a7fb393a-21ff-4791-b12f-60f912d319c5"),
    # ── Halfcheetah ────────────────────────────────────────────────────────
    ("halfcheetah_RDT_state_rnd", "halfcheetah-medium-replay-v2", "RDT", "random",      "state", 0.3, "",
     f"{R}/2024062305/halfcheetah-medium-replay-v2/RDT_halfcheetah-medium-replay-v2_rnd_obs_0_20260425030325_9ee73e68-9902-4b5d-912e-d8d71e8ffef4"),
    ("halfcheetah_DT_state_rnd",  "halfcheetah-medium-replay-v2", "DT",  "random",      "state", 0.3, "",
     f"{R}/2023082100/halfcheetah-medium-replay-v2/DT_halfcheetah-medium-replay-v2_rnd_obs_0_20260425034935_c9102537-87e5-4a23-a379-f0fa8ccdd807"),
    ("halfcheetah_RDT_state_adv", "halfcheetah-medium-replay-v2", "RDT", "adversarial", "state", 0.3, "",
     f"{R}/2024062305/halfcheetah-medium-replay-v2/RDT_halfcheetah-medium-replay-v2_adv_obs_0_20260425042729_1b70b5db-f6bd-4f15-8a57-ae6b12caa01b"),
    ("halfcheetah_DT_state_adv",  "halfcheetah-medium-replay-v2", "DT",  "adversarial", "state", 0.3, "",
     f"{R}/2023082100/halfcheetah-medium-replay-v2/DT_halfcheetah-medium-replay-v2_adv_obs_0_20260425050740_3198f9c3-966f-4883-8974-b6e48d023fbd"),
]

TARGET_KEYS_HIGH = {
    "walker2d-medium-replay-v2":    ("eval/5000_normalized_score_mean", "eval/5000_normalized_score_std"),
    "hopper-medium-replay-v2":      ("eval/3600_normalized_score_mean", "eval/3600_normalized_score_std"),
    "halfcheetah-medium-replay-v2": ("eval/12000_normalized_score_mean","eval/12000_normalized_score_std"),
}
TARGET_KEYS_LOW = {
    "walker2d-medium-replay-v2":    ("eval/2500_normalized_score_mean", "eval/2500_normalized_score_std"),
    "hopper-medium-replay-v2":      ("eval/1800_normalized_score_mean", "eval/1800_normalized_score_std"),
    "halfcheetah-medium-replay-v2": ("eval/6000_normalized_score_mean", "eval/6000_normalized_score_std"),
}

rows_out = []
for label, env, alg, mode, ctype, rate, note, edir in EXPERIMENTS:
    csv_path = os.path.join(edir, "eval_00", "progress.csv")
    row = read_csv(csv_path)
    mk_h, sk_h = TARGET_KEYS_HIGH[env]
    mk_l, sk_l = TARGET_KEYS_LOW[env]
    mean_h = get(row, mk_h)
    std_h  = get(row, sk_h)
    mean_l = get(row, mk_l)
    std_l  = get(row, sk_l)
    rows_out.append({
        "label":            label,
        "env":              env,
        "algorithm":        alg,
        "corruption_mode":  mode,
        "corruption_type":  ctype,
        "corruption_rate":  rate,
        "note":             note,
        "score_mean_high":  f"{mean_h:.2f}",
        "score_std_high":   f"{std_h:.2f}",
        "score_mean_low":   f"{mean_l:.2f}",
        "score_std_low":    f"{std_l:.2f}",
    })

out_path = "all_results.csv"
fields = ["label","env","algorithm","corruption_mode","corruption_type",
          "corruption_rate","note","score_mean_high","score_std_high",
          "score_mean_low","score_std_low"]
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows_out)

# Pretty-print
print(f"\n{'='*100}")
print(f"{'Label':<42} {'Alg':>4} {'Mode':>12} {'Type':>7} {'Rate':>5}  {'Score(hi)':>10} {'±':>2} {'Std':>6}  {'Note'}")
print(f"{'='*100}")
for r in rows_out:
    print(f"{r['label']:<42} {r['algorithm']:>4} {r['corruption_mode']:>12} {r['corruption_type']:>7} {r['corruption_rate']:>5}  "
          f"{r['score_mean_high']:>10} {'+/-':>2} {r['score_std_high']:>6}  {r['note']}")
print(f"{'='*100}")
print(f"\nSaved to {out_path}")
