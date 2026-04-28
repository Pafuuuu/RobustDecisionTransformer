#!/usr/bin/env python3
"""
Generate all candidate poster figures.
Saves to figs/ directory.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch

os.makedirs("figs", exist_ok=True)

# ─── Global style ───────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.labelsize":   13,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  11,
    "legend.framealpha": 0.9,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "axes.grid.axis":   "y",
    "grid.alpha":       0.3,
    "grid.linewidth":   0.8,
    "figure.dpi":       150,
    "savefig.dpi":      200,
    "savefig.facecolor": "white",
    "savefig.bbox":     "tight",
})

# ─── Color palette ──────────────────────────────────────────────
C_RDT   = "#1565C0"   # blue
C_DT    = "#D84315"   # orange-red
C_RIQL  = "#6A1B9A"   # purple
C_BLK   = "#2E7D32"   # green  (block)
C_RAND  = "#42A5F5"   # light blue (random scatter)
C_NONE  = "#6A1B9A"   # purple (no corruption)
C_GOOD  = "#2E7D32"   # confirmed green
C_WARN  = "#F57F17"   # partial orange
C_BAD   = "#B71C1C"   # failed red

def _bar_labels(ax, bars, vals, color, fs=10, dy=0.5):
    for bar, v in zip(bars, vals):
        if v is not None:
            ax.text(bar.get_x() + bar.get_width()/2, v + dy,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=fs, fontweight="bold", color=color)

def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)

def save(name):
    plt.tight_layout()
    plt.savefig(f"figs/{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  OK  figs/{name}.png")


# ════════════════════════════════════════════════════════════════
# FIG 01 — Multi-environment main results (mixed random)
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))

envs      = ["Walker2d", "Hopper", "HalfCheetah"]
rdt_m     = [17.6,  40.7,  8.1]
rdt_e     = [11.5,  10.8,  5.6]
dt_m      = [16.1,  24.1,  4.5]
dt_e      = [ 8.1,   7.9,  2.6]
riql_m    = [ 4.6]
riql_e    = [ 6.0]

x = np.arange(len(envs)); bw = 0.25
b1 = ax.bar(x - bw,  rdt_m, bw, color=C_RDT,  label="RDT",      zorder=3)
b2 = ax.bar(x,       dt_m,  bw, color=C_DT,   label="DT",       zorder=3)
b3 = ax.bar(x[0]+bw, riql_m, bw, color=C_RIQL, label="RIQL (TD)", zorder=3)


_bar_labels(ax, b1, rdt_m, C_RDT)
_bar_labels(ax, b2, dt_m,  C_DT)
_bar_labels(ax, b3, riql_m, C_RIQL)

ax.set_xticks(x); ax.set_xticklabels(envs, fontsize=13)
ax.set_ylabel("D4RL Normalized Score")
ax.set_ylim(0, 58)
ax.set_title("Replication: Mixed Random Corruption  (obs + act + rew,  rate = 0.3)",
             fontweight="bold")
ax.legend(loc="upper right")
ax.annotate("RIQL only\ntested here", xy=(x[0]+bw, 6), xytext=(x[0]+bw+0.35, 16),
            fontsize=9, color=C_RIQL, arrowprops=dict(arrowstyle="->", color=C_RIQL, lw=1))
_style(ax)
save("fig01_main_results")


# ════════════════════════════════════════════════════════════════
# FIG 02 — Walker2d individual attack type breakdown
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))

atypes  = ["State\n(obs only)", "Action\n(act only)", "Reward\n(rew only)", "Mixed\n(all 3)"]
rdt_ind = [24.3,  31.6,  40.3,  17.6]
dt_ind  = [22.7,  27.7,  37.6,  16.1]

x2 = np.arange(len(atypes)); bw2 = 0.30
br = ax.bar(x2 - bw2/2, rdt_ind, bw2, color=C_RDT,  label="RDT", zorder=3)
bd = ax.bar(x2 + bw2/2, dt_ind,  bw2, color=C_DT,   label="DT",  zorder=3)

# Highlight state column — special finding zone
ax.axvspan(-0.5, 0.5, alpha=0.07, color="orange", zorder=0)
ax.text(0, 27, "★ DT leads\non state", ha="center", fontsize=10,
        color="#E65100", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="#FFF3E0", ec="#E65100", lw=1.2))

_bar_labels(ax, br, rdt_ind, C_RDT)
_bar_labels(ax, bd, dt_ind,  C_DT)

ax.set_xticks(x2); ax.set_xticklabels(atypes, fontsize=12)
ax.set_ylabel("D4RL Normalized Score")
ax.set_ylim(0, 52)
ax.set_title("Walker2d — Individual Attack Type Comparison  (random, rate = 0.3)",
             fontweight="bold")
ax.legend(loc="upper right",
          handles=[mpatches.Patch(color=C_RDT, label="RDT"),
                   mpatches.Patch(color=C_DT,  label="DT")])
_style(ax)
save("fig02_attack_types_walker2d")


# ════════════════════════════════════════════════════════════════
# FIG 03 — Adversarial vs random severity (Walker2d mixed)
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))

modes = ["Random\nMixed", "Adversarial\nMixed"]
rdt_s = [17.6,  6.9]
dt_s  = [16.1, 11.0]
rdt_se= [11.5,  6.6]
dt_se = [ 8.1,  7.0]

x3 = np.arange(2); bw3 = 0.30
br3 = ax.bar(x3 - bw3/2, rdt_s, bw3, color=C_RDT,  label="RDT", zorder=3)
bd3 = ax.bar(x3 + bw3/2, dt_s,  bw3, color=C_DT,   label="DT",  zorder=3)


_bar_labels(ax, br3, rdt_s, C_RDT, dy=0.3)
_bar_labels(ax, bd3, dt_s,  C_DT,  dy=0.3)

# Drop arrows
for i, (r0, r1) in enumerate([(rdt_s[0], rdt_s[1]), (dt_s[0], dt_s[1])]):
    clr = C_RDT if i == 0 else C_DT
    xpos = -bw3/2 if i == 0 else +bw3/2
    ax.annotate("", xy=(1+xpos, r1+1.5), xytext=(0+xpos, r0-1.5),
                arrowprops=dict(arrowstyle="->", color=clr, lw=1.5, linestyle="dashed"))

ax.text(1, 14.5, "DT more\nresilient", ha="center", fontsize=11,
        color=C_DT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0", ec=C_DT, lw=1.3))

ax.set_xticks(x3); ax.set_xticklabels(modes, fontsize=13)
ax.set_ylabel("D4RL Normalized Score")
ax.set_ylim(0, 30)
ax.set_title("Walker2d — Attack Severity Comparison\nRDT collapses faster under adversarial attack",
             fontweight="bold")
ax.legend(handles=[mpatches.Patch(color=C_RDT, label="RDT"),
                   mpatches.Patch(color=C_DT,  label="DT")], loc="upper right")
_style(ax)
save("fig03_adversarial_severity")


# ════════════════════════════════════════════════════════════════
# FIG 04 — Block vs random corruption (Walker2d mixed)
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))

algs      = ["RDT", "DT"]
rand_m    = [17.6, 16.1]; rand_e  = [11.5, 8.1]
block_m   = [19.9, 18.1]; block_e = [12.0, 11.9]

x4 = np.arange(2); bw4 = 0.30
br4 = ax.bar(x4 - bw4/2, rand_m,  bw4, color=C_RAND, label="Random (scattered)", zorder=3)
bb4 = ax.bar(x4 + bw4/2, block_m, bw4, color=C_BLK,  label="Block (contiguous)", zorder=3)


_bar_labels(ax, br4, rand_m,  "#0D47A1", dy=0.4)
_bar_labels(ax, bb4, block_m, "#1B5E20", dy=0.4)

# Delta arrows
for i, (r, b) in enumerate(zip(rand_m, block_m)):
    xc = i + bw4/2 + 0.19
    ax.annotate("", xy=(xc, b), xytext=(xc, r),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
    ax.text(xc + 0.06, (r+b)/2, f"+{b-r:.1f}", va="center", fontsize=10,
            color="#2E7D32", fontweight="bold")

ax.set_xticks(x4); ax.set_xticklabels(algs, fontsize=14, fontweight="bold")
ax.set_ylabel("D4RL Normalized Score")
ax.set_ylim(0, 40)
ax.set_title("Walker2d — Block vs Random Corruption  (mixed, rate = 0.3)\n"
             "Block corruption consistently easier than random scatter",
             fontweight="bold")
ax.legend(handles=[mpatches.Patch(color=C_RAND, label="Random (scattered)"),
                   mpatches.Patch(color=C_BLK,  label="Block (contiguous)")],
          loc="upper left")
_style(ax)
save("fig04_block_vs_random")


# ════════════════════════════════════════════════════════════════
# FIG 05 — State attack anomaly: DT ≥ RDT across environments
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=False)
fig.suptitle("Special Finding: State (obs) Corruption — DT ≥ RDT in Most Settings",
             fontsize=14, fontweight="bold", y=1.01)

# Left: random obs attack
ax5L = axes[0]
envs5  = ["Walker2d", "Hopper", "HalfCheetah"]
rdt5r  = [24.3, 27.6,  8.6]
dt5r   = [22.7, 31.9,  7.3]
rdt5re = [13.5,  6.3,  6.1]
dt5re  = [13.6,  6.2,  4.5]

x5 = np.arange(3); bw5 = 0.32
b5r = ax5L.bar(x5 - bw5/2, rdt5r, bw5, color=C_RDT, label="RDT", zorder=3)
b5d = ax5L.bar(x5 + bw5/2, dt5r,  bw5, color=C_DT,  label="DT",  zorder=3)
_bar_labels(ax5L, b5r, rdt5r, C_RDT, fs=9)
_bar_labels(ax5L, b5d, dt5r,  C_DT,  fs=9)
# mark where DT > RDT
for i, (r, d) in enumerate(zip(rdt5r, dt5r)):
    if d > r:
        ax5L.text(i, max(r, d) + 3.5, "★", ha="center", fontsize=14, color="#E65100")
ax5L.set_xticks(x5); ax5L.set_xticklabels(envs5, fontsize=11)
ax5L.set_ylabel("D4RL Normalized Score"); ax5L.set_ylim(0, 50)
ax5L.set_title("Random Obs Attack", fontweight="bold")
ax5L.legend(handles=[mpatches.Patch(color=C_RDT, label="RDT"),
                     mpatches.Patch(color=C_DT,  label="DT")], loc="upper right")
_style(ax5L)

# Right: adversarial obs attack
ax5R = axes[1]
rdt5a  = [35.7,  6.3]   # Hopper, HalfCheetah (walker2d no obs-only adv)
dt5a   = [35.8,  7.9]
rdt5ae = [ 9.0,  4.4]
dt5ae  = [ 8.2,  6.0]
envs5a = ["Hopper", "HalfCheetah"]

x5a = np.arange(2); bw5a = 0.32
b5ar = ax5R.bar(x5a - bw5a/2, rdt5a, bw5a, color=C_RDT, label="RDT", zorder=3)
b5ad = ax5R.bar(x5a + bw5a/2, dt5a,  bw5a, color=C_DT,  label="DT",  zorder=3)
_bar_labels(ax5R, b5ar, rdt5a, C_RDT, fs=9)
_bar_labels(ax5R, b5ad, dt5a,  C_DT,  fs=9)
for i, (r, d) in enumerate(zip(rdt5a, dt5a)):
    if d > r:
        ax5R.text(i, max(r, d) + 2.5, "★", ha="center", fontsize=14, color="#E65100")
ax5R.set_xticks(x5a); ax5R.set_xticklabels(envs5a, fontsize=11)
ax5R.set_ylabel("D4RL Normalized Score"); ax5R.set_ylim(0, 50)
ax5R.set_title("Adversarial Obs Attack", fontweight="bold")
ax5R.legend(handles=[mpatches.Patch(color=C_RDT, label="RDT"),
                     mpatches.Patch(color=C_DT,  label="DT")], loc="upper right")
_style(ax5R)

fig.text(0.5, -0.04,
         "★ = DT outperforms RDT — embedding dropout does not reliably improve state robustness",
         ha="center", fontsize=11, color="#E65100", style="italic")
save("fig05_state_attack_anomaly")


# ════════════════════════════════════════════════════════════════
# FIG 06 — AntMaze new dataset results
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
fig.suptitle("New Dataset: AntMaze-medium-play-v0  (Mixed Random, obs+act+rew, rate=0.3)",
             fontsize=14, fontweight="bold", y=1.01)

# Left: normalized score bar
ax6L = axes[0]
algs6  = ["RDT", "DT"]
scores = {"target=1.0": ([0.0, 12.0], [C_RDT, C_DT]),
          "target=0.5": ([0.0,  0.0], [C_RDT, C_DT])}

x6 = np.arange(2); bw6 = 0.30
b6r = ax6L.bar(x6[0] - bw6/2, 0.0,  bw6, color=C_RDT, label="RDT (target=1.0)", zorder=3)
b6d = ax6L.bar(x6[0] + bw6/2, 12.0, bw6, color=C_DT,  label="DT  (target=1.0)", zorder=3)
ax6L.bar(x6[1] - bw6/2, 0.0, bw6, color=C_RDT, alpha=0.4, label="RDT (target=0.5)", zorder=3, hatch="//")
ax6L.bar(x6[1] + bw6/2, 0.0, bw6, color=C_DT,  alpha=0.4, label="DT  (target=0.5)", zorder=3, hatch="//")

ax6L.text(x6[0] + bw6/2, 12.5, "12.0", ha="center", va="bottom",
          fontsize=11, fontweight="bold", color=C_DT)
ax6L.text(x6[0] - bw6/2, 0.3, "0.0", ha="center", va="bottom",
          fontsize=11, fontweight="bold", color=C_RDT)
ax6L.text(x6[1], 0.5, "Both 0.0", ha="center", va="bottom",
          fontsize=10, color="#555")

ax6L.set_xticks(x6); ax6L.set_xticklabels(["Target = 1.0", "Target = 0.5"], fontsize=12)
ax6L.set_ylabel("D4RL Normalized Score  (0–100)"); ax6L.set_ylim(0, 22)
ax6L.set_title("Normalized Score by Target Return", fontweight="bold")
ax6L.legend(fontsize=9, loc="upper right")
_style(ax6L)

# Right: explanation text box
ax6R = axes[1]
ax6R.axis("off")
explanation = (
    "Why AntMaze fails for both algorithms:\n\n"
    "① Sparse binary reward — reward = 1 only\n"
    "   at goal, 0 everywhere else (1000 steps)\n\n"
    "② Mixed corruption at 30% destroys the\n"
    "   few goal-reaching reward signals\n\n"
    "③ RDT's self-correction flags rare reward=1\n"
    "   samples as 'outliers' and overwrites them\n"
    "   → model never sees successful navigation\n\n"
    "④ DT (target=1.0) achieves 12% success\n"
    "   because it attempts the goal without\n"
    "   self-correction removing key transitions\n\n"
    "⚠  Result: Dense-reward robustness claims\n"
    "   do NOT transfer to sparse-reward tasks"
)
ax6R.text(0.05, 0.95, explanation, transform=ax6R.transAxes,
          va="top", ha="left", fontsize=11, linespacing=1.5,
          bbox=dict(boxstyle="round,pad=0.6", fc="#FFF9C4", ec="#F57F17", lw=1.5))
save("fig06_antmaze")


# ════════════════════════════════════════════════════════════════
# FIG 07 — Full Walker2d picture: all attack modes
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 6))

# All walker2d conditions (best target return for each)
labels7  = ["No\nCorruption\n(RDT only)", "Obs\nOnly", "Act\nOnly", "Rew\nOnly",
             "Mixed\nRandom", "Mixed\nBlock", "Mixed\nAdversarial"]
rdt7     = [40.5,  24.3,  31.6,  40.3,  17.6,  19.9,   6.9]
rdt7_e   = [20.9,  13.5,  20.9,  19.5,  11.5,  12.0,   6.6]
dt7      = [None,  22.7,  27.7,  37.6,  16.1,  18.1,  11.0]
dt7_e    = [None,  13.6,  18.9,  25.5,   8.1,  11.9,   7.0]

x7 = np.arange(len(labels7)); bw7 = 0.35
br7 = ax.bar(x7 - bw7/2, rdt7, bw7, color=C_RDT, label="RDT", zorder=3)
dt7_plot = [v if v is not None else 0 for v in dt7]
bd7 = ax.bar(x7 + bw7/2, dt7_plot, bw7, color=C_DT, label="DT", zorder=3)
# hide bar 0 for DT (no measurement)
bd7[0].set_alpha(0)


for i, (r, d) in enumerate(zip(rdt7, dt7)):
    ax.text(i - bw7/2, r + 0.8, f"{r:.1f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=C_RDT)
    if d is not None:
        ax.text(i + bw7/2, d + 0.8, f"{d:.1f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=C_DT)

# Shade regions
ax.axvspan(-0.6, 0.6,   alpha=0.07, color="purple",  label="Baseline")
ax.axvspan(0.4,  3.6,   alpha=0.07, color="blue",    label="Individual attacks")
ax.axvspan(3.4,  5.6,   alpha=0.07, color="green",   label="Mixed (random/block)")
ax.axvspan(5.4,  6.6,   alpha=0.07, color="red",     label="Mixed adversarial")

ax.set_xticks(x7); ax.set_xticklabels(labels7, fontsize=11)
ax.set_ylabel("D4RL Normalized Score")
ax.set_ylim(0, 60)
ax.set_title("Walker2d: Complete Corruption Landscape  (all attack modes, rate=0.3)",
             fontweight="bold")
ax.legend(handles=[mpatches.Patch(color=C_RDT,  label="RDT"),
                   mpatches.Patch(color=C_DT,   label="DT"),
                   mpatches.Patch(color="purple", alpha=0.3, label="Baseline"),
                   mpatches.Patch(color="blue",   alpha=0.3, label="Individual attacks"),
                   mpatches.Patch(color="green",  alpha=0.3, label="Mixed random/block"),
                   mpatches.Patch(color="red",    alpha=0.3, label="Mixed adversarial")],
          loc="upper right", fontsize=9, ncol=2)
_style(ax)
save("fig07_walker2d_full")


# ════════════════════════════════════════════════════════════════
# FIG 08 — Hopper & HalfCheetah detail
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.suptitle("Generalization: Hopper & HalfCheetah Results", fontsize=14,
             fontweight="bold", y=1.01)

for ax8, env8, rdt8, dt8, rdt8e, dt8e, ylim8 in [
    (axes[0], "Hopper-medium-replay-v2",
     [27.6, 40.7, 35.7],   # obs-only, mixed, adv-obs
     [31.9, 24.1, 35.8],
     [ 6.3, 10.8,  9.0],
     [ 6.2,  7.9,  8.2], 55),
    (axes[1], "HalfCheetah-medium-replay-v2",
     [ 8.6,  8.1,  6.3],
     [ 7.3,  4.5,  7.9],
     [ 6.1,  5.6,  4.4],
     [ 4.5,  2.6,  6.0], 20),
]:
    labels8 = ["Random\nObs Only", "Mixed\nRandom", "Adversarial\nObs Only"]
    x8 = np.arange(3); bw8 = 0.30
    br8 = ax8.bar(x8 - bw8/2, rdt8, bw8, color=C_RDT, label="RDT", zorder=3)
    bd8 = ax8.bar(x8 + bw8/2, dt8,  bw8, color=C_DT,  label="DT",  zorder=3)
    _bar_labels(ax8, br8, rdt8, C_RDT, fs=9, dy=0.3)
    _bar_labels(ax8, bd8, dt8,  C_DT,  fs=9, dy=0.3)
    # Star where DT > RDT
    for i, (r, d) in enumerate(zip(rdt8, dt8)):
        if d > r + 1:
            ax8.text(i, max(r, d) + ylim8*0.04, "★", ha="center",
                     fontsize=13, color="#E65100")
    ax8.set_xticks(x8); ax8.set_xticklabels(labels8, fontsize=11)
    ax8.set_ylabel("D4RL Normalized Score"); ax8.set_ylim(0, ylim8)
    ax8.set_title(env8, fontweight="bold")
    ax8.legend(handles=[mpatches.Patch(color=C_RDT, label="RDT"),
                        mpatches.Patch(color=C_DT,  label="DT")], loc="upper right")
    _style(ax8)

fig.text(0.5, -0.04, "★ = DT outperforms RDT",
         ha="center", fontsize=11, color="#E65100", style="italic")
save("fig08_hopper_halfcheetah")


# ════════════════════════════════════════════════════════════════
# FIG 09 — Summary results table
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7))
ax.axis("off")

rows = [
    # [Environment, Attack, RDT, DT, RIQL, Notes]
    ["Walker2d",      "No corruption",      "40.5 ± 20.9", "—",            "—",          "Baseline"],
    ["Walker2d",      "Obs only (random)",   "24.3 ± 13.5", "22.7 ± 13.6", "—",          ""],
    ["Walker2d",      "Act only (random)",   "31.6 ± 20.9", "27.7 ± 18.9", "—",          ""],
    ["Walker2d",      "Rew only (random)",   "40.3 ± 19.5", "37.6 ± 25.5", "—",          ""],
    ["Walker2d",      "Mixed random",        "17.6 ± 11.5", "16.1 ± 8.1",  "4.6 ± 6.0",  "★ Claim 1"],
    ["Walker2d",      "Mixed block",         "19.9 ± 12.0", "18.1 ± 11.9", "—",          "★ Our ext."],
    ["Walker2d",      "Mixed adversarial",   " 6.9 ±  6.6", "11.0 ± 7.0",  "—",          "⚠ DT wins"],
    ["Hopper",        "Obs only (random)",   "27.6 ±  6.3", "31.9 ± 6.2",  "—",          "⚠ DT wins"],
    ["Hopper",        "Mixed random",        "40.7 ± 10.8", "24.1 ± 7.9",  "—",          "★ Claim 2"],
    ["Hopper",        "Obs adversarial",     "35.7 ±  9.0", "35.8 ± 8.2",  "—",          "≈ Equal"],
    ["HalfCheetah",   "Obs only (random)",   " 8.6 ±  6.1", " 7.3 ± 4.5",  "—",          ""],
    ["HalfCheetah",   "Mixed random",        " 8.1 ±  5.6", " 4.5 ± 2.6",  "—",          "★ Claim 2"],
    ["HalfCheetah",   "Obs adversarial",     " 6.3 ±  4.4", " 7.9 ± 6.0",  "—",          "⚠ DT wins"],
    ["AntMaze",       "Mixed random",        " 0.0",        "12.0",         "—",          "★ New dataset"],
]

col_labels = ["Environment", "Attack Mode", "RDT", "DT", "RIQL", "Notes"]
col_widths = [0.14, 0.21, 0.15, 0.15, 0.10, 0.15]

tbl = ax.table(cellText=rows, colLabels=col_labels,
               colWidths=col_widths, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(11)

# Header styling
for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor("#1565C0")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")

# Row shading + highlights
env_colors = {"Walker2d": "#EBF3FD", "Hopper": "#F3FBF0",
              "HalfCheetah": "#FFF8F0", "AntMaze": "#FFF9F0"}
for i, row in enumerate(rows, start=1):
    bg = env_colors.get(row[0], "white")
    for j in range(len(col_labels)):
        tbl[(i, j)].set_facecolor(bg)
    # Highlight "DT wins" rows
    note = row[-1]
    if "DT wins" in note:
        tbl[(i, 3)].set_facecolor("#FFCDD2")   # DT column highlight
        tbl[(i, 3)].set_text_props(fontweight="bold")
    elif "Claim 2" in note or "Claim 1" in note:
        tbl[(i, 2)].set_facecolor("#BBDEFB")   # RDT column highlight
        tbl[(i, 2)].set_text_props(fontweight="bold")
    elif "New dataset" in note:
        for j in [2, 3]:
            tbl[(i, j)].set_facecolor("#FFE0B2")
    tbl[(i, 0)].set_text_props(fontweight="bold")

tbl.scale(1, 1.55)
ax.set_title("Complete Results Summary — D4RL Normalized Score (0–100)\n"
             "Blue highlight = RDT wins (Claim 2),  Red highlight = DT wins (anomaly)",
             fontsize=13, fontweight="bold", pad=14)
save("fig09_summary_table")


# ════════════════════════════════════════════════════════════════
# FIG 10 — Claim verification visual summary
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.axis("off")

claims = [
    # (claim_text, setting, rdt_score, dt_score, riql_score, verdict, verdict_color)
    ("DT > TD-learning",
     "Walker2d, mixed random",
     "16.1", "4.6 (RIQL)", "✓ CONFIRMED", C_GOOD),
    ("RDT > DT (action)",
     "Walker2d, act-only random",
     "31.6", "27.7", "✓ CONFIRMED", C_GOOD),
    ("RDT > DT (reward)",
     "Walker2d, rew-only random",
     "40.3", "37.6", "✓ CONFIRMED", C_GOOD),
    ("RDT > DT (state)",
     "Hopper, obs-only random",
     "27.6", "31.9", "✗ FAILS", C_BAD),
    ("RDT > DT (mixed adv)",
     "Walker2d, adv mixed",
     " 6.9", "11.0", "✗ FAILS", C_BAD),
    ("RDT > DT (mixed)",
     "Hopper, mixed random",
     "40.7", "24.1", "✓ CONFIRMED", C_GOOD),
    ("Block > Random",
     "Walker2d, mixed (RDT)",
     "19.9", "17.6 (rand)", "★ NEW FINDING", C_WARN),
    ("New dataset",
     "AntMaze, mixed random",
     " 0.0", "12.0", "★ NEW FINDING", C_WARN),
]

rows_c  = [[c[0], c[1], c[2], c[3], c[4]] for c in claims]
col_c   = ["Claim", "Setting", "RDT Score", "DT Score", "Verdict"]
col_w_c = [0.20, 0.28, 0.14, 0.14, 0.18]

tbl2 = ax.table(cellText=rows_c, colLabels=col_c,
                colWidths=col_w_c, loc="center", cellLoc="center")
tbl2.auto_set_font_size(False); tbl2.set_fontsize(12)

# Header
for j in range(5):
    tbl2[(0, j)].set_facecolor("#37474F")
    tbl2[(0, j)].set_text_props(color="white", fontweight="bold")

# Row coloring by verdict
for i, c in enumerate(claims, start=1):
    vc = c[-1]
    for j in range(5):
        tbl2[(i, j)].set_facecolor(
            "#E8F5E9" if vc == C_GOOD else
            "#FFEBEE" if vc == C_BAD  else "#FFF9C4"
        )
    tbl2[(i, 4)].set_facecolor(vc)
    tbl2[(i, 4)].set_text_props(color="white", fontweight="bold")
    if vc == C_BAD:
        tbl2[(i, 3)].set_text_props(fontweight="bold")  # DT score bold when DT wins
    elif vc == C_GOOD:
        tbl2[(i, 2)].set_text_props(fontweight="bold")  # RDT score bold when RDT wins

tbl2.scale(1, 1.7)
ax.set_title("Claim Verification Summary",
             fontsize=15, fontweight="bold", pad=16)
save("fig10_claim_summary")


# ════════════════════════════════════════════════════════════════
# FIG 11 — Score radar / heatmap: all envs × all conditions
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 5))

# Heatmap matrix: rows=conditions, cols=env×alg
conds = ["Obs only\n(random)", "Mixed\nrandom", "Mixed\nblock", "Mixed\nadversarial"]
combos = ["Walker2d\nRDT", "Walker2d\nDT", "Hopper\nRDT", "Hopper\nDT",
          "HalfCheetah\nRDT", "HalfCheetah\nDT"]

matrix = np.array([
    # obs-only random:
    [24.3,  22.7,  27.6,  31.9,   8.6,   7.3],
    # mixed random:
    [17.6,  16.1,  40.7,  24.1,   8.1,   4.5],
    # mixed block (only walker2d):
    [19.9,  18.1, np.nan, np.nan, np.nan, np.nan],
    # mixed adversarial (only walker2d):
    [ 6.9,  11.0, np.nan, np.nan, np.nan, np.nan],
])

# Normalize per column for display color (0=lowest, 1=highest in that column)
mat_disp = np.ma.masked_invalid(matrix)

im = ax.imshow(mat_disp, cmap="RdYlGn", aspect="auto", vmin=0, vmax=45)

ax.set_xticks(range(len(combos))); ax.set_xticklabels(combos, fontsize=10)
ax.set_yticks(range(len(conds)));  ax.set_yticklabels(conds, fontsize=11)
ax.set_title("Score Heatmap: All Environments × Corruption Conditions\n"
             "(green = higher score, red = lower, gray = not measured)",
             fontsize=13, fontweight="bold")

# Annotate cells
for i in range(len(conds)):
    for j in range(len(combos)):
        v = matrix[i, j]
        if not np.isnan(v):
            clr = "white" if v < 15 else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=clr)
        else:
            ax.text(j, i, "N/A", ha="center", va="center",
                    fontsize=10, color="#888")

# Vertical separator between algorithms per env
for x_sep in [1.5, 3.5]:
    ax.axvline(x_sep, color="white", linewidth=2.5)

plt.colorbar(im, ax=ax, label="Normalized Score", shrink=0.8)
save("fig11_heatmap")


# ════════════════════════════════════════════════════════════════
# FIG 12 — Block corruption diagram (conceptual)
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(11, 5.5))
fig.suptitle("Block vs Random Corruption — Corruption Pattern Visualization",
             fontsize=14, fontweight="bold")

np.random.seed(42)
T = 80   # timesteps in a trajectory

def plot_traj(ax12, title, indices, color_c, color_ok):
    for t in range(T):
        c = color_c if t in indices else color_ok
        ax12.add_patch(plt.Rectangle((t, 0), 0.9, 1, color=c, linewidth=0))
    ax12.set_xlim(0, T); ax12.set_ylim(0, 1.4)
    ax12.set_yticks([]); ax12.set_xlabel("Timestep", fontsize=11)
    ax12.set_title(title, fontsize=12, fontweight="bold", pad=4)
    ax12.text(T/2, 1.15, f"{len(indices)}/{T} timesteps corrupted  ({len(indices)/T*100:.0f}%)",
              ha="center", fontsize=10, color="#555")
    ax12.spines[["top", "right", "left"]].set_visible(False)
    ax12.grid(False)

# Random: pick 30% uniformly
rand_idx = set(np.where(np.random.rand(T) < 0.3)[0])
plot_traj(axes[0], "Random Scatter — 30% of timesteps independently corrupted",
          rand_idx, "#EF5350", "#A5D6A7")

# Block: contiguous window
block_size = int(T * 0.3)
block_start = 28
block_idx = set(range(block_start, block_start + block_size))
plot_traj(axes[1], "Block Corruption — contiguous 30% window at random start",
          block_idx, "#EF5350", "#A5D6A7")

# Legend
for ax12 in axes:
    ax12.add_patch(plt.Rectangle((T-18, 1.05), 2.5, 0.25, color="#EF5350"))
    ax12.text(T-14.5, 1.18, "Corrupted", fontsize=9)
    ax12.add_patch(plt.Rectangle((T-10, 1.05), 2.5, 0.25, color="#A5D6A7"))
    ax12.text(T-6.5, 1.18, "Clean", fontsize=9)

plt.tight_layout()
save("fig12_block_pattern")


print("\nAll figures saved to figs/")
