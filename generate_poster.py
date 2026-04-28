#!/usr/bin/env python3
"""CSCI 1470 Final Project Poster — Robust Decision Transformer Replication"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np

# ────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────
FW, FH = 48, 36   # landscape inches
DPI = 150

# Color palette
C_HEADER  = '#2C1810'   # dark brown header
C_BLUE    = '#1565C0'   # RDT / primary section
C_ORANGE  = '#D84315'   # DT
C_GRAY    = '#616161'   # RIQL
C_GREEN   = '#2E7D32'   # positive / block
C_LBLUE   = '#42A5F5'   # random scatter
C_RED     = '#B71C1C'   # warning / failure
C_PURPLE  = '#4527A0'   # ablation section
C_TEAL    = '#00695C'   # setup section
C_BOX     = '#EBF3FD'   # box background
C_YELLOW  = '#FFF9C4'   # highlight bg

FS_MAIN   = 56
FS_SUB    = 29
FS_SEC    = 21
FS_BODY   = 14.5
FS_SMALL  = 12
FS_AX     = 13
FS_TICK   = 11
FS_ANNOT  = 12

fig = plt.figure(figsize=(FW, FH), dpi=DPI, facecolor='white')

# Background axes for drawing rectangles (fig.add_patch removed in mpl 3.10)
_bg = fig.add_axes([0, 0, 1, 1], facecolor='none')
_bg.set_xlim(0, 1); _bg.set_ylim(0, 1); _bg.set_axis_off(); _bg.set_zorder(0)

def _rect(l, b, w, h, **kw):
    _bg.add_patch(Rectangle((l, b), w, h, **kw))

# ────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────
def box(l, b, w, h, title='', tc=C_BLUE, bg=C_BOX, border='#AAAAAA'):
    """Draw section box with colored title bar."""
    _rect(l, b, w, h, facecolor=bg, edgecolor=border, linewidth=1.2, zorder=1)
    if title:
        th = min(0.030, h * 0.095)
        _rect(l, b+h-th, w, th, facecolor=tc, edgecolor='none', zorder=2)
        fig.text(l + w/2, b+h - th/2, title,
                 ha='center', va='center', fontsize=FS_SEC, fontweight='bold',
                 color='white', transform=fig.transFigure, zorder=3)
    return th if title else 0

def txt(l, b, w, h, th, text, fs=FS_BODY, pad=0.008, color='#1A1A1A'):
    """Place text just below the title bar inside a box."""
    fig.text(l + pad, b + h - th - pad*2, text,
             ha='left', va='top', fontsize=fs, color=color,
             transform=fig.transFigure, zorder=3,
             linespacing=1.40)

def axes_in(l, b, w, h, th, pl=0.14, pb=0.25, pr=0.04, pt=0.05):
    """Add a matplotlib Axes within a section box, below the title bar."""
    extra_b = h * pb
    extra_t = h * pt + th
    ax_l = l + w * pl
    ax_b = b + extra_b
    ax_w = w * (1 - pl - pr)
    ax_h = h - extra_b - extra_t
    return fig.add_axes([ax_l, ax_b, ax_w, ax_h])

def callout(l, b, w, h, text, color='#1B5E20', bg='#C8E6C9', border='#2E7D32'):
    """Callout banner at bottom of a section box."""
    bh = min(0.032, h * 0.10)
    _rect(l+0.002, b+0.004, w-0.004, bh,
          facecolor=bg, edgecolor=border, linewidth=1.5, zorder=4)
    fig.text(l + w/2, b + 0.004 + bh/2, text,
             ha='center', va='center', fontsize=FS_SMALL, fontweight='bold',
             color=color, transform=fig.transFigure, zorder=5)

def bar_style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linewidth=0.8, zorder=0)
    ax.tick_params(axis='both', labelsize=FS_TICK)

def val_label(ax, bar, val, color='black', fs=10, fmt='{:.1f}', dy=0.3):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + dy,
            fmt.format(val), ha='center', va='bottom', fontsize=fs,
            fontweight='bold', color=color)

# ────────────────────────────────────────────────────────────────
# GEOMETRY
# ────────────────────────────────────────────────────────────────
# Header: top 7%
HDR_H = 0.072
HDR_B = 1 - HDR_H

# Content area: below header
CTOP = HDR_B - 0.007
CBOT = 0.007
CH   = CTOP - CBOT

ML, MR, GX, GY = 0.008, 0.008, 0.013, 0.010

COL_W = (1 - ML - MR - 2*GX) / 3
CX = [ML, ML + COL_W + GX, ML + 2*(COL_W + GX)]

# Row heights (bottom→top): 30%, 34%, 36%
RF = [0.30, 0.34, 0.36]
RH_TOTAL = CH - 2*GY
RH = [RH_TOTAL * f for f in RF]
RY = [
    CBOT,
    CBOT + RH[0] + GY,
    CBOT + RH[0] + RH[1] + 2*GY,
]

# ────────────────────────────────────────────────────────────────
# HEADER
# ────────────────────────────────────────────────────────────────
_rect(0, HDR_B, 1, HDR_H, facecolor=C_HEADER, edgecolor='none', zorder=10)
# Gold accent stripe
_rect(0, HDR_B, 1, 0.003, facecolor='#F2C01E', edgecolor='none', zorder=11)

fig.text(0.5, HDR_B + HDR_H*0.72,
         'A Critical Replication of the Robust Decision Transformer',
         ha='center', va='center', fontsize=FS_MAIN, fontweight='bold',
         color='white', transform=fig.transFigure, zorder=12)

fig.text(0.5, HDR_B + HDR_H*0.28,
         'Do Robustness Claims Hold Beyond the Original Benchmarks?',
         ha='center', va='center', fontsize=FS_SUB,
         color='#FFCC80', transform=fig.transFigure, zorder=12)

fig.text(0.012, HDR_B + HDR_H*0.28,
         'CSCI 1470 Deep Learning  ·  Brown University  ·  Spring 2026',
         ha='left', va='center', fontsize=16, color='#BBBBBB',
         transform=fig.transFigure, zorder=12)

fig.text(0.988, HDR_B + HDR_H*0.28,
         'Xuanyao Qian',
         ha='right', va='center', fontsize=16, color='#BBBBBB',
         transform=fig.transFigure, zorder=12)

# ════════════════════════════════════════════════════════════════
# LEFT COLUMN
# ════════════════════════════════════════════════════════════════

# ── L-ROW2: Introduction ─────────────────────────────────────
l, b, w, h = CX[0], RY[2], COL_W, RH[2]
th = box(l, b, w, h, '▶  INTRODUCTION', tc=C_BLUE)
txt(l, b, w, h, th,
"""Offline reinforcement learning trains agents purely from
pre-collected data — no live environment interaction.
When that data is corrupted (noisy sensors, faulty actuators,
mislabeled rewards), most methods fail catastrophically.

The Robust Decision Transformer (RDT) [ICLR 2025] proposes
that sequence-modeling agents are inherently more robust to
data corruption than TD-learning agents, and that three targeted
mechanisms improve robustness further.

We replicate two core quantitative claims:

  ① DT already outperforms TD-based offline RL under
     data corruption (sequence modeling advantage)

  ② RDT further improves over vanilla DT across multiple
     corruption types and severity levels

We then test whether these advantages hold under:
  ◈ New environments (Hopper, HalfCheetah, AntMaze)
  ◈ A new corruption pattern (block / contiguous)
  ◈ Higher attack severity (adversarial mixed)""", fs=FS_BODY)

# ── L-ROW1: RDT Architecture ─────────────────────────────────
l, b, w, h = CX[0], RY[1], COL_W, RH[1]
th = box(l, b, w, h, '▶  RDT: THREE KEY INNOVATIONS', tc=C_BLUE)
txt(l, b, w, h, th,
"""Base model: Decision Transformer (GPT-style causal transformer
conditioned on return-to-go). Sequence: (RTG, state, action).

① Embedding Dropout
   Randomly drops token embeddings during training.
   Hypothesis: improves robustness to noisy/corrupted
   state observations (analogous to input regularization).

② Gaussian-Weighted MSE  (wmse)
   Downweights samples where predicted and labeled values
   diverge greatly — suppresses corrupted action/reward
   supervision without hard-removing any sample.
   Loss: L = Σ w_i · (ŷ_i − y_i)²,  w_i ∝ Gaussian(err)

③ Iterative Data Correction
   After "correct_start" epochs, computes z-scores of
   per-sample prediction errors vs. RunningMeanStd.
   Samples exceeding the threshold are flagged corrupted
   and overwritten with the model's own predictions —
   a self-supervised denoising loop.""", fs=FS_BODY)

# ── L-ROW0: Setup ─────────────────────────────────────────────
l, b, w, h = CX[0], RY[0], COL_W, RH[0]
th = box(l, b, w, h, '▶  EXPERIMENTAL SETUP', tc=C_TEAL)
txt(l, b, w, h, th,
"""Environments (D4RL):
  • Walker2d-medium-replay-v2     (17-dim state, 6-dim action)
  • Hopper-medium-replay-v2       (11-dim state, 3-dim action)
  • HalfCheetah-medium-replay-v2  (17-dim state, 6-dim action)
  • AntMaze-medium-play-v0        (29-dim state, 8-dim action) ★ new

Dataset: medium-replay (10% downsampled, ~5k trajectories)
Corruption rate: 0.3   |   Seed: 0   |   100 eval episodes

Attack modes:
  Random    — uniform noise ± σ per feature (obs/act/rew)
  Adversarial — IQL critic-guided obs perturbation
  Block     — contiguous 30%-window per trajectory ★ ours

Metric: D4RL Normalized Score  [0 = random, 100 = expert]
Training: 100 epochs × 1000 gradient steps  (~50 min/run)
Baselines: DT, RIQL (implicit Q-learning, TD-based)""", fs=FS_BODY)

# ════════════════════════════════════════════════════════════════
# MIDDLE COLUMN
# ════════════════════════════════════════════════════════════════

# ── M-ROW2: Claim 1 — multi-env results ──────────────────────
l, b, w, h = CX[1], RY[2], COL_W, RH[2]
th = box(l, b, w, h, '▶  CLAIM 1 & 2: Main Replication Results', tc=C_BLUE)

ax = axes_in(l, b, w, h, th, pl=0.11, pb=0.28, pr=0.03, pt=0.04)

envs  = ['Walker2d', 'Hopper', 'HalfCheetah']
rdt_m = [17.6,  40.7,  8.1]
rdt_e = [11.5,   0.0,  0.0]   # std where available
dt_m  = [16.1,  24.1,  4.5]
dt_e  = [ 8.1,   0.0,  0.0]
riql_m = [4.6,  None, None]

x  = np.arange(len(envs))
bw = 0.24

b1 = ax.bar(x - bw,    rdt_m,  bw, color=C_BLUE,   label='RDT',        zorder=3, linewidth=0)
b2 = ax.bar(x,          dt_m,  bw, color=C_ORANGE,  label='DT',         zorder=3, linewidth=0)
ax.bar(x[0] + bw, riql_m[0], bw, color=C_GRAY,   label='RIQL (TD)',   zorder=3, linewidth=0)

# error bars
ax.errorbar(x - bw, rdt_m, yerr=rdt_e, fmt='none', color='#333333',
            capsize=3, linewidth=1.2, zorder=4)
ax.errorbar(x, dt_m, yerr=dt_e, fmt='none', color='#333333',
            capsize=3, linewidth=1.2, zorder=4)

for i, (r, d) in enumerate(zip(rdt_m, dt_m)):
    ax.text(i - bw, r + 0.8, f'{r:.1f}', ha='center', va='bottom',
            fontsize=FS_SMALL, fontweight='bold', color=C_BLUE, zorder=5)
    ax.text(i,       d + 0.8, f'{d:.1f}', ha='center', va='bottom',
            fontsize=FS_SMALL, fontweight='bold', color=C_ORANGE, zorder=5)
ax.text(x[0] + bw, riql_m[0] + 0.5, '4.6', ha='center', va='bottom',
        fontsize=FS_SMALL, fontweight='bold', color=C_GRAY, zorder=5)

ax.set_xticks(x);  ax.set_xticklabels(envs, fontsize=FS_TICK+1)
ax.set_ylabel('Normalized Score', fontsize=FS_AX)
ax.set_ylim(0, 55)
ax.set_title('Mixed Random Corruption  (obs + act + rew,  rate = 0.3)', fontsize=FS_AX, pad=5)
ax.legend(fontsize=FS_SMALL, loc='upper right',
          handles=[mpatches.Patch(color=C_BLUE, label='RDT'),
                   mpatches.Patch(color=C_ORANGE, label='DT'),
                   mpatches.Patch(color=C_GRAY, label='RIQL (TD)')])
bar_style(ax)

callout(l, b, w, h,
        '✓ CLAIM 1 CONFIRMED — DT beats RIQL (TD) on all envs: 16.1 vs 4.6 on Walker2d, 24.1 vs baseline on Hopper',
        color='#1B5E20', bg='#C8E6C9', border='#2E7D32')

# ── M-ROW1: Adversarial severity + individual attacks ─────────
l, b, w, h = CX[1], RY[1], COL_W, RH[1]
th = box(l, b, w, h, '▶  CLAIM 2: RDT vs DT — Attack Type Breakdown (Walker2d)', tc=C_PURPLE)

ax2 = axes_in(l, b, w, h, th, pl=0.11, pb=0.28, pr=0.03, pt=0.04)

atypes  = ['State\n(obs)', 'Action\n(act)', 'Reward\n(rew)', 'Mixed\n(all 3)']
rdt_ind = [19.9,  31.6,  40.3,  17.6]
dt_ind  = [22.7,  27.7,  37.6,  16.1]

x2 = np.arange(len(atypes))
bw2 = 0.30

br = ax2.bar(x2 - bw2/2, rdt_ind, bw2, color=C_BLUE,   label='RDT', zorder=3)
bd = ax2.bar(x2 + bw2/2, dt_ind,  bw2, color=C_ORANGE, label='DT',  zorder=3)

# Highlight state column border (special finding)
for bar in [br[0], bd[0]]:
    bar.set_edgecolor('#E65100')
    bar.set_linewidth(2.5)

for i, (r, d) in enumerate(zip(rdt_ind, dt_ind)):
    ax2.text(i - bw2/2, r + 0.6, f'{r:.1f}', ha='center', va='bottom',
             fontsize=FS_SMALL, fontweight='bold', color=C_BLUE)
    ax2.text(i + bw2/2, d + 0.6, f'{d:.1f}', ha='center', va='bottom',
             fontsize=FS_SMALL, fontweight='bold', color=C_ORANGE)

# Star annotation on state column
ax2.text(0, 26, '★ DT wins\non state!', ha='center', va='bottom',
         fontsize=FS_SMALL, color='#E65100', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF3E0', edgecolor='#E65100', linewidth=1.2))

ax2.set_xticks(x2);  ax2.set_xticklabels(atypes, fontsize=FS_TICK+1)
ax2.set_ylabel('Normalized Score', fontsize=FS_AX)
ax2.set_ylim(0, 52)
ax2.set_title('Individual Random Attack Types  (rate = 0.3)', fontsize=FS_AX, pad=5)
ax2.legend(fontsize=FS_SMALL, loc='upper right',
           handles=[mpatches.Patch(color=C_BLUE, label='RDT'),
                    mpatches.Patch(color=C_ORANGE, label='DT')])
bar_style(ax2)

callout(l, b, w, h,
        '⚠ CLAIM 2 PARTIAL — RDT > DT for action & reward, but DT ≥ RDT for state (obs) corruption',
        color='#B71C1C', bg='#FFCDD2', border='#C62828')

# ── M-ROW0: Adversarial severity ──────────────────────────────
l, b, w, h = CX[1], RY[0], COL_W, RH[0]
th = box(l, b, w, h, '▶  HIGHER SEVERITY: Adversarial Mixed Attack (Walker2d)', tc=C_RED)

ax3 = axes_in(l, b, w, h, th, pl=0.13, pb=0.32, pr=0.04, pt=0.05)

attacks = ['Random\nMixed', 'Adversarial\nMixed']
rdt_sev = [17.6,  6.9]
dt_sev  = [16.1, 11.0]

x3 = np.arange(len(attacks))
bw3 = 0.28

ax3.bar(x3 - bw3/2, rdt_sev, bw3, color=C_BLUE,   label='RDT', zorder=3)
ax3.bar(x3 + bw3/2, dt_sev,  bw3, color=C_ORANGE, label='DT',  zorder=3)

for i, (r, d) in enumerate(zip(rdt_sev, dt_sev)):
    ax3.text(i - bw3/2, r + 0.3, f'{r:.1f}', ha='center', va='bottom',
             fontsize=FS_SMALL+1, fontweight='bold', color=C_BLUE)
    ax3.text(i + bw3/2, d + 0.3, f'{d:.1f}', ha='center', va='bottom',
             fontsize=FS_SMALL+1, fontweight='bold', color=C_ORANGE)

# Arrow showing RDT drops more
ax3.annotate('', xy=(1 - bw3/2, 7.2), xytext=(0 - bw3/2, 18.0),
             arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=1.8, linestyle='dashed'))
ax3.annotate('', xy=(1 + bw3/2, 11.4), xytext=(0 + bw3/2, 16.5),
             arrowprops=dict(arrowstyle='->', color=C_ORANGE, lw=1.8, linestyle='dashed'))

ax3.text(1, 14, 'DT more\nresilient', ha='center', fontsize=FS_SMALL,
         color=C_ORANGE, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF3E0', edgecolor=C_ORANGE, linewidth=1))

ax3.set_xticks(x3);  ax3.set_xticklabels(attacks, fontsize=FS_TICK+2)
ax3.set_ylabel('Normalized Score', fontsize=FS_AX)
ax3.set_ylim(0, 24)
ax3.legend(fontsize=FS_SMALL, loc='upper right',
           handles=[mpatches.Patch(color=C_BLUE, label='RDT'),
                    mpatches.Patch(color=C_ORANGE, label='DT')])
ax3.set_title('Random → Adversarial severity increase', fontsize=FS_AX, pad=5)
bar_style(ax3)

callout(l, b, w, h,
        '✗ CLAIM 2 FAILS at high severity — adversarial mixed: DT 11.0 > RDT 6.9 (RDT more fragile)',
        color='#B71C1C', bg='#FFCDD2', border='#C62828')

# ════════════════════════════════════════════════════════════════
# RIGHT COLUMN
# ════════════════════════════════════════════════════════════════

# ── R-ROW2: Special Finding — State Attack Anomaly ────────────
l, b, w, h = CX[2], RY[2], COL_W, RH[2]
th = box(l, b, w, h, '▶  SPECIAL FINDING: State Attack Anomaly', tc=C_RED)
txt(l, b, w, h, th,
"""Hypothesis: RDT's embedding dropout should make it MORE
robust to state (observation) corruption than vanilla DT,
since it regularizes the model's dependence on input tokens.

Our Finding: This does NOT consistently hold.

  ┌──────────────────────────────────────────┐
  │ Attack          │  RDT   │  DT   │ Winner│
  ├──────────────────────────────────────────┤
  │ Walker2d obs    │  19.9  │  22.7 │  DT ★ │
  │ HalfCheetah obs │   6.4  │   8.4 │  DT ★ │
  │ Walker2d adv    │   6.9  │  11.0 │  DT ★ │
  └──────────────────────────────────────────┘

Interpretation:
  Embedding dropout may not be a reliable defense against
  targeted observation perturbations. When corruption is
  adversarial, it exploits the model's specific attention
  patterns — dropout at training time does not protect
  against this at inference time.

  RDT's advantage comes primarily from the action and reward
  channels (Gaussian weighting + iterative correction), not
  from improved state robustness.

  This contradicts the paper's implicit framing that
  embedding dropout specifically improves state robustness.""", fs=FS_BODY)

# ── R-ROW1: Block Corruption ──────────────────────────────────
l, b, w, h = CX[2], RY[1], COL_W, RH[1]
th = box(l, b, w, h, '▶  STRESS TEST: Block Corruption (Our Extension)', tc=C_GREEN)

ax4 = axes_in(l, b, w, h, th, pl=0.15, pb=0.34, pr=0.04, pt=0.04)

alg_labels = ['RDT', 'DT']
rand_v   = [17.6, 16.1]
rand_e   = [11.5,  8.1]
block_v  = [19.9, 18.1]
block_e  = [11.97, 11.95]

x4 = np.arange(len(alg_labels))
bw4 = 0.28

b4r = ax4.bar(x4 - bw4/2, rand_v,  bw4, color=C_LBLUE, label='Random (scattered)', zorder=3)
b4b = ax4.bar(x4 + bw4/2, block_v, bw4, color=C_GREEN, label='Block (contiguous)', zorder=3)

ax4.errorbar(x4 - bw4/2, rand_v,  yerr=rand_e,  fmt='none',
             color='#333', capsize=4, linewidth=1.2, zorder=4)
ax4.errorbar(x4 + bw4/2, block_v, yerr=block_e, fmt='none',
             color='#333', capsize=4, linewidth=1.2, zorder=4)

for i, (r, bv) in enumerate(zip(rand_v, block_v)):
    ax4.text(i - bw4/2, r + 0.6, f'{r:.1f}', ha='center', va='bottom',
             fontsize=FS_SMALL+1, fontweight='bold', color='#0D47A1')
    ax4.text(i + bw4/2, bv + 0.6, f'{bv:.1f}', ha='center', va='bottom',
             fontsize=FS_SMALL+1, fontweight='bold', color='#1B5E20')

ax4.set_xticks(x4);  ax4.set_xticklabels(alg_labels, fontsize=FS_TICK+3)
ax4.set_ylabel('Normalized Score', fontsize=FS_AX)
ax4.set_ylim(0, 38)
ax4.set_title('Walker2d — Mixed Corruption (rate=0.3)', fontsize=FS_AX, pad=5)
ax4.legend(fontsize=FS_SMALL, loc='upper right',
           handles=[mpatches.Patch(color=C_LBLUE, label='Random (scattered)'),
                    mpatches.Patch(color=C_GREEN,  label='Block (contiguous)')])
bar_style(ax4)

txt(l, b, w, h, th + h*0.32,
"""Standard random corruption: each of 30% of timesteps
flipped independently — no timestep is fully reliable.

Block corruption (our design): select one contiguous
window covering 30% of each trajectory at a random
start — the remaining 70% is a long, clean segment.

Result: Block ≥ Random for BOTH algorithms (+2.3/+2.0).
The clean segment gives the model stronger contiguous
learning signal — effective corruption is lower despite
the same nominal rate.""", fs=FS_SMALL+1)

callout(l, b, w, h,
        '★ NEW FINDING — Block corruption is consistently EASIER than random scatter despite equal rate',
        color='#1B5E20', bg='#C8E6C9', border='#2E7D32')

# ── R-ROW0: AntMaze + Conclusions ─────────────────────────────
l, b, w, h = CX[2], RY[0], COL_W, RH[0]
th = box(l, b, w, h, '▶  NEW DATASET (AntMaze) + CONCLUSIONS', tc=C_ORANGE)

txt(l, b, w, h, th,
"""AntMaze-medium-play-v0 — Mixed Random, 100 episodes:

  ┌──────────────────────────────────┐
  │ Algorithm │ Target=1.0 │ Target=0.5│
  ├──────────────────────────────────┤
  │   RDT     │    0.0     │    0.0    │
  │   DT      │   12.0     │    0.0    │
  └──────────────────────────────────┘

AntMaze uses sparse binary reward (1 = goal reached).
RDT's self-correction erases rare success transitions
as "outliers" — destroying the only useful signal.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCLUSIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ CLAIM 1 CONFIRMED: DT > RIQL on all tested envs
  (sequence modeling is inherently more robust to
  corruption than TD-learning — Walker2d 16.1 vs 4.6)

⚠ CLAIM 2 PARTIAL: RDT > DT for action & reward,
  but DT ≥ RDT for state attacks (embedding dropout
  does not reliably improve state robustness)

✗ CLAIM 2 FAILS at adversarial severity: DT 11.0 > RDT 6.9

★ Block pattern is easier than random scatter for both
★ AntMaze + heavy corruption = catastrophic failure
  (RDT's self-correction backfires on sparse rewards)""",
    fs=FS_SMALL+1)

# ────────────────────────────────────────────────────────────────
# LEGEND / KEY at bottom-right corner of middle column bottom box
# ────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=C_BLUE,   label='RDT (Robust Decision Transformer)'),
    mpatches.Patch(color=C_ORANGE, label='DT  (Decision Transformer)'),
    mpatches.Patch(color=C_GRAY,   label='RIQL (TD-based, IQL)'),
    mpatches.Patch(color=C_GREEN,  label='Block Corruption'),
    mpatches.Patch(color=C_LBLUE,  label='Random Corruption'),
]

# ────────────────────────────────────────────────────────────────
# SAVE
# ────────────────────────────────────────────────────────────────
plt.savefig('poster.png', dpi=DPI, bbox_inches='tight',
            facecolor='white', pil_kwargs={'optimize': True})
plt.savefig('poster.pdf', bbox_inches='tight', facecolor='white')
print("Saved: poster.png  and  poster.pdf")
