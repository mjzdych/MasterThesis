"""
trajectory_analysis.py

Derives standing/propagating trajectory maps from predicted CC and BC fields,
validates against observed is_heatwave, and produces all thesis figures.

Prerequisites:
  - inference_only.py must have been run first to generate pred/true/times .npy files
  - CC, BC, and HW runs must exist for the chosen REGION and ABLATION

Outputs (saved to BASE_OUT/trajectories_{REGION}_{ABLATION}/):
  fig1_trajectory_sequence.png   — spatial maps over test period
  fig2_temporal_evolution.png    — time series of trajectory fractions vs HW
  fig3_climatological_maps.png   — mean CC, BC, dominant type
  fig4_jaccard_validation.png    — propagating → future HW overlap
  trajectory_summary.txt         — all stats
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy import stats as sp_stats

# ============================================================
# CONFIG — change REGION and ABLATION as needed
# ============================================================

BASE_OUT = "/gpfs/home2/mzdych/thesis/experiments"

REGION   = "full_europe_2010"   # options: scandinavia_2018 / iberia_2003 /
                                #          eastern_europe_2010 / full_europe_2010 etc.
ABLATION = "cn_era5"            # cn_era5 / cn_only / era5_only

CC_DIR  = os.path.join(BASE_OUT, f"{REGION}_cc_{ABLATION}")
BC_DIR  = os.path.join(BASE_OUT, f"{REGION}_bc_{ABLATION}")
HW_DIR  = os.path.join(BASE_OUT, f"{REGION}_hw_{ABLATION}")
OUT_DIR = os.path.join(BASE_OUT, f"trajectories_{REGION}_{ABLATION}")
os.makedirs(OUT_DIR, exist_ok=True)

# Thresholds: top-40% of test distribution for each metric
# Based on Mondal (2021): high CC = standing, high BC = propagating
CC_Q = 0.60
BC_Q = 0.60

# ============================================================
# STEP 1 — LOAD PREDICTIONS
# ============================================================

print(f"Loading predictions for region={REGION}  ablation={ABLATION}")
print(f"  CC dir : {CC_DIR}")
print(f"  BC dir : {BC_DIR}")
print(f"  HW dir : {HW_DIR}")

# Check all required files exist before proceeding
for label, d in [("CC", CC_DIR), ("BC", BC_DIR), ("HW", HW_DIR)]:
    for fname in ["pred_TEST.npy", "true_TEST.npy", "times_TEST.npy"]:
        path = os.path.join(d, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {label} file: {path}\n"
                f"Run inference_only.py first."
            )

# Load — all arrays are (N, 1, H, W), squeeze channel dim → (N, H, W)
pred_cc_norm = np.load(os.path.join(CC_DIR, "pred_TEST.npy"))[:, 0]
true_cc_norm = np.load(os.path.join(CC_DIR, "true_TEST.npy"))[:, 0]

pred_bc_norm = np.load(os.path.join(BC_DIR, "pred_TEST.npy"))[:, 0]
true_bc_norm = np.load(os.path.join(BC_DIR, "true_TEST.npy"))[:, 0]

# HW: pred is sigmoid probabilities, true is raw 0/1 (no normalisation for binary)
pred_hw_prob = np.load(os.path.join(HW_DIR, "pred_TEST.npy"))[:, 0]
true_hw      = np.load(os.path.join(HW_DIR, "true_TEST.npy"))[:, 0].astype(int)

# Times — load from CC dir (all three should be identical)
test_times = np.load(os.path.join(CC_DIR, "times_TEST.npy"), allow_pickle=True)
test_dates = pd.DatetimeIndex(test_times)

print(f"  Shapes — CC:{pred_cc_norm.shape}  BC:{pred_bc_norm.shape}  HW:{true_hw.shape}")
print(f"  Test period: {test_dates[0].date()} → {test_dates[-1].date()}")

# ============================================================
# STEP 2 — INVERSE TRANSFORM CC AND BC
# ============================================================

# CC: standard z-score only (no log1p)
y_mean_cc = float(np.load(os.path.join(CC_DIR, "y_mean.npy")))
y_std_cc  = float(np.load(os.path.join(CC_DIR, "y_std.npy")))
pred_cc   = np.clip(pred_cc_norm * y_std_cc + y_mean_cc, 0.0, None)
true_cc   = np.clip(true_cc_norm * y_std_cc + y_mean_cc, 0.0, None)

# BC: log1p + z-score → inverse: un-zscore then expm1
y_mean_bc = float(np.load(os.path.join(BC_DIR, "y_mean.npy")))
y_std_bc  = float(np.load(os.path.join(BC_DIR, "y_std.npy")))
pred_bc   = np.clip(np.expm1(pred_bc_norm * y_std_bc + y_mean_bc), 0.0, None)
true_bc   = np.clip(np.expm1(true_bc_norm * y_std_bc + y_mean_bc), 0.0, None)

print(f"  CC range (pred): [{pred_cc.min():.4f}, {pred_cc.max():.4f}]")
print(f"  BC range (pred): [{pred_bc.min():.6f}, {pred_bc.max():.6f}]")
print(f"  HW positive rate: {true_hw.mean():.4f}")

# ============================================================
# STEP 3 — CLASSIFY PIXELS: STANDING / PROPAGATING
# ============================================================
# Mondal (2021):
#   High CC + Low BC  → Standing  (locally cohesive, seldom propagates far)
#   Low CC  + High BC → Propagating (bottleneck node, relays HW to other regions)
#   High CC + High BC → Transitional
#   Low CC  + Low BC  → Inactive

cc_thresh = np.quantile(pred_cc, CC_Q)
bc_thresh = np.quantile(pred_bc, BC_Q)
print(f"\nClassification thresholds (q={CC_Q}):")
print(f"  CC threshold: {cc_thresh:.4f}")
print(f"  BC threshold: {bc_thresh:.6f}")

standing     = (pred_cc >= cc_thresh) & (pred_bc <  bc_thresh)
propagating  = (pred_cc <  cc_thresh) & (pred_bc >= bc_thresh)
transitional = (pred_cc >= cc_thresh) & (pred_bc >= bc_thresh)
# inactive = neither → label 0

traj_map = np.zeros_like(pred_cc, dtype=np.int8)
traj_map[standing]     = 1
traj_map[propagating]  = 2
traj_map[transitional] = 3

print("\nTrajectory label distribution (mean over test period):")
for label, name in [(0,"inactive"),(1,"standing"),(2,"propagating"),(3,"transitional")]:
    print(f"  {name:15s}: {(traj_map == label).mean():.3f}")

# ============================================================
# STEP 4 — VALIDATE: PROPAGATING PIXELS → FUTURE HW
# ============================================================
# Key thesis claim: predicted propagating nodes at day t
# spatially overlap with observed HW at day t+k
# Operationalises Wang et al. (2025) R_HO predictability window

print("\n--- Propagating → HW Jaccard validation ---")
print(f"{'Lead':>6}  {'Propag Jaccard':>16}  {'Standing Jaccard':>18}  {'N':>5}")

jaccard_prop  = {}
jaccard_stand = {}

for lead in [1, 2, 3]:
    T = traj_map.shape[0]
    j_prop, j_stand = [], []
    for t in range(T - lead):
        hw_tk = true_hw[t + lead].astype(float)
        if hw_tk.sum() == 0:
            continue

        prop_t  = (traj_map[t] == 2).astype(float)
        stand_t = (traj_map[t] == 1).astype(float)

        if prop_t.sum() > 0:
            inter = (prop_t * hw_tk).sum()
            union = ((prop_t + hw_tk) > 0).sum()
            j_prop.append(inter / (union + 1e-8))

        if stand_t.sum() > 0:
            inter = (stand_t * hw_tk).sum()
            union = ((stand_t + hw_tk) > 0).sum()
            j_stand.append(inter / (union + 1e-8))

    jaccard_prop[lead]  = j_prop
    jaccard_stand[lead] = j_stand
    mp = np.mean(j_prop)  if j_prop  else 0.0
    ms = np.mean(j_stand) if j_stand else 0.0
    print(f"  {lead:>4}d  {mp:>16.4f}  {ms:>18.4f}  {len(j_prop):>5}")

# ============================================================
# STEP 5 — TEMPORAL SUMMARY TABLE
# ============================================================

print(f"\n{'Date':<12} {'Standing%':>10} {'Propag%':>10} "
      f"{'Trans%':>10} {'Inactive%':>10} {'HW%':>8}")
for t in range(len(test_dates)):
    s   = (traj_map[t] == 1).mean() * 100
    p   = (traj_map[t] == 2).mean() * 100
    tr  = (traj_map[t] == 3).mean() * 100
    inc = (traj_map[t] == 0).mean() * 100
    hw  = true_hw[t].mean() * 100
    print(f"{str(test_dates[t].date()):<12} {s:>10.1f} {p:>10.1f} "
          f"{tr:>10.1f} {inc:>10.1f} {hw:>8.1f}")

# ============================================================
# FIGURE SETUP
# ============================================================

cmap_traj   = mcolors.ListedColormap(["#f0f0f0", "#2166ac", "#d6604d", "#762a83"])
bounds      = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm_traj   = mcolors.BoundaryNorm(bounds, cmap_traj.N)
traj_labels = ["Inactive", "Standing\n(high CC)", "Propagating\n(high BC)", "Transitional"]
region_title = REGION.replace("_", " ").title()

# ============================================================
# FIGURE 1 — TRAJECTORY SEQUENCE
# ============================================================

plot_days = list(range(0, len(test_dates), max(1, len(test_dates) // 12)))[:12]
ncols = 4
nrows = -(-len(plot_days) // ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3))
axes = axes.flatten()

for i, t in enumerate(plot_days):
    ax = axes[i]
    ax.imshow(traj_map[t], cmap=cmap_traj, norm=norm_traj,
              origin="upper", aspect="auto")
    if true_hw[t].sum() > 0:
        ax.contour(true_hw[t], levels=[0.5], colors="black",
                   linewidths=1.5, linestyles="--")
    ax.set_title(str(test_dates[t].date()), fontsize=8)
    ax.axis("off")

for j in range(len(plot_days), len(axes)):
    axes[j].axis("off")

legend_elements = [Patch(facecolor=cmap_traj(i / 3), label=traj_labels[i])
                   for i in range(4)]
legend_elements.append(
    plt.Line2D([0], [0], color="black", linestyle="--",
               linewidth=1.5, label="Observed HW")
)
fig.legend(handles=legend_elements, loc="lower center", ncol=5,
           fontsize=8, bbox_to_anchor=(0.5, 0.0))
fig.suptitle(
    f"Heatwave Trajectory — {region_title}\n"
    f"Standing (high CC) vs Propagating (high BC)  |  Mondal (2021)",
    fontsize=11, y=1.01,
)
plt.tight_layout(rect=[0, 0.06, 1, 1])
path = os.path.join(OUT_DIR, "fig1_trajectory_sequence.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {path}")

# ============================================================
# FIGURE 2 — TEMPORAL EVOLUTION
# ============================================================

t_axis     = np.arange(len(test_dates))
stand_frac = np.array([(traj_map[t] == 1).mean() for t in t_axis])
prop_frac  = np.array([(traj_map[t] == 2).mean() for t in t_axis])
hw_frac    = np.array([true_hw[t].mean()          for t in t_axis])

fig, ax1 = plt.subplots(figsize=(12, 4))
ax2 = ax1.twinx()

ax1.fill_between(t_axis, stand_frac, alpha=0.5, color="#2166ac", label="Standing fraction")
ax1.fill_between(t_axis, prop_frac,  alpha=0.5, color="#d6604d", label="Propagating fraction")
ax2.plot(t_axis, hw_frac, color="black", linewidth=1.5,
         linestyle="--", label="Observed HW fraction")

ax1.set_xlabel("Test day")
ax1.set_ylabel("Fraction of pixels")
ax2.set_ylabel("HW pixel fraction")
tick_step = max(1, len(t_axis) // 10)
ax1.set_xticks(t_axis[::tick_step])
ax1.set_xticklabels(
    [str(test_dates[t].date()) for t in t_axis[::tick_step]],
    rotation=45, ha="right", fontsize=7,
)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
ax1.set_title(f"Temporal evolution of trajectory types — {region_title}")
plt.tight_layout()
path = os.path.join(OUT_DIR, "fig2_temporal_evolution.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved: {path}")

# ============================================================
# FIGURE 3 — CLIMATOLOGICAL MAPS
# ============================================================

fig, axes = plt.subplots(1, 4, figsize=(18, 4))

im0 = axes[0].imshow(pred_cc.mean(axis=0), cmap="YlOrRd",
                     origin="upper", aspect="auto")
axes[0].set_title("Mean predicted CC")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(pred_bc.mean(axis=0), cmap="PuBu",
                     origin="upper", aspect="auto")
axes[1].set_title("Mean predicted BC")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

im2 = axes[2].imshow(true_cc.mean(axis=0), cmap="YlOrRd",
                     origin="upper", aspect="auto")
axes[2].set_title("Mean observed CC")
plt.colorbar(im2, ax=axes[2], fraction=0.046)

# Dominant trajectory type per pixel — fixed for SciPy 1.9+
traj_mode = sp_stats.mode(traj_map, axis=0, keepdims=False).mode
im3 = axes[3].imshow(traj_mode, cmap=cmap_traj, norm=norm_traj,
                     origin="upper", aspect="auto")
axes[3].set_title("Dominant trajectory type")
legend_elements = [Patch(facecolor=cmap_traj(i / 3), label=traj_labels[i])
                   for i in range(4)]
axes[3].legend(handles=legend_elements, loc="lower left", fontsize=6)

plt.suptitle(f"Climatological CN maps — {region_title}", fontsize=12)
plt.tight_layout()
path = os.path.join(OUT_DIR, "fig3_climatological_maps.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved: {path}")

# ============================================================
# FIGURE 4 — JACCARD VALIDATION BAR CHART
# ============================================================

leads      = [1, 2, 3]
prop_means = [np.mean(jaccard_prop[l])  if jaccard_prop[l]  else 0.0 for l in leads]
prop_stds  = [np.std(jaccard_prop[l])   if jaccard_prop[l]  else 0.0 for l in leads]
stand_means= [np.mean(jaccard_stand[l]) if jaccard_stand[l] else 0.0 for l in leads]
stand_stds = [np.std(jaccard_stand[l])  if jaccard_stand[l] else 0.0 for l in leads]

x   = np.arange(len(leads))
w   = 0.35
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - w/2, prop_means,  w, yerr=prop_stds,  capsize=5,
       color="#d6604d", alpha=0.85, label="Propagating → HW")
ax.bar(x + w/2, stand_means, w, yerr=stand_stds, capsize=5,
       color="#2166ac", alpha=0.85, label="Standing → HW")
ax.set_xticks(x)
ax.set_xticklabels([f"Lead +{l}d" for l in leads])
ax.set_ylabel("Mean Jaccard overlap")
ax.set_title(
    f"Predicted node type → future HW overlap\n{region_title}"
)
ax.legend()
plt.tight_layout()
path = os.path.join(OUT_DIR, "fig4_jaccard_validation.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved: {path}")

# ============================================================
# SAVE SUMMARY STATS
# ============================================================

with open(os.path.join(OUT_DIR, "trajectory_summary.txt"), "w") as f:
    f.write(f"Region        : {REGION}\n")
    f.write(f"Ablation      : {ABLATION}\n")
    f.write(f"CC threshold  (q={CC_Q}): {cc_thresh:.4f}\n")
    f.write(f"BC threshold  (q={BC_Q}): {bc_thresh:.6f}\n\n")
    f.write("Label distribution (mean over test period):\n")
    for label, name in [(0,"inactive"),(1,"standing"),(2,"propagating"),(3,"transitional")]:
        f.write(f"  {name:15s}: {(traj_map == label).mean():.4f}\n")
    f.write("\nPropagating → HW Jaccard:\n")
    for lead in leads:
        mu = np.mean(jaccard_prop[lead])  if jaccard_prop[lead]  else 0.0
        sd = np.std(jaccard_prop[lead])   if jaccard_prop[lead]  else 0.0
        f.write(f"  lead={lead}d  mean={mu:.4f}  std={sd:.4f}  n={len(jaccard_prop[lead])}\n")
    f.write("\nStanding → HW Jaccard (baseline):\n")
    for lead in leads:
        mu = np.mean(jaccard_stand[lead]) if jaccard_stand[lead] else 0.0
        sd = np.std(jaccard_stand[lead])  if jaccard_stand[lead] else 0.0
        f.write(f"  lead={lead}d  mean={mu:.4f}  std={sd:.4f}  n={len(jaccard_stand[lead])}\n")

print(f"\nAll outputs saved to: {OUT_DIR}")
print("Done.")
