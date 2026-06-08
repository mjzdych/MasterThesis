"""
trajectory_analysis_cc_dc.py

Derives standing/propagating trajectory maps from predicted CC and DC fields.
Replaces BC with DC after BC was found to be uniformly near-zero in the
gridded European CN (see thesis Section X).

Classification (Mondal 2021 + van der Geest 2024):
  Standing    : high CC, low DC  — locally cohesive, few long-range connections
  Propagating : low CC, high DC  — many connections, not locally clustered (relay node)
  Transitional: high CC, high DC — active hub (both locally cohesive AND well-connected)
  Inactive    : low CC, low DC   — weakly connected, not part of active HW network

Prerequisites:
  - inference_only.py run for CC, DC, and HW tasks for chosen REGION/ABLATION

Outputs saved to BASE_OUT/trajectories_{REGION}_{ABLATION}_ccdc/:
  fig1_trajectory_sequence.png
  fig2_temporal_evolution.png
  fig3_climatological_maps.png
  fig4_jaccard_validation.png
  fig5_cc_dc_diff_distribution.png
  trajectory_summary.txt
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
# CONFIG
# ============================================================

BASE_OUT = "/gpfs/home2/mzdych/thesis/experiments"

REGION   = "iberia_2003"   # change per region
ABLATION = "cn_era5"

CC_DIR  = os.path.join(BASE_OUT, f"{REGION}_cc_{ABLATION}")
DC_DIR  = os.path.join(BASE_OUT, f"{REGION}_dc_{ABLATION}")
HW_DIR  = os.path.join(BASE_OUT, f"{REGION}_hw_{ABLATION}")
OUT_DIR = os.path.join(BASE_OUT, f"trajectories_{REGION}_{ABLATION}_ccdc")
os.makedirs(OUT_DIR, exist_ok=True)

# Classification margin: how much larger CC_norm must exceed DC_norm
# (or vice versa) to be classified as standing/propagating vs transitional
MARGIN     = 0.15

# Pixels below this quantile on BOTH metrics → inactive
INACTIVE_Q = 0.40

# ============================================================
# STEP 1 — CHECK FILES + LOAD
# ============================================================

print(f"Region={REGION}  Ablation={ABLATION}")
print(f"  CC  : {CC_DIR}")
print(f"  DC  : {DC_DIR}")
print(f"  HW  : {HW_DIR}")

for label, d in [("CC", CC_DIR), ("DC", DC_DIR), ("HW", HW_DIR)]:
    for fname in ["pred_TEST.npy", "true_TEST.npy", "times_TEST.npy"]:
        path = os.path.join(d, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {label} file: {path}\n"
                f"Run inference_only.py first for task={label.lower()}."
            )

# Load — (N, 1, H, W) → squeeze to (N, H, W)
pred_cc_norm = np.load(os.path.join(CC_DIR, "pred_TEST.npy"))[:, 0]
true_cc_norm = np.load(os.path.join(CC_DIR, "true_TEST.npy"))[:, 0]
pred_dc_norm = np.load(os.path.join(DC_DIR, "pred_TEST.npy"))[:, 0]
true_dc_norm = np.load(os.path.join(DC_DIR, "true_TEST.npy"))[:, 0]
pred_hw_prob = np.load(os.path.join(HW_DIR, "pred_TEST.npy"))[:, 0]
true_hw      = np.load(os.path.join(HW_DIR, "true_TEST.npy"))[:, 0].astype(int)
test_times   = np.load(os.path.join(CC_DIR, "times_TEST.npy"), allow_pickle=True)
test_dates   = pd.DatetimeIndex(test_times)

print(f"  Shapes — CC:{pred_cc_norm.shape}  DC:{pred_dc_norm.shape}  HW:{true_hw.shape}")
print(f"  Test period: {test_dates[0].date()} → {test_dates[-1].date()}")

# ============================================================
# STEP 2 — INVERSE TRANSFORM
# ============================================================

# CC: standard z-score (no log1p)
y_mean_cc = float(np.load(os.path.join(CC_DIR, "y_mean.npy")))
y_std_cc  = float(np.load(os.path.join(CC_DIR, "y_std.npy")))
pred_cc   = np.clip(pred_cc_norm * y_std_cc + y_mean_cc, 0.0, None)
true_cc   = np.clip(true_cc_norm * y_std_cc + y_mean_cc, 0.0, None)

# DC: standard z-score (no log1p)
y_mean_dc = float(np.load(os.path.join(DC_DIR, "y_mean.npy")))
y_std_dc  = float(np.load(os.path.join(DC_DIR, "y_std.npy")))
pred_dc   = np.clip(pred_dc_norm * y_std_dc + y_mean_dc, 0.0, None)
true_dc   = np.clip(true_dc_norm * y_std_dc + y_mean_dc, 0.0, None)

print(f"\n  CC range (pred): [{pred_cc.min():.4f}, {pred_cc.max():.4f}]  "
      f"mean={pred_cc.mean():.4f}")
print(f"  DC range (pred): [{pred_dc.min():.6f}, {pred_dc.max():.6f}]  "
      f"mean={pred_dc.mean():.6f}")
print(f"  HW positive rate: {true_hw.mean():.4f}")

# Sanity check: DC should have meaningful range
dc_range = pred_dc.max() - pred_dc.min()
cc_range = pred_cc.max() - pred_cc.min()
print(f"\n  CC range span: {cc_range:.4f}")
print(f"  DC range span: {dc_range:.6f}")
if dc_range < 1e-4:
    print("  WARNING: DC range very small — classification may be unreliable")
else:
    print("  DC range OK for classification")

# ============================================================
# STEP 3 — NORMALISE TO [0,1] AND CLASSIFY
# ============================================================

cc_norm01 = (pred_cc - pred_cc.min()) / (cc_range + 1e-8)
dc_norm01 = (pred_dc - pred_dc.min()) / (dc_range + 1e-8)

cc_iq = np.quantile(cc_norm01, INACTIVE_Q)
dc_iq = np.quantile(dc_norm01, INACTIVE_Q)

inactive     = (cc_norm01 < cc_iq) & (dc_norm01 < dc_iq)
standing     = (~inactive) & ((cc_norm01 - dc_norm01) >  MARGIN)
propagating  = (~inactive) & ((dc_norm01 - cc_norm01) >  MARGIN)
transitional = (~inactive) & (~standing) & (~propagating)

traj_map = np.zeros_like(pred_cc, dtype=np.int8)
traj_map[standing]     = 1
traj_map[propagating]  = 2
traj_map[transitional] = 3

print(f"\nClassification (MARGIN={MARGIN}, INACTIVE_Q={INACTIVE_Q}):")
for label, name in [(0,"inactive"),(1,"standing"),(2,"propagating"),(3,"transitional")]:
    print(f"  {name:15s}: {(traj_map == label).mean():.3f}")

# ============================================================
# STEP 4 — JACCARD VALIDATION
# ============================================================

print(f"\n{'Lead':>6}  {'Propag Jaccard':>16}  {'Standing Jaccard':>18}  {'N':>5}")
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
# STEP 5 — TEMPORAL TABLE
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

cmap_traj    = mcolors.ListedColormap(["#f0f0f0", "#2166ac", "#d6604d", "#762a83"])
bounds       = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm_traj    = mcolors.BoundaryNorm(bounds, cmap_traj.N)
traj_labels  = ["Inactive", "Standing\n(high CC)", "Propagating\n(high DC)", "Transitional"]
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
legend_elements.append(plt.Line2D([0], [0], color="black", linestyle="--",
                                   linewidth=1.5, label="Observed HW"))
fig.legend(handles=legend_elements, loc="lower center", ncol=5,
           fontsize=8, bbox_to_anchor=(0.5, 0.0))
fig.suptitle(
    f"Heatwave Trajectory — {region_title}\n"
    f"Standing (high CC) vs Propagating (high DC)  |  margin={MARGIN}",
    fontsize=11, y=1.01)
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
trans_frac = np.array([(traj_map[t] == 3).mean() for t in t_axis])
hw_frac    = np.array([true_hw[t].mean()          for t in t_axis])

fig, ax1 = plt.subplots(figsize=(12, 4))
ax2 = ax1.twinx()
ax1.fill_between(t_axis, stand_frac, alpha=0.5, color="#2166ac", label="Standing (high CC)")
ax1.fill_between(t_axis, prop_frac,  alpha=0.5, color="#d6604d", label="Propagating (high DC)")
ax1.fill_between(t_axis, trans_frac, alpha=0.25, color="#762a83", label="Transitional")
ax2.plot(t_axis, hw_frac, color="black", linewidth=1.5,
         linestyle="--", label="Observed HW fraction")
ax1.set_xlabel("Test day")
ax1.set_ylabel("Fraction of pixels")
ax2.set_ylabel("HW pixel fraction")
tick_step = max(1, len(t_axis) // 10)
ax1.set_xticks(t_axis[::tick_step])
ax1.set_xticklabels([str(test_dates[t].date()) for t in t_axis[::tick_step]],
                    rotation=45, ha="right", fontsize=7)
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

fig, axes = plt.subplots(1, 5, figsize=(22, 4))

im0 = axes[0].imshow(pred_cc.mean(axis=0), cmap="YlOrRd", origin="upper", aspect="auto")
axes[0].set_title("Mean predicted CC")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(pred_dc.mean(axis=0), cmap="YlGn", origin="upper", aspect="auto")
axes[1].set_title("Mean predicted DC")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

im2 = axes[2].imshow(true_cc.mean(axis=0), cmap="YlOrRd", origin="upper", aspect="auto")
axes[2].set_title("Mean observed CC")
plt.colorbar(im2, ax=axes[2], fraction=0.046)

im3 = axes[3].imshow(true_dc.mean(axis=0), cmap="YlGn", origin="upper", aspect="auto")
axes[3].set_title("Mean observed DC")
plt.colorbar(im3, ax=axes[3], fraction=0.046)

traj_mode = sp_stats.mode(traj_map, axis=0, keepdims=False).mode
im4 = axes[4].imshow(traj_mode, cmap=cmap_traj, norm=norm_traj,
                     origin="upper", aspect="auto")
axes[4].set_title("Dominant trajectory type")
legend_elements = [Patch(facecolor=cmap_traj(i / 3), label=traj_labels[i])
                   for i in range(4)]
axes[4].legend(handles=legend_elements, loc="lower left", fontsize=6)

plt.suptitle(f"Climatological CN maps — {region_title}", fontsize=12)
plt.tight_layout()
path = os.path.join(OUT_DIR, "fig3_climatological_maps.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved: {path}")

# ============================================================
# FIGURE 4 — JACCARD VALIDATION
# ============================================================

leads      = [1, 2, 3]
prop_means = [np.mean(jaccard_prop[l])  if jaccard_prop[l]  else 0.0 for l in leads]
prop_stds  = [np.std(jaccard_prop[l])   if jaccard_prop[l]  else 0.0 for l in leads]
stand_means= [np.mean(jaccard_stand[l]) if jaccard_stand[l] else 0.0 for l in leads]
stand_stds = [np.std(jaccard_stand[l])  if jaccard_stand[l] else 0.0 for l in leads]

x = np.arange(len(leads))
w = 0.35
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - w/2, prop_means,  w, yerr=prop_stds,  capsize=5,
       color="#d6604d", alpha=0.85, label="Propagating (high DC) → HW")
ax.bar(x + w/2, stand_means, w, yerr=stand_stds, capsize=5,
       color="#2166ac", alpha=0.85, label="Standing (high CC) → HW")
ax.set_xticks(x)
ax.set_xticklabels([f"Lead +{l}d" for l in leads])
ax.set_ylabel("Mean Jaccard overlap")
ax.set_title(f"Predicted node type → future HW overlap\n{region_title}")
ax.legend()
plt.tight_layout()
path = os.path.join(OUT_DIR, "fig4_jaccard_validation.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved: {path}")

# ============================================================
# FIGURE 5 — CC−DC DIFFERENCE DISTRIBUTION
# ============================================================

diff = (cc_norm01 - dc_norm01).flatten()
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(diff, bins=100, color="#555", alpha=0.7)
ax.axvline( MARGIN, color="#2166ac", linestyle="--", linewidth=2,
            label=f"Standing threshold (+{MARGIN})")
ax.axvline(-MARGIN, color="#d6604d", linestyle="--", linewidth=2,
            label=f"Propagating threshold (−{MARGIN})")
ax.set_xlabel("CC_norm − DC_norm")
ax.set_ylabel("Count")
ax.set_title(f"Distribution of CC−DC difference — {region_title}\n"
             f"Values outside dashed lines → classified as standing/propagating")
ax.legend()
plt.tight_layout()
path = os.path.join(OUT_DIR, "fig5_cc_dc_diff_distribution.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved: {path}")

# ============================================================
# SAVE SUMMARY
# ============================================================

with open(os.path.join(OUT_DIR, "trajectory_summary.txt"), "w") as f:
    f.write(f"Region        : {REGION}\n")
    f.write(f"Ablation      : {ABLATION}\n")
    f.write(f"Metrics used  : CC + DC  (BC dropped — near-zero in gridded CN)\n")
    f.write(f"Method        : relative (CC_norm - DC_norm)\n")
    f.write(f"Margin        : {MARGIN}\n")
    f.write(f"Inactive Q    : {INACTIVE_Q}\n\n")
    f.write(f"CC range      : [{pred_cc.min():.4f}, {pred_cc.max():.4f}]\n")
    f.write(f"DC range      : [{pred_dc.min():.6f}, {pred_dc.max():.6f}]\n\n")
    f.write("Label distribution (mean over test period):\n")
    for label, name in [(0,"inactive"),(1,"standing"),(2,"propagating"),(3,"transitional")]:
        f.write(f"  {name:15s}: {(traj_map == label).mean():.4f}\n")
    f.write("\nPropagating (high DC) → HW Jaccard:\n")
    for lead in leads:
        mu = np.mean(jaccard_prop[lead])  if jaccard_prop[lead]  else 0.0
        sd = np.std(jaccard_prop[lead])   if jaccard_prop[lead]  else 0.0
        f.write(f"  lead={lead}d  mean={mu:.4f}  std={sd:.4f}  n={len(jaccard_prop[lead])}\n")
    f.write("\nStanding (high CC) → HW Jaccard:\n")
    for lead in leads:
        mu = np.mean(jaccard_stand[lead]) if jaccard_stand[lead] else 0.0
        sd = np.std(jaccard_stand[lead])  if jaccard_stand[lead] else 0.0
        f.write(f"  lead={lead}d  mean={mu:.4f}  std={sd:.4f}  n={len(jaccard_stand[lead])}\n")

print(f"\nAll outputs saved to: {OUT_DIR}")
print("Done.")
