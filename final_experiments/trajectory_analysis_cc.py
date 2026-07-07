"""
trajectory_analysis_cc_fixed.py

CC-only trajectory analysis with geographic maps using geopandas + NUTS boundaries.
Uses pcolormesh with real lat/lon arrays from NetCDF for pixel-perfect NUTS alignment.

FIX: Thresholds are now derived from the model's own predicted CC values across
VAL + TEST periods (not the binary CC variable in the raw NetCDF, which is a
pre-thresholded 0/1 label and unsuitable for continuous quantile thresholds).

Prerequisites: inference_only.py run for CC and HW tasks.
Outputs: BASE_OUT/results_{REGION}_{ABLATION}_cconly/
"""

import os
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import geopandas as gpd
import xarray as xr
from scipy.ndimage import label as nd_label

# ============================================================
# CONFIG
# ============================================================

# Snellius 
# DATA_FILE = "/gpfs/home2/mzdych/thesis/full_processed_training_dataset.nc"
# BASE_OUT  = "/gpfs/home2/mzdych/thesis/experiments"
# NUTS_FILE = "/gpfs/home2/mzdych/thesis/NUTS_RG_01M_2024_4326_LEVL_2.geojson"

# for running convlstm change the paths of CC_DIR and HW_DIR and BASE_OUT just experiments, plus ALREADY_DENORMALIZED set to False

# Local
DATA_FILE = "full_processed_training_dataset.nc"
BASE_OUT  = "experiments"
NUTS_FILE = "NUTS_RG_01M_2024_4326_LEVL_2.geojson"
SEQ_LEN = 14

REGION   = "scandinavia_2018"
ABLATION = "cn_era5"

CC_HIGH_Q          = 0.70
CC_LOW_Q           = 0.40
MIN_COMPONENT_SIZE = 50

# ── Figure 8: Predicted vs True CC comparison ────────────────────────────────
# Set the start date for the 12-consecutive-day comparison panel.
# Must be a date within the test period (format: "YYYY-MM-DD").
# Set to None to auto-select (uses the day of peak active fraction).
COMPARISON_START_DATE = "2003-07-15"   # ← change this to any date in the test period
N_COMPARISON_DAYS     = 5             # number of consecutive days to show
# ─────────────────────────────────────────────────────────────────────────────

REGION_BOUNDS = {
    "eastern_europe_2010": {"lat_min": 45, "lat_max": 55, "lon_min": 20,  "lon_max": 40},
    "iberia_2003":         {"lat_min": 36, "lat_max": 44, "lon_min": -10, "lon_max":  5},
    "scandinavia_2018":    {"lat_min": 55, "lat_max": 65, "lon_min":  5,  "lon_max": 30},
    "full_europe_2003":    {"lat_min": 35, "lat_max": 71, "lon_min": -25, "lon_max": 45},
    "full_europe_2010":    {"lat_min": 35, "lat_max": 71, "lon_min": -25, "lon_max": 45},
    "full_europe_2018":    {"lat_min": 35, "lat_max": 71, "lon_min": -25, "lon_max": 45},
    "north_europe_2018":   {"lat_min": 55, "lat_max": 71, "lon_min": -25, "lon_max": 45},
    "mediterranean_2003":  {"lat_min": 30, "lat_max": 48, "lon_min": -10, "lon_max": 40},
}

# CC_DIR  = os.path.join(BASE_OUT, f"{REGION}_cc_{ABLATION}_transformer") # _transformer
# HW_DIR  = os.path.join(BASE_OUT, f"{REGION}_hw_{ABLATION}_transformer")
CC_DIR  = os.path.join(BASE_OUT, f"{REGION}_cc_{ABLATION}") # _transformer
HW_DIR  = os.path.join(BASE_OUT, f"{REGION}_hw_{ABLATION}")
OUT_DIR = os.path.join(BASE_OUT, f"trajectories_{REGION}_{ABLATION}_cconly")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# STEP 1 — LOAD MODEL OUTPUTS
# ============================================================

b       = REGION_BOUNDS[REGION]
LAT_MIN = b["lat_min"]; LAT_MAX = b["lat_max"]
LON_MIN = b["lon_min"]; LON_MAX = b["lon_max"]

print(f"Region={REGION}  Ablation={ABLATION}")
print(f"  Requested bounds: lat=[{LAT_MIN},{LAT_MAX}]  lon=[{LON_MIN},{LON_MAX}]")

for lbl, d in [("CC", CC_DIR), ("HW", HW_DIR)]:
    for fname in ["pred_TEST.npy", "true_TEST.npy", "times_TEST.npy"]:
        p = os.path.join(d, fname)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {lbl} file: {p}")

# pred_cc_norm = np.load(os.path.join(CC_DIR, "pred_TEST.npy"))[:, 0]
# true_cc_norm = np.load(os.path.join(CC_DIR, "true_TEST.npy"))[:, 0]
# pred_hw_prob = np.load(os.path.join(HW_DIR, "pred_TEST.npy"))[:, 0]
# true_hw      = np.load(os.path.join(HW_DIR, "true_TEST.npy"))[:, 0].astype(int)

def load_spatial_npy(path, H=None, W=None):
    """Load pred/true npy handling both (T,1,H,W) and flat/2D shapes."""
    arr = np.load(path)
    if arr.ndim == 4:
        return arr[:, 0]           # (T,1,H,W) → (T,H,W)  — ConvLSTM format
    elif arr.ndim == 3:
        return arr                 # (T,H,W)   already correct
    elif arr.ndim == 2:
        return arr                 # (T, H*W)  — unlikely but handle
    elif arr.ndim == 1:
        # fully flattened — need H and W to reshape
        assert H is not None and W is not None, \
            f"Cannot reshape 1D array without H and W: {path}"
        T = len(arr) // (H * W)
        return arr.reshape(T, H, W)
    else:
        raise ValueError(f"Unexpected shape {arr.shape} in {path}")

# Load CC (shape known already)
pred_cc_norm = load_spatial_npy(os.path.join(CC_DIR, "pred_TEST.npy"))
true_cc_norm = load_spatial_npy(os.path.join(CC_DIR, "true_TEST.npy"))

# Get H, W from CC shape
H_tmp, W_tmp = pred_cc_norm.shape[1], pred_cc_norm.shape[2]

# Load HW with fallback reshape if flat
pred_hw_prob = load_spatial_npy(os.path.join(HW_DIR, "pred_TEST.npy"), H_tmp, W_tmp)
true_hw      = load_spatial_npy(os.path.join(HW_DIR, "true_TEST.npy"), H_tmp, W_tmp).astype(int)

test_times   = np.load(os.path.join(CC_DIR, "times_TEST.npy"), allow_pickle=True)
test_dates = pd.DatetimeIndex(test_times)[SEQ_LEN:]

y_mean_cc = float(np.load(os.path.join(CC_DIR, "y_mean.npy")))
y_std_cc  = float(np.load(os.path.join(CC_DIR, "y_std.npy")))
# pred_cc   = np.clip(pred_cc_norm * y_std_cc + y_mean_cc, 0.0, None)
# true_cc   = np.clip(true_cc_norm * y_std_cc + y_mean_cc, 0.0, None)


ALREADY_DENORMALIZED = True  # set True for transformer outputs

if ALREADY_DENORMALIZED:
    pred_cc = np.clip(pred_cc_norm, 0.0, None)
    true_cc = np.clip(true_cc_norm, 0.0, None)
else:
    pred_cc = np.clip(pred_cc_norm * y_std_cc + y_mean_cc, 0.0, None)
    true_cc = np.clip(true_cc_norm * y_std_cc + y_mean_cc, 0.0, None)

H, W = pred_cc.shape[1], pred_cc.shape[2]
print(f"  pred_cc shape: {pred_cc.shape}  →  {H} rows × {W} cols")
print(f"  Test period: {test_dates[0].date()} → {test_dates[-1].date()}")

# ============================================================
# STEP 2 — LOAD REAL LAT/LON FROM NETCDF AND MATCH TO pred_cc
# ============================================================

ds_coords = xr.open_dataset(DATA_FILE)
lat_full  = ds_coords["lat"].values   # ascending: 35.8 → 71.0
lon_full  = ds_coords["lon"].values   # ascending: -25.0 → 41.0
ds_coords.close()

lat_idx = np.where((lat_full >= LAT_MIN) & (lat_full <= LAT_MAX))[0]
lon_idx = np.where((lon_full >= LON_MIN) & (lon_full <= LON_MAX))[0]

LAT_ARR = lat_full[lat_idx]   # ascending S→N
LON_ARR = lon_full[lon_idx]

print(f"  Coord grid from NetCDF: {len(LAT_ARR)} lat × {len(LON_ARR)} lon")

if len(LAT_ARR) != H:
    print(f"  WARNING: lat count {len(LAT_ARR)} != pred_cc rows {H} — trimming to match")
    LAT_ARR = LAT_ARR[:H] if len(LAT_ARR) > H else LAT_ARR
    if len(LAT_ARR) != H:
        LAT_ARR = lat_full[lat_idx][-H:]
    print(f"  After trim: {len(LAT_ARR)} lat values")

if len(LON_ARR) != W:
    print(f"  WARNING: lon count {len(LON_ARR)} != pred_cc cols {W} — trimming to match")
    LON_ARR = LON_ARR[:W] if len(LON_ARR) > W else LON_ARR
    if len(LON_ARR) != W:
        LON_ARR = lon_full[lon_idx][-W:]
    print(f"  After trim: {len(LON_ARR)} lon values")

assert len(LAT_ARR) == H, f"Cannot reconcile lat: {len(LAT_ARR)} vs {H}"
assert len(LON_ARR) == W, f"Cannot reconcile lon: {len(LON_ARR)} vs {W}"

LAT_MIN_REAL = float(LAT_ARR.min())
LAT_MAX_REAL = float(LAT_ARR.max())
LON_MIN_REAL = float(LON_ARR.min())
LON_MAX_REAL = float(LON_ARR.max())

print(f"  Actual grid bounds: lat=[{LAT_MIN_REAL:.2f},{LAT_MAX_REAL:.2f}]  "
      f"lon=[{LON_MIN_REAL:.2f},{LON_MAX_REAL:.2f}]")

# North-first lat array (matches pred_cc row 0 = northernmost)
LAT_ARR_NF = LAT_ARR[::-1]

# 2D meshgrid for pcolormesh
LON2D, LAT2D = np.meshgrid(LON_ARR, LAT_ARR_NF)

# ============================================================
# STEP 3 — LAND MASK
# ============================================================

def load_land_mask(data_file, lat_arr, lon_arr, target_shape):
    """Load land mask, select region by actual coords, flip to north-first."""
    ds = xr.open_dataset(data_file)
    lm_region = ds["land_mask"].isel(time=0).sel(
        lat=lat_arr, lon=lon_arr, method="nearest"
    ).values.astype(bool)   # south-first
    ds.close()
    lm_nf = lm_region[::-1, :]   # flip to north-first
    print(f"  Land mask shape (N-first): {lm_nf.shape}")
    assert lm_nf.shape == target_shape, \
        f"Land mask {lm_nf.shape} != pred_cc {target_shape}"
    print(f"  NW corner land={lm_nf[0,0]}  "
          f"(lat≈{lat_arr.max():.2f}°N, lon≈{lon_arr.min():.2f}°)")
    print(f"  SE corner land={lm_nf[-1,-1]}  "
          f"(lat≈{lat_arr.min():.2f}°N, lon≈{lon_arr.max():.2f}°)")
    print(f"  Land fraction: {lm_nf.mean():.3f}")
    return lm_nf


print("Loading land mask...")
land_mask = load_land_mask(DATA_FILE, LAT_ARR, LON_ARR, target_shape=(H, W))

# Sea → NaN for all statistics; HW sea stays 0
pred_cc[:, ~land_mask] = np.nan
true_cc[:, ~land_mask] = np.nan
true_hw[:, ~land_mask] = 0

print(f"  CC range (land): [{np.nanmin(pred_cc):.4f}, {np.nanmax(pred_cc):.4f}]")
print(f"  HW positive rate (land): {true_hw[:, land_mask].mean():.4f}")

# ============================================================
# STEP 4 — NUTS BOUNDARIES
# ============================================================

print("Loading NUTS boundaries...")
nuts = gpd.read_file(NUTS_FILE).to_crs("EPSG:4326")
nuts = nuts.cx[LON_MIN_REAL:LON_MAX_REAL, LAT_MIN_REAL:LAT_MAX_REAL]
print(f"  NUTS features clipped to region: {len(nuts)}")

# ============================================================
# STEP 5 — CC THRESHOLDS FROM MODEL PREDICTIONS (VAL + TEST)
# ============================================================
# The raw NetCDF "CC" variable is a binary 0/1 label (pre-thresholded),
# not the continuous clustering coefficient the model predicts.
# Thresholds must therefore be derived from the model's own output.
# We use predicted CC from both VAL and TEST periods, giving a
# multi-period distribution that is physically meaningful and
# comparable across different test events.

print("\nComputing CC thresholds from model predictions (VAL + TEST)...")

val_path = os.path.join(CC_DIR, "pred_VAL.npy")
if os.path.exists(val_path):
    pred_val_norm = load_spatial_npy(val_path, H, W)
    if ALREADY_DENORMALIZED:
        pred_val_cc = np.clip(pred_val_norm, 0.0, None)
    else:
        pred_val_cc = np.clip(pred_val_norm * y_std_cc + y_mean_cc, 0.0, None)
    # Apply the same land mask (spatial grid is identical across splits)
    pred_val_cc[:, ~land_mask] = np.nan
    cc_val_land  = pred_val_cc[:, land_mask].ravel()
    cc_val_land  = cc_val_land[~np.isnan(cc_val_land)]
    print(f"  VAL land-pixel-days : {len(cc_val_land):,}")
    use_val = True
else:
    print("  pred_VAL.npy not found — using TEST period only for thresholds")
    cc_val_land = np.array([])
    use_val = False

cc_test_land = pred_cc[:, land_mask].ravel()
cc_test_land = cc_test_land[~np.isnan(cc_test_land)]
print(f"  TEST land-pixel-days: {len(cc_test_land):,}")

# Combined distribution for threshold calculation
cc_all_land = np.concatenate([cc_val_land]) if use_val else cc_test_land

cc_thresh_high = np.quantile(cc_all_land, CC_HIGH_Q)
cc_thresh_low  = np.quantile(cc_all_land, CC_LOW_Q)

# Keep cc_test_vals alias for Figure 6
cc_test_vals = cc_test_land
cc_ref_vals  = cc_all_land   # VAL+TEST reference distribution (replaces old cc_clim_vals)
ref_label    = "VAL+TEST (model pred.)" if use_val else "TEST (model pred.)"

print(f"\nCC thresholds from {ref_label}:")
print(f"  Active       (q{int(CC_HIGH_Q*100)}): CC >= {cc_thresh_high:.4f}")
print(f"  Transitional (q{int(CC_LOW_Q*100)}):  CC >= {cc_thresh_low:.4f}")
print(f"  Inactive:          CC <  {cc_thresh_low:.4f}")

# ============================================================
# STEP 6 — CLASSIFICATION
# ============================================================

traj_map = np.zeros_like(pred_cc, dtype=np.int8)
traj_map[pred_cc >= cc_thresh_high] = 1
traj_map[(pred_cc >= cc_thresh_low) & (pred_cc < cc_thresh_high)] = 2
traj_map[:, ~land_mask] = 0   # force sea inactive

print("\nLabel distribution (land only):")
for lbl, name in [(0,"inactive"),(1,"active"),(2,"transitional")]:
    frac = (traj_map[:, land_mask] == lbl).mean()
    print(f"  {name:12s}: {frac:.3f}")

# ============================================================
# STEP 7 — JACCARD VALIDATION
# ============================================================

print(f"\n{'Lead':>6}  {'Active Jaccard':>16}  {'Trans Jaccard':>15}  {'N':>5}")
jaccard_active = {}
jaccard_trans  = {}

for lead in [1, 2, 3]:
    T = traj_map.shape[0]
    j_act, j_tr = [], []
    for t in range(T - lead):
        hw_tk = true_hw[t + lead].astype(float)
        if hw_tk.sum() == 0:
            continue
        act_t   = (traj_map[t] == 1).astype(float)
        trans_t = (traj_map[t] == 2).astype(float)
        if act_t.sum() > 0:
            j_act.append((act_t * hw_tk).sum() /
                         (((act_t + hw_tk) > 0).sum() + 1e-8))
        if trans_t.sum() > 0:
            j_tr.append((trans_t * hw_tk).sum() /
                        (((trans_t + hw_tk) > 0).sum() + 1e-8))
    jaccard_active[lead] = j_act
    jaccard_trans[lead]  = j_tr
    ma = np.mean(j_act) if j_act else 0.0
    mt = np.mean(j_tr)  if j_tr  else 0.0
    print(f"  {lead:>4}d  {ma:>16.4f}  {mt:>15.4f}  {len(j_act):>5}")

# ============================================================
# STEP 8 — LCC CENTROID TRACKING
# ============================================================

def pixel_to_latlon(row_idx, col_idx):
    row_i = int(np.clip(np.round(row_idx), 0, H - 1))
    col_i = int(np.clip(np.round(col_idx), 0, W - 1))
    return float(LAT2D[row_i, col_i]), float(LON2D[row_i, col_i])


T_map = traj_map.shape[0]

centroid_lat = np.full(T_map, np.nan)
centroid_lon = np.full(T_map, np.nan)
lcc_size     = np.zeros(T_map)
n_components = np.zeros(T_map)

for t in range(T_map):
    act_mask = (traj_map[t] == 1)
    if act_mask.sum() < MIN_COMPONENT_SIZE:
        continue
    labeled, n_feat = nd_label(act_mask)
    if n_feat == 0:
        continue
    sizes = np.array([(labeled == i).sum() for i in range(1, n_feat + 1)])
    valid = np.where(sizes >= MIN_COMPONENT_SIZE)[0] + 1
    n_components[t] = len(valid)
    if len(valid) == 0:
        continue
    largest    = valid[np.argmax(sizes[valid - 1])]
    rows, cols = np.where(labeled == largest)
    lcc_size[t] = sizes[largest - 1]
    centroid_lat[t], centroid_lon[t] = pixel_to_latlon(rows.mean(), cols.mean())

# Map traj indices back to dates for display
# test_dates covers the full test period; traj predictions start SEQ_LEN days in
traj_dates = test_dates[:T_map]

# ── Observed centroid (from true_hw LCC) ──────────────────────
obs_centroid_lat = np.full(T_map, np.nan)
obs_centroid_lon = np.full(T_map, np.nan)

for t in range(T_map):
    hw_mask = true_hw[t].astype(bool)
    hw_mask[~land_mask] = False
    if hw_mask.sum() < MIN_COMPONENT_SIZE:
        continue
    labeled_hw, n_feat_hw = nd_label(hw_mask)
    if n_feat_hw == 0:
        continue
    sizes_hw = np.array([(labeled_hw == i).sum()
                         for i in range(1, n_feat_hw + 1)])
    largest_hw = np.argmax(sizes_hw) + 1
    if sizes_hw[largest_hw - 1] < MIN_COMPONENT_SIZE:
        continue
    rows_hw, cols_hw = np.where(labeled_hw == largest_hw)
    obs_centroid_lat[t], obs_centroid_lon[t] = pixel_to_latlon(
        rows_hw.mean(), cols_hw.mean()
    )

print(f"  Predicted centroids valid: {(~np.isnan(centroid_lat)).sum()} / {T_map}")
print(f"  Observed  centroids valid: {(~np.isnan(obs_centroid_lat)).sum()} / {T_map}")

print("\nLarge centroid jumps (>5° in one day):")
found_jump = False
for t in range(1, T_map):
    if np.isnan(centroid_lat[t]) or np.isnan(centroid_lat[t-1]):
        continue
    dlat = abs(centroid_lat[t] - centroid_lat[t-1])
    dlon = abs(centroid_lon[t] - centroid_lon[t-1])
    if dlat > 5 or dlon > 5:
        print(f"  {traj_dates[t].date()}: Δlat={dlat:.1f}° Δlon={dlon:.1f}°  "
              f"({centroid_lon[t-1]:.1f}→{centroid_lon[t]:.1f}°)")
        found_jump = True
if not found_jump:
    print("  None detected.")

# ============================================================
# STEP 9 — TEMPORAL TABLE
# ============================================================

T_map  = traj_map.shape[0]  # already set in Step 8, reassigned for clarity
t_axis = np.arange(T_map)
act_frac = np.array([(traj_map[t, land_mask] == 1).mean() for t in t_axis])
tr_frac  = np.array([(traj_map[t, land_mask] == 2).mean() for t in t_axis])
hw_frac  = np.array([true_hw[t, land_mask].mean()          for t in t_axis])

print(f"\n{'Date':<12} {'Active%':>8} {'HW%':>8} {'LCC_size':>10} "
      f"{'N_comp':>8} {'Lat':>8} {'Lon':>8}")
for t in range(T_map):
    print(f"{str(traj_dates[t].date()):<12} "
          f"{act_frac[t]*100:>8.1f} "
          f"{hw_frac[t]*100:>8.1f} "
          f"{lcc_size[t]:>10.0f} {n_components[t]:>8.0f} "
          f"{centroid_lat[t]:>8.2f} {centroid_lon[t]:>8.2f}")

# ============================================================
# FIGURE HELPERS
# ============================================================

cmap_traj   = mcolors.ListedColormap(["#f0f0f0", "#d73027", "#fdae61"])
bnds        = [-0.5, 0.5, 1.5, 2.5]
norm_traj   = mcolors.BoundaryNorm(bnds, cmap_traj.N)
traj_labels = ["Inactive", "Active (high CC)", "Transitional (mid CC)"]
rtitle      = REGION.replace("_", " ").title()
MID_LAT     = (LAT_MIN_REAL + LAT_MAX_REAL) / 2
GEO_ASPECT  = 1.0 / np.cos(np.radians(MID_LAT))
leads       = [1, 2, 3]


def setup_geo_ax(ax):
    ax.set_aspect(GEO_ASPECT)
    ax.set_xlim(LON_MIN_REAL, LON_MAX_REAL)
    ax.set_ylim(LAT_MIN_REAL, LAT_MAX_REAL)


def geo_ticks(ax, n_lon=4, n_lat=4, fontsize=7):
    lons = np.linspace(LON_MIN_REAL, LON_MAX_REAL, n_lon)
    lats = np.linspace(LAT_MIN_REAL, LAT_MAX_REAL, n_lat)
    ax.set_xticks(lons)
    ax.set_yticks(lats)
    ax.set_xticklabels(
        [f"{v:.0f}°E" if v >= 0 else f"{abs(v):.0f}°W" for v in lons],
        fontsize=fontsize)
    ax.set_yticklabels([f"{v:.0f}°N" for v in lats], fontsize=fontsize)


# ============================================================
# FIGURE 0 — DEBUG LAND MASK
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))
ax.pcolormesh(LON2D, LAT2D, land_mask.astype(float),
              cmap="Greens", vmin=0, vmax=1, shading="nearest")
nuts.boundary.plot(ax=ax, linewidth=0.5, color="black", alpha=0.7)
setup_geo_ax(ax)
ax.set_title("Land mask (green=land) + NUTS borders — alignment check")
ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
plt.tight_layout()
path = os.path.join(OUT_DIR, "debug_land_mask.png")
plt.savefig(path, dpi=120); plt.close()
print(f"\nSaved: {path}")

# ============================================================
# FIGURE 1 — TRAJECTORY SEQUENCE
# ============================================================

T_map     = traj_map.shape[0]
plot_days = list(range(0, T_map, max(1, T_map // 12)))[:12]
ncols = 4
nrows = -(-len(plot_days) // ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
axes = axes.flatten()

for i, t in enumerate(plot_days):
    ax = axes[i]
    ax.pcolormesh(LON2D, LAT2D, traj_map[t],
                  cmap=cmap_traj, norm=norm_traj, shading="nearest", zorder=1)
    if true_hw[t].sum() > 0:
        ax.contour(LON2D, LAT2D, true_hw[t], levels=[0.5],
                   colors="black", linewidths=1.2, linestyles="--", zorder=3)
    nuts.boundary.plot(ax=ax, linewidth=0.4, color="black", alpha=0.7, zorder=4)
    if not np.isnan(centroid_lat[t]):
        ax.plot(centroid_lon[t], centroid_lat[t], "w*", markersize=12,
                markeredgecolor="black", markeredgewidth=0.8, zorder=5)
    setup_geo_ax(ax)
    ax.set_title(str(traj_dates[t].date()), fontsize=8)
    if i % ncols == 0:
        ax.set_yticks(np.linspace(LAT_MIN_REAL, LAT_MAX_REAL, 3))
        ax.set_yticklabels(
            [f"{v:.0f}°N" for v in np.linspace(LAT_MIN_REAL, LAT_MAX_REAL, 3)],
            fontsize=6)
    else:
        ax.set_yticks([])
    if i >= (nrows - 1) * ncols:
        ax.set_xticks(np.linspace(LON_MIN_REAL, LON_MAX_REAL, 3))
        ax.set_xticklabels(
            [f"{v:.0f}°E" if v >= 0 else f"{abs(v):.0f}°W"
             for v in np.linspace(LON_MIN_REAL, LON_MAX_REAL, 3)], fontsize=6)
    else:
        ax.set_xticks([])

for j in range(len(plot_days), len(axes)):
    axes[j].axis("off")

legend_elements = [Patch(facecolor=cmap_traj(k / 2), label=traj_labels[k])
                   for k in range(3)]
legend_elements += [
    plt.Line2D([0],[0], color="black", linestyle="--",
               linewidth=1.2, label="Observed HW"),
    plt.Line2D([0],[0], marker="*", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=10,
               label="LCC centroid", linestyle="None"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=5,
           fontsize=8, bbox_to_anchor=(0.5, 0.0))
fig.suptitle(
    f"Heatwave Trajectory — {rtitle}\n"
    f"Active: CC ≥ q{int(CC_HIGH_Q*100)} (model pred. distribution)  |  ★ = LCC centroid",
    fontsize=11, y=1.01)
plt.tight_layout(rect=[0, 0.06, 1, 1])
path = os.path.join(OUT_DIR, "fig1_trajectory_sequence.png")
plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
print(f"Saved: {path}")

# ============================================================
# FIGURE 2 — TEMPORAL EVOLUTION
# ============================================================

fig, ax1 = plt.subplots(figsize=(12, 4))
ax2 = ax1.twinx()
ax1.fill_between(t_axis, act_frac, alpha=0.6, color="#d73027", label="Active (high CC)")
ax1.fill_between(t_axis, tr_frac,  alpha=0.3, color="#fdae61", label="Transitional")
ax2.plot(t_axis, hw_frac, color="black", linewidth=1.5,
         linestyle="--", label="Observed HW fraction")
ax1.set_xlabel("Test day"); ax1.set_ylabel("Fraction of land pixels")
ax2.set_ylabel("HW land pixel fraction")
step = max(1, len(t_axis) // 10)
ax1.set_xticks(t_axis[::step])
ax1.set_xticklabels([str(traj_dates[t].date()) for t in t_axis[::step]],
                    rotation=45, ha="right", fontsize=7)
l1, lb1 = ax1.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1+l2, lb1+lb2, loc="upper left", fontsize=9)
ax1.set_title(f"Temporal evolution of active CC fraction (land only) — {rtitle}")
plt.tight_layout()
path = os.path.join(OUT_DIR, "fig2_temporal_evolution.png")
plt.savefig(path, dpi=150); plt.close()
print(f"Saved: {path}")

# ============================================================
# FIGURE 3 — PREDICTED vs OBSERVED LCC CENTROID COMPARISON
# ============================================================
# Single panel: predicted centroids (circles) vs observed centroids
# (squares), both colour-coded by test day index so the temporal
# progression and spatial agreement are immediately readable.

fig, ax = plt.subplots(figsize=(8, 7))

# Light land background
bg = np.where(land_mask, 0.5, np.nan)
ax.pcolormesh(LON2D, LAT2D, bg,
              cmap="Greys", vmin=0, vmax=1,
              shading="nearest", alpha=0.18, zorder=1)
nuts.boundary.plot(ax=ax, linewidth=0.5,
                   color="black", alpha=0.55, zorder=2)

cmap_track = plt.cm.plasma
norm_track  = mcolors.Normalize(vmin=0, vmax=T_map)

valid_pred = ~np.isnan(centroid_lat)
valid_obs  = ~np.isnan(obs_centroid_lat)

# Predicted track
if valid_pred.sum() > 1:
    ax.plot(centroid_lon[valid_pred], centroid_lat[valid_pred],
            color="#d73027", linewidth=1.2, alpha=0.55, zorder=3)
    sc_pred = ax.scatter(
        centroid_lon[valid_pred], centroid_lat[valid_pred],
        c=t_axis[valid_pred], cmap=cmap_track, norm=norm_track,
        s=70, zorder=5, edgecolors="#d73027",
        linewidths=1.4, marker="o", label="Predicted centroid"
    )
    # Start / end markers
    ax.plot(centroid_lon[valid_pred][0], centroid_lat[valid_pred][0],
            "g^", markersize=11, zorder=7, label="Pred. start")
    ax.plot(centroid_lon[valid_pred][-1], centroid_lat[valid_pred][-1],
            "rs", markersize=11, zorder=7, label="Pred. end")

# Observed track
if valid_obs.sum() > 1:
    ax.plot(obs_centroid_lon[valid_obs], obs_centroid_lat[valid_obs],
            color="#2166ac", linewidth=1.2, alpha=0.55, zorder=3)
    sc_obs = ax.scatter(
        obs_centroid_lon[valid_obs], obs_centroid_lat[valid_obs],
        c=t_axis[valid_obs], cmap=cmap_track, norm=norm_track,
        s=70, zorder=4, edgecolors="#2166ac",
        linewidths=1.4, marker="s", label="Observed centroid"
    )

# Shared colorbar (use pred scatter as reference)
ref_sc = sc_pred if valid_pred.sum() > 1 else sc_obs
cb = plt.colorbar(ref_sc, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("Test day index", fontsize=8)
# Annotate colorbar with actual dates at tick positions
tick_vals = cb.get_ticks()
tick_vals = [v for v in tick_vals if 0 <= v < T_map]
cb.set_ticks(tick_vals)
cb.set_ticklabels(
    [str(traj_dates[int(v)].date()) if int(v) < T_map else ""
     for v in tick_vals],
    fontsize=7
)

setup_geo_ax(ax)
geo_ticks(ax)
ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
ax.set_title(
    f"Predicted vs Observed LCC centroid — {rtitle}\n"
    f"Circles = predicted · Squares = observed · "
    f"Colour = temporal progression",
    fontsize=9
)
plt.tight_layout()
path = os.path.join(OUT_DIR, "fig3_centroid_comparison.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")

# ============================================================
# FIGURE 4 — JACCARD VALIDATION
# ============================================================

act_means = [np.mean(jaccard_active[l]) if jaccard_active[l] else 0.0 for l in leads]
act_stds  = [np.std(jaccard_active[l])  if jaccard_active[l] else 0.0 for l in leads]
tr_means  = [np.mean(jaccard_trans[l])  if jaccard_trans[l]  else 0.0 for l in leads]
tr_stds   = [np.std(jaccard_trans[l])   if jaccard_trans[l]  else 0.0 for l in leads]

x = np.arange(len(leads)); w = 0.35
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x-w/2, act_means, w, yerr=act_stds, capsize=5,
       color="#d73027", alpha=0.85, label="Active (high CC) → HW")
ax.bar(x+w/2, tr_means,  w, yerr=tr_stds,  capsize=5,
       color="#fdae61", alpha=0.85, label="Transitional → HW")
ax.set_xticks(x)
ax.set_xticklabels([f"Lead +{l}d" for l in leads])
ax.set_ylabel("Mean Jaccard overlap")
ax.set_title(f"Predicted CC activity → future HW overlap\n{rtitle}")
ax.legend(); plt.tight_layout()
path = os.path.join(OUT_DIR, "fig4_jaccard_validation.png")
plt.savefig(path, dpi=150); plt.close()
print(f"Saved: {path}")

# ============================================================
# FIGURE 5 — LCC SIZE OVER TIME
# ============================================================

fig, ax1 = plt.subplots(figsize=(12, 3))
ax2 = ax1.twinx()
ax1.bar(t_axis, lcc_size, color="#d73027", alpha=0.6, label="LCC size (pixels)")
ax2.plot(t_axis, hw_frac, color="black", linewidth=1.5,
         linestyle="--", label="Observed HW fraction (land)")
ax1.set_xlabel("Test day")
ax1.set_ylabel("Largest connected component (pixels)")
ax2.set_ylabel("HW land pixel fraction")
step = max(1, len(t_axis) // 10)
ax1.set_xticks(t_axis[::step])
ax1.set_xticklabels([str(traj_dates[t].date()) for t in t_axis[::step]],
                    rotation=45, ha="right", fontsize=7)
l1, lb1 = ax1.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1+l2, lb1+lb2, loc="upper left", fontsize=9)
ax1.set_title(f"Active CC region size over time — {rtitle}")
plt.tight_layout()
path = os.path.join(OUT_DIR, "fig5_lcc_size.png")
plt.savefig(path, dpi=150); plt.close()
print(f"Saved: {path}")

# ============================================================
# FIGURE 6 — CC DISTRIBUTION (VAL+TEST reference vs TEST period)
# ============================================================
# Blue = VAL+TEST combined (multi-period reference distribution)
# Red  = TEST period only (the specific heatwave event being analysed)
# This shows how the 2003 test event sits relative to the broader
# model output distribution, and where the thresholds fall.

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

p99_ref  = np.percentile(cc_ref_vals,  99)
p99_test = np.percentile(cc_test_vals, 99)
p99_zoom = max(p99_ref, p99_test)

ax = axes[0]
ax.hist(cc_ref_vals,  bins=150, color="#4393c3", alpha=0.75,
        edgecolor="none", label=ref_label, density=True)
ax.hist(cc_test_vals, bins=150, color="#d73027", alpha=0.55,
        edgecolor="none", label=f"Test period ({test_dates[0].year})", density=True)
ax.axvline(cc_thresh_low,  color="#fdae61", linewidth=2, linestyle="--",
           label=f"q{int(CC_LOW_Q*100)} = {cc_thresh_low:.4f}")
ax.axvline(cc_thresh_high, color="#8b0000", linewidth=2, linestyle="--",
           label=f"q{int(CC_HIGH_Q*100)} = {cc_thresh_high:.4f}")
ax.set_xlabel("Predicted CC (land pixels)"); ax.set_ylabel("Density")
ax.set_title("CC distribution — full range\n(model prediction thresholds)")
ax.legend(fontsize=8)

ax2 = axes[1]
ax2.hist(cc_ref_vals[cc_ref_vals   <= p99_zoom], bins=150,
         color="#4393c3", alpha=0.75, edgecolor="none",
         label=ref_label, density=True)
ax2.hist(cc_test_vals[cc_test_vals <= p99_zoom], bins=150,
         color="#d73027", alpha=0.55, edgecolor="none",
         label=f"Test period ({test_dates[0].year})", density=True)
ax2.axvline(cc_thresh_low,  color="#fdae61", linewidth=2, linestyle="--",
            label=f"q{int(CC_LOW_Q*100)} = {cc_thresh_low:.4f}")
ax2.axvline(cc_thresh_high, color="#8b0000", linewidth=2, linestyle="--",
            label=f"q{int(CC_HIGH_Q*100)} = {cc_thresh_high:.4f}")
ax2.axvspan(0,             cc_thresh_low,  alpha=0.07, color="grey",    label="Inactive")
ax2.axvspan(cc_thresh_low, cc_thresh_high, alpha=0.07, color="#fdae61", label="Transitional")
ax2.axvspan(cc_thresh_high, p99_zoom,      alpha=0.07, color="#d73027", label="Active")
ax2.set_xlim(0, p99_zoom)
ax2.set_xlabel("Predicted CC (land pixels, zoomed to p99)"); ax2.set_ylabel("Density")
ax2.set_title(f"CC distribution — zoomed (p99={p99_zoom:.4f})")
ax2.legend(fontsize=8)

print(f"\nCC distribution summary:")
print(f"  {'':20s}  {ref_label:>20}  {f'Test {test_dates[0].year}':>12}")
print(f"  {'n':20s}  {len(cc_ref_vals):>20,}  {len(cc_test_vals):>12,}")
print(f"  {'mean':20s}  {cc_ref_vals.mean():>20.4f}  {cc_test_vals.mean():>12.4f}")
print(f"  {'median':20s}  {np.median(cc_ref_vals):>20.4f}  {np.median(cc_test_vals):>12.4f}")
print(f"  {'std':20s}  {cc_ref_vals.std():>20.4f}  {cc_test_vals.std():>12.4f}")
print(f"  {'p75':20s}  {np.percentile(cc_ref_vals,75):>20.4f}  {np.percentile(cc_test_vals,75):>12.4f}")
print(f"  {'p90':20s}  {np.percentile(cc_ref_vals,90):>20.4f}  {np.percentile(cc_test_vals,90):>12.4f}")
print(f"  {'p99':20s}  {p99_ref:>20.4f}  {p99_test:>12.4f}")
print(f"  {'max':20s}  {cc_ref_vals.max():>20.4f}  {cc_test_vals.max():>12.4f}")

fig.suptitle(
    f"CC distribution — {rtitle}\n"
    f"Thresholds from {ref_label}, land pixels only",
    fontsize=11)
plt.tight_layout()
path = os.path.join(OUT_DIR, "fig6_cc_distribution.png")
plt.savefig(path, dpi=150); plt.close()
print(f"Saved: {path}")

# ============================================================
# FIGURE 7 — PREDICTED CC vs OBSERVED CC (key days)
# ============================================================

lcc_series = lcc_size[:T_map]
onset_t = next((t for t in range(T_map) if lcc_size[t] > MIN_COMPONENT_SIZE), 0)
peak_t  = int(np.argmax(lcc_series))

# Decay: first day after peak where active fraction drops below 50% of peak value
peak_act = act_frac[peak_t]
decay_t  = peak_t
for t in range(peak_t + 1, T_map):
    if act_frac[t] < peak_act * 0.5:
        decay_t = t
        break
# If no clear decay found, use last day with any active pixels
if decay_t == peak_t:
    for t in range(T_map - 1, peak_t, -1):
        if lcc_size[t] > 0:
            decay_t = t
            break

key_days   = [onset_t, peak_t, decay_t]
day_labels = ["Onset", "Peak", "Decay"]

# Shared colormap and scale for CC
vmin_cc = 0.0
vmax_cc = float(np.nanpercentile(
    np.concatenate([pred_cc[key_days].ravel(),
                    true_cc[key_days].ravel()]), 98))
cmap_cc = "YlOrRd"

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for col, (t, label) in enumerate(zip(key_days, day_labels)):

    pred_day = np.where(land_mask, pred_cc[t], np.nan)
    true_day = np.where(land_mask, true_cc[t], np.nan)
    active_contour = (traj_map[t] == 1).astype(float)

    for row, (data, row_label) in enumerate(
            [(pred_day, "Predicted CC"), (true_day, "Observed CC")]):

        ax = axes[row, col]
        pc = ax.pcolormesh(LON2D, LAT2D, data,
                           cmap=cmap_cc, vmin=vmin_cc, vmax=vmax_cc,
                           shading="nearest", zorder=1)

        # Active CC contour on both rows
        if active_contour.sum() > 0:
            ax.contour(LON2D, LAT2D, active_contour,
                       levels=[0.5], colors="#2166ac",
                       linewidths=1.5, zorder=3)

        nuts.boundary.plot(ax=ax, linewidth=0.4,
                           color="black", alpha=0.6, zorder=4)

        if row == 0 and not np.isnan(centroid_lat[t]):
            ax.plot(centroid_lon[t], centroid_lat[t], "w*",
                    markersize=14, markeredgecolor="black",
                    markeredgewidth=0.8, zorder=5)

        setup_geo_ax(ax)
        geo_ticks(ax, fontsize=6)

        if row == 0:
            ax.set_title(f"{label}  —  {traj_dates[t].date()}\n{row_label}",
                         fontsize=9)
        else:
            ax.set_title(row_label, fontsize=9)

        if col == 2:
            plt.colorbar(pc, ax=ax, fraction=0.046,
                         label="CC value", pad=0.02)

        if col == 0:
            ax.set_ylabel(row_label, fontsize=10, fontweight="bold")

# Shared legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], color="#2166ac", linewidth=1.5,
           label="Predicted active CC boundary (q70 threshold)"),
    plt.Line2D([0],[0], marker="*", color="w",
               markerfacecolor="white", markeredgecolor="black",
               markersize=10, label="LCC centroid", linestyle="None"),
]
fig.legend(handles=legend_elements, loc="lower center",
           ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.0))

fig.suptitle(
    f"Predicted vs Observed CC — {rtitle}\n"
    f"Blue contour = predicted active region (CC ≥ q70) · ★ = LCC centroid",
    fontsize=11)
plt.tight_layout(rect=[0, 0.06, 1, 1])
path = os.path.join(OUT_DIR, "fig7_pred_vs_obs_cc.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ============================================================
# STEP 7b — PERSISTENCE BASELINE
# ============================================================
# "Tomorrow's heatwave looks like today's heatwave" — Jaccard between
# true_hw[t] and true_hw[t+lead], land pixels only. This is the trivial
# baseline that the model's active-CC Jaccard (Step 7) should beat.
 
print(f"\n{'Lead':>6}  {'Persistence Jaccard':>20}  {'Active CC Jaccard':>18}  "
      f"{'Delta':>8}  {'N':>5}")
jaccard_persistence = {}
 
for lead in [1, 2, 3]:
    T = traj_map.shape[0]
    j_pers = []
    for t in range(T - lead):
        hw_today    = true_hw[t].astype(bool)
        hw_tomorrow = true_hw[t + lead].astype(bool)
        hw_today[~land_mask]    = False
        hw_tomorrow[~land_mask] = False
        if hw_tomorrow.sum() == 0:
            continue
        intersection = (hw_today & hw_tomorrow).sum()
        union        = (hw_today | hw_tomorrow).sum()
        j_pers.append(intersection / (union + 1e-8))
    jaccard_persistence[lead] = j_pers
    mp = np.mean(j_pers) if j_pers else 0.0
    ma = np.mean(jaccard_active[lead]) if jaccard_active[lead] else 0.0
    print(f"  {lead:>4}d  {mp:>20.4f}  {ma:>18.4f}  {ma-mp:>+8.4f}  {len(j_pers):>5}")
 
print("\nInterpretation: if 'Delta' > 0, the active-CC region predicts")
print("future HW location better than simply persisting today's HW mask.")

 
# ============================================================
# STEP 7c — DAILY DISCRIMINATION CHECK
# ============================================================
# Per-day spatial correlation between predicted and true CC.
# Tests whether the model tracks day-to-day evolution (varying,
# mostly-positive correlation that is higher on "active" days)
# rather than reproducing a fixed climatological pattern every day
# (which would give flat/low correlation regardless of the day's
# true activity).
 
print("\nComputing per-day spatial correlation (pred vs true CC)...")
 
daily_corr        = np.full(T_map, np.nan)
daily_active_frac = np.full(T_map, np.nan)
 
for t in range(T_map):
    p  = pred_cc[t][land_mask].ravel()
    tr = true_cc[t][land_mask].ravel()
    valid = ~(np.isnan(p) | np.isnan(tr))
    p, tr = p[valid], tr[valid]
 
    daily_active_frac[t] = (tr > 0.5).mean()
    if tr.std() > 0 and p.std() > 0:
        daily_corr[t] = np.corrcoef(p, tr)[0, 1]
 
valid_days = ~np.isnan(daily_corr)
print(f"  Days with defined correlation: {valid_days.sum()} / {T_map}")
print(f"  Mean daily correlation (all valid days)   : {np.nanmean(daily_corr):.3f}")
 
active_days = valid_days & (daily_active_frac > 0.01)
quiet_days  = valid_days & (daily_active_frac <= 0.01)
if active_days.sum() > 0:
    print(f"  Mean daily correlation (active days, n={active_days.sum():>3}) "
          f": {np.nanmean(daily_corr[active_days]):.3f}")
if quiet_days.sum() > 0:
    print(f"  Mean daily correlation (quiet days,  n={quiet_days.sum():>3}) "
          f": {np.nanmean(daily_corr[quiet_days]):.3f}")
 
# Figure: daily correlation + active fraction over time
fig, ax1 = plt.subplots(figsize=(12, 4))
ax2 = ax1.twinx()
 
ax1.plot(t_axis, daily_corr, color="#2166ac", marker="o", markersize=3,
         linewidth=1.2, label="Daily spatial corr. (pred vs true CC)")
ax1.axhline(0, color="grey", linewidth=0.8, linestyle="--")
ax1.set_ylabel("Pearson correlation (pred vs true, per day)", color="#2166ac")
ax1.set_ylim(-1, 1)
ax1.tick_params(axis="y", labelcolor="#2166ac")
 
ax2.fill_between(t_axis, daily_active_frac, alpha=0.25, color="#d73027",
                 label="True active fraction (CC > 0.5)")
ax2.set_ylabel("Fraction of land pixels with true CC > 0.5", color="#d73027")
ax2.tick_params(axis="y", labelcolor="#d73027")
 
step = max(1, len(t_axis) // 10)
ax1.set_xticks(t_axis[::step])
ax1.set_xticklabels([str(traj_dates[t].date()) for t in t_axis[::step]],
                    rotation=45, ha="right", fontsize=7)
 
l1, lb1 = ax1.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1+l2, lb1+lb2, loc="upper left", fontsize=8)
ax1.set_title(
    f"Daily discrimination check — {rtitle}\n"
    f"Does the model track day-to-day CC evolution, or just a fixed pattern?",
    fontsize=10
)
plt.tight_layout()
path = os.path.join(OUT_DIR, "fig_daily_discrimination.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")
 

# STEP 7c — DAILY DISCRIMINATION CHECK
# ============================================================
# Per-day spatial correlation between predicted and true CC.
# Tests whether the model tracks day-to-day evolution (varying,
# mostly-positive correlation that is higher on "active" days)
# rather than reproducing a fixed climatological pattern every day
# (which would give flat/low correlation regardless of the day's
# true activity).
 
print("\nComputing per-day spatial correlation (pred vs true CC)...")
 
daily_corr        = np.full(T_map, np.nan)
daily_active_frac = np.full(T_map, np.nan)
 
for t in range(T_map):
    p  = pred_cc[t][land_mask].ravel()
    tr = true_cc[t][land_mask].ravel()
    valid = ~(np.isnan(p) | np.isnan(tr))
    p, tr = p[valid], tr[valid]
 
    daily_active_frac[t] = (tr > 0.5).mean()
    if tr.std() > 0 and p.std() > 0:
        daily_corr[t] = np.corrcoef(p, tr)[0, 1]
 
valid_days = ~np.isnan(daily_corr)
print(f"  Days with defined correlation: {valid_days.sum()} / {T_map}")
print(f"  Mean daily correlation (all valid days)   : {np.nanmean(daily_corr):.3f}")
 
active_days = valid_days & (daily_active_frac > 0.01)
quiet_days  = valid_days & (daily_active_frac <= 0.01)
if active_days.sum() > 0:
    print(f"  Mean daily correlation (active days, n={active_days.sum():>3}) "
          f": {np.nanmean(daily_corr[active_days]):.3f}")
if quiet_days.sum() > 0:
    print(f"  Mean daily correlation (quiet days,  n={quiet_days.sum():>3}) "
          f": {np.nanmean(daily_corr[quiet_days]):.3f}")
 
# Figure: daily correlation + active fraction over time
fig, ax1 = plt.subplots(figsize=(12, 4))
ax2 = ax1.twinx()
 
ax1.plot(t_axis, daily_corr, color="#2166ac", marker="o", markersize=3,
         linewidth=1.2, label="Daily spatial corr. (pred vs true CC)")
ax1.axhline(0, color="grey", linewidth=0.8, linestyle="--")
ax1.set_ylabel("Pearson correlation (pred vs true, per day)", color="#2166ac")
ax1.set_ylim(-1, 1)
ax1.tick_params(axis="y", labelcolor="#2166ac")
 
ax2.fill_between(t_axis, daily_active_frac, alpha=0.25, color="#d73027",
                 label="True active fraction (CC > 0.5)")
ax2.set_ylabel("Fraction of land pixels with true CC > 0.5", color="#d73027")
ax2.tick_params(axis="y", labelcolor="#d73027")
 
step = max(1, len(t_axis) // 10)
ax1.set_xticks(t_axis[::step])
ax1.set_xticklabels([str(traj_dates[t].date()) for t in t_axis[::step]],
                    rotation=45, ha="right", fontsize=7)
 
l1, lb1 = ax1.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1+l2, lb1+lb2, loc="upper left", fontsize=8)
ax1.set_title(
    f"Daily discrimination check — {rtitle}\n"
    f"Does the model track day-to-day CC evolution, or just a fixed pattern?",
    fontsize=10
)
plt.tight_layout()
path = os.path.join(OUT_DIR, "fig_daily_discrimination.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ============================================================
# FIGURE 8 — N-DAY TRUE vs PREDICTED CC COMPARISON
# ============================================================
# Layout: N_COMPARISON_DAYS rows, each row = one day.
#   Left panel  → True CC
#   Right panel → Predicted CC
# Shared colorbar, shared colour scale across all days.
# Start date controlled by COMPARISON_START_DATE in CONFIG.
# Set to None to auto-pick the peak active-fraction day.

# ── Resolve start index ──────────────────────────────────────────────────────
if COMPARISON_START_DATE is not None:
    req_date = pd.Timestamp(COMPARISON_START_DATE)
    diffs    = np.abs((traj_dates - req_date).total_seconds())
    start_t  = int(np.argmin(diffs))
    actual_start = traj_dates[start_t].date()
    if str(actual_start) != COMPARISON_START_DATE:
        print(f"\n[Fig 8] Requested {COMPARISON_START_DATE} not in test period — "
              f"snapping to nearest: {actual_start}")
else:
    start_t      = int(np.argmax(act_frac))
    actual_start = traj_dates[start_t].date()
    print(f"\n[Fig 8] Auto start date (peak active fraction): {actual_start}")

# Clamp window to available data
end_t    = min(start_t + N_COMPARISON_DAYS, T_map)
day_idxs = list(range(start_t, end_t))
n_days   = len(day_idxs)

if n_days == 0:
    print("[Fig 8] WARNING: no days in window — skipping figure.")
else:
    print(f"[Fig 8] Plotting {n_days} days: {actual_start} → "
          f"{traj_dates[day_idxs[-1]].date()}")

    # ── Shared colour scale ───────────────────────────────────────────────────
    combined = np.concatenate([
        pred_cc[day_idxs][:, land_mask].ravel(),
        true_cc[day_idxs][:, land_mask].ravel(),
    ])
    combined = combined[~np.isnan(combined)]
    vmin_f8  = 0.0
    vmax_f8  = float(np.nanpercentile(combined, 98))
    cmap_f8  = "viridis"

    # ── Layout: n_days rows × 2 cols ─────────────────────────────────────────
    map_w  = 5.0    # width of each map panel (inches)
    map_h  = 2.6    # height of each map panel (inches)
    fig_w  = map_w * 2 + 1.8   # +1.8 for colorbar + margins
    fig_h  = map_h * n_days + 0.5  # tight: just enough for suptitle

    fig, axes = plt.subplots(
        n_days, 2,
        figsize=(fig_w, fig_h),
        gridspec_kw={"hspace": 0.10, "wspace": 0.04},
    )
    if n_days == 1:
        axes = axes.reshape(1, 2)

    # Column headers on the very first row
    axes[0, 0].set_title("True CC",      fontsize=10, fontweight="bold", pad=4)
    axes[0, 1].set_title("Predicted CC", fontsize=10, fontweight="bold", pad=4)

    for row, t in enumerate(day_idxs):
        date_str = str(traj_dates[t].date())
        true_day = np.where(land_mask, true_cc[t], np.nan)
        pred_day = np.where(land_mask, pred_cc[t], np.nan)

        for col, data in enumerate([true_day, pred_day]):
            ax = axes[row, col]
            ax.pcolormesh(
                LON2D, LAT2D, data,
                cmap=cmap_f8, vmin=vmin_f8, vmax=vmax_f8,
                shading="nearest", zorder=1,
            )
            nuts.boundary.plot(ax=ax, linewidth=0.3,
                               color="black", alpha=0.55, zorder=4)
            setup_geo_ax(ax)

            # Date label on the left edge of each row
            if col == 0:
                ax.set_ylabel(date_str, fontsize=8, labelpad=4)

            # Axis ticks only on bottom row
            if row == n_days - 1:
                ax.set_xticks(np.linspace(LON_MIN_REAL, LON_MAX_REAL, 4))
                ax.set_xticklabels(
                    [f"{v:.0f}°E" if v >= 0 else f"{abs(v):.0f}°W"
                     for v in np.linspace(LON_MIN_REAL, LON_MAX_REAL, 4)],
                    fontsize=6)
            else:
                ax.set_xticks([])

            ax.set_yticks([])

    # ── Shared colorbar ───────────────────────────────────────────────────────
    fig.subplots_adjust(top=0.96, right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.04, 0.018, 0.90])
    sm = plt.cm.ScalarMappable(
        cmap=cmap_f8,
        norm=mcolors.Normalize(vmin=vmin_f8, vmax=vmax_f8))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="CC value")

    fig.suptitle(
        f"True vs Predicted CC — {rtitle}  |  "
        f"{actual_start} → {traj_dates[day_idxs[-1]].date()}",
        fontsize=11, y=0.99,
    )

    path = os.path.join(OUT_DIR, "fig8_true_vs_pred_cc.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# ============================================================
# SAVE SUMMARY
# ============================================================

with open(os.path.join(OUT_DIR, "trajectory_summary.txt"), "w") as f:
    f.write(f"Region          : {REGION}\n")
    f.write(f"Ablation        : {ABLATION}\n")
    f.write(f"Bounds (actual) : lat=[{LAT_MIN_REAL:.2f},{LAT_MAX_REAL:.2f}]  "
            f"lon=[{LON_MIN_REAL:.2f},{LON_MAX_REAL:.2f}]\n")
    f.write(f"Method          : CC-only, LCC centroid, model-pred thresholds ({ref_label})\n")
    f.write(f"CC_HIGH_Q       : {CC_HIGH_Q}  → {cc_thresh_high:.4f}\n")
    f.write(f"CC_LOW_Q        : {CC_LOW_Q}   → {cc_thresh_low:.4f}\n")
    f.write(f"MIN_COMPONENT   : {MIN_COMPONENT_SIZE} pixels\n\n")
    f.write(f"Grid            : {H} rows × {W} cols\n")
    f.write(f"Land pixels     : {land_mask.sum()} / {H*W} ({land_mask.mean():.3f})\n")
    f.write(f"CC range (pred) : [{np.nanmin(pred_cc):.4f}, {np.nanmax(pred_cc):.4f}]\n")
    f.write(f"HW positive rate: {true_hw[:, land_mask].mean():.4f}\n\n")
    f.write("Label distribution (land only):\n")
    for lbl, name in [(0,"inactive"),(1,"active"),(2,"transitional")]:
        frac = (traj_map[:, land_mask] == lbl).mean()
        f.write(f"  {name:12s}: {frac:.4f}\n")
    f.write(f"\nCC distribution ({ref_label} vs test):\n")
    f.write(f"  {'':20s}  {ref_label:>20}  {f'Test {test_dates[0].year}':>12}\n")
    f.write(f"  {'mean':20s}  {cc_ref_vals.mean():>20.4f}  {cc_test_vals.mean():>12.4f}\n")
    f.write(f"  {'median':20s}  {np.median(cc_ref_vals):>20.4f}  {np.median(cc_test_vals):>12.4f}\n")
    f.write(f"  {'p99':20s}  {p99_ref:>20.4f}  {p99_test:>12.4f}\n")
    f.write("\nActive → HW Jaccard:\n")
    for lead in leads:
        mu = np.mean(jaccard_active[lead]) if jaccard_active[lead] else 0.0
        sd = np.std(jaccard_active[lead])  if jaccard_active[lead] else 0.0
        f.write(f"  lead={lead}d  mean={mu:.4f}  std={sd:.4f}  "
                f"n={len(jaccard_active[lead])}\n")
    f.write("\nTransitional → HW Jaccard:\n")
    for lead in leads:
        mu = np.mean(jaccard_trans[lead]) if jaccard_trans[lead] else 0.0
        sd = np.std(jaccard_trans[lead])  if jaccard_trans[lead] else 0.0
        f.write(f"  lead={lead}d  mean={mu:.4f}  std={sd:.4f}  "
                f"n={len(jaccard_trans[lead])}\n")
    f.write("\nLCC centroid positions (predicted vs observed):\n")
    f.write(f"  {'Date':<12} {'LCC_size':>10} {'N_comp':>8} "
            f"{'Pred_Lat':>10} {'Pred_Lon':>10} "
            f"{'Obs_Lat':>9} {'Obs_Lon':>9} {'Dist_deg':>10}\n")
    for t in range(T_map):
        plat = centroid_lat[t]
        plon = centroid_lon[t]
        olat = obs_centroid_lat[t]
        olon = obs_centroid_lon[t]
        if not np.isnan(plat) and not np.isnan(olat):
            dist = float(np.sqrt((plat - olat)**2 + (plon - olon)**2))
        else:
            dist = np.nan
        f.write(f"  {str(traj_dates[t].date()):<12} "
                f"{lcc_size[t]:>10.0f} {n_components[t]:>8.0f} "
                f"{plat:>10.2f} {plon:>10.2f} "
                f"{olat:>9.2f} {olon:>9.2f} "
                f"{dist:>10.3f}\n")

print(f"\nAll outputs saved to: {OUT_DIR}")
print("Done.")