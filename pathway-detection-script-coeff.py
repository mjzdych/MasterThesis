"""
Heatwave propagation pathway analysis using HDBSCAN.
v7: CN variables (CC, BC, OD, ID) + land mask + UMAP dimensionality reduction.

Key improvements over v6:
  1. Centroid weighted by CN divergence (OD-ID) instead of uniform mask
  2. Land mask applied before centroid computation
  3. CN-enriched trajectory features (per-day spatial mean of CC, BC, OD-ID)
  4. UMAP reduces high-dimensional feature space before HDBSCAN

Input:  full_processed_training_dataset.nc
Output: pathway_detection_folder/
"""

import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import hdbscan
from scipy.interpolate import interp1d
from scipy.ndimage import label as scipy_label
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from umap import UMAP

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
NC_FILE        = "full_processed_training_dataset.nc"
N_TRACK_POINTS = 10
MIN_COMP_SIZE  = 1000    # cells in largest HW component
MIN_DURATION   = 3       # days

# Propagating = centroid moves > threshold over event lifetime
PROP_DISP_THRESHOLD_KM = 300   # lower than v6 — centroid is now more stable

# HDBSCAN
HDBSCAN_MIN_CLUSTER_SIZE = 10
HDBSCAN_MIN_SAMPLES      = 2
HDBSCAN_METRIC           = "euclidean"
KINEMATIC_WEIGHT         = 3.0

# UMAP — reduce combined feature space before HDBSCAN
UMAP_N_COMPONENTS  = 10   # target dimensions
UMAP_N_NEIGHBORS   = 15   # local neighbourhood size
UMAP_MIN_DIST      = 0.1  # compactness of embedding

OUT_DIR = "./pathway_detection_folder_new"
OUT_NC  = os.path.join(OUT_DIR, "pathway_hdbscan_results.nc")

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Output directory: {os.path.abspath(OUT_DIR)}")

# ── Load dataset ──────────────────────────────────────────────────────────────
print(f"Loading {NC_FILE} ...")
ds = xr.open_dataset(NC_FILE)
print(ds)

lat   = ds.lat.values
lon   = ds.lon.values
times = pd.DatetimeIndex(ds.time.values)

# Core variables
is_hw    = ds["is_heatwave"].values.astype(bool)       # (T, H, W)
z500_raw = ds["z"].values / 9.80665
z500_clim = z500_raw.mean(axis=0)
z500_anom = z500_raw - z500_clim                        # (T, H, W)

# CN variables
cc_vals  = ds["CC"].values.astype(np.float32)           # (T, H, W)
bc_vals  = ds["BC"].values.astype(np.float32)           # (T, H, W)
od_vals  = ds["OD"].values.astype(np.float32)           # (T, H, W)
id_vals  = ds["ID"].values.astype(np.float32)           # (T, H, W)
div_vals = od_vals - id_vals                            # divergence = OD - ID

# Land mask — shape (T, H, W) or (H, W) depending on dataset
land_raw = ds["land_mask"].values
if land_raw.ndim == 3:
    land_mask = land_raw[0].astype(bool)   # take first time step, static
else:
    land_mask = land_raw.astype(bool)      # (H, W)

print(f"\nLand cells: {land_mask.sum()} / {land_mask.size} "
      f"({100*land_mask.mean():.1f}%)")
print(f"Z500 anom range: {z500_anom.min():.1f} to {z500_anom.max():.1f} m")
print(f"Divergence range: {div_vals.min():.3f} to {div_vals.max():.3f}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def haversine(p1, p2):
    dlat = np.radians(p2[0] - p1[0])
    dlon = np.radians(p2[1] - p1[1])
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(p1[0]))
         * np.cos(np.radians(p2[0]))
         * np.sin(dlon / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def find_largest_component(hw_slice, land_mask=None):
    """
    Returns (size, boolean mask) of largest connected HW component.
    If land_mask provided, only considers land cells.
    """
    if land_mask is not None:
        hw_land = hw_slice & land_mask
    else:
        hw_land = hw_slice
    labeled, n = scipy_label(hw_land)
    if n == 0:
        # Fallback: try without land mask
        labeled, n = scipy_label(hw_slice)
        if n == 0:
            return 0, None
    sizes = np.bincount(labeled.ravel())[1:]
    best  = sizes.argmax() + 1
    return int(sizes.max()), labeled == best


def weighted_centroid(weight_map, lat, lon):
    """
    Compute centroid weighted by weight_map (e.g. divergence or CC).
    Weights by cos(lat) for area correction.
    Returns (lat_c, lon_c) or (nan, nan) if no valid weights.
    """
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    area_w = np.cos(np.radians(lat2d))
    w      = np.clip(weight_map, 0, None) * area_w   # only positive weights
    total  = w.sum()
    if total == 0 or np.isnan(total):
        return np.nan, np.nan
    return float((w * lat2d).sum() / total), float((w * lon2d).sum() / total)


# ── Event extraction ──────────────────────────────────────────────────────────

def extract_events(is_hw, z500_anom, div_vals, cc_vals, bc_vals,
                   times, lat, lon, land_mask):
    """
    For each event, track:
      1. Centroid weighted by CN divergence (OD-ID) within HW footprint
         → marks where heat is being exported = leading edge of propagation
      2. Per-day spatial means of CC, BC, divergence within HW footprint
         → CN signature of the event over time

    Land mask applied before centroid to exclude ocean noise.
    """
    year_list = sorted(np.unique(times.year))
    events    = []

    for yr in year_list:
        yr_mask  = times.year == yr
        yr_idx   = np.where(yr_mask)[0]
        hw_yr    = is_hw[yr_mask]
        z_yr     = z500_anom[yr_mask]
        div_yr   = div_vals[yr_mask]
        cc_yr    = cc_vals[yr_mask]
        bc_yr    = bc_vals[yr_mask]
        n_days   = len(yr_idx)

        # Component size on land only
        daily_max = np.array([
            find_largest_component(hw_yr[t], land_mask)[0]
            for t in range(n_days)
        ])
        active = daily_max >= MIN_COMP_SIZE

        t = 0
        while t < n_days:
            if not active[t]:
                t += 1
                continue
            t_start = t
            while t < n_days and active[t]:
                t += 1
            t_end    = t
            duration = t_end - t_start
            if duration < MIN_DURATION:
                continue

            peak_day  = t_start + daily_max[t_start:t_end].argmax()
            peak_size = int(daily_max[peak_day])
            track_len = min(10, n_days - t_start)

            centers     = []   # (lat, lon) per day
            cn_profiles = []   # [mean_div, mean_cc, mean_bc, mean_z500] per day

            for dt in range(track_len):
                hw_t  = hw_yr[t_start + dt]
                div_t = div_yr[t_start + dt]
                cc_t  = cc_yr[t_start + dt]
                bc_t  = bc_yr[t_start + dt]
                z_t   = z_yr[t_start + dt]

                # Largest component on land
                _, comp_mask = find_largest_component(hw_t, land_mask)
                if comp_mask is None:
                    if centers:
                        centers.append(centers[-1])
                        cn_profiles.append(cn_profiles[-1])
                    continue

                # Centroid weighted by positive divergence (source nodes)
                # Positive divergence = OD > ID = heat source = leading edge
                div_in_comp = np.where(comp_mask, div_t, 0.0)
                lat_c, lon_c = weighted_centroid(div_in_comp, lat, lon)

                # Fallback: uniform centroid if divergence all negative
                if np.isnan(lat_c):
                    lat_c, lon_c = weighted_centroid(
                        comp_mask.astype(float), lat, lon)

                centers.append([lat_c, lon_c])

                # CN spatial means within component (land only)
                n_cells = comp_mask.sum()
                if n_cells > 0:
                    cn_profiles.append([
                        float(div_t[comp_mask].mean()),
                        float(cc_t[comp_mask].mean()),
                        float(bc_t[comp_mask].mean()),
                        float(z_t[comp_mask].mean()),
                    ])
                else:
                    cn_profiles.append([0., 0., 0., 0.])

            if len(centers) < 3:
                continue

            events.append({
                "year":         yr,
                "global_start": int(yr_idx[t_start]),
                "global_end":   int(yr_idx[t_end - 1]),
                "duration":     duration,
                "peak_size":    peak_size,
                "centers":      np.array(centers),
                "cn_profiles":  np.array(cn_profiles),   # (track_len, 4)
            })

    print(f"Extracted {len(events)} events across {len(year_list)} summers")
    return events


# ── Trajectory features ───────────────────────────────────────────────────────

def build_trajectory_features(events, n_points=N_TRACK_POINTS):
    """
    Feature vector per event:
      - 20D: interpolated centroid lat/lon track
      - 40D: interpolated CN profiles (div, CC, BC, Z500) × 10 points
      - 6D:  kinematic scalars (disp, mean_step, speed, duration, dlon, dlat)
    Total: 66D → reduced by UMAP to UMAP_N_COMPONENTS
    """
    coords_list, cn_list, scalar_list, valid = [], [], [], []

    for ev in events:
        c   = ev["centers"]
        cnp = ev["cn_profiles"]

        if len(c) < 3 or np.isnan(c).any():
            continue

        t_orig = np.linspace(0, 1, len(c))
        t_new  = np.linspace(0, 1, n_points)

        # Spatial track
        lat_i = interp1d(t_orig, c[:, 0], kind="linear")(t_new)
        lon_i = interp1d(t_orig, c[:, 1], kind="linear")(t_new)
        coords_list.append(np.concatenate([lat_i, lon_i]))   # 20D

        # CN profiles — interpolate each of 4 variables to n_points
        cn_interp = []
        n_cn = min(len(cnp), len(c))
        t_cn = np.linspace(0, 1, n_cn)
        for vi in range(4):
            vals = cnp[:n_cn, vi]
            if np.isnan(vals).any():
                vals = np.nan_to_num(vals, nan=0.0)
            cn_interp.append(interp1d(t_cn, vals, kind="linear")(t_new))
        cn_list.append(np.concatenate(cn_interp))             # 40D

        # Kinematics
        total     = haversine(c[0], c[-1])
        steps     = [haversine(c[i], c[i+1]) for i in range(len(c)-1)]
        mean_step = float(np.mean(steps))
        speed     = total / ev["duration"]
        dlon      = float(c[-1, 1] - c[0, 1])
        dlat      = float(c[-1, 0] - c[0, 0])
        scalar_list.append([total, mean_step, speed, ev["duration"], dlon, dlat])

        valid.append(ev)

    coord_feats  = np.array(coords_list)    # (N, 20)
    cn_feats     = np.array(cn_list)        # (N, 40)
    scalar_feats = np.array(scalar_list)    # (N, 6)

    # Standardise each block
    sc = StandardScaler()
    ss = StandardScaler()
    sk = StandardScaler()
    coord_s  = sc.fit_transform(coord_feats)
    cn_s     = ss.fit_transform(cn_feats)
    scalar_s = sk.fit_transform(scalar_feats)

    # Combined: coords + CN profiles + kinematics (upweighted)
    combined = np.hstack([
        coord_s,
        cn_s,
        scalar_s * KINEMATIC_WEIGHT
    ])  # (N, 66)

    disps = scalar_feats[:, 0]
    steps = scalar_feats[:, 1]
    print(f"\nEvents with full tracks: {len(valid)}")
    print(f"Combined feature matrix: {combined.shape}")
    print(f"\nDisplacement stats (divergence-weighted centroid):")
    print(f"  min={disps.min():.1f}  median={np.median(disps):.1f}  "
          f"max={disps.max():.1f} km")
    print(f"  Mean daily step: median={np.median(steps):.1f} km/day")
    print(f"  Events disp > {PROP_DISP_THRESHOLD_KM} km: "
          f"{(disps > PROP_DISP_THRESHOLD_KM).sum()} / {len(disps)} "
          f"({100*(disps > PROP_DISP_THRESHOLD_KM).mean():.0f}%)")

    return coord_feats, scalar_feats, cn_feats, combined, valid


# ── UMAP dimensionality reduction ─────────────────────────────────────────────

def reduce_with_umap(combined_feats, random_state=42):
    """
    Reduce 66D feature space to UMAP_N_COMPONENTS dimensions.

    Why UMAP over PCA:
    - Preserves local neighbourhood structure (important for cluster shape)
    - Non-linear: can unroll curved manifolds in trajectory space
    - Better separation of clusters in low-dimensional embedding

    Returns reduced array and the fitted UMAP object.
    """
    print(f"\nRunning UMAP: {combined_feats.shape[1]}D → {UMAP_N_COMPONENTS}D ...")
    reducer = UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="euclidean",
        random_state=random_state,
    )
    embedding = reducer.fit_transform(combined_feats)
    print(f"UMAP done. Embedding shape: {embedding.shape}")
    return embedding, reducer


def plot_umap_embedding(embedding, labels, binary_labels,
                         filename="umap_embedding.png"):
    """Visualise UMAP 2D projection (first 2 components) coloured by cluster."""
    fpath = os.path.join(OUT_DIR, filename)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cluster colours
    unique = sorted(set(labels))
    for k in unique:
        mask = labels == k
        col  = "lightgray" if k == -1 else CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        lbl  = "noise" if k == -1 else f"C{k} (n={mask.sum()})"
        axes[0].scatter(embedding[mask, 0], embedding[mask, 1],
                        c=col, s=20, alpha=0.7, label=lbl)
    axes[0].set_title("UMAP — HDBSCAN clusters")
    axes[0].legend(fontsize=8, markerscale=1.5)
    axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")

    # Prop / standing
    colors_bin = {-1: "lightgray", 0: "steelblue", 1: "tomato"}
    labels_bin = {-1: "noise", 0: "standing", 1: "propagating"}
    for lab_val in [-1, 0, 1]:
        mask = binary_labels == lab_val
        if mask.sum() == 0:
            continue
        axes[1].scatter(embedding[mask, 0], embedding[mask, 1],
                        c=colors_bin[lab_val], s=20, alpha=0.7,
                        label=f"{labels_bin[lab_val]} (n={mask.sum()})")
    axes[1].set_title("UMAP — prop/standing")
    axes[1].legend(fontsize=8, markerscale=1.5)
    axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")

    plt.tight_layout()
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


# ── HDBSCAN ───────────────────────────────────────────────────────────────────

def run_hdbscan(embedding):
    """Run HDBSCAN on UMAP embedding (already low-dimensional)."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        cluster_selection_method="eom",
        prediction_data=True,
    )
    clusterer.fit(embedding)
    return clusterer


def summarise_clusters(labels, scalar_feats, cn_feats, events):
    unique = sorted(set(labels))
    cn_names = ["Divergence", "CC", "BC", "Z500_anom"]
    print(f"\n{'='*65}")
    print(f"  HDBSCAN  mcs={HDBSCAN_MIN_CLUSTER_SIZE}  ms={HDBSCAN_MIN_SAMPLES}")
    print(f"{'='*65}")
    for k in unique:
        mask  = labels == k
        tag   = "NOISE" if k == -1 else f"Cluster {k}"
        disp  = scalar_feats[mask, 0]
        speed = scalar_feats[mask, 2]
        dur   = scalar_feats[mask, 3]
        step  = scalar_feats[mask, 1]
        print(f"  {tag:12s}  n={mask.sum():3d}  "
              f"disp={disp.mean():5.0f}±{disp.std():3.0f} km  "
              f"step={step.mean():4.0f} km/d  "
              f"dur={dur.mean():.1f} d")

        # CN signature per cluster
        if k != -1 and mask.sum() > 0:
            # Mean of first time point of CN profile (onset conditions)
            n_pts = cn_feats.shape[1] // 4
            for vi, name in enumerate(cn_names):
                vals = cn_feats[mask, vi * n_pts]   # onset value
                print(f"    {name:12s} at onset: {vals.mean():.3f} ± {vals.std():.3f}")

    print("\n  Case study events (2003, 2010, 2018):")
    for ev, lab in zip(events, labels):
        if ev["year"] in [2003, 2010, 2018]:
            idx = events.index(ev)
            print(f"    {ev['year']}  dur={ev['duration']}d  "
                  f"disp={scalar_feats[idx,0]:.0f} km  "
                  f"step={scalar_feats[idx,1]:.0f} km/d  "
                  f"→ cluster {lab}")


# ── Prop / standing assignment ────────────────────────────────────────────────

# def assign_prop_standing(labels, scalar_feats):
#     disps           = scalar_feats[:, 0]
#     unique_clusters = [k for k in sorted(set(labels)) if k != -1]
#     binary          = np.full(len(labels), -1, dtype=np.int8)
#     low_disp        = disps <= PROP_DISP_THRESHOLD_KM
#     high_disp       = ~low_disp

#     cluster_speeds = {}
#     for k in unique_clusters:
#         mask = (labels == k) & high_disp
#         cluster_speeds[k] = scalar_feats[mask, 2].mean() if mask.sum() > 0 else 0.0

#     threshold = np.median(list(cluster_speeds.values())) if cluster_speeds else 0.0
#     cluster_to_binary = {k: (1 if cluster_speeds.get(k, 0) > threshold else 0)
#                          for k in unique_clusters}
#     cluster_to_binary[-1] = -1

#     for i, (lab, is_low) in enumerate(zip(labels, low_disp)):
#         if lab == -1:
#             binary[i] = -1
#         elif is_low:
#             binary[i] = 0
#         else:
#             binary[i] = cluster_to_binary[lab]

#     print(f"\n  Displacement threshold: {PROP_DISP_THRESHOLD_KM} km")
#     print(f"  Forced standing (low disp): {low_disp.sum()} events")
#     print(f"  Cluster speeds: "
#           + "  ".join([f"C{k}={v:.0f}" for k, v in cluster_speeds.items()]))
#     print(f"  Speed threshold: {threshold:.0f} km/d")
#     print(f"  Mapping: {cluster_to_binary}")

#     return binary, cluster_to_binary
def assign_prop_standing_direct(scalar_feats, labels,
                                 disp_threshold=300,    # km
                                 step_threshold=150):   # km/day mean step
    """
    Standing  : net displacement < disp_threshold  AND  mean step < step_threshold
    Propagating: net displacement >= disp_threshold OR   mean step >= step_threshold
    Noise     : HDBSCAN label == -1
    """
    disps  = scalar_feats[:, 0]   # net displacement
    steps  = scalar_feats[:, 1]   # mean daily step
    
    binary = np.full(len(labels), -1, dtype=np.int8)
    
    for i, lab in enumerate(labels):
        if lab == -1:
            binary[i] = -1
        elif disps[i] < disp_threshold and steps[i] < step_threshold:
            binary[i] = 0   # standing
        else:
            binary[i] = 1   # propagating
    
    return binary

# ── Sensitivity sweep ─────────────────────────────────────────────────────────

def plot_sensitivity(embedding, filename="hdbscan_sensitivity.png"):
    fpath    = os.path.join(OUT_DIR, filename)
    mcs_vals = [3, 5, 7, 10, 15]
    ms_vals  = [2, 3, 5]
    results  = []
    for mcs in mcs_vals:
        for ms in ms_vals:
            cl = hdbscan.HDBSCAN(
                min_cluster_size=mcs, min_samples=ms,
                metric="euclidean", cluster_selection_method="eom"
            ).fit(embedding)
            n_cl    = len(set(cl.labels_)) - (1 if -1 in cl.labels_ else 0)
            noise_f = (cl.labels_ == -1).mean()
            results.append((mcs, ms, n_cl, noise_f))

    df = pd.DataFrame(results, columns=["min_cluster_size", "min_samples",
                                         "n_clusters", "noise_frac"])
    print("\nSensitivity analysis (UMAP embedding):")
    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ms in ms_vals:
        sub = df[df.min_samples == ms]
        axes[0].plot(sub.min_cluster_size, sub.n_clusters, "o-",
                     label=f"min_samples={ms}")
        axes[1].plot(sub.min_cluster_size, sub.noise_frac * 100, "o-",
                     label=f"min_samples={ms}")
    for ax in axes:
        ax.axvline(HDBSCAN_MIN_CLUSTER_SIZE, color="red",
                   linestyle="--", linewidth=1.2, label="chosen mcs")
    axes[0].set_xlabel("min_cluster_size"); axes[0].set_ylabel("Clusters")
    axes[0].set_title("Cluster count"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("min_cluster_size"); axes[1].set_ylabel("Noise %")
    axes[1].set_title("Noise fraction"); axes[1].legend(fontsize=8)
    plt.suptitle("UMAP + CN features sensitivity", fontsize=10)
    plt.tight_layout()
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")
    return df


# ── Displacement diagnostic ───────────────────────────────────────────────────

def plot_displacement_distribution(scalar_feats,
                                    filename="displacement_distribution.png"):
    fpath  = os.path.join(OUT_DIR, filename)
    disps  = scalar_feats[:, 0]
    steps  = scalar_feats[:, 1]
    speeds = scalar_feats[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(disps, bins=40, color="steelblue", alpha=0.7, edgecolor="white")
    axes[0].axvline(PROP_DISP_THRESHOLD_KM, color="red", linestyle="--",
                    linewidth=2, label=f"threshold={PROP_DISP_THRESHOLD_KM} km")
    axes[0].set_xlabel("Net displacement (km)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Net displacement\n(want: peak near 0 + tail at >300 km)")
    axes[0].legend()

    axes[1].hist(steps, bins=40, color="forestgreen", alpha=0.7, edgecolor="white")
    axes[1].set_xlabel("Mean daily step (km/day)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Mean daily step\n(standing: <30 km/d)")

    axes[2].hist(speeds, bins=40, color="tomato", alpha=0.7, edgecolor="white")
    axes[2].set_xlabel("Speed (km/day)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Speed distribution")

    plt.suptitle("Trajectory kinematics — divergence-weighted centroid", fontsize=11)
    plt.tight_layout()
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


# ── Map plots ─────────────────────────────────────────────────────────────────

CLUSTER_COLORS = ["tomato", "steelblue", "forestgreen", "darkorange",
                  "purple", "gold", "cyan", "magenta", "brown", "lime"]


def plot_pathway_map(events, labels, filename="pathways_hdbscan.png"):
    fpath      = os.path.join(OUT_DIR, filename)
    unique     = sorted(set(labels))
    n_clusters = sum(1 for k in unique if k != -1)

    fig, ax = plt.subplots(figsize=(14, 7),
                           subplot_kw={"projection": ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.4)
    ax.add_feature(cfeature.LAND,      facecolor="lightgray", alpha=0.3)
    ax.set_extent([-25, 41, 35, 71])
    ax.set_title(f"Heatwave centroid pathways — HDBSCAN+UMAP+CN "
                 f"({n_clusters} clusters)", fontsize=11)

    for ev, lab in zip(events, labels):
        c    = ev["centers"]
        col  = "lightgray" if lab == -1 else CLUSTER_COLORS[lab % len(CLUSTER_COLORS)]
        zord = 1 if lab == -1 else 2
        ax.plot(c[:, 1], c[:, 0], "-o", color=col, linewidth=1.0,
                markersize=2.5, alpha=0.5, zorder=zord,
                transform=ccrs.PlateCarree())
        ax.plot(c[0, 1], c[0, 0], "o", color=col, markersize=5,
                alpha=0.8, zorder=zord+1, transform=ccrs.PlateCarree())

    t_new = np.linspace(0, 1, N_TRACK_POINTS)
    for k in unique:
        if k == -1:
            continue
        mask   = labels == k
        c_list = [ev["centers"] for ev, m in zip(events, mask) if m]
        ilats, ilons = [], []
        for c in c_list:
            if len(c) < 3:
                continue
            t_orig = np.linspace(0, 1, len(c))
            ilats.append(interp1d(t_orig, c[:, 0])(t_new))
            ilons.append(interp1d(t_orig, c[:, 1])(t_new))
        if ilats:
            col = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
            ml, mn = np.mean(ilats, axis=0), np.mean(ilons, axis=0)
            ax.plot(mn, ml, "-", color=col, linewidth=3.5, zorder=5,
                    transform=ccrs.PlateCarree(),
                    label=f"C{k} (n={mask.sum()})")
            ax.plot(mn[0], ml[0], "*", color=col, markersize=14,
                    zorder=6, transform=ccrs.PlateCarree())

    case_colors = {2003: "black", 2010: "lime", 2018: "navy"}
    for ev, lab in zip(events, labels):
        if ev["year"] in case_colors:
            c = ev["centers"]
            ax.plot(c[0, 1], c[0, 0], "D", color=case_colors[ev["year"]],
                    markersize=10, zorder=10, transform=ccrs.PlateCarree(),
                    label=f"{ev['year']} (C{lab})")

    ax.legend(loc="lower left", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


def plot_prop_standing_map(events, binary_labels,
                           filename="prop_standing_hdbscan.png"):
    fpath = os.path.join(OUT_DIR, filename)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    for ax, lab_val, title in zip(axes, [0, 1],
                                   ["Standing events", "Propagating events"]):
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.4)
        ax.add_feature(cfeature.LAND,      facecolor="lightgray", alpha=0.3)
        ax.set_extent([-25, 41, 35, 71])
        n = (binary_labels == lab_val).sum()
        ax.set_title(f"{title} (n={n})", fontsize=11)

    for ev, lab in zip(events, binary_labels):
        if lab == -1:
            continue
        c   = ev["centers"]
        col = "steelblue" if lab == 0 else "tomato"
        axes[lab].plot(c[:, 1], c[:, 0], "-", color=col, linewidth=0.9,
                       alpha=0.5, transform=ccrs.PlateCarree())
        axes[lab].plot(c[0, 1], c[0, 0], "o", color=col, markersize=4,
                       alpha=0.8, transform=ccrs.PlateCarree())

    case_colors = {2003: "black", 2010: "lime", 2018: "navy"}
    for ev, lab in zip(events, binary_labels):
        if ev["year"] in case_colors and lab != -1:
            c = ev["centers"]
            axes[lab].plot(c[0, 1], c[0, 0], "*",
                           color=case_colors[ev["year"]], markersize=12,
                           zorder=10, transform=ccrs.PlateCarree(),
                           label=str(ev["year"]))
    for ax in axes:
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


def plot_feature_distributions(scalar_feats, labels,
                                filename="hdbscan_features.png"):
    fpath      = os.path.join(OUT_DIR, filename)
    unique     = sorted(set(labels))
    feat_names = ["Total displacement (km)", "Mean step (km/day)",
                  "Speed (km/day)", "Duration (days)"]
    fig, axes  = plt.subplots(2, 2, figsize=(12, 8))
    for ax, fi, name in zip(axes.ravel(), range(4), feat_names):
        for k in unique:
            mask = labels == k
            col  = "lightgray" if k == -1 else CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
            ax.hist(scalar_feats[mask, fi], bins=20, alpha=0.6,
                    color=col, label="noise" if k==-1 else f"C{k}", density=True)
        ax.set_xlabel(name); ax.set_ylabel("Density"); ax.legend(fontsize=8)
    plt.suptitle("Feature distributions — CN+UMAP+HDBSCAN", fontsize=12)
    plt.tight_layout()
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Extract events — divergence-weighted centroid, land only
    events = extract_events(
        is_hw, z500_anom, div_vals, cc_vals, bc_vals,
        times, lat, lon, land_mask
    )

    # 2. Build features (coords + CN profiles + kinematics)
    coord_feats, scalar_feats, cn_feats, combined_feats, valid_events = \
        build_trajectory_features(events)

    # 3. Displacement diagnostic
    plot_displacement_distribution(scalar_feats)

    # 4. UMAP dimensionality reduction
    embedding, umap_model = reduce_with_umap(combined_feats)

    # 5. Sensitivity sweep on UMAP embedding
    plot_sensitivity(embedding)

    # 6. Run HDBSCAN on embedding
    clusterer  = run_hdbscan(embedding)
    labels     = clusterer.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac = (labels == -1).mean()
    print(f"\nHDBSCAN: {n_clusters} clusters, noise={noise_frac*100:.1f}%")

    non_noise = labels != -1
    if non_noise.sum() > 1 and n_clusters > 1:
        sil = silhouette_score(embedding[non_noise], labels[non_noise])
        print(f"Silhouette score: {sil:.3f}")

    summarise_clusters(labels, scalar_feats, cn_feats, valid_events)

    # 7. Assign prop / standing
    binary_labels, cluster_map = assign_prop_standing(labels, scalar_feats)
    prop_frac = (binary_labels == 1).sum() / max((binary_labels >= 0).sum(), 1)
    print(f"\nPropagating: {(binary_labels==1).sum()}  "
          f"Standing: {(binary_labels==0).sum()}  "
          f"Noise: {(binary_labels==-1).sum()}")
    print(f"Propagating fraction: {prop_frac*100:.1f}%  (Wang et al. ~40–60%)")

    # 8. Figures
    plot_umap_embedding(embedding, labels, binary_labels)
    plot_pathway_map(valid_events, labels)
    plot_prop_standing_map(valid_events, binary_labels)
    plot_feature_distributions(scalar_feats, labels)

    # 9. Daily label array
    daily_labels = np.full(len(times), -1, dtype=np.int8)
    for ev, lab in zip(valid_events, binary_labels):
        daily_labels[ev["global_start"]:ev["global_end"] + 1] = lab

    print(f"\nDaily label array:")
    print(f"  Standing days:    {(daily_labels == 0).sum()}")
    print(f"  Propagating days: {(daily_labels == 1).sum()}")
    print(f"  No-event days:    {(daily_labels == -1).sum()}")

    # 10. Save NetCDF
    out_ds = ds.copy()
    out_ds["hdbscan_cluster"] = xr.DataArray(
        np.array([labels[valid_events.index(ev)]
                  if ev in valid_events else -9
                  for ev in events], dtype=np.int8),
        dims=["event"],
        attrs={"description": "HDBSCAN cluster label (-1=noise)",
               "tracker": "divergence_weighted_centroid",
               "features": "coords+CN(CC,BC,OD-ID,Z500)+kinematics",
               "reduction": f"UMAP_{UMAP_N_COMPONENTS}D"}
    )
    out_ds["event_label_hdbscan"] = xr.DataArray(
        daily_labels, dims=["time"], coords={"time": ds.time},
        attrs={"description": "0=standing, 1=propagating, -1=no-event",
               "prop_disp_threshold_km": PROP_DISP_THRESHOLD_KM}
    )
    out_ds.to_netcdf(OUT_NC, format="NETCDF4", engine="netcdf4",
                     encoding={v: {"zlib": True, "complevel": 4}
                                for v in out_ds.data_vars})
    print(f"\nSaved: {OUT_NC}")
    print(f"All outputs: {os.path.abspath(OUT_DIR)}/")
    print("Done.")
