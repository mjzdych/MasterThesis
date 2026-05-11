"""
Heatwave propagation pathway analysis using HDBSCAN.
v4: Combined coord + kinematic features to break dominant cluster.

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

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
NC_FILE          = "full_processed_training_dataset.nc"
N_TRACK_POINTS   = 10
MIN_COMP_SIZE    = 1000
MIN_DURATION     = 3
SEARCH_RADIUS_KM = 500

# Tuned: use combined coord+kinematic features
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES      = 2
HDBSCAN_METRIC           = "euclidean"

# Weight for kinematic features relative to spatial coords
# Increase to give more influence to speed/displacement in clustering
KINEMATIC_WEIGHT = 3.0

OUT_DIR = "./pathway_detection_folder_centroid"
OUT_NC  = os.path.join(OUT_DIR, "pathway_hdbscan_results.nc")

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Output directory: {os.path.abspath(OUT_DIR)}")

# ── Load dataset ──────────────────────────────────────────────────────────────
print(f"Loading {NC_FILE} ...")
ds = xr.open_dataset(NC_FILE)

lat   = ds.lat.values
lon   = ds.lon.values
times = pd.DatetimeIndex(ds.time.values)

is_hw    = ds["is_heatwave"].values.astype(bool)
z500_raw = ds["z"].values / 9.80665
z500_clim = z500_raw.mean(axis=0)
z500_anom = z500_raw - z500_clim

# ── Helpers ───────────────────────────────────────────────────────────────────

def haversine_matrix(center_lat, center_lon, lat_grid, lon_grid):
    dlat = np.radians(lat_grid - center_lat)
    dlon = np.radians(lon_grid - center_lon)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(center_lat))
         * np.cos(np.radians(lat_grid))
         * np.sin(dlon / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def haversine(p1, p2):
    return haversine_matrix(p1[0], p1[1],
                            np.array([p2[0]]), np.array([p2[1]]))[0]


def find_largest_component_size(hw_slice):
    labeled, n = scipy_label(hw_slice)
    if n == 0:
        return 0, None
    sizes = np.bincount(labeled.ravel())[1:]
    return int(sizes.max()), labeled == (sizes.argmax() + 1)


# ── Event extraction ──────────────────────────────────────────────────────────

def extract_events(is_hw, z500_anom, times, lat, lon):
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    year_list = sorted(np.unique(times.year))
    events    = []

    for yr in year_list:
        yr_mask = times.year == yr
        yr_idx  = np.where(yr_mask)[0]
        hw_yr   = is_hw[yr_mask]
        z_yr    = z500_anom[yr_mask]
        n_days  = len(yr_idx)

        daily_max = np.array([find_largest_component_size(hw_yr[t])[0]
                               for t in range(n_days)])
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
            centers   = []
            cur_lat, cur_lon = None, None

            for dt in range(track_len):
                z_t  = z_yr[t_start + dt]
                hw_t = hw_yr[t_start + dt]
                if dt == 0:
                    masked = np.where(hw_t, z_t, -np.inf)
                    if masked.max() == -np.inf:
                        masked = z_t
                    idx = np.unravel_index(masked.argmax(), masked.shape)
                else:
                    dist   = haversine_matrix(cur_lat, cur_lon, lat_grid, lon_grid)
                    search = np.where(dist < SEARCH_RADIUS_KM, z_t, -np.inf)
                    if search.max() == -np.inf:
                        search = np.where(dist < 800, z_t, -np.inf)
                    if search.max() == -np.inf:
                        search = z_t
                    idx = np.unravel_index(search.argmax(), search.shape)
                cur_lat = lat[idx[0]]
                cur_lon = lon[idx[1]]
                centers.append([cur_lat, cur_lon])

            events.append({
                "year":         yr,
                "global_start": int(yr_idx[t_start]),
                "global_end":   int(yr_idx[t_end - 1]),
                "duration":     duration,
                "peak_size":    peak_size,
                "centers":      np.array(centers),
            })

    print(f"Extracted {len(events)} events across {len(year_list)} summers")
    return events


# ── Trajectory features ───────────────────────────────────────────────────────

def build_trajectory_features(events, n_points=N_TRACK_POINTS):
    """
    Returns:
      coord_feats   : (N, n_points*2)  interpolated lat/lon
      scalar_feats  : (N, 4)           [total_disp, mean_step, speed, duration]
      combined_feats: (N, n_points*2 + 4)  used for HDBSCAN
      valid         : list of events
    """
    coords_list, scalar_list, valid = [], [], []

    for ev in events:
        c = ev["centers"]
        if len(c) < 3:
            continue
        t_orig = np.linspace(0, 1, len(c))
        t_new  = np.linspace(0, 1, n_points)
        lat_i  = interp1d(t_orig, c[:, 0], kind="linear")(t_new)
        lon_i  = interp1d(t_orig, c[:, 1], kind="linear")(t_new)
        coords_list.append(np.concatenate([lat_i, lon_i]))

        total     = haversine(c[0], c[-1])
        steps     = [haversine(c[i], c[i+1]) for i in range(len(c)-1)]
        mean_step = float(np.mean(steps))
        speed     = total / ev["duration"]
        scalar_list.append([total, mean_step, speed, ev["duration"]])
        valid.append(ev)

    coord_feats  = np.array(coords_list)   # (N, 20)
    scalar_feats = np.array(scalar_list)   # (N, 4)

    # Standardise each block separately, then upweight kinematics
    scaler_coord  = StandardScaler()
    scaler_scalar = StandardScaler()
    coord_scaled  = scaler_coord.fit_transform(coord_feats)
    scalar_scaled = scaler_scalar.fit_transform(scalar_feats)

    # Combined: spatial shape + kinematic behaviour
    combined = np.hstack([coord_scaled,
                          scalar_scaled * KINEMATIC_WEIGHT])   # (N, 24)

    print(f"Events with full tracks: {len(valid)}")
    print(f"Feature matrix: {combined.shape}  "
          f"(20 coord + 4 kinematic × weight {KINEMATIC_WEIGHT})")
    return coord_feats, scalar_feats, combined, valid


# ── HDBSCAN ───────────────────────────────────────────────────────────────────

def run_hdbscan(combined_feats):
    """Cluster on pre-scaled combined features (no extra scaling needed)."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        cluster_selection_method="eom",
        prediction_data=True,
    )
    clusterer.fit(combined_feats)
    return clusterer


def summarise_clusters(labels, scalar_feats, events):
    unique = sorted(set(labels))
    print(f"\n{'='*60}")
    print(f"  HDBSCAN  mcs={HDBSCAN_MIN_CLUSTER_SIZE}  "
          f"ms={HDBSCAN_MIN_SAMPLES}  "
          f"kinematic_weight={KINEMATIC_WEIGHT}")
    print(f"{'='*60}")
    for k in unique:
        mask  = labels == k
        tag   = "NOISE" if k == -1 else f"Cluster {k}"
        disp  = scalar_feats[mask, 0]
        speed = scalar_feats[mask, 2]
        dur   = scalar_feats[mask, 3]
        print(f"  {tag:12s}  n={mask.sum():3d}  "
              f"disp={disp.mean():5.0f}±{disp.std():4.0f} km  "
              f"speed={speed.mean():4.0f} km/d  "
              f"dur={dur.mean():.1f} d")

    print("\n  Case study events (2003, 2010, 2018):")
    for ev, lab in zip(events, labels):
        if ev["year"] in [2003, 2010, 2018]:
            idx = events.index(ev)
            print(f"    {ev['year']}  dur={ev['duration']}d  "
                  f"disp={scalar_feats[idx,0]:.0f} km  "
                  f"speed={scalar_feats[idx,2]:.0f} km/d  "
                  f"→ cluster {lab}")


# ── Prop / standing assignment ────────────────────────────────────────────────

def assign_prop_standing(labels, scalar_feats):
    """
    Speed-based binary assignment:
      - Per-cluster mean speed computed
      - Clusters above median cluster speed → propagating (1)
      - Clusters at/below median           → standing   (0)
      - Noise (-1) stays -1
    """
    unique_clusters = [k for k in sorted(set(labels)) if k != -1]
    if not unique_clusters:
        return labels.copy(), {}

    cluster_speeds    = {k: scalar_feats[labels == k, 2].mean()
                         for k in unique_clusters}
    threshold         = np.median(list(cluster_speeds.values()))
    cluster_to_binary = {k: (1 if cluster_speeds[k] > threshold else 0)
                         for k in unique_clusters}
    cluster_to_binary[-1] = -1

    print(f"\n  Cluster speeds: "
          + "  ".join([f"C{k}={v:.0f} km/d" for k, v in cluster_speeds.items()]))
    print(f"  Speed threshold (median): {threshold:.0f} km/d")
    print(f"  Mapping: {cluster_to_binary}")

    binary = np.array([cluster_to_binary[l] for l in labels])
    return binary, cluster_to_binary


# ── Sensitivity sweep ─────────────────────────────────────────────────────────

def plot_sensitivity(combined_feats, filename="hdbscan_sensitivity.png"):
    fpath    = os.path.join(OUT_DIR, filename)
    mcs_vals = [3, 5, 7, 10, 15]
    ms_vals  = [2, 3, 5]
    results  = []

    for mcs in mcs_vals:
        for ms in ms_vals:
            cl = hdbscan.HDBSCAN(
                min_cluster_size=mcs, min_samples=ms,
                metric="euclidean", cluster_selection_method="eom"
            ).fit(combined_feats)
            n_cl    = len(set(cl.labels_)) - (1 if -1 in cl.labels_ else 0)
            noise_f = (cl.labels_ == -1).mean()
            results.append((mcs, ms, n_cl, noise_f))

    df = pd.DataFrame(results, columns=["min_cluster_size", "min_samples",
                                         "n_clusters", "noise_frac"])
    print("\nSensitivity analysis (combined features):")
    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ms in ms_vals:
        sub = df[df.min_samples == ms]
        axes[0].plot(sub.min_cluster_size, sub.n_clusters,
                     "o-", label=f"min_samples={ms}")
        axes[1].plot(sub.min_cluster_size, sub.noise_frac * 100,
                     "o-", label=f"min_samples={ms}")
    for ax in axes:
        ax.axvline(HDBSCAN_MIN_CLUSTER_SIZE, color="red",
                   linestyle="--", linewidth=1.2, label="chosen mcs")
    axes[0].set_xlabel("min_cluster_size"); axes[0].set_ylabel("Number of clusters")
    axes[0].set_title("Cluster count sensitivity"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("min_cluster_size"); axes[1].set_ylabel("Noise fraction (%)")
    axes[1].set_title("Noise fraction sensitivity"); axes[1].legend(fontsize=8)
    plt.suptitle(f"Combined features (kinematic_weight={KINEMATIC_WEIGHT})",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")
    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

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
    ax.set_title(f"Heatwave propagation pathways — HDBSCAN "
                 f"({n_clusters} clusters, mcs={HDBSCAN_MIN_CLUSTER_SIZE}, "
                 f"ms={HDBSCAN_MIN_SAMPLES}, kw={KINEMATIC_WEIGHT})",
                 fontsize=10)

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
                    label=f"Cluster {k} (n={mask.sum()})")
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
    for ax, title in zip(axes, ["Standing events (HDBSCAN)",
                                 "Propagating events (HDBSCAN)"]):
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.4)
        ax.add_feature(cfeature.LAND,      facecolor="lightgray", alpha=0.3)
        ax.set_extent([-25, 41, 35, 71])
        ax.set_title(title, fontsize=11)

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
    feat_names = ["Total displacement (km)", "Mean step (km)",
                  "Speed (km/day)", "Duration (days)"]
    fig, axes  = plt.subplots(2, 2, figsize=(12, 8))
    for ax, fi, name in zip(axes.ravel(), range(4), feat_names):
        for k in unique:
            mask = labels == k
            col  = "lightgray" if k == -1 else CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
            ax.hist(scalar_feats[mask, fi], bins=15, alpha=0.6,
                    color=col, label="noise" if k==-1 else f"C{k}", density=True)
        ax.set_xlabel(name); ax.set_ylabel("Density"); ax.legend(fontsize=8)
    plt.suptitle(f"Feature distributions — HDBSCAN (kw={KINEMATIC_WEIGHT})",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Extract events
    events = extract_events(is_hw, z500_anom, times, lat, lon)

    # 2. Build features (coord + kinematic combined)
    coord_feats, scalar_feats, combined_feats, valid_events = \
        build_trajectory_features(events)

    # 3. Sensitivity sweep on combined features
    plot_sensitivity(combined_feats)

    # 4. Run HDBSCAN
    clusterer  = run_hdbscan(combined_feats)
    labels     = clusterer.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac = (labels == -1).mean()
    print(f"\nHDBSCAN: {n_clusters} clusters, noise={noise_frac*100:.1f}%")

    non_noise = labels != -1
    if non_noise.sum() > 1 and n_clusters > 1:
        sil = silhouette_score(combined_feats[non_noise], labels[non_noise])
        print(f"Silhouette score: {sil:.3f}")

    summarise_clusters(labels, scalar_feats, valid_events)

    # 5. Assign prop / standing
    binary_labels, cluster_map = assign_prop_standing(labels, scalar_feats)
    print(f"\nPropagating: {(binary_labels==1).sum()}  "
          f"Standing: {(binary_labels==0).sum()}  "
          f"Noise: {(binary_labels==-1).sum()}")

    # 6. Figures
    plot_pathway_map(valid_events, labels)
    plot_prop_standing_map(valid_events, binary_labels)
    plot_feature_distributions(scalar_feats, labels)

    # 7. Daily label array
    daily_labels = np.full(len(times), -1, dtype=np.int8)
    for ev, lab in zip(valid_events, binary_labels):
        daily_labels[ev["global_start"]:ev["global_end"] + 1] = lab

    print(f"\nDaily label array:")
    print(f"  Standing days:    {(daily_labels == 0).sum()}")
    print(f"  Propagating days: {(daily_labels == 1).sum()}")
    print(f"  No-event days:    {(daily_labels == -1).sum()}")

    # 8. Save NetCDF
    out_ds = ds.copy()
    out_ds["hdbscan_cluster"] = xr.DataArray(
        np.array([labels[valid_events.index(ev)]
                  if ev in valid_events else -9
                  for ev in events], dtype=np.int8),
        dims=["event"],
        attrs={"description": "HDBSCAN cluster label (-1=noise)",
               "min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
               "min_samples": HDBSCAN_MIN_SAMPLES,
               "kinematic_weight": KINEMATIC_WEIGHT}
    )
    out_ds["event_label_hdbscan"] = xr.DataArray(
        daily_labels, dims=["time"], coords={"time": ds.time},
        attrs={"description": "0=standing, 1=propagating, -1=no-event"}
    )
    out_ds.to_netcdf(OUT_NC, format="NETCDF4", engine="netcdf4",
                     encoding={v: {"zlib": True, "complevel": 4}
                                for v in out_ds.data_vars})
    print(f"\nSaved: {OUT_NC}")
    print(f"All outputs: {os.path.abspath(OUT_DIR)}/")
    print("Done.")
