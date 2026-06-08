import os
os.environ["OMP_NUM_THREADS"] = "8"

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, jaccard_score,
    average_precision_score
)

# ============================================================
# PATHS
# ============================================================

DATA_FILE = "/gpfs/home2/mzdych/thesis/full_processed_training_dataset.nc"
BASE_OUT  = "/gpfs/home2/mzdych/thesis/experiments"
os.makedirs(BASE_OUT, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# FIXED TRAINING SETTINGS  (not tuned)
# ============================================================

SEQ_LEN       = 14
BATCH_SIZE    = 4
N_EPOCHS      = 100
LR            = 5e-4
ES_PATIENCE   = 7
ES_MIN_EPOCHS = 40

# ============================================================
# BEST HYPERPARAMS PER TASK  (from grid search)
# ============================================================

BEST_HP = {
    "CC":  {"HIDDEN_DIM": 64,  "DROPOUT": 0.0, "WEIGHT_DECAY": 1e-4},
    "BC":  {"HIDDEN_DIM": 256, "DROPOUT": 0.2, "WEIGHT_DECAY": 1e-4},
    "DC":  {"HIDDEN_DIM": 256, "DROPOUT": 0.0, "WEIGHT_DECAY": 1e-3},
    "HW":  {"HIDDEN_DIM": 128, "DROPOUT": 0.3, "WEIGHT_DECAY": 1e-3},
    # "OD":  {"HIDDEN_DIM": 64,  "DROPOUT": 0.0, "WEIGHT_DECAY": 1e-4},
    # "ID":  {"HIDDEN_DIM": 64,  "DROPOUT": 0.0, "WEIGHT_DECAY": 1e-4},
    # "ND":  {"HIDDEN_DIM": 64,  "DROPOUT": 0.0, "WEIGHT_DECAY": 1e-4},
}

# ============================================================
# FEATURE-SET DEFINITIONS FOR ABLATION
# ============================================================

# CN features = complex-network-derived variables
CN_FEATURES  = ["BC", "DC", "ID", "OD", "CC"]

# ERA5 features = physical/meteorological variables
ERA5_FEATURES = ["swvl1", "land_mask", "u", "v", "z"]

# Per-task CN feature sets (exclude the target variable itself)
TASK_CN_FEATURES = {
    "CC":  ["BC", "DC", "ID", "OD"],          # CC is the target → excluded
    "BC":  ["CC", "DC", "ID", "OD"],          # BC is the target → excluded
    "DC":  ["BC", "CC", "ID", "OD"],          # DC is the target → excluded
    # "OD":  ["BC", "CC", "DC", "ID"],          # OD is the target → excluded
    # "ID":  ["BC", "CC", "DC", "OD"],          # ID is the target → excluded
    # "ND":  ["BC", "CC", "DC", "ID", "OD"],    # ND = OD-ID, none excluded
    "HW":  ["BC", "DC", "ID", "OD", "CC"],    # HW is binary, all CN kept
}

def get_feature_sets(task):
    """Return (full, cn_only, era5_only) feature lists for a given task."""
    cn   = TASK_CN_FEATURES[task]
    era5 = ERA5_FEATURES
    # HW also uses is_heatwave as a feature (for regression tasks it's a predictor)
    if task in ("CC", "BC", "DC", "OD", "ID", "ND"):
        cn_full = cn + ["is_heatwave"]
    else:
        cn_full = cn  # HW task: is_heatwave IS the target, not a feature
    full     = cn_full + era5
    cn_only  = cn_full
    era5_only = era5
    return full, cn_only, era5_only

# ============================================================
# EXPERIMENT LIST
# ── Automatically generates CN+ERA5 / CN-only / ERA5-only
#    for every (task, region) combination.
# ============================================================

REGIONS = [
    # "full_europe_2003", # done
    # "full_europe_2010", # done
    # "full_europe_2018", # done
    # "north_europe_2010", # done
    "north_europe_2018" # done
    # "south_europe_2003" # done
    # "eastern_europe_2010", # done
    # "iberia_2003", # done
    # "scandinavia_2018", # done
    # "mediterranean_2003" # done

]
# 23211028 - ee, iberia, mediterranean, scandi
# 23211155 - full europe 2010, south 2003
# 23211503 - north 2010 and 2018
# 23211549 - europe 2018

TASKS = ["DC"]

# Build experiments programmatically
EXPERIMENTS = []

for region in REGIONS:
    for task in TASKS:
        full_feats, cn_feats, era5_feats = get_feature_sets(task)

        # ── Full (CN + ERA5) ─────────────────────────────────
        EXPERIMENTS.append({
            "TASK":             task,
            "REGION":           region,
            "COEFFS_OVERRIDE":  full_feats,
            "RUN_SUFFIX":       "cn_era5",
        })

        # ── CN-only ──────────────────────────────────────────
        EXPERIMENTS.append({
            "TASK":             task,
            "REGION":           region,
            "COEFFS_OVERRIDE":  cn_feats,
            "RUN_SUFFIX":       "cn_only",
        })

        # ── ERA5-only ─────────────────────────────────────────
        EXPERIMENTS.append({
            "TASK":             task,
            "REGION":           region,
            "COEFFS_OVERRIDE":  era5_feats,
            "RUN_SUFFIX":       "era5_only",
        })

# ============================================================
# REGION REGISTRY
# ============================================================

REGION_CONFIG = {
    "full_europe_2003": {
        "lat_min": 35, "lat_max": 71, "lon_min": -25, "lon_max": 45,
        "split_type": "year",
        "train_start": "1990-06-01", "train_end": "2001-08-31",
        "val_start":   "2001-06-01", "val_end":   "2002-08-31",
        "test_start":  "2003-06-01", "test_end":  "2003-08-31",
    },
    "full_europe_2010": {
        "lat_min": 35, "lat_max": 71, "lon_min": -25, "lon_max": 45,
        "split_type": "year",
        "train_start": "1990-06-01", "train_end": "2008-08-31",
        "val_start":   "2008-06-01", "val_end":   "2009-08-31",
        "test_start":  "2010-06-01", "test_end":  "2010-08-31",
    },
    "full_europe_2018": {
        "lat_min": 35, "lat_max": 71, "lon_min": -25, "lon_max": 45,
        "split_type": "year",
        "train_start": "1990-06-01", "train_end": "2015-08-31",
        "val_start":   "2016-06-01", "val_end":   "2017-08-31",
        "test_start":  "2018-06-01", "test_end":  "2018-08-31",
    },
    "north_europe_2010": {
        "lat_min": 55, "lat_max": 71, "lon_min": -25, "lon_max": 45,
        "split_type": "year",
        "train_start": "1990-06-01", "train_end": "2007-08-31",
        "val_start":   "2008-06-01", "val_end":   "2009-08-31",
        "test_start":  "2010-06-01", "test_end":  "2010-08-31",
    },
    "north_europe_2018": {
        "lat_min": 55, "lat_max": 71, "lon_min": -25, "lon_max": 45,
        "split_type": "year",
        "train_start": "1990-06-01", "train_end": "2015-08-31",
        "val_start":   "2016-06-01", "val_end":   "2017-08-31",
        "test_start":  "2018-06-01", "test_end":  "2018-08-31",
    },
    "south_europe_2003": {
        "lat_min": 35, "lat_max": 55, "lon_min": -25, "lon_max": 45,
        "split_type": "year",
        "train_start": "1990-06-01", "train_end": "2000-08-31",
        "val_start":   "2001-06-01", "val_end":   "2002-08-31",
        "test_start":  "2003-06-01", "test_end":  "2003-08-31",
    },
    "iberia_2003": {
        "lat_min": 36, "lat_max": 44, "lon_min": -10, "lon_max": 5,
        "split_type": "event",
        "train_start": "1990-06-01", "train_end": "2002-08-31",
        "val_start":   "2003-06-01", "val_end":   "2003-07-14",
        "test_start":  "2003-07-15", "test_end":  "2003-08-31",
    },
    "mediterranean_2003": {
        "lat_min": 30, "lat_max": 48, "lon_min": -10, "lon_max": 40,
        "split_type": "event",
        "train_start": "1990-06-01", "train_end": "2002-08-31",
        "val_start":   "2003-06-01", "val_end":   "2003-07-14",
        "test_start":  "2003-07-15", "test_end":  "2003-08-31",
    },
    "eastern_europe_2010": {
        "lat_min": 45, "lat_max": 55, "lon_min": 20, "lon_max": 40,
        "split_type": "event",
        "train_start": "1990-06-01", "train_end": "2009-08-31",
        "val_start":   "2010-06-01", "val_end":   "2010-06-30",
        "test_start":  "2010-07-01", "test_end":  "2010-08-31",
    },
    "scandinavia_2018": {
        "lat_min": 55, "lat_max": 65, "lon_min": 5,  "lon_max": 30,
        "split_type": "event",
        "train_start": "1990-06-01", "train_end": "2017-08-31",
        "val_start":   "2018-06-01", "val_end":   "2018-07-14",
        "test_start":  "2018-07-15", "test_end":  "2018-08-31",
    },
}

# ============================================================
# TASK DEFAULTS
# ============================================================

TASK_DEFAULTS = {
    "CC":  {
        "coeffs":    ["BC", "DC", "ID", "OD", "is_heatwave", "swvl1", "land_mask", "u", "v", "z"],
        "target":    "CC_target_next_day",
        "task_type": "regression",
    },
    "BC":  {
        "coeffs":    ["CC", "DC", "ID", "OD", "is_heatwave", "swvl1", "land_mask", "u", "v", "z"],
        "target":    "BC_target_next_day",
        "task_type": "regression",
    },
    "DC":  {
        "coeffs":    ["BC", "CC", "ID", "OD", "is_heatwave", "swvl1", "land_mask", "u", "v", "z"],
        "target":    "DC_target_next_day",
        "task_type": "regression",
    },
    "OD":  {
        "coeffs":    ["BC", "CC", "DC", "ID", "is_heatwave", "swvl1", "land_mask", "u", "v", "z"],
        "target":    "OD_target_next_day",
        "task_type": "regression",
    },
    "ID":  {
        "coeffs":    ["BC", "CC", "DC", "OD", "is_heatwave", "swvl1", "land_mask", "u", "v", "z"],
        "target":    "ID_target_next_day",
        "task_type": "regression",
    },
    "ND":  {
        "coeffs":    ["BC", "CC", "DC", "ID", "OD", "is_heatwave", "swvl1", "land_mask", "u", "v", "z"],
        "target":    "ND_target_next_day",
        "task_type": "regression",
    },
    "HW":  {
        "coeffs":    ["BC", "DC", "ID", "OD", "CC", "swvl1", "land_mask", "u", "v", "z"],
        "target":    "is_heatwave",
        "task_type": "binary_spatial",
    },
    "CLASS": {
        "coeffs":    ["BC", "DC", "ID", "OD", "swvl1", "land_mask", "u", "v", "z"],
        "target":    "event_label",
        "task_type": "binary_scalar",
    },
}

# ============================================================
# TRANSFORMS
# ============================================================

FEATURE_TRANSFORMS = {
    "BC": "log1p", "DC": "standard", "CC": "standard",
    "ID": "log1p", "OD": "log1p",    "is_heatwave": "standard",
    "swvl1": "standard", "land_mask": "standard",
    "u": "standard", "v": "standard", "z": "standard",
}

TARGET_TRANSFORM = {
    "CC_target_next_day": "standard",
    "BC_target_next_day": "log1p",
    "DC_target_next_day": "standard",
    "OD_target_next_day": "log1p",
    "ID_target_next_day": "log1p",
    "ND_target_next_day": "standard",
    "is_heatwave":        None,
    "event_label":        None,
}

NON_NEGATIVE_TARGETS = {
    "CC_target_next_day", "BC_target_next_day", "DC_target_next_day",
    "OD_target_next_day", "ID_target_next_day",
}

# ============================================================
# CLASSES
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs     = F.softmax(logits, dim=1)
        probs_pos = probs[:, 1]
        t         = targets.float()
        p_t       = probs_pos * t + (1 - probs_pos) * (1 - t)
        alpha_t   = self.alpha * t + (1 - self.alpha) * (1 - t)
        focal     = alpha_t * (1 - p_t) ** self.gamma
        ce        = F.cross_entropy(logits, targets, reduction="none")
        return (focal * ce).mean()


class SeqDataset(Dataset):
    def __init__(self, X, y, times, seq_len, task_type):
        self.X = X; self.y = y; self.times = times
        self.seq_len = seq_len; self.task_type = task_type
        self.indices = []
        for yr in np.unique(times.year):
            yr_idx = np.where(times.year == yr)[0]
            for i in range(len(yr_idx) - seq_len):
                self.indices.append((yr_idx[i], yr_idx[i + seq_len]))

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        start, target = self.indices[idx]
        X_seq = torch.tensor(self.X[start:start + self.seq_len], dtype=torch.float32)
        if self.task_type in ("regression", "binary_spatial"):
            y_out = torch.tensor(self.y[target][None, :, :], dtype=torch.float32)
        elif self.task_type == "binary_scalar":
            y_out = torch.tensor(int(self.y[target]), dtype=torch.long)
        return X_seq, y_out


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim,
            kernel_size=kernel_size, padding=padding
        )

    def forward(self, x, h, c):
        gates      = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next


class SingleHeadConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, task_type,
                 dropout=0.0, kernel_size=3, n_classes=2):
        super().__init__()
        self.task_type = task_type
        self.cell      = ConvLSTMCell(input_dim, hidden_dim, kernel_size)

        if task_type == "regression":
            self.head = nn.Sequential(
                nn.Dropout2d(p=dropout),
                nn.Conv2d(hidden_dim, 1, kernel_size=1)
            )
        elif task_type == "binary_spatial":
            self.head = nn.Sequential(
                nn.Dropout2d(p=dropout),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, 1, kernel_size=1)
            )
        elif task_type == "binary_scalar":
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, n_classes)
            )

    def forward(self, x):
        B, L, C, H, W = x.shape
        h = torch.zeros(B, self.cell.hidden_dim, H, W, device=x.device)
        c = torch.zeros(B, self.cell.hidden_dim, H, W, device=x.device)
        for t in range(L):
            h, c = self.cell(x[:, t], h, c)
        return self.head(h)

# ============================================================
# HELPERS
# ============================================================

def make_mask(times, start, end, split_type):
    date_mask = (times >= np.datetime64(start)) & (times <= np.datetime64(end))
    if split_type == "year":
        return date_mask & times.month.isin([6, 7, 8])
    return date_mask


def fmt_split(f, split, m):
    f.write(f"=== {split} ===\n")
    for k, v in m.items():
        f.write(f"  {k}: {v}\n")
    f.write("\n")

# ============================================================
# SINGLE EXPERIMENT
# ============================================================

def run_experiment(exp, clean_ds, times):

    TASK      = exp["TASK"]
    REGION    = exp["REGION"]
    suffix    = exp.get("RUN_SUFFIX", "")
    run_name  = f"{REGION}_{TASK.lower()}" + (f"_{suffix}" if suffix else "")
    OUT_DIR   = os.path.join(BASE_OUT, run_name)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Skip completed runs ───────────────────────────────────────────────
    if os.path.exists(os.path.join(OUT_DIR, "DONE")):
        print(f"[SKIP] {run_name} — already done.")
        return run_name, None

    print(f"\n{'='*65}")
    print(f"  START  {run_name}")
    print(f"{'='*65}")

    # ── Resolve settings ──────────────────────────────────────────────────
    rc        = REGION_CONFIG[REGION]
    td        = TASK_DEFAULTS[TASK]
    TASK_TYPE = td["task_type"]
    COEFFS    = exp.get("COEFFS_OVERRIDE", td["coeffs"])
    TARGET    = td["target"]
    hp        = {**BEST_HP[TASK], **exp.get("HP_OVERRIDE", {})}
    HIDDEN_DIM   = hp["HIDDEN_DIM"]
    DROPOUT      = hp["DROPOUT"]
    WEIGHT_DECAY = hp["WEIGHT_DECAY"]
    SPLIT_TYPE   = rc["split_type"]

    print(f"  Task     : {TASK} ({TASK_TYPE})")
    print(f"  Region   : {REGION}  "
          f"lat=[{rc['lat_min']},{rc['lat_max']}]  "
          f"lon=[{rc['lon_min']},{rc['lon_max']}]")
    print(f"  Features : {COEFFS}")
    print(f"  Target   : {TARGET}")
    print(f"  HP       : HD={HIDDEN_DIM}  DO={DROPOUT}  WD={WEIGHT_DECAY}")

    # ── Slice + build arrays ──────────────────────────────────────────────
    ds = clean_ds.sel(
        lat=slice(rc["lat_min"], rc["lat_max"]),
        lon=slice(rc["lon_min"], rc["lon_max"])
    )

    X_xr    = xr.concat([ds[var] for var in COEFFS], dim="channel").assign_coords(channel=COEFFS)
    Xt_vals = X_xr.transpose("time", "channel", "lat", "lon").values.astype(np.float32)

    if TASK_TYPE == "regression":
        if TARGET == "ND_target_next_day":
            od_raw = ds["OD"].transpose("time", "lat", "lon").values.astype(np.float32)
            id_raw = ds["ID"].transpose("time", "lat", "lon").values.astype(np.float32)
            y_vals = np.roll(od_raw - id_raw, shift=-1, axis=0); y_vals[-1] = 0.0
        elif TARGET in ds:
            y_vals = ds[TARGET].transpose("time", "lat", "lon").values.astype(np.float32)
        else:
            raw_var  = TARGET.replace("_target_next_day", "")
            raw_vals = ds[raw_var].transpose("time", "lat", "lon").values.astype(np.float32)
            y_vals   = np.roll(raw_vals, shift=-1, axis=0); y_vals[-1] = 0.0
    elif TASK_TYPE == "binary_spatial":
        y_raw  = ds[TARGET].values.astype(np.float32)
        y_vals = np.roll(y_raw, shift=-1, axis=0); y_vals[-1] = 0.0
    elif TASK_TYPE == "binary_scalar":
        y_vals = clean_ds[TARGET].values.astype(np.int8)

    # ── Split ─────────────────────────────────────────────────────────────
    train_mask = make_mask(times, rc["train_start"], rc["train_end"], SPLIT_TYPE)
    val_mask   = make_mask(times, rc["val_start"],   rc["val_end"],   SPLIT_TYPE)
    test_mask  = make_mask(times, rc["test_start"],  rc["test_end"],  SPLIT_TYPE)

    Xt_tr,  Xt_val,  Xt_te  = Xt_vals[train_mask], Xt_vals[val_mask], Xt_vals[test_mask]
    y_tr,   y_val,   y_te   = y_vals[train_mask],   y_vals[val_mask],  y_vals[test_mask]
    tms_tr, tms_val, tms_te = times[train_mask],    times[val_mask],   times[test_mask]

    print(f"  Train: {tms_tr[0].date()} → {tms_tr[-1].date()}  ({len(tms_tr)} days)")
    print(f"  Val  : {tms_val[0].date()} → {tms_val[-1].date()}  ({len(tms_val)} days)")
    print(f"  Test : {tms_te[0].date()} → {tms_te[-1].date()}  ({len(tms_te)} days)")

    # ── Feature normalisation ─────────────────────────────────────────────
    channel_mean = np.zeros((1, len(COEFFS), 1, 1), dtype=np.float32)
    channel_std  = np.ones( (1, len(COEFFS), 1, 1), dtype=np.float32)

    for ci, feat in enumerate(COEFFS):
        transform = FEATURE_TRANSFORMS.get(feat, "standard")
        if transform == "log1p":
            Xt_tr[:, ci]  = np.log1p(Xt_tr[:, ci])
            Xt_val[:, ci] = np.log1p(Xt_val[:, ci])
            Xt_te[:, ci]  = np.log1p(Xt_te[:, ci])
        mu  = Xt_tr[:, ci].mean()
        sig = Xt_tr[:, ci].std()
        if sig < 1e-8: sig = 1.0
        Xt_tr[:, ci]  = (Xt_tr[:, ci]  - mu) / sig
        Xt_val[:, ci] = (Xt_val[:, ci] - mu) / sig
        Xt_te[:, ci]  = (Xt_te[:, ci]  - mu) / sig
        channel_mean[0, ci, 0, 0] = mu
        channel_std[ 0, ci, 0, 0] = sig

    np.save(os.path.join(OUT_DIR, "channel_mean.npy"), channel_mean)
    np.save(os.path.join(OUT_DIR, "channel_std.npy"),  channel_std)

    # ── Target normalisation ──────────────────────────────────────────────
    y_mean, y_std = 0.0, 1.0
    if TASK_TYPE == "regression":
        tgt_transform = TARGET_TRANSFORM.get(TARGET, "standard")
        if tgt_transform == "log1p":
            y_tr  = np.log1p(y_tr)
            y_val = np.log1p(y_val)
            y_te  = np.log1p(y_te)
        y_mean = float(y_tr.mean())
        y_std  = float(y_tr.std()) + 1e-8
        y_tr   = (y_tr  - y_mean) / y_std
        y_val  = (y_val - y_mean) / y_std
        y_te   = (y_te  - y_mean) / y_std
        np.save(os.path.join(OUT_DIR, "y_mean.npy"), np.array(y_mean))
        np.save(os.path.join(OUT_DIR, "y_std.npy"),  np.array(y_std))
        print(f"  Target normalised — mean={y_mean:.6f}  std={y_std:.6f}")

    # ── Class balance ─────────────────────────────────────────────────────
    hw_pos_weight = None
    if TASK_TYPE == "binary_spatial":
        hw_pos_frac   = y_tr.mean()
        hw_pos_weight = torch.tensor(
            min((1.0 - hw_pos_frac) / (hw_pos_frac + 1e-6), 10.0),
            dtype=torch.float32, device=DEVICE
        )
        print(f"  pos_weight (capped): {hw_pos_weight:.2f}")

    elif TASK_TYPE == "binary_scalar":
        valid   = y_tr[y_tr >= 0]
        n_stand = int((valid == 0).sum())
        n_prop  = int((valid == 1).sum())
        print(f"  Train — standing: {n_stand}  propagating: {n_prop}  "
              f"ratio: {n_stand/max(n_prop,1):.1f}:1")

    # ── Datasets + loaders ────────────────────────────────────────────────
    train_ds = SeqDataset(Xt_tr,  y_tr,  tms_tr,  SEQ_LEN, TASK_TYPE)
    val_ds   = SeqDataset(Xt_val, y_val, tms_val, SEQ_LEN, TASK_TYPE)
    test_ds  = SeqDataset(Xt_te,  y_te,  tms_te,  SEQ_LEN, TASK_TYPE)
    print(f"  Sequences — train:{len(train_ds)}  val:{len(val_ds)}  test:{len(test_ds)}")

    if TASK_TYPE == "binary_scalar":
        seq_labels     = np.array([int(y_tr[t]) for _, t in train_ds.indices])
        prop_mask_seq  = seq_labels == 1
        stand_mask_seq = seq_labels == 0
        no_ev_mask_seq = seq_labels == -1
        sample_weights = np.ones(len(seq_labels))
        if prop_mask_seq.sum() > 0 and stand_mask_seq.sum() > 0:
            w_prop = stand_mask_seq.sum() / prop_mask_seq.sum()
            sample_weights[prop_mask_seq]  = w_prop
            sample_weights[stand_mask_seq] = 1.0
            sample_weights[no_ev_mask_seq] = 0.3
        sampler = WeightedRandomSampler(
            torch.tensor(sample_weights, dtype=torch.float32),
            num_samples=len(train_ds), replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  sampler=sampler, shuffle=False,
                                  num_workers=2, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=2, pin_memory=True)

    val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = SingleHeadConvLSTM(
        input_dim=len(COEFFS),
        hidden_dim=HIDDEN_DIM,
        task_type=TASK_TYPE,
        dropout=DROPOUT,
        kernel_size=3,
        n_classes=2,
    ).to(DEVICE)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss ──────────────────────────────────────────────────────────────
    if TASK_TYPE == "regression":
        criterion = nn.HuberLoss(delta=1.0)
    elif TASK_TYPE == "binary_spatial":
        criterion = nn.BCEWithLogitsLoss(pos_weight=hw_pos_weight)
    elif TASK_TYPE == "binary_scalar":
        criterion = FocalLoss(alpha=0.75, gamma=2.0).to(DEVICE)

    def compute_loss(pred, target):
        if TASK_TYPE == "binary_scalar":
            valid = target >= 0
            if valid.sum() == 0:
                return torch.tensor(0.0, device=DEVICE, requires_grad=True)
            return criterion(pred[valid], target[valid].long())
        return criterion(pred, target)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # ── Training loop ─────────────────────────────────────────────────────
    history       = {"train_loss": [], "val_loss": []}
    best_val_loss = np.inf
    es_counter    = 0
    best_epoch    = 0

    print("\n  Training...")
    for epoch in range(N_EPOCHS):
        model.train()
        tl = 0.0
        for X, y in train_loader:
            X = X.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            loss = compute_loss(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tl += loss.item()
        tl /= len(train_loader)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                vl += compute_loss(model(X), y).item()
        vl /= len(val_loader)

        scheduler.step(vl)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)

        if vl < best_val_loss:
            best_val_loss = vl
            best_epoch    = epoch + 1
            es_counter    = 0
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pt"))
        else:
            es_counter += 1

        print(
            f"  Epoch {epoch+1:03d}/{N_EPOCHS} | "
            f"train={tl:.4f}  val={vl:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"es={es_counter}/{ES_PATIENCE}",
            flush=True
        )

        if es_counter >= ES_PATIENCE and epoch + 1 >= ES_MIN_EPOCHS:
            print(f"  Early stopping at epoch {epoch+1} (best: {best_epoch})")
            break

    print(f"  Training done. Best epoch: {best_epoch}  Best val loss: {best_val_loss:.4f}")

    # ── Loss curve ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"],   label="val")
    ax.axvline(best_epoch - 1, color="gray", linestyle="--",
               label=f"best epoch {best_epoch}")
    ax.set_title(f"{TASK} — {REGION} — {suffix} — loss curve  "
                 f"(HD={HIDDEN_DIM} DO={DROPOUT} WD={WEIGHT_DECAY})")
    ax.set_xlabel("Epoch"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_curve.png"), dpi=150)
    plt.close()

    # ── Inverse transform ─────────────────────────────────────────────────
    def inverse_transform_target(arr):
        out = arr * y_std + y_mean
        if TASK_TYPE == "regression":
            if TARGET_TRANSFORM.get(TARGET) == "log1p":
                out = np.expm1(out)
            if TARGET in NON_NEGATIVE_TARGETS:
                out = np.clip(out, 0.0, None)
        return out

    # ── Evaluation ────────────────────────────────────────────────────────
    model.load_state_dict(
        torch.load(os.path.join(OUT_DIR, "best_model.pt"), map_location=DEVICE)
    )
    model.eval()

    def evaluate(loader, split_name):
        all_pred, all_true = [], []
        with torch.no_grad():
            for X, y in loader:
                X    = X.to(DEVICE, non_blocking=True)
                pred = model(X)
                if TASK_TYPE == "regression":
                    all_pred.append(pred.cpu().numpy())
                    all_true.append(y.numpy())
                elif TASK_TYPE == "binary_spatial":
                    all_pred.append(torch.sigmoid(pred).cpu().numpy())
                    all_true.append(y.numpy())
                elif TASK_TYPE == "binary_scalar":
                    probs = torch.softmax(pred, dim=1).cpu().numpy()
                    y_np  = y.numpy(); valid = y_np >= 0
                    all_pred.extend(probs[valid, 1].tolist())
                    all_true.extend(y_np[valid].tolist())

        print(f"\n  === {split_name} ===")
        metrics = {}

        if TASK_TYPE == "regression":
            pred_norm = np.concatenate(all_pred)
            true_norm = np.concatenate(all_true)
            mae_norm  = float(np.mean(np.abs(pred_norm - true_norm)))
            r2_norm   = float(1 - np.sum((true_norm - pred_norm)**2) /
                              (np.sum((true_norm - true_norm.mean())**2) + 1e-8))
            pred_arr  = inverse_transform_target(pred_norm)
            true_arr  = inverse_transform_target(true_norm)
            mae       = float(np.mean(np.abs(pred_arr - true_arr)))
            rmse      = float(np.sqrt(np.mean((pred_arr - true_arr)**2)))
            r2        = float(1 - np.sum((true_arr - pred_arr)**2) /
                              (np.sum((true_arr - true_arr.mean())**2) + 1e-8))
            pearson   = float(np.corrcoef(pred_arr.flatten(), true_arr.flatten())[0, 1])
            print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  Pearson={pearson:.4f}")
            metrics = dict(mae_norm=mae_norm, r2_norm=r2_norm,
                           mae=mae, rmse=rmse, r2=r2, pearson=pearson)

        elif TASK_TYPE == "binary_spatial":
            pred_arr = np.concatenate(all_pred).flatten()
            true_arr = np.concatenate(all_true).flatten().astype(int)
            bin_arr  = (pred_arr >= 0.5).astype(int)
            f1     = float(f1_score(true_arr, bin_arr,  zero_division=0))
            prec   = float(precision_score(true_arr, bin_arr, zero_division=0))
            rec    = float(recall_score(true_arr, bin_arr,  zero_division=0))
            iou    = float(jaccard_score(true_arr, bin_arr,  zero_division=0))
            pr_auc = float(average_precision_score(true_arr, pred_arr))
            roc    = float(roc_auc_score(true_arr, pred_arr)
                           if true_arr.sum() > 0 and (1 - true_arr).sum() > 0
                           else float("nan"))
            best_f1, best_th = 0.0, 0.5
            for th in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
                p_th  = (pred_arr >= th).astype(int)
                f1_th = float(f1_score(true_arr, p_th, zero_division=0))
                pr_th = float(precision_score(true_arr, p_th, zero_division=0))
                re_th = float(recall_score(true_arr, p_th, zero_division=0))
                print(f"    th={th:.2f}  F1={f1_th:.3f}  Prec={pr_th:.3f}  Rec={re_th:.3f}")
                if f1_th > best_f1:
                    best_f1, best_th = f1_th, th
            print(f"  ROC={roc:.4f}  PR-AUC={pr_auc:.4f}  "
                  f"best F1={best_f1:.4f} @ th={best_th}")
            metrics = dict(f1=f1, prec=prec, rec=rec, iou=iou,
                           pr_auc=pr_auc, roc=roc,
                           best_thresh=best_th, best_f1=best_f1)

        elif TASK_TYPE == "binary_scalar":
            probs_arr = np.array(all_pred)
            true_arr  = np.array(all_true)
            pred_arr  = (probs_arr >= 0.5).astype(int)
            acc  = float(accuracy_score(true_arr, pred_arr))
            prec = float(precision_score(true_arr, pred_arr, zero_division=0))
            rec  = float(recall_score(true_arr, pred_arr,    zero_division=0))
            f1   = float(f1_score(true_arr, pred_arr,        zero_division=0))
            cm   = confusion_matrix(true_arr, pred_arr)
            best_f1, best_th = 0.0, 0.5
            for th in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
                p_th  = (probs_arr >= th).astype(int)
                f1_th = float(f1_score(true_arr, p_th, zero_division=0))
                if f1_th > best_f1:
                    best_f1, best_th = f1_th, th
            print(f"  Acc={acc:.4f}  F1={best_f1:.4f}@th={best_th}  "
                  f"Prec={prec:.4f}  Rec={rec:.4f}")
            print(f"  Confusion matrix:\n{cm}")
            metrics = dict(acc=acc, prec=prec, rec=rec, f1=f1,
                           cm=cm, best_thresh=best_th, best_f1=best_f1)

        return metrics

    val_metrics  = evaluate(val_loader,  "VAL")
    test_metrics = evaluate(test_loader, "TEST")

    # ── Save metrics.txt ──────────────────────────────────────────────────
    with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"Task           : {TASK}  ({TASK_TYPE})\n")
        f.write(f"Ablation       : {suffix}\n")
        f.write(f"Features       : {COEFFS}\n")
        f.write(f"Target         : {TARGET}\n")
        f.write(f"Context region : lat=[{rc['lat_min']},{rc['lat_max']}]  "
                f"lon=[{rc['lon_min']},{rc['lon_max']}]\n")
        f.write(f"Target region  : lat=[{rc['lat_min']},{rc['lat_max']}]  "
                f"lon=[{rc['lon_min']},{rc['lon_max']}]\n")
        f.write(f"Train  : {rc['train_start']} → {rc['train_end']}\n")
        f.write(f"Val    : {rc['val_start']} → {rc['val_end']}\n")
        f.write(f"Test   : {rc['test_start']} → {rc['test_end']}\n")
        f.write(f"Best epoch     : {best_epoch}  "
                f"(ES patience={ES_PATIENCE}, min epochs={ES_MIN_EPOCHS})\n")
        f.write(f"HIDDEN_DIM={HIDDEN_DIM}  DROPOUT={DROPOUT}  "
                f"WEIGHT_DECAY={WEIGHT_DECAY}\n")
        f.write(f"Target transform : {TARGET_TRANSFORM.get(TARGET, 'standard')}\n")
        f.write(f"y_mean={y_mean:.6f}  y_std={y_std:.6f}\n\n")
        fmt_split(f, "VAL",  val_metrics)
        fmt_split(f, "TEST", test_metrics)

    # ── Save checkpoint ───────────────────────────────────────────────────
    torch.save({
        "model_state_dict":   model.state_dict(),
        "task":               TASK,
        "task_type":          TASK_TYPE,
        "region":             REGION,
        "ablation":           suffix,
        "split_type":         SPLIT_TYPE,
        "coeffs":             COEFFS,
        "target":             TARGET,
        "seq_len":            SEQ_LEN,
        "hidden_dim":         HIDDEN_DIM,
        "dropout":            DROPOUT,
        "weight_decay":       WEIGHT_DECAY,
        "channel_mean":       channel_mean,
        "channel_std":        channel_std,
        "feature_transforms": FEATURE_TRANSFORMS,
        "target_transform":   TARGET_TRANSFORM.get(TARGET, "standard"),
        "y_mean":             y_mean,
        "y_std":              y_std,
        "history":            history,
        "best_val_loss":      best_val_loss,
        "best_epoch":         best_epoch,
        "region_config":      rc,
        "val_metrics":        val_metrics,
        "test_metrics":       test_metrics,
    }, os.path.join(OUT_DIR, "final_checkpoint.pt"))

    open(os.path.join(OUT_DIR, "DONE"), "w").close()
    print(f"\n  DONE  →  {OUT_DIR}")
    return run_name, test_metrics

# ============================================================
# MAIN
# ============================================================

print(f"Device           : {DEVICE}")
print(f"Total experiments: {len(EXPERIMENTS)}")
print(f"Output base      : {BASE_OUT}\n")

# Print experiment plan
print("Experiment plan:")
for i, exp in enumerate(EXPERIMENTS):
    label = f"{exp['REGION']}_{exp['TASK'].lower()}_{exp.get('RUN_SUFFIX','')}"
    feats = exp.get("COEFFS_OVERRIDE", TASK_DEFAULTS[exp["TASK"]]["coeffs"])
    print(f"  [{i+1:02d}] {label:55s}  features={feats}")
print()

print("Loading dataset (once)...")
clean_ds = xr.open_dataset(DATA_FILE)
times    = pd.DatetimeIndex(clean_ds.time.values)
print(f"Dataset loaded   : {dict(clean_ds.dims)}\n")

summary = []

for i, exp in enumerate(EXPERIMENTS):
    label = f"{exp['REGION']}_{exp['TASK'].lower()}_{exp.get('RUN_SUFFIX','')}"
    print(f"[{i+1}/{len(EXPERIMENTS)}]  {label}")

    try:
        run_name, test_metrics = run_experiment(exp, clean_ds, times)
        summary.append({"run": run_name, "status": "OK", "metrics": test_metrics})
    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()
        summary.append({"run": label, "status": f"ERROR: {e}", "metrics": None})

# ── Final summary ─────────────────────────────────────────────────────────
print(f"\n\n{'='*75}")
print(f"  BATCH COMPLETE — {len(EXPERIMENTS)} experiments  "
      f"({len(REGIONS)} regions × {len(TASKS)} tasks × 3 ablations)")
print(f"{'='*75}")
for s in summary:
    m = s["metrics"]
    if m is None:
        print(f"  {s['run']:60s}  {s['status']}")
    elif "r2" in m:
        print(f"  {s['run']:60s}  R²={m['r2']:.3f}  P={m['pearson']:.3f}")
    elif "roc" in m:
        print(f"  {s['run']:60s}  ROC={m['roc']:.3f}  PR={m['pr_auc']:.3f}  "
              f"F1={m['best_f1']:.3f}")
    else:
        print(f"  {s['run']:60s}  {s['status']}")


