import os
os.environ["OMP_NUM_THREADS"] = "8"

import xarray as xr
import numpy as np
import pandas as pd
import itertools
import json
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    precision_score, recall_score
)

# ============================
# GRID SEARCH CONFIG
# ============================
# Tune one TASK at a time. Set TASK and TUNING_REGION, then run.
# Best config is saved to GRID_OUT_DIR/best_config.json
# All results are saved to GRID_OUT_DIR/grid_results.csv

# ── Set these before running ──────────────────────────────────────────────
TASK = "HW"   # "CC" | "BC" | "DC" | "HW"

# Tuning region — fixed per task type:
#   regression (CC/BC/DC) → "eastern_europe_2010"  (stable, strong signal)
#   detection  (HW)       → "scandinavia_2018"      (hardest, most informative)
TUNING_REGION = "eastern_europe_2010"

DATA_FILE    = "/gpfs/home2/mzdych/thesis/full_processed_training_dataset.nc"
GRID_OUT_DIR = f"/gpfs/home2/mzdych/thesis/gridsearch_scandi_{TASK.lower()}"
os.makedirs(GRID_OUT_DIR, exist_ok=True)

# ── Search space ──────────────────────────────────────────────────────────
# SEQ_LEN and LR are fixed — both validated in prior runs.
# We search the three parameters most likely to fix overfitting.
SEARCH_SPACE = {
    "HIDDEN_DIM":   [64, 128, 256],
    "DROPOUT":      [0.0, 0.1, 0.2, 0.3],
    "WEIGHT_DECAY": [1e-4, 1e-3, 5e-3],
}

# Fixed hyperparameters (not tuned)
SEQ_LEN       = 14
BATCH_SIZE    = 4
N_EPOCHS      = 100
LR            = 5e-4
ES_PATIENCE   = 7
ES_MIN_EPOCHS = 40

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device       : {DEVICE}")
print(f"Task         : {TASK}")
print(f"Tuning region: {TUNING_REGION}")

# ============================
# REGION REGISTRY
# ============================

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
        "lat_min": 55, "lat_max": 65, "lon_min": 5, "lon_max": 30,
        "split_type": "event",
        "train_start": "1990-06-01", "train_end": "2017-08-31",
        "val_start":   "2018-06-01", "val_end":   "2018-07-14",
        "test_start":  "2018-07-15", "test_end":  "2018-08-31",
    },
}

# ============================
# TASK CONFIG
# ============================

TASK_CONFIGS = {
    "CC": {
        "coeffs":    ["BC", "DC", "ID", "OD", "is_heatwave", "swvl1", "land_mask", "u", "v", "z"],
        "target":    "CC_target_next_day",
        "task_type": "regression",
    },
    "BC": {
        "coeffs":    ["CC", "DC", "ID", "OD", "is_heatwave", "swvl1", "land_mask", "u", "v", "z"],
        "target":    "BC_target_next_day",
        "task_type": "regression",
    },
    "DC": {
        "coeffs":    ["BC", "CC", "ID", "OD", "is_heatwave", "swvl1", "land_mask", "u", "v", "z"],
        "target":    "DC_target_next_day",
        "task_type": "regression",
    },
    "HW": {
        "coeffs":    ["BC", "DC", "ID", "OD", "CC", "swvl1", "land_mask", "u", "v", "z"],
        "target":    "is_heatwave",
        "task_type": "binary_spatial",
    },
}

COEFFS    = TASK_CONFIGS[TASK]["coeffs"]
TARGET    = TASK_CONFIGS[TASK]["target"]
TASK_TYPE = TASK_CONFIGS[TASK]["task_type"]

FEATURE_TRANSFORMS = {
    "BC": "log1p", "DC": "standard", "CC": "standard",
    "ID": "log1p", "OD": "log1p", "is_heatwave": "standard",
    "swvl1": "standard", "land_mask": "standard",
    "u": "standard", "v": "standard", "z": "standard",
}

TARGET_TRANSFORM = {
    "CC_target_next_day": "standard",
    "BC_target_next_day": "log1p",
    "DC_target_next_day": "standard",
    "is_heatwave":        None,
}

NON_NEGATIVE_TARGETS = {
    "CC_target_next_day", "BC_target_next_day", "DC_target_next_day",
}

# ============================
# DATA LOADING (once, outside loop)
# ============================

rc         = REGION_CONFIG[TUNING_REGION]
LAT_MIN    = rc["lat_min"];  LAT_MAX    = rc["lat_max"]
LON_MIN    = rc["lon_min"];  LON_MAX    = rc["lon_max"]
SPLIT_TYPE = rc["split_type"]

print(f"\nLoading dataset (loaded once, reused across configs)...")
clean_ds = xr.open_dataset(DATA_FILE)
ds       = clean_ds.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
times    = pd.DatetimeIndex(clean_ds.time.values)

X_xr    = xr.concat([ds[var] for var in COEFFS], dim="channel").assign_coords(channel=COEFFS)
Xt_vals = X_xr.transpose("time", "channel", "lat", "lon").values.astype(np.float32)

if TASK_TYPE == "regression":
    if TARGET in ds:
        y_vals = ds[TARGET].transpose("time", "lat", "lon").values.astype(np.float32)
    else:
        raw_var  = TARGET.replace("_target_next_day", "")
        raw_vals = ds[raw_var].transpose("time", "lat", "lon").values.astype(np.float32)
        y_vals   = np.roll(raw_vals, shift=-1, axis=0);  y_vals[-1] = 0.0
elif TASK_TYPE == "binary_spatial":
    y_raw  = ds[TARGET].values.astype(np.float32)
    y_vals = np.roll(y_raw, shift=-1, axis=0);  y_vals[-1] = 0.0

print(f"Xt_vals: {Xt_vals.shape}  y_vals: {y_vals.shape}")

def make_mask(times, start, end, split_type):
    date_mask = (times >= np.datetime64(start)) & (times <= np.datetime64(end))
    if split_type == "year":
        return date_mask & times.month.isin([6, 7, 8])
    return date_mask

train_mask = make_mask(times, rc["train_start"], rc["train_end"], SPLIT_TYPE)
val_mask   = make_mask(times, rc["val_start"],   rc["val_end"],   SPLIT_TYPE)
test_mask  = make_mask(times, rc["test_start"],  rc["test_end"],  SPLIT_TYPE)

# Raw (un-normalised) splits — normalisation happens inside the loop
Xt_tr_raw  = Xt_vals[train_mask].copy()
Xt_val_raw = Xt_vals[val_mask].copy()
Xt_te_raw  = Xt_vals[test_mask].copy()
y_tr_raw   = y_vals[train_mask].copy()
y_val_raw  = y_vals[val_mask].copy()
y_te_raw   = y_vals[test_mask].copy()
tms_tr     = times[train_mask]
tms_val    = times[val_mask]
tms_te     = times[test_mask]

print(f"Train: {Xt_tr_raw.shape}  {tms_tr[0].date()} → {tms_tr[-1].date()}")
print(f"Val  : {Xt_val_raw.shape}  {tms_val[0].date()} → {tms_val[-1].date()}")
print(f"Test : {Xt_te_raw.shape}  {tms_te[0].date()} → {tms_te[-1].date()}")

# ============================
# CLASSES
# ============================

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
        return X_seq, y_out


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                              kernel_size=kernel_size, padding=padding)

    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next


class SingleHeadConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, task_type, dropout=0.0, kernel_size=3):
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

    def forward(self, x):
        B, L, C, H, W = x.shape
        h = torch.zeros(B, self.cell.hidden_dim, H, W, device=x.device)
        c = torch.zeros(B, self.cell.hidden_dim, H, W, device=x.device)
        for t in range(L):
            h, c = self.cell(x[:, t], h, c)
        return self.head(h)

# ============================
# HELPER: normalise one split
# ============================

def normalise_data(Xt_tr_raw, Xt_val_raw, Xt_te_raw,
                   y_tr_raw,  y_val_raw,  y_te_raw):
    """Returns normalised copies. Does NOT modify the raw arrays."""
    Xt_tr  = Xt_tr_raw.copy();  Xt_val = Xt_val_raw.copy()
    Xt_te  = Xt_te_raw.copy()
    y_tr   = y_tr_raw.copy();   y_val  = y_val_raw.copy()
    y_te   = y_te_raw.copy()

    for ci, feat in enumerate(COEFFS):
        if FEATURE_TRANSFORMS.get(feat) == "log1p":
            Xt_tr[:, ci]  = np.log1p(Xt_tr[:, ci])
            Xt_val[:, ci] = np.log1p(Xt_val[:, ci])
            Xt_te[:, ci]  = np.log1p(Xt_te[:, ci])
        mu  = Xt_tr[:, ci].mean()
        sig = max(Xt_tr[:, ci].std(), 1e-8)
        Xt_tr[:, ci]  = (Xt_tr[:, ci]  - mu) / sig
        Xt_val[:, ci] = (Xt_val[:, ci] - mu) / sig
        Xt_te[:, ci]  = (Xt_te[:, ci]  - mu) / sig

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

    return Xt_tr, Xt_val, Xt_te, y_tr, y_val, y_te, y_mean, y_std

# ============================
# HELPER: one training run
# ============================

def run_one(cfg, run_id, save_model=False):
    """
    Train with config cfg. Returns dict of val metrics.
    cfg keys: HIDDEN_DIM, DROPOUT, WEIGHT_DECAY
    """
    hidden_dim   = cfg["HIDDEN_DIM"]
    dropout      = cfg["DROPOUT"]
    weight_decay = cfg["WEIGHT_DECAY"]

    run_dir = os.path.join(GRID_OUT_DIR, f"run_{run_id:03d}")
    os.makedirs(run_dir, exist_ok=True)

    # ── Normalise ────────────────────────────────────────────────────────
    Xt_tr, Xt_val, Xt_te, y_tr, y_val, y_te, y_mean, y_std = normalise_data(
        Xt_tr_raw, Xt_val_raw, Xt_te_raw,
        y_tr_raw,  y_val_raw,  y_te_raw
    )

    # ── Datasets ─────────────────────────────────────────────────────────
    train_ds = SeqDataset(Xt_tr,  y_tr,  tms_tr,  SEQ_LEN, TASK_TYPE)
    val_ds   = SeqDataset(Xt_val, y_val, tms_val, SEQ_LEN, TASK_TYPE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = SingleHeadConvLSTM(
        input_dim=len(COEFFS),
        hidden_dim=hidden_dim,
        task_type=TASK_TYPE,
        dropout=dropout,
        kernel_size=3,
    ).to(DEVICE)

    # ── Loss ──────────────────────────────────────────────────────────────
    if TASK_TYPE == "regression":
        criterion = nn.HuberLoss(delta=1.0)
    elif TASK_TYPE == "binary_spatial":
        hw_pos_frac   = y_tr.mean()
        hw_pos_weight = torch.tensor(
            min((1.0 - hw_pos_frac) / (hw_pos_frac + 1e-6), 10.0),
            dtype=torch.float32, device=DEVICE
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=hw_pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = np.inf
    es_counter    = 0
    best_epoch    = 0
    train_losses  = []
    val_losses    = []

    for epoch in range(N_EPOCHS):
        model.train()
        tl = 0.0
        for X, y in train_loader:
            X = X.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
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
                vl += criterion(model(X), y).item()
        vl /= len(val_loader)

        scheduler.step(vl)
        train_losses.append(tl)
        val_losses.append(vl)

        if vl < best_val_loss:
            best_val_loss = vl
            best_epoch    = epoch + 1
            es_counter    = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
        else:
            es_counter += 1

        if es_counter >= ES_PATIENCE and epoch + 1 >= ES_MIN_EPOCHS:
            break

    # ── Val metrics ───────────────────────────────────────────────────────
    model.load_state_dict(
        torch.load(os.path.join(run_dir, "best_model.pt"), map_location=DEVICE)
    )
    model.eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(DEVICE, non_blocking=True)
            pred = model(X)
            if TASK_TYPE == "regression":
                all_pred.append(pred.cpu().numpy())
                all_true.append(y.numpy())
            elif TASK_TYPE == "binary_spatial":
                all_pred.append(torch.sigmoid(pred).cpu().numpy())
                all_true.append(y.numpy())

    if TASK_TYPE == "regression":
        pred_norm = np.concatenate(all_pred)
        true_norm = np.concatenate(all_true)
        r2_val    = float(1 - np.sum((true_norm - pred_norm)**2) /
                          (np.sum((true_norm - true_norm.mean())**2) + 1e-8))
        pearson   = float(np.corrcoef(pred_norm.flatten(),
                                      true_norm.flatten())[0, 1])
        val_metric        = r2_val   # primary selection metric
        val_metric_name   = "val_r2"
        extra = {"val_pearson": pearson}

    elif TASK_TYPE == "binary_spatial":
        pred_flat = np.concatenate(all_pred).flatten()
        true_flat = np.concatenate(all_true).flatten().astype(int)
        roc       = float(roc_auc_score(true_flat, pred_flat)
                          if true_flat.sum() > 0 else float("nan"))
        pr_auc    = float(average_precision_score(true_flat, pred_flat))
        val_metric        = roc      # primary selection metric
        val_metric_name   = "val_roc_auc"
        extra = {"val_pr_auc": pr_auc}

    # ── Save loss curve ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses,   label="val")
    ax.axvline(best_epoch - 1, color="gray", linestyle="--",
               label=f"best epoch {best_epoch}")
    ax.set_title(f"Run {run_id:03d} | HD={hidden_dim} DO={dropout} WD={weight_decay}")
    ax.set_xlabel("Epoch"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=100)
    plt.close()

    result = {
        "run_id":       run_id,
        "HIDDEN_DIM":   hidden_dim,
        "DROPOUT":      dropout,
        "WEIGHT_DECAY": weight_decay,
        val_metric_name: val_metric,
        "best_epoch":   best_epoch,
        "best_val_loss": best_val_loss,
        **extra,
    }

    print(
        f"  Run {run_id:03d} | HD={hidden_dim:3d}  DO={dropout:.2f}  "
        f"WD={weight_decay:.0e} | {val_metric_name}={val_metric:.4f}  "
        f"epoch={best_epoch}",
        flush=True
    )

    # Save per-run config + result
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result

# ============================
# GRID SEARCH LOOP
# ============================

keys   = list(SEARCH_SPACE.keys())
values = list(SEARCH_SPACE.values())
grid   = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

total = len(grid)
print(f"\nGrid size: {total} configs")
print(f"Estimated time: ~{total * 5 // 60}h {total * 5 % 60}min "
      f"(assuming 5 min/run)\n")

# ── CSV header ────────────────────────────────────────────────────────────
csv_path = os.path.join(GRID_OUT_DIR, "grid_results.csv")
primary_metric = "val_r2" if TASK_TYPE == "regression" else "val_roc_auc"
secondary_metric = "val_pearson" if TASK_TYPE == "regression" else "val_pr_auc"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "run_id", "HIDDEN_DIM", "DROPOUT", "WEIGHT_DECAY",
        primary_metric, secondary_metric, "best_epoch", "best_val_loss"
    ])
    writer.writeheader()

# ── Run all configs ───────────────────────────────────────────────────────
all_results = []

for run_id, cfg in enumerate(grid):
    result = run_one(cfg, run_id)
    all_results.append(result)

    # Append to CSV after each run (safe if job is killed mid-way)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run_id", "HIDDEN_DIM", "DROPOUT", "WEIGHT_DECAY",
            primary_metric, secondary_metric, "best_epoch", "best_val_loss"
        ])
        writer.writerow({k: result.get(k, "") for k in [
            "run_id", "HIDDEN_DIM", "DROPOUT", "WEIGHT_DECAY",
            primary_metric, secondary_metric, "best_epoch", "best_val_loss"
        ]})

# ============================
# PICK BEST CONFIG
# ============================

valid_results = [r for r in all_results if not np.isnan(r.get(primary_metric, float("nan")))]
best_result   = max(valid_results, key=lambda r: r[primary_metric])

print(f"\n{'='*60}")
print(f"  GRID SEARCH COMPLETE — {total} configs")
print(f"  Task: {TASK}  Region: {TUNING_REGION}")
print(f"{'='*60}")
print(f"  Best config (run {best_result['run_id']:03d}):")
print(f"    HIDDEN_DIM   = {best_result['HIDDEN_DIM']}")
print(f"    DROPOUT      = {best_result['DROPOUT']}")
print(f"    WEIGHT_DECAY = {best_result['WEIGHT_DECAY']}")
print(f"    {primary_metric}   = {best_result[primary_metric]:.4f}")
print(f"    {secondary_metric} = {best_result.get(secondary_metric, 'n/a')}")
print(f"    best_epoch   = {best_result['best_epoch']}")

# Save best config
best_cfg_path = os.path.join(GRID_OUT_DIR, "best_config.json")
with open(best_cfg_path, "w") as f:
    json.dump(best_result, f, indent=2)
print(f"\n  Best config saved to: {best_cfg_path}")

# ── Top 5 table ───────────────────────────────────────────────────────────
sorted_results = sorted(valid_results, key=lambda r: r[primary_metric], reverse=True)
print(f"\n  Top 5 configs:")
print(f"  {'Run':>4}  {'HD':>4}  {'DO':>5}  {'WD':>8}  {primary_metric:>12}  {'epoch':>6}")
print(f"  {'-'*52}")
for r in sorted_results[:5]:
    print(f"  {r['run_id']:>4}  {r['HIDDEN_DIM']:>4}  {r['DROPOUT']:>5.2f}  "
          f"{r['WEIGHT_DECAY']:>8.0e}  {r[primary_metric]:>12.4f}  {r['best_epoch']:>6}")

# ── Summary plot: primary metric vs config index ──────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle(f"Grid search — {TASK} on {TUNING_REGION}", fontsize=12)

for ax, param in zip(axes, ["HIDDEN_DIM", "DROPOUT", "WEIGHT_DECAY"]):
    vals    = [r[param] for r in valid_results]
    metrics = [r[primary_metric] for r in valid_results]
    ax.scatter(vals, metrics, alpha=0.6)
    ax.set_xlabel(param)
    ax.set_ylabel(primary_metric)
    ax.set_title(param)
    if param == "WEIGHT_DECAY":
        ax.set_xscale("log")

plt.tight_layout()
plt.savefig(os.path.join(GRID_OUT_DIR, "grid_summary.png"), dpi=150)
plt.close()
print(f"\n  Summary plot saved to: {GRID_OUT_DIR}/grid_summary.png")
print(f"  Full results CSV  : {csv_path}")
print(f"\nDone.")
