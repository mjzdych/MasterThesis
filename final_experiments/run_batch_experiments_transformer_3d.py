"""
run_batch_experiments_transformer.py

Drop-in Spatiotemporal Transformer replacement for SingleHeadConvLSTM.
Identical data pipeline, training loop, evaluation, and output format.
Only the model class changes.

Architecture — SpatioTemporalTransformer:
  1. CNN spatial encoder  : (B, L, C, H, W) → (B, L, D, H, W)  [local spatial features]
  2. Temporal attention   : attends across L=14 timesteps per spatial location
  3. Spatial attention    : attends across H×W locations at the final step
  4. Same task heads as ConvLSTM (regression / binary_spatial / binary_scalar)

Usage:
  - Set REGIONS, TASKS, MODEL = "transformer" at the top
  - Outputs go to BASE_OUT/{region}_{task}_{suffix}_transformer/
  - metrics.txt format identical to ConvLSTM runs for easy comparison
"""

import os
os.environ["OMP_NUM_THREADS"] = "8"

import math
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, jaccard_score, average_precision_score
)

# ============================================================
# PATHS
# ============================================================

DATA_FILE = "/gpfs/home2/mzdych/thesis/full_processed_training_dataset.nc"
BASE_OUT  = "/gpfs/home2/mzdych/thesis/experiments_transformer_7d"
os.makedirs(BASE_OUT, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# SETTINGS
# ============================================================

SEQ_LEN       = 14
BATCH_SIZE    = 4
N_EPOCHS      = 100
LR            = 5e-4
ES_PATIENCE   = 7
ES_MIN_EPOCHS = 40

# ── Forecast horizon ─────────────────────────────────────────────────────────
# How many days ahead to predict.  Change this to 3 or 7 for t+3 / t+7 runs.
FORECAST_HORIZON = 7   # {1, 3, 7}

# Transformer-specific hyperparams (kept consistent across tasks)
# Embed dim matched to ConvLSTM hidden dims where possible
TRANSFORMER_HP = {
    "CC":  {"EMBED_DIM": 64,  "N_HEADS": 4, "N_LAYERS": 2, "DROPOUT": 0.1, "WEIGHT_DECAY": 1e-4},
    "BC":  {"EMBED_DIM": 128, "N_HEADS": 4, "N_LAYERS": 2, "DROPOUT": 0.2, "WEIGHT_DECAY": 1e-4},
    "DC":  {"EMBED_DIM": 128, "N_HEADS": 4, "N_LAYERS": 2, "DROPOUT": 0.1, "WEIGHT_DECAY": 1e-3},
    "HW":  {"EMBED_DIM": 128, "N_HEADS": 4, "N_LAYERS": 2, "DROPOUT": 0.2, "WEIGHT_DECAY": 1e-3},
}

# ============================================================
# REGIONS + TASKS TO RUN
# ============================================================

REGIONS = [
    # "full_europe_2003",
    # "full_europe_2010",
    # "full_europe_2018",
    # "north_europe_2010"
    # "north_europe_2018",
    # "south_europe_2003"


    "iberia_2003",
    "mediterranean_2003",
    "eastern_europe_2010",
    "scandinavia_2018"
]

TASKS = ["BC", "DC"]   # focus on the two most important tasks

# ============================================================
# COPY ALL CONFIG FROM ORIGINAL SCRIPT 
# ============================================================

CN_FEATURES   = ["BC", "DC", "ID", "OD", "CC"]
ERA5_FEATURES = ["swvl1", "land_mask", "u", "v", "z"]

TASK_CN_FEATURES = {
    "CC":  ["BC", "DC", "ID", "OD"],
    "HW":  ["BC", "DC", "ID", "OD", "CC"],
}

#   "BC":  ["CC", "DC", "ID", "OD"],
#     "DC":  ["BC", "CC", "ID", "OD"],

def get_feature_sets(task):
    cn   = TASK_CN_FEATURES[task]
    era5 = ERA5_FEATURES
    if task in ("CC", "BC", "DC"):
        cn_full = cn + ["is_heatwave"]
    else:
        cn_full = cn
    return cn_full + era5, cn_full, era5

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

TASK_DEFAULTS = {
    "CC": {"target": "CC_target_next_day", "task_type": "regression"},
    "BC": {"target": "BC_target_next_day", "task_type": "regression"},
    "DC": {"target": "DC_target_next_day", "task_type": "regression"},
    "HW": {"target": "is_heatwave",        "task_type": "binary_spatial"},
}

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
    "is_heatwave":        None,
}

NON_NEGATIVE_TARGETS = {
    "CC_target_next_day", "BC_target_next_day", "DC_target_next_day",
}

# ============================================================
# DATASET  (identical to original)
# ============================================================

class SeqDataset(Dataset):
    def __init__(self, X, y, times, seq_len, task_type):
        self.X = X; self.y = y; self.times = times
        self.seq_len = seq_len; self.task_type = task_type
        self.indices = []
        for yr in np.unique(times.year):
            yr_idx = np.where(times.year == yr)[0]
            # y_vals is already rolled by -horizon via np.roll in run_experiment,
            # so we only need the standard seq_len offset here.
            for i in range(len(yr_idx) - seq_len):
                self.indices.append((yr_idx[i], yr_idx[i + seq_len]))

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        start, target = self.indices[idx]
        X_seq = torch.tensor(self.X[start:start + self.seq_len], dtype=torch.float32)
        y_out = torch.tensor(self.y[target][None, :, :], dtype=torch.float32)
        return X_seq, y_out


# ============================================================
# MODEL — SpatioTemporalTransformer
# ============================================================

class PositionalEncoding1D(nn.Module):
    """Standard sinusoidal positional encoding for temporal dimension."""
    def __init__(self, embed_dim, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() *
                        (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)   # (max_len, embed_dim)

    def forward(self, x):
        # x: (B, L, D, H, W) — add temporal PE to D dimension
        B, L, D, H, W = x.shape
        pe = self.pe[:L, :].view(1, L, D, 1, 1)
        return x + pe


class SpatialCNNEncoder(nn.Module):
    """
    Lightweight CNN that maps each timestep's spatial grid to an embedding.
    Preserves H×W — no downsampling — so spatial resolution is maintained.
    input_dim C → embed_dim D via two depthwise-separable conv layers.
    """
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            # Pointwise projection
            nn.Conv2d(input_dim, embed_dim, kernel_size=1),
            nn.GroupNorm(min(8, embed_dim), embed_dim),
            nn.GELU(),
            # Depthwise spatial mixing (3×3, preserves H×W)
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1,
                      groups=embed_dim),
            nn.GroupNorm(min(8, embed_dim), embed_dim),
            nn.GELU(),
            # Pointwise projection out
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )

    def forward(self, x):
        # x: (B*L, C, H, W) → (B*L, D, H, W)
        return self.net(x)


class TemporalTransformerLayer(nn.Module):
    """
    Self-attention across L timesteps, applied independently per spatial location.
    Operates on (B*H*W, L, D) — efficient: no spatial mixing here.
    """
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x):
        # x: (N, L, D) where N = B*H*W
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class SpatialTransformerLayer(nn.Module):
    """
    Self-attention across H×W spatial locations at a single timestep.
    Operates on (B, H*W, D).
    Uses window-based attention for large grids to keep memory tractable.
    """
    def __init__(self, embed_dim, n_heads, dropout, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(embed_dim, n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x, H, W):
        # x: (B, H*W, D)
        B, N, D = x.shape
        ws = self.window_size

        if H * W <= ws * ws * 4:
            # small grid — full spatial attention
            attn_out, _ = self.attn(x, x, x)
        else:
            # window attention: reshape into non-overlapping windows
            # pad H and W to multiples of ws
            H_pad = math.ceil(H / ws) * ws
            W_pad = math.ceil(W / ws) * ws
            x_2d = x.view(B, H, W, D)
            # pad
            pad_h = H_pad - H
            pad_w = W_pad - W
            x_2d = F.pad(x_2d.permute(0,3,1,2),
                         (0, pad_w, 0, pad_h)).permute(0,2,3,1)
            # partition into windows
            x_win = x_2d.view(B, H_pad//ws, ws, W_pad//ws, ws, D)
            x_win = x_win.permute(0,1,3,2,4,5).contiguous()
            x_win = x_win.view(B * (H_pad//ws) * (W_pad//ws), ws*ws, D)
            attn_win, _ = self.attn(x_win, x_win, x_win)
            # unpartition
            attn_win = attn_win.view(B, H_pad//ws, W_pad//ws, ws, ws, D)
            attn_win = attn_win.permute(0,1,3,2,4,5).contiguous()
            attn_win = attn_win.view(B, H_pad, W_pad, D)
            attn_out = attn_win[:, :H, :W, :].contiguous().view(B, H*W, D)

        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class SpatioTemporalTransformer(nn.Module):
    """
    Full spatiotemporal transformer.
    Input : (B, L, C, H, W)
    Output: same as SingleHeadConvLSTM for each task type

    Pipeline:
      1. CNN spatial encoder per timestep  → (B, L, D, H, W)
      2. Temporal positional encoding
      3. n_layers × [temporal attention → spatial attention]
      4. Take final timestep hidden state  → (B, D, H, W)
      5. Task head (identical to ConvLSTM)
    """
    def __init__(self, input_dim, embed_dim, n_heads, n_layers,
                 task_type, dropout=0.1, window_size=8):
        super().__init__()
        self.task_type  = task_type
        self.embed_dim  = embed_dim
        self.n_layers   = n_layers

        self.cnn_encoder = SpatialCNNEncoder(input_dim, embed_dim)
        self.pos_enc     = PositionalEncoding1D(embed_dim, max_len=SEQ_LEN + 4)

        self.temp_layers  = nn.ModuleList([
            TemporalTransformerLayer(embed_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.spat_layers  = nn.ModuleList([
            SpatialTransformerLayer(embed_dim, n_heads, dropout, window_size)
            for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(embed_dim)

        # Task heads — identical to ConvLSTM
        if task_type == "regression":
            self.head = nn.Sequential(
                nn.Dropout2d(p=dropout),
                nn.Conv2d(embed_dim, 1, kernel_size=1)
            )
        elif task_type == "binary_spatial":
            self.head = nn.Sequential(
                nn.Dropout2d(p=dropout),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(embed_dim, 1, kernel_size=1)
            )

    def forward(self, x):
        B, L, C, H, W = x.shape

        # 1. CNN encode each timestep
        x_flat = x.view(B * L, C, H, W)
        z_flat = self.cnn_encoder(x_flat)          # (B*L, D, H, W)
        z = z_flat.view(B, L, self.embed_dim, H, W)

        # 2. Temporal positional encoding
        z = self.pos_enc(z)

        # 3. Interleaved temporal + spatial attention layers
        for temp_layer, spat_layer in zip(self.temp_layers, self.spat_layers):

            # Temporal attention: (B*H*W, L, D)
            z_t = z.permute(0, 3, 4, 1, 2).contiguous()   # (B, H, W, L, D)
            z_t = z_t.view(B * H * W, L, self.embed_dim)
            z_t = temp_layer(z_t)
            z_t = z_t.view(B, H, W, L, self.embed_dim)
            z   = z_t.permute(0, 3, 4, 1, 2).contiguous() # (B, L, D, H, W)

            # Spatial attention on final timestep only (most recent context)
            z_last = z[:, -1, :, :, :]                     # (B, D, H, W)
            z_s    = z_last.permute(0, 2, 3, 1).contiguous().view(B, H*W, self.embed_dim)
            z_s    = spat_layer(z_s, H, W)
            z_s    = z_s.view(B, H, W, self.embed_dim).permute(0, 3, 1, 2)
            # Write back into last timestep slot
            z = torch.cat([z[:, :-1], z_s.unsqueeze(1)], dim=1)

        # 4. Take final timestep
        h = z[:, -1, :, :, :]                              # (B, D, H, W)
        h = self.norm_out(h.permute(0,2,3,1)).permute(0,3,1,2)

        # 5. Task head
        return self.head(h)


# ============================================================
# HELPERS  (identical to original)
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma

    def forward(self, logits, targets):
        probs     = torch.sigmoid(logits).squeeze(1)
        t         = targets.squeeze(1).float()
        p_t       = probs * t + (1 - probs) * (1 - t)
        alpha_t   = self.alpha * t + (1 - self.alpha) * (1 - t)
        bce       = F.binary_cross_entropy_with_logits(
                        logits.squeeze(1), t, reduction="none")
        focal     = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal.mean()


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


def inverse_transform(arr, transform, mean, std):
    if transform == "standard":
        return arr * std + mean
    elif transform == "log1p":
        return np.expm1(np.clip(arr * std + mean, -10, 20))
    return arr


# ============================================================
# BUILD EXPERIMENTS
# ============================================================

EXPERIMENTS = []
for region in REGIONS:
    for task in TASKS:
        full_feats, cn_feats, era5_feats = get_feature_sets(task) 
        EXPERIMENTS.append({
            "TASK": task, "REGION": region,
            "COEFFS": full_feats, "SUFFIX": "cn_era5",
        })

# ============================================================
# SINGLE EXPERIMENT
# ============================================================

def run_experiment(exp, clean_ds, times):

    TASK      = exp["TASK"]
    REGION    = exp["REGION"]
    COEFFS    = exp["COEFFS"]
    suffix    = exp["SUFFIX"]
    horizon   = exp.get("FORECAST_HORIZON", FORECAST_HORIZON)
    horizon_tag = f"t{horizon}"
    run_name  = f"{REGION}_{TASK.lower()}_{horizon_tag}_{suffix}_transformer"
    OUT_DIR   = os.path.join(BASE_OUT, run_name)
    os.makedirs(OUT_DIR, exist_ok=True)

    if os.path.exists(os.path.join(OUT_DIR, "DONE")):
        print(f"[SKIP] {run_name}")
        return run_name, None

    print(f"\n{'='*65}")
    print(f"  START  {run_name}")
    print(f"{'='*65}")

    rc        = REGION_CONFIG[REGION]
    td        = TASK_DEFAULTS[TASK]
    TASK_TYPE = td["task_type"]
    TARGET    = td["target"]
    hp        = TRANSFORMER_HP[TASK]
    EMBED_DIM   = hp["EMBED_DIM"]
    N_HEADS     = hp["N_HEADS"]
    N_LAYERS    = hp["N_LAYERS"]
    DROPOUT     = hp["DROPOUT"]
    WEIGHT_DECAY = hp["WEIGHT_DECAY"]
    SPLIT_TYPE  = rc["split_type"]

    print(f"  Task     : {TASK} ({TASK_TYPE})")
    print(f"  Region   : {REGION}")
    print(f"  Features : {COEFFS}")
    print(f"  HP       : D={EMBED_DIM}  H={N_HEADS}  L={N_LAYERS}  DO={DROPOUT}")

    # ── Slice data ────────────────────────────────────────────────────────
    ds = clean_ds.sel(
        lat=slice(rc["lat_min"], rc["lat_max"]),
        lon=slice(rc["lon_min"], rc["lon_max"])
    )

    X_xr    = xr.concat([ds[var] for var in COEFFS], dim="channel").assign_coords(channel=COEFFS)
    Xt_vals = X_xr.transpose("time", "channel", "lat", "lon").values.astype(np.float32)

    if TASK_TYPE == "regression":
        if TARGET in ds:
            raw_var  = TARGET.replace("_target_next_day", "")
            raw_vals = ds[raw_var].transpose("time", "lat", "lon").values.astype(np.float32)
            y_vals   = np.roll(raw_vals, shift=-horizon, axis=0); y_vals[-horizon:] = 0.0
        else:
            raw_var  = TARGET.replace("_target_next_day", "")
            raw_vals = ds[raw_var].transpose("time", "lat", "lon").values.astype(np.float32)
            y_vals   = np.roll(raw_vals, shift=-horizon, axis=0); y_vals[-horizon:] = 0.0
    elif TASK_TYPE == "binary_spatial":
        y_raw  = ds[TARGET].values.astype(np.float32)
        y_vals = np.roll(y_raw, shift=-horizon, axis=0); y_vals[-horizon:] = 0.0

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
        tr = FEATURE_TRANSFORMS.get(feat, "standard")
        vals = Xt_tr[:, ci].copy()
        if tr == "log1p":
            vals = np.log1p(np.clip(vals, 0, None))
            Xt_tr[:, ci]  = np.log1p(np.clip(Xt_tr[:, ci], 0, None))
            Xt_val[:, ci] = np.log1p(np.clip(Xt_val[:, ci], 0, None))
            Xt_te[:, ci]  = np.log1p(np.clip(Xt_te[:, ci], 0, None))
        m = vals.mean(); s = vals.std() + 1e-8
        channel_mean[0, ci, 0, 0] = m
        channel_std [0, ci, 0, 0] = s
        Xt_tr[:, ci]  = (Xt_tr[:, ci]  - m) / s
        Xt_val[:, ci] = (Xt_val[:, ci] - m) / s
        Xt_te[:, ci]  = (Xt_te[:, ci]  - m) / s

    # ── Target normalisation ──────────────────────────────────────────────
    tgt_transform = TARGET_TRANSFORM.get(TARGET, "standard")
    y_mean, y_std = 0.0, 1.0

    if TASK_TYPE == "regression" and tgt_transform is not None:
        if tgt_transform == "log1p":
            y_tr  = np.log1p(np.clip(y_tr, 0, None))
            y_val = np.log1p(np.clip(y_val, 0, None))
            y_te  = np.log1p(np.clip(y_te, 0, None))
        y_mean = float(y_tr.mean()); y_std = float(y_tr.std()) + 1e-8
        y_tr   = (y_tr  - y_mean) / y_std
        y_val  = (y_val - y_mean) / y_std
        y_te   = (y_te  - y_mean) / y_std

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_ds = SeqDataset(Xt_tr,  y_tr,  tms_tr,  SEQ_LEN, TASK_TYPE)
    val_ds   = SeqDataset(Xt_val, y_val, tms_val, SEQ_LEN, TASK_TYPE)
    test_ds  = SeqDataset(Xt_te,  y_te,  tms_te,  SEQ_LEN, TASK_TYPE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    # Estimate grid size for window attention
    H_est = len(ds.lat.values)
    W_est = len(ds.lon.values)
    ws    = 8 if H_est * W_est > 256 else H_est * W_est

    model = SpatioTemporalTransformer(
        input_dim  = len(COEFFS),
        embed_dim  = EMBED_DIM,
        n_heads    = N_HEADS,
        n_layers   = N_LAYERS,
        task_type  = TASK_TYPE,
        dropout    = DROPOUT,
        window_size = ws,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}  Grid: {H_est}×{W_est}  Window: {ws}")

    # ── Loss ──────────────────────────────────────────────────────────────
    if TASK_TYPE == "regression":
        criterion = nn.MSELoss()
    elif TASK_TYPE == "binary_spatial":
        criterion = FocalLoss(alpha=0.75, gamma=2.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=LR * 0.05
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_epoch    = 0
    history       = {"train_loss": [], "val_loss": []}
    no_improve    = 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        train_losses = []
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_b)
            if TASK_TYPE == "regression":
                loss = criterion(pred, y_b)
            elif TASK_TYPE == "binary_spatial":
                loss = criterion(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                pred = model(X_b)
                if TASK_TYPE == "regression":
                    loss = criterion(pred, y_b)
                elif TASK_TYPE == "binary_spatial":
                    loss = criterion(pred, y_b)
                val_losses.append(loss.item())

        tl = np.mean(train_losses)
        vl = np.mean(val_losses)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)

        if epoch % 10 == 0 or epoch <= 5:
            print(f"  Ep {epoch:3d}  train={tl:.4f}  val={vl:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if vl < best_val_loss:
            best_val_loss = vl
            best_epoch    = epoch
            no_improve    = 0
            torch.save(model.state_dict(),
                       os.path.join(OUT_DIR, "best_model.pt"))
        else:
            no_improve += 1

        if epoch >= ES_MIN_EPOCHS and no_improve >= ES_PATIENCE:
            print(f"  Early stop at epoch {epoch} (best={best_epoch})")
            break

    # ── Load best checkpoint ──────────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model.pt"),
                                     map_location=DEVICE))
    model.eval()
    print(f"  Best epoch: {best_epoch}  val_loss={best_val_loss:.4f}")

    # ── Evaluate ──────────────────────────────────────────────────────────
    def evaluate(loader, split_name):
        all_pred, all_true = [], []
        with torch.no_grad():
            for X_b, y_b in loader:
                X_b = X_b.to(DEVICE)
                pred = model(X_b)
                if TASK_TYPE == "regression":
                    all_pred.append(pred.cpu().numpy())
                    all_true.append(y_b.numpy())
                elif TASK_TYPE == "binary_spatial":
                    all_pred.append(torch.sigmoid(pred).cpu().numpy())
                    all_true.append(y_b.numpy())

        print(f"\n  === {split_name} ===")
        metrics = {}

        if TASK_TYPE == "regression":
            pred_norm = np.concatenate(all_pred)
            true_norm = np.concatenate(all_true)
            mae_norm  = float(np.mean(np.abs(pred_norm - true_norm)))
            r2_norm   = float(1 - np.sum((true_norm - pred_norm)**2) /
                              (np.sum((true_norm - true_norm.mean())**2) + 1e-8))
            pred_arr  = inverse_transform(pred_norm, tgt_transform, y_mean, y_std)
            true_arr  = inverse_transform(true_norm, tgt_transform, y_mean, y_std)
            if TARGET in NON_NEGATIVE_TARGETS:
                pred_arr = np.clip(pred_arr, 0, None)
                true_arr = np.clip(true_arr, 0, None)
            mae     = float(np.mean(np.abs(pred_arr - true_arr)))
            rmse    = float(np.sqrt(np.mean((pred_arr - true_arr)**2)))
            r2      = float(1 - np.sum((true_arr - pred_arr)**2) /
                            (np.sum((true_arr - true_arr.mean())**2) + 1e-8))
            pearson = float(np.corrcoef(pred_arr.flatten(), true_arr.flatten())[0, 1])
            print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  Pearson={pearson:.4f}")
            metrics = dict(mae_norm=mae_norm, r2_norm=r2_norm,
                           mae=mae, rmse=rmse, r2=r2, pearson=pearson)

            # Save arrays for trajectory analysis
            np.save(os.path.join(OUT_DIR, f"pred_{split_name}.npy"), pred_arr)
            np.save(os.path.join(OUT_DIR, f"true_{split_name}.npy"), true_arr)

        elif TASK_TYPE == "binary_spatial":
            pred_arr = np.concatenate(all_pred).flatten()
            true_arr = np.concatenate(all_true).flatten().astype(int)
            bin_arr  = (pred_arr >= 0.5).astype(int)
            f1     = float(f1_score(true_arr, bin_arr, zero_division=0))
            prec   = float(precision_score(true_arr, bin_arr, zero_division=0))
            rec    = float(recall_score(true_arr, bin_arr, zero_division=0))
            iou    = float(jaccard_score(true_arr, bin_arr, zero_division=0))
            pr_auc = float(average_precision_score(true_arr, pred_arr))
            roc    = float(roc_auc_score(true_arr, pred_arr)
                           if true_arr.sum() > 0 and (1-true_arr).sum() > 0
                           else float("nan"))
            best_f1, best_th = 0.0, 0.5
            for th in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
                p_th  = (pred_arr >= th).astype(int)
                f1_th = float(f1_score(true_arr, p_th, zero_division=0))
                if f1_th > best_f1:
                    best_f1, best_th = f1_th, th
            print(f"  ROC={roc:.4f}  PR-AUC={pr_auc:.4f}  "
                  f"best F1={best_f1:.4f} @ th={best_th}")
            metrics = dict(f1=f1, prec=prec, rec=rec, iou=iou,
                           pr_auc=pr_auc, roc=roc,
                           best_thresh=best_th, best_f1=best_f1)

            np.save(os.path.join(OUT_DIR, f"pred_{split_name}.npy"), pred_arr)
            np.save(os.path.join(OUT_DIR, f"true_{split_name}.npy"), true_arr)

        return metrics

    # Also save times and normalisation for downstream use
    np.save(os.path.join(OUT_DIR, "times_TEST.npy"), tms_te)
    np.save(os.path.join(OUT_DIR, "times_VAL.npy"),  tms_val)
    np.save(os.path.join(OUT_DIR, "y_mean.npy"), np.array(y_mean))
    np.save(os.path.join(OUT_DIR, "y_std.npy"),  np.array(y_std))

    val_metrics  = evaluate(val_loader,  "VAL")
    test_metrics = evaluate(test_loader, "TEST")

    # ── Save metrics.txt ──────────────────────────────────────────────────
    with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"Task           : {TASK}  ({TASK_TYPE})\n")
        f.write(f"Forecast horizon: t+{horizon} days\n")
        f.write(f"Model          : SpatioTemporalTransformer\n")
        f.write(f"Ablation       : {suffix}\n")
        f.write(f"Features       : {COEFFS}\n")
        f.write(f"Target         : {TARGET}\n")
        f.write(f"Region         : {REGION}\n")
        f.write(f"Train  : {rc['train_start']} → {rc['train_end']}\n")
        f.write(f"Val    : {rc['val_start']} → {rc['val_end']}\n")
        f.write(f"Test   : {rc['test_start']} → {rc['test_end']}\n")
        f.write(f"Best epoch     : {best_epoch}  "
                f"(ES patience={ES_PATIENCE}, min epochs={ES_MIN_EPOCHS})\n")
        f.write(f"EMBED_DIM={EMBED_DIM}  N_HEADS={N_HEADS}  "
                f"N_LAYERS={N_LAYERS}  DROPOUT={DROPOUT}\n")
        f.write(f"Target transform : {tgt_transform}\n")
        f.write(f"y_mean={y_mean:.6f}  y_std={y_std:.6f}\n\n")
        fmt_split(f, "VAL",  val_metrics)
        fmt_split(f, "TEST", test_metrics)

    # ── Training curve ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"],   label="val")
    ax.axvline(best_epoch - 1, color="red", linestyle="--", alpha=0.5,
               label=f"best (ep {best_epoch})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"{run_name} — training curve")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_curve.png"), dpi=120)
    plt.close()

    open(os.path.join(OUT_DIR, "DONE"), "w").close()
    print(f"\n  DONE  →  {OUT_DIR}")
    return run_name, test_metrics


# ============================================================
# MAIN
# ============================================================

print(f"Device           : {DEVICE}")
print(f"Total experiments: {len(EXPERIMENTS)}")
print(f"Output base      : {BASE_OUT}\n")

print("Loading dataset...")
clean_ds = xr.open_dataset(DATA_FILE)
times    = pd.DatetimeIndex(clean_ds.time.values)
print(f"Dataset loaded: {dict(clean_ds.dims)}\n")

summary = []
for i, exp in enumerate(EXPERIMENTS):
    h     = exp.get("FORECAST_HORIZON", FORECAST_HORIZON)
    label = f"{exp['REGION']}_{exp['TASK'].lower()}_t{h}_{exp['SUFFIX']}_transformer"
    print(f"[{i+1}/{len(EXPERIMENTS)}]  {label}")
    try:
        run_name, test_metrics = run_experiment(exp, clean_ds, times)
        summary.append({"run": run_name, "status": "OK", "metrics": test_metrics})
    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()
        summary.append({"run": label, "status": f"ERROR: {e}", "metrics": None})

print(f"\n\n{'='*75}")
print(f"  TRANSFORMER BATCH COMPLETE — {len(EXPERIMENTS)} experiments")
print(f"{'='*75}")
for s in summary:
    m = s["metrics"]
    if m is None:
        print(f"  {s['run']:65s}  {s['status']}")
    elif m and "r2" in m:
        print(f"  {s['run']:65s}  R²={m['r2']:.3f}  P={m['pearson']:.3f}")
    elif m and "roc" in m:
        print(f"  {s['run']:65s}  ROC={m['roc']:.3f}  F1={m['best_f1']:.3f}")