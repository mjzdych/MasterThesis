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

# ============================
# CONFIG
# ============================

DATA_FILE = "/gpfs/home2/mzdych/thesis/full_processed_training_dataset.nc"

# ── Task switch ───────────────────────────────────────────────────────────
# Set TASK to one of: "CC" | "BC" | "DC" | "OD" | "ID" | "ND" | "HW" | "CLASS"
TASK = "CC"

if TASK == "CC":
    COEFFS    = ["BC", "DC", "ID", "OD", "is_heatwave",
                 "swvl1", "land_mask", "u", "v", "z"]
    TARGET    = "CC_target_next_day"
    TASK_TYPE = "regression"

elif TASK == "BC":
    COEFFS    = ["CC", "DC", "ID", "OD", "is_heatwave",
                 "swvl1", "land_mask", "u", "v", "z"]
    TARGET    = "BC_target_next_day"
    TASK_TYPE = "regression"

elif TASK == "OD":
    COEFFS    = ["BC", "CC", "DC", "ID", "is_heatwave",
                 "swvl1", "land_mask", "u", "v", "z"]
    TARGET    = "OD_target_next_day"
    TASK_TYPE = "regression"

elif TASK == "ID":
    COEFFS    = ["BC", "CC", "DC", "OD", "is_heatwave",
                 "swvl1", "land_mask", "u", "v", "z"]
    TARGET    = "ID_target_next_day"
    TASK_TYPE = "regression"

elif TASK == "DC":
    COEFFS    = ["BC", "CC", "ID", "OD", "is_heatwave",
                 "swvl1", "land_mask", "u", "v", "z"]
    TARGET    = "DC_target_next_day"
    TASK_TYPE = "regression"

elif TASK == "ND":
    # Network Divergence = OD - ID (built on the fly)
    # Include both OD and ID as features so model sees directed structure
    COEFFS    = ["BC", "CC", "DC", "ID", "OD", "is_heatwave",
                 "swvl1", "land_mask", "u", "v", "z"]
    TARGET    = "ND_target_next_day"
    TASK_TYPE = "regression"

elif TASK == "HW":
    COEFFS    = ["BC", "DC", "ID", "OD", "CC",
                 "swvl1", "land_mask", "u", "v", "z"]
    TARGET    = "is_heatwave"
    TASK_TYPE = "binary_spatial"

elif TASK == "CLASS":
    COEFFS    = ["BC", "DC", "ID", "OD",
                 "swvl1", "land_mask", "u", "v", "z"]
    TARGET    = "event_label"
    TASK_TYPE = "binary_scalar"

else:
    raise ValueError(f"Unknown TASK: {TASK}.")

OUT_DIR = f"/gpfs/home2/mzdych/thesis/single_head_{TASK.lower()}_eu_scandi_2018_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================
# SPATIAL CONFIG
# ============================
#
# For the small-region baseline:
#   set CONTEXT = TARGET bounds (same region for input and output)
#
# For Option A (full Europe input → small region output):
#   set CONTEXT to full Europe, TARGET to the small region
#
# ── Option A: Eastern Europe 2010 ────────────────────────────────────────
CONTEXT_LAT_MIN, CONTEXT_LAT_MAX =  35,  71   # full Europe input
CONTEXT_LON_MIN, CONTEXT_LON_MAX = -25,  45

# TARGET_LAT_MIN,  TARGET_LAT_MAX  =  45,  60   # Eastern Europe output
# TARGET_LON_MIN,  TARGET_LON_MAX  =  20,  50

# TRAIN_START = "1990-06-01";  TRAIN_END = "2009-08-31"
# VAL_START   = "2010-06-01";  VAL_END   = "2010-07-14"
# TEST_START  = "2010-07-15";  TEST_END  = "2010-08-31"

# ── Baseline (small region only — disable Option A) ───────────────────────
# Set CONTEXT = TARGET to run without the large context field
# CONTEXT_LAT_MIN, CONTEXT_LAT_MAX = 45, 60
# CONTEXT_LON_MIN, CONTEXT_LON_MAX = 20, 50
# TARGET_LAT_MIN,  TARGET_LAT_MAX  = 45, 60
# TARGET_LON_MIN,  TARGET_LON_MAX  = 20, 50

# ── Option A: Iberia 2003 ─────────────────────────────────────────────────
# CONTEXT_LAT_MIN, CONTEXT_LAT_MAX =  35,  71
# CONTEXT_LON_MIN, CONTEXT_LON_MAX = -25,  45
# TARGET_LAT_MIN,  TARGET_LAT_MAX  =  36,  44
# TARGET_LON_MIN,  TARGET_LON_MAX  = -10,   5
# TRAIN_START = "1990-06-01";  TRAIN_END = "2002-08-31"
# VAL_START   = "2003-06-01";  VAL_END   = "2003-07-26"
# TEST_START  = "2003-07-27";  TEST_END  = "2003-08-31"

# ── Option A: Scandinavia 2018 ────────────────────────────────────────────
# CONTEXT_LAT_MIN, CONTEXT_LAT_MAX =  35,  71
# CONTEXT_LON_MIN, CONTEXT_LON_MAX = -25,  45
TARGET_LAT_MIN,  TARGET_LAT_MAX  =  55,  65
TARGET_LON_MIN,  TARGET_LON_MAX  =   5,  30
TRAIN_START = "1990-06-01";  TRAIN_END = "2017-08-31"
VAL_START   = "2018-06-01";  VAL_END   = "2018-07-14"
TEST_START  = "2018-07-15";  TEST_END  = "2018-08-31"

# ── Training hyperparameters ──────────────────────────────────────────────
SEQ_LEN       = 14
BATCH_SIZE    = 4
HIDDEN_DIM    = 64
N_EPOCHS      = 100
LR            = 5e-4
WEIGHT_DECAY  = 1e-4
ES_PATIENCE   = 7
ES_MIN_EPOCHS = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
print(f"Task           : {TASK}  ({TASK_TYPE})")
print(f"Features       : {COEFFS}")
print(f"Target         : {TARGET}")
print(f"Context region : lat=[{CONTEXT_LAT_MIN},{CONTEXT_LAT_MAX}]  "
      f"lon=[{CONTEXT_LON_MIN},{CONTEXT_LON_MAX}]")
print(f"Target region  : lat=[{TARGET_LAT_MIN},{TARGET_LAT_MAX}]  "
      f"lon=[{TARGET_LON_MIN},{TARGET_LON_MAX}]")

# ============================
# TARGETED PREPROCESSING CONFIG
# ============================

FEATURE_TRANSFORMS = {
    "BC":          "log1p",
    "DC":          "standard",
    "CC":          "standard",
    "ID":          "log1p",
    "OD":          "log1p",
    "is_heatwave": "standard",
    "swvl1":       "standard",
    "land_mask":   "standard",
    "u":           "standard",
    "v":           "standard",
    "z":           "standard",
}

# ND uses standard (no log1p) because it can be negative (OD - ID)
# non-negative clip is also disabled for ND in inverse_transform_target
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

# Variables whose predictions must be >= 0 after inverse transform
# ND is NOT in this set — it can be negative
NON_NEGATIVE_TARGETS = {
    "CC_target_next_day",
    "BC_target_next_day",
    "DC_target_next_day",
    "OD_target_next_day",
    "ID_target_next_day",
}

# ============================
# FOCAL LOSS (CLASS task)
# ============================

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

# ============================
# LOAD & SLICE DATA
# ============================

print("\nLoading dataset...")
clean_ds = xr.open_dataset(DATA_FILE)

# Input: context region (full Europe for Option A, same as target for baseline)
ctx_ds = clean_ds.sel(
    lat=slice(CONTEXT_LAT_MIN, CONTEXT_LAT_MAX),
    lon=slice(CONTEXT_LON_MIN, CONTEXT_LON_MAX)
)

# Target labels: small prediction region
tgt_ds = clean_ds.sel(
    lat=slice(TARGET_LAT_MIN, TARGET_LAT_MAX),
    lon=slice(TARGET_LON_MIN, TARGET_LON_MAX)
)

print(f"Context dataset: {dict(ctx_ds.dims)}")
print(f"Target  dataset: {dict(tgt_ds.dims)}")

# ── Compute crop indices (context grid → target subgrid) ─────────────────
lats_ctx = ctx_ds.lat.values
lons_ctx = ctx_ds.lon.values

lat_idx = np.where(
    (lats_ctx >= TARGET_LAT_MIN) & (lats_ctx <= TARGET_LAT_MAX))[0]
lon_idx = np.where(
    (lons_ctx >= TARGET_LON_MIN) & (lons_ctx <= TARGET_LON_MAX))[0]

if len(lat_idx) == 0 or len(lon_idx) == 0:
    raise ValueError(
        "Target region not found within context grid. "
        "Check that TARGET bounds are inside CONTEXT bounds."
    )

LAT_SLICE = slice(lat_idx[0], lat_idx[-1] + 1)
LON_SLICE = slice(lon_idx[0], lon_idx[-1] + 1)
print(f"Crop indices — lat: {LAT_SLICE}  lon: {LON_SLICE}")

# when CONTEXT == TARGET the crop is the full array (no-op)
is_option_a = not (
    CONTEXT_LAT_MIN == TARGET_LAT_MIN and
    CONTEXT_LAT_MAX == TARGET_LAT_MAX and
    CONTEXT_LON_MIN == TARGET_LON_MIN and
    CONTEXT_LON_MAX == TARGET_LON_MAX
)
print(f"Option A (large context): {is_option_a}")

# ── Build feature array on context region ────────────────────────────────
X_xr = xr.concat(
    [ctx_ds[var] for var in COEFFS], dim="channel"
).assign_coords(channel=COEFFS)

times = pd.DatetimeIndex(clean_ds.time.values)

# ── Build target array on small region ───────────────────────────────────
if TASK_TYPE == "regression":
    if TARGET == "ND_target_next_day":
        # ND = OD - ID, shifted +1 day
        od_raw = tgt_ds["OD"].transpose(
            "time", "lat", "lon").values.astype(np.float32)
        id_raw = tgt_ds["ID"].transpose(
            "time", "lat", "lon").values.astype(np.float32)
        nd_raw          = od_raw - id_raw
        y_vals          = np.roll(nd_raw, shift=-1, axis=0)
        y_vals[-1]      = 0.0
        print(f"Built ND_target_next_day = OD - ID (shifted 1 day).")
        print(f"  ND range: [{nd_raw.min():.2f}, {nd_raw.max():.2f}]  "
              f"(can be negative)")
    elif TARGET in tgt_ds:
        y_vals = tgt_ds[TARGET].transpose(
            "time", "lat", "lon").values.astype(np.float32)
    else:
        raw_var = TARGET.replace("_target_next_day", "")
        if raw_var not in tgt_ds:
            raise KeyError(
                f"Cannot build target '{TARGET}': raw variable "
                f"'{raw_var}' not found.\n"
                f"Available: {list(tgt_ds.data_vars)}"
            )
        raw_vals        = tgt_ds[raw_var].transpose(
            "time", "lat", "lon").values.astype(np.float32)
        y_vals          = np.roll(raw_vals, shift=-1, axis=0)
        y_vals[-1]      = 0.0
        print(f"Built '{TARGET}' by shifting '{raw_var}' by 1 day.")

elif TASK_TYPE == "binary_spatial":
    y_raw  = tgt_ds[TARGET].values.astype(np.float32)
    y_vals = np.roll(y_raw, shift=-1, axis=0)
    y_vals[-1] = 0.0

elif TASK_TYPE == "binary_scalar":
    y_vals = clean_ds[TARGET].values.astype(np.int8)

print("Converting to NumPy...")
Xt_vals = X_xr.transpose(
    "time", "channel", "lat", "lon").values.astype(np.float32)
print(f"Xt_vals (context) : {Xt_vals.shape}")
print(f"y_vals  (target)  : {y_vals.shape}")

# ============================
# TRAIN / VAL / TEST SPLIT
# ============================

train_mask = ((times >= np.datetime64(TRAIN_START)) &
              (times <= np.datetime64(TRAIN_END)))
val_mask   = ((times >= np.datetime64(VAL_START)) &
              (times <= np.datetime64(VAL_END)))
test_mask  = ((times >= np.datetime64(TEST_START)) &
              (times <= np.datetime64(TEST_END)))

Xt_tr,  Xt_val,  Xt_te  = Xt_vals[train_mask], Xt_vals[val_mask], Xt_vals[test_mask]
y_tr,   y_val,   y_te   = y_vals[train_mask],   y_vals[val_mask],  y_vals[test_mask]
tms_tr, tms_val, tms_te = times[train_mask],    times[val_mask],   times[test_mask]

print(f"Train : {Xt_tr.shape}  {tms_tr[0].date()} → {tms_tr[-1].date()}")
print(f"Val   : {Xt_val.shape}  {tms_val[0].date()} → {tms_val[-1].date()}")
print(f"Test  : {Xt_te.shape}  {tms_te[0].date()} → {tms_te[-1].date()}")

# ============================
# TARGETED FEATURE NORMALIZATION
# ============================

print("\nApplying targeted feature transforms...")

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
    if sig < 1e-8:
        sig = 1.0

    Xt_tr[:, ci]  = (Xt_tr[:, ci]  - mu) / sig
    Xt_val[:, ci] = (Xt_val[:, ci] - mu) / sig
    Xt_te[:, ci]  = (Xt_te[:, ci]  - mu) / sig

    channel_mean[0, ci, 0, 0] = mu
    channel_std[ 0, ci, 0, 0] = sig

    print(f"  {feat:12s}  transform={transform:8s}  "
          f"mu={mu:.6f}  sig={sig:.6f}")

np.save(os.path.join(OUT_DIR, "channel_mean.npy"), channel_mean)
np.save(os.path.join(OUT_DIR, "channel_std.npy"),  channel_std)
print("Feature normalization done.")

# ============================
# TARGET NORMALIZATION
# ============================

y_mean, y_std = 0.0, 1.0

if TASK_TYPE == "regression":
    tgt_transform = TARGET_TRANSFORM.get(TARGET, "standard")
    print(f"\nTarget transform: {tgt_transform}  (target={TARGET})")

    if tgt_transform == "log1p":
        y_tr  = np.log1p(y_tr)
        y_val = np.log1p(y_val)
        y_te  = np.log1p(y_te)

    y_mean = float(y_tr.mean())
    y_std  = float(y_tr.std()) + 1e-8

    y_tr  = (y_tr  - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std
    y_te  = (y_te  - y_mean) / y_std

    np.save(os.path.join(OUT_DIR, "y_mean.npy"), np.array(y_mean))
    np.save(os.path.join(OUT_DIR, "y_std.npy"),  np.array(y_std))
    print(f"Target normalized — mean={y_mean:.6f}  std={y_std:.6f}")
    print(f"  y_tr after norm: min={y_tr.min():.3f}  max={y_tr.max():.3f}")

# ============================
# CLASS BALANCE
# ============================

if TASK_TYPE == "binary_spatial":
    hw_pos_frac   = y_tr.mean()
    hw_pos_weight = torch.tensor(
        (1.0 - hw_pos_frac) / (hw_pos_frac + 1e-6),
        dtype=torch.float32, device=DEVICE
    )
    print(f"HW pixel pos fraction: {hw_pos_frac:.3f}  "
          f"BCE pos_weight: {hw_pos_weight:.2f}")

elif TASK_TYPE == "binary_scalar":
    valid   = y_tr[y_tr >= 0]
    n_stand = int((valid == 0).sum())
    n_prop  = int((valid == 1).sum())
    print(f"Train — standing: {n_stand}  propagating: {n_prop}  "
          f"ratio: {n_stand/n_prop:.1f}:1")

# ============================
# DATASET
# ============================

class SeqDataset(Dataset):
    def __init__(self, X, y, times, seq_len, task_type):
        self.X         = X
        self.y         = y
        self.times     = times
        self.seq_len   = seq_len
        self.task_type = task_type
        self.indices   = []

        for yr in np.unique(times.year):
            yr_idx = np.where(times.year == yr)[0]
            for i in range(len(yr_idx) - seq_len):
                start  = yr_idx[i]
                target = yr_idx[i + seq_len]
                self.indices.append((start, target))

        print(f"  Sequences created: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start, target = self.indices[idx]
        X_seq = torch.tensor(
            self.X[start : start + self.seq_len], dtype=torch.float32)

        if self.task_type in ("regression", "binary_spatial"):
            y_out = torch.tensor(
                self.y[target][None, :, :], dtype=torch.float32)
        elif self.task_type == "binary_scalar":
            y_out = torch.tensor(int(self.y[target]), dtype=torch.long)

        return X_seq, y_out

print("\nBuilding datasets...")
train_ds = SeqDataset(Xt_tr,  y_tr,  tms_tr,  SEQ_LEN, TASK_TYPE)
val_ds   = SeqDataset(Xt_val, y_val, tms_val, SEQ_LEN, TASK_TYPE)
test_ds  = SeqDataset(Xt_te,  y_te,  tms_te,  SEQ_LEN, TASK_TYPE)

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

# ============================
# MODEL
# ============================
# ConvLSTM processes the full context grid.
# Hidden state is cropped to the target subregion before the head.
# When CONTEXT == TARGET, LAT_SLICE/LON_SLICE cover the full array (no-op).

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
                 lat_slice, lon_slice,
                 kernel_size=3, n_classes=2):
        super().__init__()
        self.task_type = task_type
        self.lat_slice = lat_slice
        self.lon_slice = lon_slice
        self.cell      = ConvLSTMCell(input_dim, hidden_dim, kernel_size)

        if task_type == "regression":
            self.head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

        elif task_type == "binary_spatial":
            self.head = nn.Sequential(
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

        # crop to target region (no-op when context == target)
        h_crop = h[:, :, self.lat_slice, self.lon_slice]

        if self.task_type == "regression":
            return self.head(h_crop)        # (B,1,H_tgt,W_tgt)
        elif self.task_type == "binary_spatial":
            return self.head(h_crop)        # raw logits (B,1,H_tgt,W_tgt)
        elif self.task_type == "binary_scalar":
            return self.head(h_crop)        # logits (B, n_classes)


model = SingleHeadConvLSTM(
    input_dim=len(COEFFS),
    hidden_dim=HIDDEN_DIM,
    task_type=TASK_TYPE,
    lat_slice=LAT_SLICE,
    lon_slice=LON_SLICE,
    kernel_size=3,
    n_classes=2
).to(DEVICE)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Option A active: {is_option_a}")

# ============================
# LOSS
# ============================

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

# ============================
# TRAINING + EARLY STOPPING
# ============================

history       = {"train_loss": [], "val_loss": []}
best_val_loss = np.inf
es_counter    = 0
best_epoch    = 0

print("\nStarting training...")

for epoch in range(N_EPOCHS):

    model.train()
    train_loss = 0.0
    for X, y in train_loader:
        X = X.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        pred = model(X)
        loss = compute_loss(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            pred = model(X)
            val_loss += compute_loss(pred, y).item()
    val_loss /= len(val_loader)

    scheduler.step(val_loss)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch    = epoch + 1
        es_counter    = 0
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pt"))
    else:
        es_counter += 1

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch+1:03d}/{N_EPOCHS} | "
        f"train={train_loss:.4f}  val={val_loss:.4f} | "
        f"lr={current_lr:.2e}  es={es_counter}/{ES_PATIENCE}",
        flush=True
    )

    if es_counter >= ES_PATIENCE and epoch + 1 >= ES_MIN_EPOCHS:
        print(f"\nEarly stopping at epoch {epoch+1} "
              f"(best epoch: {best_epoch}, best val: {best_val_loss:.4f})")
        break

print(f"\nTraining done. Best epoch: {best_epoch}  "
      f"Best val loss: {best_val_loss:.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history["train_loss"], label="train")
ax.plot(history["val_loss"],   label="val")
ax.axvline(best_epoch - 1, color="gray", linestyle="--",
           label=f"best epoch {best_epoch}")
ax.set_title(f"{TASK} — loss curve  (Option A: {is_option_a})")
ax.set_xlabel("Epoch")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_curve.png"), dpi=150)
plt.close()

# ============================
# INVERSE TRANSFORM
# ============================

def inverse_transform_target(arr):
    """Undo z-score and optional log1p. Only clip to >= 0 for non-negative targets."""
    out = arr * y_std + y_mean
    if TASK_TYPE == "regression":
        tgt_transform = TARGET_TRANSFORM.get(TARGET, "standard")
        if tgt_transform == "log1p":
            out = np.expm1(out)
        # ND can be negative — do not clip
        if TARGET in NON_NEGATIVE_TARGETS:
            out = np.clip(out, 0.0, None)
    return out

# ============================
# EVALUATION
# ============================

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
                prob = torch.sigmoid(pred).cpu().numpy()
                all_pred.append(prob)
                all_true.append(y.numpy())

            elif TASK_TYPE == "binary_scalar":
                probs = torch.softmax(pred, dim=1).cpu().numpy()
                y_np  = y.numpy()
                valid = y_np >= 0
                all_pred.extend(probs[valid, 1].tolist())
                all_true.extend(y_np[valid].tolist())

    print(f"\n{'='*50}")
    print(f"  [{split_name}]  TASK: {TASK}  Option A: {is_option_a}")
    print(f"{'='*50}")

    metrics = {}

    if TASK_TYPE == "regression":
        pred_norm = np.concatenate(all_pred)
        true_norm = np.concatenate(all_true)

        mae_norm = np.mean(np.abs(pred_norm - true_norm))
        r2_norm  = 1 - np.sum((true_norm - pred_norm)**2) / \
                       (np.sum((true_norm - true_norm.mean())**2) + 1e-8)
        print(f"  [normalized space]")
        print(f"  MAE:  {mae_norm:.4f}")
        print(f"  R²:   {r2_norm:.4f}")

        pred_arr = inverse_transform_target(pred_norm)
        true_arr = inverse_transform_target(true_norm)

        mae     = np.mean(np.abs(pred_arr - true_arr))
        rmse    = np.sqrt(np.mean((pred_arr - true_arr) ** 2))
        r2      = 1 - np.sum((true_arr - pred_arr)**2) / \
                      (np.sum((true_arr - true_arr.mean())**2) + 1e-8)
        pearson = np.corrcoef(pred_arr.flatten(), true_arr.flatten())[0, 1]

        print(f"\n  [original units]")
        print(f"  MAE:     {mae:.4f}")
        print(f"  RMSE:    {rmse:.4f}")
        print(f"  R²:      {r2:.4f}")
        print(f"  Pearson: {pearson:.4f}")
        print(f"  pred range: [{pred_arr.min():.3f}, {pred_arr.max():.3f}]")
        print(f"  true range: [{true_arr.min():.3f}, {true_arr.max():.3f}]")

        metrics = dict(
            mae_norm=mae_norm, r2_norm=r2_norm,
            mae=mae, rmse=rmse, r2=r2, pearson=pearson
        )

    elif TASK_TYPE == "binary_spatial":
        pred_arr = np.concatenate(all_pred).flatten()
        true_arr = np.concatenate(all_true).flatten().astype(int)
        bin_arr  = (pred_arr >= 0.5).astype(int)
        f1     = f1_score(true_arr, bin_arr,  zero_division=0)
        prec   = precision_score(true_arr, bin_arr, zero_division=0)
        rec    = recall_score(true_arr, bin_arr,  zero_division=0)
        iou    = jaccard_score(true_arr, bin_arr,  zero_division=0)
        pr_auc = average_precision_score(true_arr, pred_arr)
        roc    = (roc_auc_score(true_arr, pred_arr)
                  if true_arr.sum() > 0 and (1 - true_arr).sum() > 0
                  else float("nan"))
        print(f"  F1:        {f1:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  IoU:       {iou:.4f}")
        print(f"  PR-AUC:    {pr_auc:.4f}")
        print(f"  ROC-AUC:   {roc:.4f}")
        print(f"\n  Threshold sweep:")
        best_f1, best_th = 0.0, 0.5
        for th in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            p_th  = (pred_arr >= th).astype(int)
            f1_th = f1_score(true_arr, p_th, zero_division=0)
            pr_th = precision_score(true_arr, p_th, zero_division=0)
            re_th = recall_score(true_arr, p_th, zero_division=0)
            print(f"    th={th:.2f}  F1={f1_th:.3f}  Prec={pr_th:.3f}  Rec={re_th:.3f}")
            if f1_th > best_f1:
                best_f1, best_th = f1_th, th
        print(f"  Best threshold: {best_th}  F1={best_f1:.4f}")
        metrics = dict(f1=f1, prec=prec, rec=rec, iou=iou,
                       pr_auc=pr_auc, roc=roc,
                       best_thresh=best_th, best_f1=best_f1)

    elif TASK_TYPE == "binary_scalar":
        probs_arr = np.array(all_pred)
        true_arr  = np.array(all_true)
        pred_arr  = (probs_arr >= 0.5).astype(int)
        acc  = accuracy_score(true_arr, pred_arr)
        prec = precision_score(true_arr, pred_arr, zero_division=0)
        rec  = recall_score(true_arr, pred_arr,    zero_division=0)
        f1   = f1_score(true_arr, pred_arr,        zero_division=0)
        cm   = confusion_matrix(true_arr, pred_arr)
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  Confusion matrix:\n{cm}")
        print(f"\n  Threshold sweep:")
        best_f1, best_th = 0.0, 0.5
        for th in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            p_th  = (probs_arr >= th).astype(int)
            f1_th = f1_score(true_arr, p_th, zero_division=0)
            pr_th = precision_score(true_arr, p_th, zero_division=0)
            re_th = recall_score(true_arr, p_th, zero_division=0)
            print(f"    th={th:.2f}  F1={f1_th:.3f}  Prec={pr_th:.3f}  Rec={re_th:.3f}")
            if f1_th > best_f1:
                best_f1, best_th = f1_th, th
        print(f"  Best threshold: {best_th}  F1={best_f1:.4f}")
        metrics = dict(acc=acc, prec=prec, rec=rec, f1=f1,
                       cm=cm, best_thresh=best_th, best_f1=best_f1)

    return metrics

val_metrics  = evaluate(val_loader,  "VAL")
test_metrics = evaluate(test_loader, "TEST")

# ============================
# SAVE METRICS
# ============================

def fmt_split(f, split, m):
    f.write(f"=== {split} ===\n")
    for k, v in m.items():
        f.write(f"  {k}: {v}\n")
    f.write("\n")

with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
    f.write(f"Task           : {TASK}  ({TASK_TYPE})\n")
    f.write(f"Features       : {COEFFS}\n")
    f.write(f"Target         : {TARGET}\n")
    f.write(f"Option A       : {is_option_a}\n")
    f.write(f"Context region : lat=[{CONTEXT_LAT_MIN},{CONTEXT_LAT_MAX}]  "
            f"lon=[{CONTEXT_LON_MIN},{CONTEXT_LON_MAX}]\n")
    f.write(f"Target region  : lat=[{TARGET_LAT_MIN},{TARGET_LAT_MAX}]  "
            f"lon=[{TARGET_LON_MIN},{TARGET_LON_MAX}]\n")
    f.write(f"Train  : {TRAIN_START} → {TRAIN_END}\n")
    f.write(f"Val    : {VAL_START} → {VAL_END}\n")
    f.write(f"Test   : {TEST_START} → {TEST_END}\n")
    f.write(f"Best epoch     : {best_epoch}  (ES patience={ES_PATIENCE}, "
            f"min epochs={ES_MIN_EPOCHS})\n")
    f.write(f"Target transform : {TARGET_TRANSFORM.get(TARGET, 'standard')}\n")
    f.write(f"y_mean={y_mean:.6f}  y_std={y_std:.6f}\n\n")
    fmt_split(f, "VAL",  val_metrics)
    fmt_split(f, "TEST", test_metrics)

torch.save({
    "model_state_dict":    model.state_dict(),
    "task":                TASK,
    "task_type":           TASK_TYPE,
    "coeffs":              COEFFS,
    "target":              TARGET,
    "seq_len":             SEQ_LEN,
    "hidden_dim":          HIDDEN_DIM,
    "channel_mean":        channel_mean,
    "channel_std":         channel_std,
    "feature_transforms":  FEATURE_TRANSFORMS,
    "target_transform":    TARGET_TRANSFORM.get(TARGET, "standard"),
    "y_mean":              y_mean,
    "y_std":               y_std,
    "history":             history,
    "best_val_loss":       best_val_loss,
    "best_epoch":          best_epoch,
    "is_option_a":         is_option_a,
    "context_region": {
        "lat_min": CONTEXT_LAT_MIN, "lat_max": CONTEXT_LAT_MAX,
        "lon_min": CONTEXT_LON_MIN, "lon_max": CONTEXT_LON_MAX,
    },
    "target_region": {
        "lat_min": TARGET_LAT_MIN, "lat_max": TARGET_LAT_MAX,
        "lon_min": TARGET_LON_MIN, "lon_max": TARGET_LON_MAX,
    },
    "lat_slice": (LAT_SLICE.start, LAT_SLICE.stop),
    "lon_slice": (LON_SLICE.start, LON_SLICE.stop),
}, os.path.join(OUT_DIR, "final_checkpoint.pt"))

print(f"\nAll outputs saved to: {OUT_DIR}")
