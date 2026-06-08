import os
os.environ["OMP_NUM_THREADS"] = "8"
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================

DATA_FILE  = "/gpfs/home2/mzdych/thesis/full_processed_training_dataset.nc"
BASE_OUT   = "/gpfs/home2/mzdych/thesis/experiments"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN    = 14
BATCH_SIZE = 4

RUNS = [
    "south_europe_2003_dc_cn_only",
    "south_europe_2003_dc_cn_era5",
    "south_europe_2003_dc_era5_only",
    "north_europe_2010_dc_cn_only",
    "north_europe_2010_dc_cn_era5",
    "north_europe_2010_dc_era5_only",
    "north_europe_2018_dc_cn_only",
    "north_europe_2018_dc_cn_era5",
    "north_europe_2018_dc_era5_only",
    "full_europe_2003_dc_cn_only",
    "full_europe_2003_dc_cn_era5",
    "full_europe_2003_dc_era5_only",
    "full_europe_2010_dc_cn_only",
    "full_europe_2010_dc_cn_era5",
    "full_europe_2010_dc_era5_only",
    "full_europe_2018_dc_cn_only",
    "full_europe_2018_dc_cn_era5",
    "full_europe_2018_dc_era5_only",
    "eastern_europe_2010_dc_cn_only",
    "eastern_europe_2010_dc_cn_era5",
    "eastern_europe_2010_dc_era5_only",
    "iberia_2003_dc_cn_only",
    "iberia_2003_dc_cn_era5",
    "iberia_2003_dc_era5_only",
    "mediterranean_2003_dc_cn_only",
    "mediterranean_2003_dc_cn_era5",
    "mediterranean_2003_dc_era5_only",
    "scandinavia_2018_dc_cn_only",
    "scandinavia_2018_dc_cn_era5",
    "scandinavia_2018_dc_era5_only"
    

]

# ============================================================
# STATIC LOOKUPS
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
    "is_heatwave":        None,
}

NON_NEGATIVE_TARGETS = {
    "CC_target_next_day", "BC_target_next_day", "DC_target_next_day",
    "OD_target_next_day", "ID_target_next_day",
}

REGION_CONFIG = {
    "scandinavia_2018": {
        "lat_min": 55, "lat_max": 65, "lon_min": 5,  "lon_max": 30,
        "split_type": "event",
        "train_start": "1990-06-01", "train_end": "2017-08-31",
        "val_start":   "2018-06-01", "val_end":   "2018-07-14",
        "test_start":  "2018-07-15", "test_end":  "2018-08-31",
    },
    "iberia_2003": {
        "lat_min": 36, "lat_max": 44, "lon_min": -10, "lon_max": 5,
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
    "mediterranean_2003": {
        "lat_min": 30, "lat_max": 48, "lon_min": -10, "lon_max": 40,
        "split_type": "event",
        "train_start": "1990-06-01", "train_end": "2002-08-31",
        "val_start":   "2003-06-01", "val_end":   "2003-07-14",
        "test_start":  "2003-07-15", "test_end":  "2003-08-31",
    },
}

TASK_DEFAULTS = {
    "CC": {"target": "CC_target_next_day", "task_type": "regression"},
    "BC": {"target": "BC_target_next_day", "task_type": "regression"},
    "DC": {"target": "DC_target_next_day", "task_type": "regression"},
    "OD": {"target": "OD_target_next_day", "task_type": "regression"},
    "ID": {"target": "ID_target_next_day", "task_type": "regression"},
    "HW": {"target": "is_heatwave",        "task_type": "binary_spatial"},
}

# ============================================================
# MODEL
# ============================================================

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim,
            kernel_size=kernel_size, padding=padding,
        )

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
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        if task_type == "regression":
            self.head = nn.Sequential(
                nn.Dropout2d(p=dropout),
                nn.Conv2d(hidden_dim, 1, kernel_size=1),
            )
        elif task_type == "binary_spatial":
            self.head = nn.Sequential(
                nn.Dropout2d(p=dropout),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, 1, kernel_size=1),
            )

    def forward(self, x):
        B, L, C, H, W = x.shape
        h = torch.zeros(B, self.cell.hidden_dim, H, W, device=x.device)
        c = torch.zeros(B, self.cell.hidden_dim, H, W, device=x.device)
        for t in range(L):
            h, c = self.cell(x[:, t], h, c)
        return self.head(h)


# ============================================================
# DATASET
# ============================================================

class SeqDataset(Dataset):
    def __init__(self, X, y, times, seq_len, task_type):
        self.X = X
        self.y = y
        self.times = times
        self.seq_len = seq_len
        self.task_type = task_type
        self.indices = []
        for yr in np.unique(times.year):
            yr_idx = np.where(times.year == yr)[0]
            for i in range(len(yr_idx) - seq_len):
                self.indices.append((yr_idx[i], yr_idx[i + seq_len]))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start, target = self.indices[idx]
        X_seq = torch.tensor(self.X[start:start + self.seq_len], dtype=torch.float32)
        y_out = torch.tensor(self.y[target][None, :, :], dtype=torch.float32)
        return X_seq, y_out


# ============================================================
# HELPERS
# ============================================================

def make_mask(times, start, end, split_type):
    date_mask = (times >= np.datetime64(start)) & (times <= np.datetime64(end))
    if split_type == "year":
        return date_mask & times.month.isin([6, 7, 8])
    return date_mask


def parse_run_name(run_name):
    tokens   = run_name.split("_")
    ablation = tokens[-2] + "_" + tokens[-1]
    task     = tokens[-3].upper()
    region   = "_".join(tokens[:-3])
    return region, task, ablation


# ============================================================
# MAIN
# ============================================================

print(f"Device : {DEVICE}")
print(f"Runs   : {len(RUNS)}")
print("Loading dataset (once)...")
clean_ds = xr.open_dataset(DATA_FILE)
times    = pd.DatetimeIndex(clean_ds.time.values)
print(f"Dataset: {dict(clean_ds.dims)}\n")

for run_name in RUNS:
    run_dir = os.path.join(BASE_OUT, run_name)

    if os.path.exists(os.path.join(run_dir, "pred_TEST.npy")):
        print(f"[SKIP]    {run_name}")
        continue
    if not os.path.exists(os.path.join(run_dir, "best_model.pt")):
        print(f"[MISSING] {run_name} — no best_model.pt")
        continue

    print(f"\n{'='*65}")
    print(f"  {run_name}")
    print(f"{'='*65}")

    region, task, ablation = parse_run_name(run_name)
    rc        = REGION_CONFIG[region]
    td        = TASK_DEFAULTS[task]
    TASK_TYPE = td["task_type"]
    TARGET    = td["target"]
    print(f"  Region={region}  Task={task}  Ablation={ablation}")

    # ── Load config from checkpoint (weights_only=False required for PyTorch 2.6+)
    ckpt = torch.load(
        os.path.join(run_dir, "final_checkpoint.pt"),
        map_location="cpu",
        weights_only=False,   # ← fix for PyTorch 2.6 pickle error
    )
    COEFFS     = ckpt["coeffs"]
    HIDDEN_DIM = ckpt["hidden_dim"]
    DROPOUT    = ckpt["dropout"]
    y_mean     = float(ckpt.get("y_mean", 0.0))
    y_std      = float(ckpt.get("y_std",  1.0))
    print(f"  Coeffs    : {COEFFS}")
    print(f"  Hidden dim: {HIDDEN_DIM}  Dropout: {DROPOUT}")
    print(f"  y_mean={y_mean:.6f}  y_std={y_std:.6f}")

    # ── Slice dataset ─────────────────────────────────────────
    ds = clean_ds.sel(
        lat=slice(rc["lat_min"], rc["lat_max"]),
        lon=slice(rc["lon_min"], rc["lon_max"]),
    )

    X_xr    = xr.concat([ds[v] for v in COEFFS], dim="channel").assign_coords(channel=COEFFS)
    Xt_vals = X_xr.transpose("time", "channel", "lat", "lon").values.astype(np.float32)

    if TASK_TYPE == "regression":
        if TARGET in ds:
            y_vals = ds[TARGET].transpose("time", "lat", "lon").values.astype(np.float32)
        else:
            raw_var  = TARGET.replace("_target_next_day", "")
            raw_vals = ds[raw_var].transpose("time", "lat", "lon").values.astype(np.float32)
            y_vals   = np.roll(raw_vals, shift=-1, axis=0)
            y_vals[-1] = 0.0
    else:
        y_raw  = ds[TARGET].values.astype(np.float32)
        y_vals = np.roll(y_raw, shift=-1, axis=0)
        y_vals[-1] = 0.0

    # ── Split ─────────────────────────────────────────────────
    val_mask  = make_mask(times, rc["val_start"],  rc["val_end"],  rc["split_type"])
    test_mask = make_mask(times, rc["test_start"], rc["test_end"], rc["split_type"])

    Xt_val, Xt_te   = Xt_vals[val_mask], Xt_vals[test_mask]
    y_val,  y_te    = y_vals[val_mask],  y_vals[test_mask]
    tms_val, tms_te = times[val_mask],   times[test_mask]

    print(f"  Val  : {tms_val[0].date()} → {tms_val[-1].date()}  ({len(tms_val)} days)")
    print(f"  Test : {tms_te[0].date()} → {tms_te[-1].date()}  ({len(tms_te)} days)")

    # ── Normalise using SAVED stats ───────────────────────────
    ch_mean = np.load(os.path.join(run_dir, "channel_mean.npy"))
    ch_std  = np.load(os.path.join(run_dir, "channel_std.npy"))

    for ci, feat in enumerate(COEFFS):
        if FEATURE_TRANSFORMS.get(feat) == "log1p":
            Xt_val[:, ci] = np.log1p(Xt_val[:, ci])
            Xt_te[:, ci]  = np.log1p(Xt_te[:, ci])
        mu  = ch_mean[0, ci, 0, 0]
        sig = ch_std[0, ci, 0, 0]
        Xt_val[:, ci] = (Xt_val[:, ci] - mu) / sig
        Xt_te[:, ci]  = (Xt_te[:, ci]  - mu) / sig

    if TASK_TYPE == "regression":
        if TARGET_TRANSFORM.get(TARGET) == "log1p":
            y_val = np.log1p(y_val)
            y_te  = np.log1p(y_te)
        y_val = (y_val - y_mean) / y_std
        y_te  = (y_te  - y_mean) / y_std

    # ── Datasets + loaders ────────────────────────────────────
    val_ds  = SeqDataset(Xt_val, y_val, tms_val, SEQ_LEN, TASK_TYPE)
    test_ds = SeqDataset(Xt_te,  y_te,  tms_te,  SEQ_LEN, TASK_TYPE)
    val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)
    print(f"  Sequences — val:{len(val_ds)}  test:{len(test_ds)}")

    # ── Load model (weights_only=False required for PyTorch 2.6+) ──
    model = SingleHeadConvLSTM(
        input_dim=len(COEFFS),
        hidden_dim=HIDDEN_DIM,
        task_type=TASK_TYPE,
        dropout=DROPOUT,
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(
            os.path.join(run_dir, "best_model.pt"),
            map_location=DEVICE,
            weights_only=False,   # ← fix for PyTorch 2.6 pickle error
        )
    )
    model.eval()
    print(f"  Model loaded — {sum(p.numel() for p in model.parameters()):,} params")

    # ── Inference + save ──────────────────────────────────────
    def run_inference(loader, split_name, tms_split):
        all_pred, all_true = [], []
        with torch.no_grad():
            for X, y in loader:
                X   = X.to(DEVICE, non_blocking=True)
                out = model(X)
                if TASK_TYPE == "binary_spatial":
                    out = torch.sigmoid(out)
                all_pred.append(out.cpu().numpy())
                all_true.append(y.numpy())

        pred_arr = np.concatenate(all_pred)   # (N, 1, H, W)
        true_arr = np.concatenate(all_true)   # (N, 1, H, W)
        time_arr = np.array([tms_split[t] for _, t in loader.dataset.indices])

        np.save(os.path.join(run_dir, f"pred_{split_name}.npy"), pred_arr)
        np.save(os.path.join(run_dir, f"true_{split_name}.npy"), true_arr)
        np.save(os.path.join(run_dir, f"times_{split_name}.npy"), time_arr)

        print(f"  Saved {split_name}: pred={pred_arr.shape}  "
              f"true={true_arr.shape}  times={time_arr.shape}")

    run_inference(val_loader,  "VAL",  tms_val)
    run_inference(test_loader, "TEST", tms_te)
    print(f"  DONE → {run_dir}")

print(f"\n{'='*65}")
print("All inference runs complete.")
print("Next step: run trajectory_analysis.py")
print(f"{'='*65}")
