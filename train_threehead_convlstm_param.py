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
    confusion_matrix, roc_auc_score, jaccard_score
)

# ============================
# CONFIG
# ============================

DATA_FILE = "/gpfs/home2/mzdych/thesis/full_processed_training_dataset.nc"
OUT_DIR   = "/gpfs/home2/mzdych/thesis/three_head_output_north"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Region bounding box ───────────────────────────────────────────────────
# Change these four values to select a sub-region of Europe.
# Examples:
#   Mediterranean : lon_min=-10, lon_max=40, lat_min=30, lat_max=48
#   Iberia        : lon_min=-10, lon_max=5,  lat_min=36, lat_max=44
#   North Europe  : lon_min=-25, lon_max=45, lat_min=55, lat_max=71
#   Full Europe   : lon_min=-25, lon_max=45, lat_min=35, lat_max=71  (set to None to skip slicing)
LON_MIN = -25
LON_MAX =  45
LAT_MIN =  55
LAT_MAX =  71

# ── Year splits ───────────────────────────────────────────────────────────
# Adjust freely per experiment / region.
TRAIN_YEARS = list(range(1990, 2016))   # 1990–2015
VAL_YEARS   = list(range(2016, 2018))   # 2016–2017
TEST_YEARS  = list(range(2018, 2020))   # 2018–2019

# ── Input features ────────────────────────────────────────────────────────
# COEFFS = ["BC", "DC", "ID", "OD", "is_heatwave", "CC"]   # CN only
# COEFFS = ["swvl1", "land_mask", "u", "v", "z"]            # ERA5 only
COEFFS = ["BC", "DC", "ID", "OD",
           "swvl1", "land_mask", "u", "v", "z"]             # CN + ERA5

CC_TARGET = "CC_target_next_day"
HW_TARGET = "is_heatwave"

SEQ_LEN    = 14
BATCH_SIZE = 4
HIDDEN_DIM = 32
N_EPOCHS   = 50
LR         = 1e-3

# Loss weights
W_CC    = 1.0
W_CLASS = 5.0
W_HW    = 2.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
print("Torch CUDA available:", torch.cuda.is_available())
print(f"Region  : lat=[{LAT_MIN}, {LAT_MAX}]  lon=[{LON_MIN}, {LON_MAX}]")
print(f"Train years : {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}")
print(f"Val years   : {VAL_YEARS[0]}–{VAL_YEARS[-1]}")
print(f"Test years  : {TEST_YEARS[0]}–{TEST_YEARS[-1]}")

# ============================
# FOCAL LOSS
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

print("Loading dataset...")
clean_ds = xr.open_dataset(DATA_FILE)

# Spatial sub-region slice
clean_ds = clean_ds.sel(
    lat=slice(LAT_MIN, LAT_MAX),
    lon=slice(LON_MIN, LON_MAX)
)
print(f"Dataset after spatial slice: {clean_ds}")

X_xr = xr.concat(
    [clean_ds[var] for var in COEFFS], dim="channel"
).assign_coords(channel=COEFFS)

y_cc_xr = clean_ds[CC_TARGET]

y_hw_raw  = clean_ds[HW_TARGET].values.astype(np.float32)
y_hw_vals = np.roll(y_hw_raw, shift=-1, axis=0)
y_hw_vals[-1] = 0.0

labels = clean_ds["event_label"].values.astype(np.int8)
times  = pd.DatetimeIndex(clean_ds.time.values)

print("Converting to NumPy...")
Xt_vals  = X_xr.transpose("time", "channel", "lat", "lon").values.astype(np.float32)
ycc_vals = y_cc_xr.transpose("time", "lat", "lon").values.astype(np.float32)

print("Xt_vals :", Xt_vals.shape)
print("ycc_vals:", ycc_vals.shape)
print("y_hw_vals:", y_hw_vals.shape)

# ============================
# TRAIN / VAL / TEST SPLIT
# ============================

train_mask = times.year.isin(TRAIN_YEARS)
val_mask   = times.year.isin(VAL_YEARS)
test_mask  = times.year.isin(TEST_YEARS)

Xt_tr,   Xt_val,   Xt_te   = Xt_vals[train_mask],   Xt_vals[val_mask],   Xt_vals[test_mask]
ycc_tr,  ycc_val,  ycc_te  = ycc_vals[train_mask],   ycc_vals[val_mask],  ycc_vals[test_mask]
yhw_tr,  yhw_val,  yhw_te  = y_hw_vals[train_mask],  y_hw_vals[val_mask], y_hw_vals[test_mask]
lbl_tr,  lbl_val,  lbl_te  = labels[train_mask],     labels[val_mask],    labels[test_mask]
tms_tr,  tms_val,  tms_te  = times[train_mask],      times[val_mask],     times[test_mask]

print(f"Train : {Xt_tr.shape}  {tms_tr[0].date()} → {tms_tr[-1].date()}")
print(f"Val   : {Xt_val.shape}  {tms_val[0].date()} → {tms_val[-1].date()}")
print(f"Test  : {Xt_te.shape}  {tms_te[0].date()} → {tms_te[-1].date()}")

# ============================
# NORMALIZATION (fit on train)
# ============================

print("Normalizing...")
channel_mean = Xt_tr.mean(axis=(0, 2, 3), keepdims=True)
channel_std  = Xt_tr.std(axis=(0, 2, 3),  keepdims=True)
channel_std[channel_std == 0] = 1.0

Xt_tr  = (Xt_tr  - channel_mean) / channel_std
Xt_val = (Xt_val - channel_mean) / channel_std
Xt_te  = (Xt_te  - channel_mean) / channel_std

np.save(os.path.join(OUT_DIR, "channel_mean.npy"), channel_mean)
np.save(os.path.join(OUT_DIR, "channel_std.npy"),  channel_std)
print("Normalization done.")

# ============================
# CLASS BALANCE INFO
# ============================

valid_train_labels = lbl_tr[lbl_tr >= 0]
n_standing    = int((valid_train_labels == 0).sum())
n_propagating = int((valid_train_labels == 1).sum())
print(f"Train HW days — standing: {n_standing}  propagating: {n_propagating}")
print(f"Imbalance ratio: {n_standing/n_propagating:.1f}:1")

hw_pos_frac   = yhw_tr.mean()
hw_pos_weight = torch.tensor(
    (1.0 - hw_pos_frac) / (hw_pos_frac + 1e-6),
    dtype=torch.float32, device=DEVICE
)
print(f"HW pixel positive fraction: {hw_pos_frac:.3f}  BCE pos_weight: {hw_pos_weight:.2f}")

# ============================
# DATASET
# ============================

class SeqDataset(Dataset):
    def __init__(self, X, y_cc, y_hw, labels, times, seq_len=14):
        self.X       = X
        self.y_cc    = y_cc
        self.y_hw    = y_hw
        self.labels  = labels
        self.times   = times
        self.seq_len = seq_len
        self.indices = []

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
        X_seq = self.X[start : start + self.seq_len]
        y_cc  = self.y_cc[target][None, :, :]
        y_hw  = self.y_hw[target][None, :, :]
        label = int(self.labels[target])
        return (
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_cc,  dtype=torch.float32),
            torch.tensor(y_hw,  dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )

print("Building datasets...")
train_ds = SeqDataset(Xt_tr,  ycc_tr,  yhw_tr,  lbl_tr, tms_tr,  seq_len=SEQ_LEN)
val_ds   = SeqDataset(Xt_val, ycc_val, yhw_val, lbl_val, tms_val, seq_len=SEQ_LEN)
test_ds  = SeqDataset(Xt_te,  ycc_te,  yhw_te,  lbl_te,  tms_te,  seq_len=SEQ_LEN)

# WeightedRandomSampler (train only)
seq_labels     = np.array([int(lbl_tr[t]) for _, t in train_ds.indices])
prop_mask_seq  = seq_labels == 1
stand_mask_seq = seq_labels == 0
no_ev_mask_seq = seq_labels == -1

sample_weights = np.ones(len(seq_labels))
if prop_mask_seq.sum() > 0 and stand_mask_seq.sum() > 0:
    w_prop = stand_mask_seq.sum() / prop_mask_seq.sum()
    sample_weights[prop_mask_seq]  = w_prop
    sample_weights[stand_mask_seq] = 1.0
    sample_weights[no_ev_mask_seq] = 0.3
    print(f"Sampler — prop weight: {w_prop:.2f}  stand: 1.0  no-event: 0.3")

sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.float32),
    num_samples=len(train_ds), replacement=True
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          sampler=sampler, shuffle=False,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)

# ============================
# MODEL
# ============================

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


class TripleHeadConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, kernel_size=3, n_classes=2):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)

        self.cc_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

        self.hw_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.hw_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

        self.class_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        B, L, C, H, W = x.shape
        h = torch.zeros(B, self.cell.hidden_dim, H, W, device=x.device)
        c = torch.zeros(B, self.cell.hidden_dim, H, W, device=x.device)
        for t in range(L):
            h, c = self.cell(x[:, t], h, c)

        cc_pred      = torch.sigmoid(self.cc_head(h))
        hw_pred      = self.hw_head(F.relu(self.hw_proj(h)))
        class_logits = self.class_head(F.relu(self.class_proj(h)))
        return cc_pred, hw_pred, class_logits


model = TripleHeadConvLSTM(
    input_dim=len(COEFFS), hidden_dim=HIDDEN_DIM,
    kernel_size=3, n_classes=2
).to(DEVICE)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================
# LOSS
# ============================

focal_loss = FocalLoss(alpha=0.75, gamma=2.0).to(DEVICE)
bce_loss   = nn.BCEWithLogitsLoss(pos_weight=hw_pos_weight)

def compute_loss(cc_pred, hw_pred, class_logits, y_cc, y_hw, y_class):
    losses = {}
    losses["cc"]  = nn.MSELoss()(cc_pred, y_cc)
    losses["hw"]  = bce_loss(hw_pred, y_hw)
    valid_mask = y_class >= 0
    if valid_mask.sum() > 0:
        losses["class"] = focal_loss(
            class_logits[valid_mask], y_class[valid_mask].long()
        )
    else:
        losses["class"] = torch.tensor(0.0, device=DEVICE)
    total = W_CC * losses["cc"] + W_HW * losses["hw"] + W_CLASS * losses["class"]
    return total, losses

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5
)

# ============================
# TRAINING
# ============================

history = {k: [] for k in
           ["train_total", "val_total",
            "train_cc",    "val_cc",
            "train_hw",    "val_hw",
            "train_class", "val_class"]}

best_val_loss = np.inf
print("\nStarting training...")

for epoch in range(N_EPOCHS):
    model.train()
    tr = {"total": 0.0, "cc": 0.0, "hw": 0.0, "class": 0.0}

    for X, y_cc, y_hw, y_class in train_loader:
        X       = X.to(DEVICE,       non_blocking=True)
        y_cc    = y_cc.to(DEVICE,    non_blocking=True)
        y_hw    = y_hw.to(DEVICE,    non_blocking=True)
        y_class = y_class.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        cc_pred, hw_pred, class_logits = model(X)
        loss, losses = compute_loss(cc_pred, hw_pred, class_logits,
                                    y_cc, y_hw, y_class)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        tr["total"] += loss.item()
        tr["cc"]    += losses["cc"].item()
        tr["hw"]    += losses["hw"].item()
        tr["class"] += losses["class"].item()

    for k in tr:
        tr[k] /= len(train_loader)

    model.eval()
    vl = {"total": 0.0, "cc": 0.0, "hw": 0.0, "class": 0.0}

    with torch.no_grad():
        for X, y_cc, y_hw, y_class in val_loader:
            X       = X.to(DEVICE,       non_blocking=True)
            y_cc    = y_cc.to(DEVICE,    non_blocking=True)
            y_hw    = y_hw.to(DEVICE,    non_blocking=True)
            y_class = y_class.to(DEVICE, non_blocking=True)

            cc_pred, hw_pred, class_logits = model(X)
            loss, losses = compute_loss(cc_pred, hw_pred, class_logits,
                                        y_cc, y_hw, y_class)
            vl["total"] += loss.item()
            vl["cc"]    += losses["cc"].item()
            vl["hw"]    += losses["hw"].item()
            vl["class"] += losses["class"].item()

    for k in vl:
        vl[k] /= len(val_loader)

    scheduler.step(vl["total"])

    for k in ["total", "cc", "hw", "class"]:
        history[f"train_{k}"].append(tr[k])
        history[f"val_{k}"].append(vl[k])

    if vl["total"] < best_val_loss:
        best_val_loss = vl["total"]
        torch.save(model.state_dict(),
                   os.path.join(OUT_DIR, "best_model.pt"))

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch+1:03d}/{N_EPOCHS} | "
        f"train={tr['total']:.4f} "
        f"(cc={tr['cc']:.4f} hw={tr['hw']:.4f} cls={tr['class']:.4f}) | "
        f"val={vl['total']:.4f} "
        f"(cc={vl['cc']:.4f} hw={vl['hw']:.4f} cls={vl['class']:.4f}) | "
        f"lr={current_lr:.2e}",
        flush=True
    )

print(f"\nTraining finished. Best val loss: {best_val_loss:.4f}")

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, key, title in zip(axes,
                           ["total", "cc", "hw", "class"],
                           ["Total", "CC (MSE)", "HW (BCE)", "Class (Focal)"]):
    ax.plot(history[f"train_{key}"], label="train")
    ax.plot(history[f"val_{key}"],   label="val")
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150)
plt.close()

# ============================
# EVALUATION HELPER
# ============================

def evaluate(loader, split_name):
    model.load_state_dict(
        torch.load(os.path.join(OUT_DIR, "best_model.pt"), map_location=DEVICE)
    )
    model.eval()

    all_cc_pred, all_cc_true = [], []
    all_hw_pred, all_hw_true = [], []
    all_class_true, all_class_pred, all_class_prob = [], [], []

    with torch.no_grad():
        for X, y_cc, y_hw, y_class in loader:
            X = X.to(DEVICE, non_blocking=True)
            cc_pred, hw_pred, class_logits = model(X)

            all_cc_pred.append(cc_pred.cpu().numpy())
            all_cc_true.append(y_cc.numpy())

            hw_prob = torch.sigmoid(hw_pred).cpu().numpy()
            all_hw_pred.append(hw_prob)
            all_hw_true.append(y_hw.numpy())

            probs = torch.softmax(class_logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            y_np  = y_class.numpy()
            valid = y_np >= 0

            all_class_true.extend(y_np[valid].tolist())
            all_class_pred.extend(preds[valid].tolist())
            all_class_prob.extend(probs[valid, 1].tolist())

    # CC
    cc_pred_arr = np.concatenate(all_cc_pred)
    cc_true_arr = np.concatenate(all_cc_true)
    mae  = np.mean(np.abs(cc_pred_arr - cc_true_arr))
    rmse = np.sqrt(np.mean((cc_pred_arr - cc_true_arr) ** 2))
    r2   = 1 - np.sum((cc_true_arr - cc_pred_arr)**2) / \
               (np.sum((cc_true_arr - cc_true_arr.mean())**2) + 1e-8)

    # HW
    hw_pred_arr = np.concatenate(all_hw_pred).flatten()
    hw_true_arr = np.concatenate(all_hw_true).flatten().astype(int)
    hw_bin_arr  = (hw_pred_arr >= 0.5).astype(int)
    hw_f1   = f1_score(hw_true_arr, hw_bin_arr,  zero_division=0)
    hw_prec = precision_score(hw_true_arr, hw_bin_arr, zero_division=0)
    hw_rec  = recall_score(hw_true_arr, hw_bin_arr,  zero_division=0)
    hw_iou  = jaccard_score(hw_true_arr, hw_bin_arr,  zero_division=0)
    hw_roc  = (roc_auc_score(hw_true_arr, hw_pred_arr)
               if hw_true_arr.sum() > 0 and (1 - hw_true_arr).sum() > 0
               else float("nan"))

    # Prop/standing
    y_true     = np.array(all_class_true)
    y_pred     = np.array(all_class_pred)
    probs_prop = np.array(all_class_prob)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred,    zero_division=0)
    f1   = f1_score(y_true, y_pred,        zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    # Threshold tuning
    best_f1, best_thresh = 0.0, 0.5
    print(f"\n=== [{split_name}] Threshold tuning (prop/standing) ===")
    for th in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        p_th  = (probs_prop >= th).astype(int)
        f1_th = f1_score(y_true, p_th, zero_division=0)
        re_th = recall_score(y_true, p_th, zero_division=0)
        pr_th = precision_score(y_true, p_th, zero_division=0)
        print(f"  th={th:.2f}  F1={f1_th:.3f}  Prec={pr_th:.3f}  Rec={re_th:.3f}")
        if f1_th > best_f1:
            best_f1, best_thresh = f1_th, th

    print(f"\n=== [{split_name}] CC regression ===")
    print(f"  MAE:  {mae:.4f}  RMSE: {rmse:.4f}  R²: {r2:.4f}")
    print(f"\n=== [{split_name}] is_heatwave spatial ===")
    print(f"  F1: {hw_f1:.3f}  Precision: {hw_prec:.3f}  Recall: {hw_rec:.3f}")
    print(f"  IoU: {hw_iou:.3f}  ROC-AUC: {hw_roc:.3f}")
    print(f"\n=== [{split_name}] Standing vs Propagating ===")
    print(f"  Accuracy: {acc:.3f}  Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}")
    print(f"  Confusion matrix:\n{cm}")
    print(f"  Best threshold: {best_thresh}  (F1={best_f1:.3f})")

    return dict(
        mae=mae, rmse=rmse, r2=r2,
        hw_f1=hw_f1, hw_prec=hw_prec, hw_rec=hw_rec,
        hw_iou=hw_iou, hw_roc=hw_roc,
        acc=acc, prec=prec, rec=rec, f1=f1,
        cm=cm, best_thresh=best_thresh, best_f1=best_f1
    )

# ============================
# RUN EVALUATION
# ============================

val_metrics  = evaluate(val_loader,  "VAL")
test_metrics = evaluate(test_loader, "TEST")

# ============================
# SAVE METRICS
# ============================

def write_metrics(f, split, m):
    f.write(f"=== {split} — CC regression ===\n")
    f.write(f"MAE: {m['mae']}  RMSE: {m['rmse']}  R2: {m['r2']}\n\n")
    f.write(f"=== {split} — is_heatwave spatial ===\n")
    f.write(f"F1: {m['hw_f1']}  Precision: {m['hw_prec']}  Recall: {m['hw_rec']}\n")
    f.write(f"IoU: {m['hw_iou']}  ROC-AUC: {m['hw_roc']}\n\n")
    f.write(f"=== {split} — prop/standing classification ===\n")
    f.write(f"Accuracy: {m['acc']}  Precision: {m['prec']}  Recall: {m['rec']}  F1: {m['f1']}\n")
    f.write(f"Confusion matrix:\n{m['cm']}\n")
    f.write(f"Best threshold: {m['best_thresh']}  F1={m['best_f1']}\n\n")

with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
    f.write(f"Region : lat=[{LAT_MIN}, {LAT_MAX}]  lon=[{LON_MIN}, {LON_MAX}]\n")
    f.write(f"Train  : {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}\n")
    f.write(f"Val    : {VAL_YEARS[0]}–{VAL_YEARS[-1]}\n")
    f.write(f"Test   : {TEST_YEARS[0]}–{TEST_YEARS[-1]}\n\n")
    write_metrics(f, "VAL",  val_metrics)
    write_metrics(f, "TEST", test_metrics)

torch.save({
    "model_state_dict": model.state_dict(),
    "coeffs":           COEFFS,
    "cc_target":        CC_TARGET,
    "hw_target":        HW_TARGET,
    "seq_len":          SEQ_LEN,
    "hidden_dim":       HIDDEN_DIM,
    "channel_mean":     channel_mean,
    "channel_std":      channel_std,
    "history":          history,
    "best_val_loss":    best_val_loss,
    "region":           {"lat_min": LAT_MIN, "lat_max": LAT_MAX,
                         "lon_min": LON_MIN, "lon_max": LON_MAX},
    "train_years":      TRAIN_YEARS,
    "val_years":        VAL_YEARS,
    "test_years":       TEST_YEARS,
}, os.path.join(OUT_DIR, "final_checkpoint.pt"))

print(f"\nAll outputs saved to: {OUT_DIR}")
