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
    confusion_matrix
)

# ============================
# CONFIG
# ============================

DATA_FILE = "/gpfs/home2/mzdych/thesis/full_processed_training_dataset.nc"
OUT_DIR   = "/gpfs/home2/mzdych/thesis/output2"
os.makedirs(OUT_DIR, exist_ok=True)

COEFFS = ["BC", "DC", "ID", "OD", "is_heatwave",
          "swvl1", "land_mask", "u", "v", "z"]
TARGET  = "CC_target_next_day"

SEQ_LEN    = 14
BATCH_SIZE = 4
HIDDEN_DIM = 32
N_EPOCHS   = 50
LR         = 1e-3
W_CC       = 1.0
W_CLASS    = 5.0

VAL_YEARS = [2016, 2017, 2018, 2019, 2020]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
print("Torch CUDA available:", torch.cuda.is_available())

# ============================
# FOCAL LOSS
# ============================

class FocalLoss(nn.Module):
    """
    Focal loss for class imbalance.
    gamma=2 focuses learning on hard/misclassified examples.
    alpha weights the rare (propagating) class higher.
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha   # weight for positive (propagating) class
        self.gamma = gamma

    def forward(self, logits, targets):
        # logits: (N, 2), targets: (N,) long
        probs     = F.softmax(logits, dim=1)
        probs_pos = probs[:, 1]                        # P(propagating)

        # binary targets
        t = targets.float()

        # focal weight
        p_t   = probs_pos * t + (1 - probs_pos) * (1 - t)
        alpha = self.alpha * t + (1 - self.alpha) * (1 - t)
        focal = alpha * (1 - p_t) ** self.gamma

        ce   = F.cross_entropy(logits, targets, reduction="none")
        loss = (focal * ce).mean()
        return loss

# ============================
# LOAD DATA
# ============================

print("Loading dataset...")
clean_ds = xr.open_dataset(DATA_FILE)
print(clean_ds)

X_xr = xr.concat(
    [clean_ds[var] for var in COEFFS], dim="channel"
).assign_coords(channel=COEFFS)

y_xr   = clean_ds[TARGET]
labels = clean_ds["event_label"].values.astype(np.int8)
times  = pd.DatetimeIndex(clean_ds.time.values)

print("Converting to NumPy...")
Xt_vals = X_xr.transpose("time", "channel", "lat", "lon").values.astype(np.float32)
yt_vals = y_xr.transpose("time", "lat", "lon").values.astype(np.float32)
print("Xt_vals:", Xt_vals.shape)
print("yt_vals:", yt_vals.shape)

# ============================
# TRAIN / VAL SPLIT
# ============================

val_mask   = times.year.isin(VAL_YEARS)
train_mask = ~val_mask

Xt_tr,  Xt_val  = Xt_vals[train_mask],  Xt_vals[val_mask]
yt_tr,  yt_val  = yt_vals[train_mask],  yt_vals[val_mask]
lbl_tr, lbl_val = labels[train_mask],   labels[val_mask]
tms_tr, tms_val = times[train_mask],    times[val_mask]

print(f"Train: {Xt_tr.shape}  {tms_tr[0].date()} → {tms_tr[-1].date()}")
print(f"Val:   {Xt_val.shape}  {tms_val[0].date()} → {tms_val[-1].date()}")

# ============================
# NORMALIZATION
# ============================

print("Normalizing...")
channel_mean = Xt_tr.mean(axis=(0, 2, 3), keepdims=True)
channel_std  = Xt_tr.std(axis=(0, 2, 3),  keepdims=True)
channel_std[channel_std == 0] = 1.0

Xt_tr  = (Xt_tr  - channel_mean) / channel_std
Xt_val = (Xt_val - channel_mean) / channel_std

np.save(os.path.join(OUT_DIR, "channel_mean.npy"), channel_mean)
np.save(os.path.join(OUT_DIR, "channel_std.npy"),  channel_std)
print("Normalization done.")

# ============================
# CLASS BALANCE INFO
# ============================

valid_train_labels = lbl_tr[lbl_tr >= 0]
n_standing    = int((valid_train_labels == 0).sum())
n_propagating = int((valid_train_labels == 1).sum())
total_hw      = n_standing + n_propagating

print(f"Train HW days — standing: {n_standing}  propagating: {n_propagating}")
print(f"Imbalance ratio: {n_standing/n_propagating:.1f}:1")

# ============================
# DATASET
# ============================

class SeqDataset(Dataset):
    def __init__(self, X, y, labels, times, seq_len=14):
        self.X       = X
        self.y       = y
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
        y_map = self.y[target][None, :, :]
        label = int(self.labels[target])
        return (
            torch.tensor(X_seq,  dtype=torch.float32),
            torch.tensor(y_map,  dtype=torch.float32),
            torch.tensor(label,  dtype=torch.long),
        )

print("Building datasets...")
train_ds = SeqDataset(Xt_tr,  yt_tr,  lbl_tr, tms_tr,  seq_len=SEQ_LEN)
val_ds   = SeqDataset(Xt_val, yt_val, lbl_val, tms_val, seq_len=SEQ_LEN)

# ── WeightedRandomSampler: oversample propagating sequences ──────────────────
seq_labels = np.array([int(lbl_tr[t]) for _, t in train_ds.indices])

# weight per sequence: propagating gets n_standing/n_propagating times more
sample_weights = np.ones(len(seq_labels))
hw_mask_seq    = seq_labels >= 0
prop_mask_seq  = seq_labels == 1
stand_mask_seq = seq_labels == 0
no_ev_mask_seq = seq_labels == -1

if prop_mask_seq.sum() > 0 and stand_mask_seq.sum() > 0:
    w_prop  = stand_mask_seq.sum() / prop_mask_seq.sum()
    sample_weights[prop_mask_seq]  = w_prop
    sample_weights[stand_mask_seq] = 1.0
    sample_weights[no_ev_mask_seq] = 0.3   # downsample no-event days

print(f"Sampler — prop weight: {w_prop:.2f}  stand weight: 1.0  no-event: 0.3")

sampler = WeightedRandomSampler(
    weights     = torch.tensor(sample_weights, dtype=torch.float32),
    num_samples = len(train_ds),
    replacement = True
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)

# ============================
# MODEL
# ============================

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding         = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim,
            kernel_size=kernel_size, padding=padding
        )

    def forward(self, x, h, c):
        combined   = torch.cat([x, h], dim=1)
        gates      = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i);  f = torch.sigmoid(f)
        o = torch.sigmoid(o);  g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class MultiHeadConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, kernel_size=3, n_classes=2):
        super().__init__()
        self.cell    = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        self.cc_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

        # separate projection before classification head
        # so gradients don't interfere with CC head
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
        class_logits = self.class_head(F.relu(self.class_proj(h)))
        return cc_pred, class_logits


model = MultiHeadConvLSTM(
    input_dim=len(COEFFS), hidden_dim=HIDDEN_DIM,
    kernel_size=3, n_classes=2
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# ============================
# LOSS
# ============================

focal_loss = FocalLoss(alpha=0.75, gamma=2.0).to(DEVICE)

def compute_loss(cc_pred, class_logits, y_cc, y_class,
                 w_cc=W_CC, w_class=W_CLASS):
    losses = {}
    losses["cc"] = nn.MSELoss()(cc_pred, y_cc)

    valid_mask = y_class >= 0
    if valid_mask.sum() > 0:
        losses["class"] = focal_loss(
            class_logits[valid_mask], y_class[valid_mask].long()
        )
    else:
        losses["class"] = torch.tensor(0.0, device=DEVICE)

    total = w_cc * losses["cc"] + w_class * losses["class"]
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
            "train_class", "val_class"]}

best_val_loss = np.inf
print("\nStarting training...")

for epoch in range(N_EPOCHS):

    model.train()
    tr = {"total": 0.0, "cc": 0.0, "class": 0.0}

    for X, y_cc, y_class in train_loader:
        X       = X.to(DEVICE,       non_blocking=True)
        y_cc    = y_cc.to(DEVICE,    non_blocking=True)
        y_class = y_class.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        cc_pred, class_logits = model(X)
        loss, losses = compute_loss(cc_pred, class_logits, y_cc, y_class)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        tr["total"] += loss.item()
        tr["cc"]    += losses["cc"].item()
        tr["class"] += losses["class"].item()

    for k in tr:
        tr[k] /= len(train_loader)

    model.eval()
    vl = {"total": 0.0, "cc": 0.0, "class": 0.0}

    with torch.no_grad():
        for X, y_cc, y_class in val_loader:
            X       = X.to(DEVICE,       non_blocking=True)
            y_cc    = y_cc.to(DEVICE,    non_blocking=True)
            y_class = y_class.to(DEVICE, non_blocking=True)

            cc_pred, class_logits = model(X)
            loss, losses = compute_loss(cc_pred, class_logits, y_cc, y_class)

            vl["total"] += loss.item()
            vl["cc"]    += losses["cc"].item()
            vl["class"] += losses["class"].item()

    for k in vl:
        vl[k] /= len(val_loader)

    scheduler.step(vl["total"])

    for k in ["total", "cc", "class"]:
        history[f"train_{k}"].append(tr[k])
        history[f"val_{k}"].append(vl[k])

    if vl["total"] < best_val_loss:
        best_val_loss = vl["total"]
        torch.save(model.state_dict(),
                   os.path.join(OUT_DIR, "best_model.pt"))

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch+1:03d}/{N_EPOCHS} | "
        f"train={tr['total']:.4f} (cc={tr['cc']:.4f} cls={tr['class']:.4f}) | "
        f"val={vl['total']:.4f} (cc={vl['cc']:.4f} cls={vl['class']:.4f}) | "
        f"lr={current_lr:.2e}",
        flush=True
    )

print(f"\nTraining finished. Best val loss: {best_val_loss:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, key, title in zip(axes,
                           ["total", "cc", "class"],
                           ["Total loss", "CC loss (MSE)", "Class loss (Focal)"]):
    ax.plot(history[f"train_{key}"], label="train")
    ax.plot(history[f"val_{key}"],   label="val")
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150)
plt.close()

# ============================
# EVALUATION
# ============================

model.load_state_dict(
    torch.load(os.path.join(OUT_DIR, "best_model.pt"), map_location=DEVICE)
)
model.eval()

all_cc_pred, all_cc_true       = [], []
all_class_true, all_class_pred = [], []
all_class_prob                 = []

with torch.no_grad():
    for X, y_cc, y_class in val_loader:
        X = X.to(DEVICE, non_blocking=True)
        cc_pred, class_logits = model(X)

        all_cc_pred.append(cc_pred.cpu().numpy())
        all_cc_true.append(y_cc.numpy())

        probs = torch.softmax(class_logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        y_np  = y_class.numpy()
        valid = y_np >= 0

        all_class_true.extend(y_np[valid].tolist())
        all_class_pred.extend(preds[valid].tolist())
        all_class_prob.extend(probs[valid, 1].tolist())

cc_pred_arr = np.concatenate(all_cc_pred)
cc_true_arr = np.concatenate(all_cc_true)

mae  = np.mean(np.abs(cc_pred_arr - cc_true_arr))
rmse = np.sqrt(np.mean((cc_pred_arr - cc_true_arr) ** 2))
r2   = 1 - np.sum((cc_true_arr - cc_pred_arr)**2) / \
           (np.sum((cc_true_arr - cc_true_arr.mean())**2) + 1e-8)

y_true     = np.array(all_class_true)
y_pred     = np.array(all_class_pred)
probs_prop = np.array(all_class_prob)

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec  = recall_score(y_true, y_pred,    zero_division=0)
f1   = f1_score(y_true, y_pred,        zero_division=0)
cm   = confusion_matrix(y_true, y_pred)

print("\n=== CC regression ===")
print(f"  MAE:  {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²:   {r2:.4f}")

print("\n=== Standing vs Propagating classification ===")
print(f"  Accuracy:  {acc:.3f}")
print(f"  Precision: {prec:.3f}")
print(f"  Recall:    {rec:.3f}")
print(f"  F1:        {f1:.3f}")
print(f"  Confusion matrix:\n{cm}")

print("\n=== Threshold tuning ===")
best_f1, best_thresh = 0.0, 0.5
for th in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    preds_th = (probs_prop >= th).astype(int)
    f1_th    = f1_score(y_true, preds_th, zero_division=0)
    rec_th   = recall_score(y_true, preds_th, zero_division=0)
    prec_th  = precision_score(y_true, preds_th, zero_division=0)
    print(f"  th={th:.2f}  F1={f1_th:.3f}  Prec={prec_th:.3f}  Rec={rec_th:.3f}")
    if f1_th > best_f1:
        best_f1, best_thresh = f1_th, th

print(f"\n  Best threshold: {best_thresh}  (F1={best_f1:.3f})")

print(f"\nProp prob stats — mean: {probs_prop.mean():.3f}  "
      f"min: {probs_prop.min():.3f}  max: {probs_prop.max():.3f}")

with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
    f.write("=== CC regression ===\n")
    f.write(f"MAE:  {mae}\nRMSE: {rmse}\nR2:   {r2}\n\n")
    f.write("=== Classification ===\n")
    f.write(f"Accuracy:  {acc}\nPrecision: {prec}\n"
            f"Recall:    {rec}\nF1:        {f1}\n")
    f.write(f"Confusion matrix:\n{cm}\n")
    f.write(f"\nBest threshold: {best_thresh}  F1={best_f1}\n")

torch.save({
    "model_state_dict": model.state_dict(),
    "coeffs":           COEFFS,
    "target":           TARGET,
    "seq_len":          SEQ_LEN,
    "hidden_dim":       HIDDEN_DIM,
    "channel_mean":     channel_mean,
    "channel_std":      channel_std,
    "history":          history,
    "best_val_loss":    best_val_loss,
}, os.path.join(OUT_DIR, "final_checkpoint.pt"))

print(f"\nAll outputs saved to: {OUT_DIR}")
