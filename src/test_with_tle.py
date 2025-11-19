"""
test_with_tle.py — Unified Propagation + TDNN Correction Pipeline
------------------------------------------------------------------
Integrates:
✅ SP3/TLE propagation (from prop_verification.py)
✅ Orbit component plots (X, Y, Z) with epoch markers
✅ TDNN model for SGP4 orbit error correction

Workflow:
1. Load SP3 truth data (ECEF → TEME)
2. Propagate SGP4 orbits using TLE segments
3. Plot X/Y/Z coordinate comparisons and 3D error over time
4. Compute propagation errors (SGP4–SP3)
5. Train a TDNN to model and correct these errors
6. Compare SGP4 vs TDNN-corrected performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
from astropy.coordinates import ITRS, TEME
from astropy.time import Time
import astropy.units as u
import glob, os

# ==============================================================
# 1. TDNN Correction Model
# ==============================================================
class TDNNCorrector(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(TDNNCorrector, self).__init__()
        self.tdnn1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.tdnn2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu1(self.tdnn1(x))
        x = self.relu2(self.tdnn2(x))
        x = x.permute(0, 2, 1)
        return self.fc(x)[:, -1, :]


# ==============================================================
# 2. Orbit Propagation and Error Computation
# ==============================================================
def load_sp3(filepath):
    """Read JCET/ILRS-style SP3 file and extract position (X,Y,Z) in km."""
    times, xs, ys, zs = [], [], [], []
    current_time = None
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("*"):
                parts = line.split()
                year, month, day, hour, minute = map(int, parts[1:6])
                sec = float(parts[6])
                current_time = datetime(year, month, day, hour, minute, int(sec))
            elif line.startswith("P") and current_time is not None:
                vals = line.split()
                try:
                    x, y, z = map(float, vals[-3:])
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                    times.append(current_time)
                except Exception:
                    continue
    return pd.DataFrame({"time_sp3": times, "x_truth": xs, "y_truth": ys, "z_truth": zs})


def ecef_to_teme(sp3_df):
    """Convert SP3 coordinates from ECEF/ITRF to TEME frame."""
    times = Time(sp3_df["time_sp3"].values, scale="utc")
    itrs = ITRS(x=sp3_df["x_truth"].values * u.km,
                y=sp3_df["y_truth"].values * u.km,
                z=sp3_df["z_truth"].values * u.km,
                obstime=times)
    teme = itrs.transform_to(TEME(obstime=times))
    sp3_df["x_truth"] = teme.x.to(u.km).value
    sp3_df["y_truth"] = teme.y.to(u.km).value
    sp3_df["z_truth"] = teme.z.to(u.km).value
    return sp3_df


def propagate_tle(filepath, time_list):
    """Propagate a sequence of TLE segments across their epoch windows."""
    lines = open(filepath).read().strip().splitlines()
    tle_pairs = [(lines[i], lines[i + 1]) for i in range(0, len(lines), 2)]

    tle_epochs = []
    for line1, _ in tle_pairs:
        year = int(line1[18:20])
        year += 2000 if year < 57 else 1900
        day_of_year = float(line1[20:32])
        tle_epochs.append(datetime(year, 1, 1) + timedelta(days=day_of_year - 1))

    df_list = []
    for i, (line1, line2) in enumerate(tle_pairs):
        sat = Satrec.twoline2rv(line1, line2)
        start_time = tle_epochs[i]
        end_time = tle_epochs[i + 1] if i + 1 < len(tle_epochs) else time_list[-1]
        valid_times = [t for t in time_list if start_time <= t < end_time]
        if not valid_times:
            continue
        pos = []
        for t in valid_times:
            jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
            e, r, v = sat.sgp4(jd, fr)
            pos.append(r if e == 0 else [np.nan, np.nan, np.nan])
        df_temp = pd.DataFrame(pos, columns=["x_sgp4", "y_sgp4", "z_sgp4"])
        df_temp["time"] = valid_times
        df_list.append(df_temp)

    df = pd.concat(df_list, ignore_index=True)
    df = df.drop_duplicates(subset=["time"]).sort_values(by="time").reset_index(drop=True)
    return df, tle_epochs


def compute_errors(sp3_df, sgp4_df):
    """Compute SGP4 vs SP3 positional errors."""
    merged = pd.merge(sp3_df, sgp4_df, left_on="time_sp3", right_on="time", how="inner")
    merged["dx"] = merged["x_truth"] - merged["x_sgp4"]
    merged["dy"] = merged["y_truth"] - merged["y_sgp4"]
    merged["dz"] = merged["z_truth"] - merged["z_sgp4"]
    merged["error_norm_km"] = np.sqrt(merged["dx"]**2 + merged["dy"]**2 + merged["dz"]**2)
    return merged


# ==============================================================
# 3. Load SP3/TLE Data, Propagate, and Plot
# ==============================================================
sp3_files = sorted(glob.glob("../Data/Lag1_*.sp3"))
tle_files = sorted(glob.glob("../Data/Lag1TLE_*.txt"))

print("\n=== Checking for SP3 and TLE files ===")
print("SP3 files found:", sp3_files)
print("TLE files found:", tle_files)

if not sp3_files or not tle_files:
    raise RuntimeError("❌ No matching SP3/TLE files found in ../Data/")

datasets = []
for sp3_file, tle_file in zip(sp3_files, tle_files):
    print(f"\n→ Processing {os.path.basename(sp3_file)} with {os.path.basename(tle_file)}")
    sp3_df = ecef_to_teme(load_sp3(sp3_file))
    sgp4_df, tle_epochs = propagate_tle(tle_file, list(sp3_df["time_sp3"]))
    merged = compute_errors(sp3_df, sgp4_df)
    print(f"Merged samples: {len(merged)} (SP3: {len(sp3_df)}, SGP4: {len(sgp4_df)})")
    print(merged[['dx','dy','dz']].describe())

    datasets.append(merged)

    # --- X/Y/Z Component Plots ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    coords = ["x", "y", "z"]
    for i, ax in enumerate(axes):
        c = coords[i]
        ax.plot(sp3_df["time_sp3"], sp3_df[f"{c}_truth"], label=f"SP3 {c.upper()}", color="blue")
        ax.plot(sgp4_df["time"], sgp4_df[f"{c}_sgp4"], label=f"SGP4 {c.upper()}", color="orange", linestyle="--")
        for epoch in tle_epochs:
            if sp3_df["time_sp3"].iloc[0] <= epoch <= sp3_df["time_sp3"].iloc[-1]:
                ax.axvline(epoch, color="gray", linestyle=":", alpha=0.7)
        ax.set_ylabel(f"{c.upper()} (km)")
        ax.legend()
        ax.grid(True)
    axes[-1].set_xlabel("Time (UTC)")
    plt.suptitle(f"SP3 vs SGP4 Orbit Components — {os.path.basename(sp3_file)}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=False)
    plt.pause(2)

    # --- 3D Error Plot ---
    plt.figure(figsize=(10, 4))
    plt.plot(merged["time_sp3"], merged["error_norm_km"], color="red")
    for epoch in tle_epochs:
        if merged["time_sp3"].iloc[0] <= epoch <= merged["time_sp3"].iloc[-1]:
            plt.axvline(epoch, color="gray", linestyle=":", alpha=0.7)
    plt.title(f"SGP4–SP3 3D Error — {os.path.basename(sp3_file)}")
    plt.xlabel("Time (UTC)")
    plt.ylabel("3D Error (km)")
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)

data = pd.concat(datasets, ignore_index=True)
print(f"✅ Combined all datasets — Total samples: {len(data)}")


# ==============================================================
# 4. TDNN Training & Evaluation
# ==============================================================
sequence_length = 10
input_dim = 3
output_dim = 3

X, y = [], []
for i in range(len(data) - sequence_length):
    seq = data[['x_sgp4', 'y_sgp4', 'z_sgp4']].iloc[i:i+sequence_length].values
    label = data[['dx', 'dy', 'dz']].iloc[i+sequence_length].values
    X.append(seq)
    y.append(label)

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32)

train_split, val_split = 0.7, 0.15
train_end = int(train_split * len(X))
val_end = int((train_split + val_split) * len(X))

train_loader = DataLoader(TensorDataset(X[:train_end], y[:train_end]), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X[train_end:val_end], y[train_end:val_end]), batch_size=32)
test_X, test_y = X[val_end:], y[val_end:]

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        train_loss = sum(criterion(model(xb), yb).item() for xb, yb in train_loader) / len(train_loader)
        model.eval()
        val_loss = sum(criterion(model(xb), yb).item() for xb, yb in val_loader) / len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}")
    return model, train_losses, val_losses

model = TDNNCorrector(input_dim, output_dim)
trained_model, train_losses, val_losses = train_model(model, train_loader, val_loader)

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('TDNN Training Loss')
plt.legend()
plt.grid(True)
plt.show(block=False)
plt.pause(2)

trained_model.eval()
with torch.no_grad():
    pred_err = trained_model(test_X)
    corrected = test_X[:, -1, :] + pred_err
truth = test_X[:, -1, :] + test_y
sgp4 = test_X[:, -1, :]

sgp4_err = torch.norm(truth - sgp4, dim=1)
tdnn_err = torch.norm(truth - corrected, dim=1)

plt.figure(figsize=(12, 6))
plt.plot(sgp4_err.numpy(), label='SGP4 Error', color='orange', alpha=0.6)
plt.plot(tdnn_err.numpy(), label='TDNN Corrected Error', color='green')
plt.xlabel('Sample Index')
plt.ylabel('3D Error (km)')
plt.title('3D Error Comparison: SGP4 vs TDNN')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n========== RMS Error Summary ==========")
print(f"SGP4 Mean 3D Error  : {sgp4_err.mean():.3f} km")
print(f"SGP4 Max 3D Error   : {sgp4_err.max():.3f} km")
print(f"TDNN Mean 3D Error  : {tdnn_err.mean():.3f} km")
print(f"TDNN Max 3D Error   : {tdnn_err.max():.3f} km")
print("=======================================")
