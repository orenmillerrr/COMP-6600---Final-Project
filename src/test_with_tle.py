import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import georinex as grx
from sgp4.api import Satrec, jday
import glob
from datetime import datetime
import os
from astropy.coordinates import EarthLocation, TEME, ITRS
from astropy.time import Time
import astropy.units as u

# =========================================
# Step 1: Define the TDNN correction model
# =========================================
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
        output = self.fc(x)
        # Only keep last time step prediction
        return output[:, -1, :]


# =========================================
# Step 2: Load and preprocess the real data
# =========================================

def load_sp3(filepath):
    """
    Custom SP3 reader for JCET/ILRS-style files (handles PLxx / VLxx entries).
    Extracts position (X,Y,Z) in km for each epoch line (* ...).
    """
    from datetime import datetime

    times, xs, ys, zs = [], [], [], []
    current_time = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Epoch line: starts with '*  YYYY MM DD hh mm ss'
            if line.startswith("*"):
                parts = line.split()
                try:
                    year, month, day, hour, minute = map(int, parts[1:6])
                    sec = float(parts[6])
                    current_time = datetime(year, month, day, hour, minute, int(sec))
                except Exception:
                    current_time = None

            # Position line: starts with 'P' or 'PL'
            elif line.startswith("P") and current_time is not None:
                try:
                    # Some lines have an ID like PL51 or P 51
                    values = line.split()
                    # Last three numeric values are X, Y, Z in meters
                    nums = [float(v) for v in values[-3:]]
                    xs.append(nums[0])
                    ys.append(nums[1])
                    zs.append(nums[2])
                    times.append(current_time)
                except Exception:
                    continue
            # Skip velocity lines (VLxx)
            else:
                continue

    if not times:
        raise ValueError(f"No position records found in {filepath}")

    df = pd.DataFrame({
        "time_sp3": times,
        "x_truth": xs,
        "y_truth": ys,
        "z_truth": zs
    })
    return df

def propagate_tle(filepath, time_list):
    lines = open(filepath).read().strip().splitlines()
    tle_pairs = [(lines[i], lines[i + 1]) for i in range(0, len(lines), 2)]
    all_positions = []

    for line1, line2 in tle_pairs:
        sat = Satrec.twoline2rv(line1, line2)
        pos = []
        for t in time_list:
            jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
            e, r, v = sat.sgp4(jd, fr)
            pos.append(r if e == 0 else [np.nan, np.nan, np.nan])
        all_positions.append(pos)

    mean_positions = np.nanmean(np.array(all_positions), axis=0)
    df = pd.DataFrame(mean_positions, columns=["x_sgp4", "y_sgp4", "z_sgp4"])
    df["time"] = time_list
    return df

def compute_errors(sp3_df, sgp4_df):
    merged = pd.concat([sp3_df.reset_index(drop=True), sgp4_df[['x_sgp4', 'y_sgp4', 'z_sgp4']]], axis=1)
    merged['dx'] = merged['x_truth'] - merged['x_sgp4']
    merged['dy'] = merged['y_truth'] - merged['y_sgp4']
    merged['dz'] = merged['z_truth'] - merged['z_sgp4']
    merged['error_norm_km'] = np.sqrt(merged['dx']**2 + merged['dy']**2 + merged['dz']**2)
    return merged

def ecef_to_teme(sp3_df):
    """Convert SP3 positions (ECEF/ITRF) to TEME frame for comparison with SGP4."""
    itrs = ITRS(x=sp3_df["x_truth"].values * u.km,
                y=sp3_df["y_truth"].values * u.km,
                z=sp3_df["z_truth"].values * u.km,
                obstime=Time(sp3_df["time_sp3"].values))
    teme = itrs.transform_to(TEME(obstime=Time(sp3_df["time_sp3"].values)))
    sp3_df["x_truth"], sp3_df["y_truth"], sp3_df["z_truth"] = (
        teme.x.to(u.km).value,
        teme.y.to(u.km).value,
        teme.z.to(u.km).value
    )
    return sp3_df

# Load and merge all available SP3/TLE pairs
sp3_files = sorted(glob.glob("../Data/Lag1_*.sp3"))
tle_files = sorted(glob.glob("../Data/Lag1TLE_*.txt"))

print("\n=== Checking for SP3 and TLE files ===")
print("SP3 files found:", sp3_files)
print("TLE files found:", tle_files)

datasets = []
if not sp3_files or not tle_files:
    print("❌ No matching files found. Check your 'Data/' folder path and filenames.")
    print("Expected names like: Data/Lag1_30225.sp3 and Data/Lag1TLE_30225.txt")
    exit()

print("\nProcessing SP3/TLE pairs...")
for sp3_file, tle_file in zip(sp3_files, tle_files):
    print(f"  → {os.path.basename(sp3_file)} with {os.path.basename(tle_file)}")

    try:
        sp3_df = load_sp3(sp3_file)
        sp3_df = ecef_to_teme(sp3_df)
        # print(sp3_df.head(10))
        # print("X range:", sp3_df['x_truth'].min(), "→", sp3_df['x_truth'].max())
        # print("Y range:", sp3_df['y_truth'].min(), "→", sp3_df['y_truth'].max())
        # print("Z range:", sp3_df['z_truth'].min(), "→", sp3_df['z_truth'].max())

        if sp3_df.empty:
            print(f"⚠️ Skipping {sp3_file}, no data read.")
            continue

        # --- Extract SP3 start date/time ---
        sp3_start = sp3_df['time_sp3'].iloc[0]
        sp3_end = sp3_df['time_sp3'].iloc[-1]
        print(f"     SP3 Start Time : {sp3_start}  →  End Time : {sp3_end}")

        # --- Read TLE and extract its first epoch time ---
        with open(tle_file, 'r') as f:
            lines = f.readlines()
        tle_epochs = []
        for line in lines:
            if line.startswith('1 '):
                try:
                    year_str = line[18:20]
                    year = int(year_str)
                    year += 2000 if year < 57 else 1900
                    day_of_year = float(line[20:32])
                    tle_epochs.append((year, day_of_year))
                except Exception:
                    pass
        if tle_epochs:
            tle_start = tle_epochs[0]
            tle_end = tle_epochs[-1]
            import datetime
            tle_start_dt = datetime.datetime(tle_start[0], 1, 1) + datetime.timedelta(days=tle_start[1] - 1)
            tle_end_dt = datetime.datetime(tle_end[0], 1, 1) + datetime.timedelta(days=tle_end[1] - 1)
            print(f"     TLE First Epoch : {tle_start_dt.strftime('%Y-%m-%d %H:%M:%S')}  →  Last Epoch : {tle_end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("     ⚠️ No valid TLE epochs found!")
        sgp4_df = propagate_tle(tle_file, list(sp3_df['time_sp3']))
        merged = compute_errors(sp3_df, sgp4_df)
        datasets.append(merged)
        print(f"✅ Successfully processed {os.path.basename(sp3_file)}")

                # --- Plot comparison between SGP4 and SP3 for this dataset ---
        # plt.figure(figsize=(10, 6))
        # plt.plot(sp3_df["time_sp3"], sp3_df["x_truth"], label="SP3 X (truth)", color="blue")
        # plt.plot(sgp4_df["time"], sgp4_df["x_sgp4"], label="SGP4 X (pred)", color="orange", linestyle="--")
        # plt.title(f"SGP4 vs SP3 X-Coordinate — {os.path.basename(sp3_file)}")
        # plt.xlabel("Time (UTC)")
        # plt.ylabel("X Position (km)")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

        # plt.figure(figsize=(10, 6))
        # plt.plot(sp3_df["time_sp3"], sp3_df["y_truth"], label="SP3 Y (truth)", color="blue")
        # plt.plot(sgp4_df["time"], sgp4_df["y_sgp4"], label="SGP4 Y (pred)", color="orange", linestyle="--")
        # plt.title(f"SGP4 vs SP3 Y-Coordinate — {os.path.basename(sp3_file)}")
        # plt.xlabel("Time (UTC)")
        # plt.ylabel("Y Position (km)")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

        # plt.figure(figsize=(10, 6))
        # plt.plot(sp3_df["time_sp3"], sp3_df["z_truth"], label="SP3 Z (truth)", color="blue")
        # plt.plot(sgp4_df["time"], sgp4_df["z_sgp4"], label="SGP4 Z (pred)", color="orange", linestyle="--")
        # plt.title(f"SGP4 vs SP3 Z-Coordinate — {os.path.basename(sp3_file)}")
        # plt.xlabel("Time (UTC)")
        # plt.ylabel("Z Position (km)")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

        # Optional: plot total position error magnitude
        plt.figure(figsize=(10, 4))
        plt.plot(merged["time_sp3"], merged["error_norm_km"], color="red")
        plt.title(f"SGP4–SP3 Position Error — {os.path.basename(sp3_file)}")
        plt.xlabel("Time (UTC)")
        plt.ylabel("3D Error Magnitude (km)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Error processing {sp3_file} / {tle_file}: {e}")

if not datasets:
    raise RuntimeError("❌ No datasets were processed successfully. Check file contents or formats.")

data = pd.concat(datasets, ignore_index=True)
print("✅ Combined all datasets successfully.")

# =========================================
# Step 3: Prepare sequences for TDNN input
# =========================================
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

# Split into train/val/test
train_split = 0.7
val_split = 0.15
test_split = 0.15
total_samples = len(X)

train_end = int(train_split * total_samples)
val_end = int((train_split + val_split) * total_samples)

train_X, val_X, test_X = X[:train_end], X[train_end:val_end], X[val_end:]
train_y, val_y, test_y = y[:train_end], y[train_end:val_end], y[val_end:]

train_dataset = TensorDataset(train_X, train_y)
val_dataset = TensorDataset(val_X, val_y)
test_dataset = TensorDataset(test_X, test_y)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=32)

# =========================================
# Step 4: Train the model
# =========================================
def train_model(model, train_loader, val_loader, epochs=500, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                running_loss += loss.item()
        val_losses.append(running_loss / len(val_loader))
        print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_losses[-1]:.5f}, Val Loss={val_losses[-1]:.5f}")

    return model, train_losses, val_losses

model = TDNNCorrector(input_dim, output_dim)
trained_model, train_losses, val_losses = train_model(model, train_dataloader, val_dataloader)

# =========================================
# Step 5: Plot training and validation loss
# =========================================
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('TDNN Training with Real SP3/TLE Data')
plt.legend()
plt.grid(True)
plt.show()

# =========================================
# Step 6: Evaluate on test data
# =========================================
trained_model.eval()
with torch.no_grad():
    predicted_errors = trained_model(test_X)
    corrected_positions = test_X[:, -1, :] + predicted_errors

# Compute true and uncorrected (SGP4) positions
true_positions = (test_X[:, -1, :] + test_y)
sgp4_positions = test_X[:, -1, :]

# Convert to numpy for plotting
pred_np = corrected_positions.numpy()
true_np = true_positions.numpy()
sgp4_np = sgp4_positions.numpy()

# Plot one coordinate comparison
plt.figure(figsize=(12, 6))
plt.plot(sgp4_np[:, 0], label='SGP4 X', color='orange', alpha=0.6)
plt.plot(true_np[:, 0], label='True X', color='blue', alpha=0.6)
plt.plot(pred_np[:, 0], label='TDNN Corrected X', color='green')
plt.title('TDNN Orbit Correction (X Coordinate)')
plt.xlabel('Sample Index')
plt.ylabel('X Position (km)')
plt.legend()
plt.grid(True)
plt.show()


print("✅ Training complete and orbit correction results plotted.")

# =============================================================
# Step 7: Evaluate and compare TDNN vs SGP4 vs Truth accuracy
# =============================================================

# Compute 3D RMS error for each sample
sgp4_error = torch.norm(true_positions - sgp4_positions, dim=1)     # |truth - sgp4|
tdnn_error = torch.norm(true_positions - corrected_positions, dim=1)  # |truth - tdnn|

# Convert to numpy
sgp4_error_np = sgp4_error.numpy()
tdnn_error_np = tdnn_error.numpy()

# --- 1. Plot 3D error over time ---
plt.figure(figsize=(12, 6))
plt.plot(sgp4_error_np, label='SGP4 Error (km)', color='orange', alpha=0.6)
plt.plot(tdnn_error_np, label='TDNN Corrected Error (km)', color='green')
plt.title('3D Position Error Magnitude: SGP4 vs TDNN')
plt.xlabel('Sample Index (time progression)')
plt.ylabel('3D Error (km)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 2. Print summary statistics ---
import numpy as np
print("========== RMS Error Summary ==========")
print(f"SGP4 Mean 3D Error  : {np.mean(sgp4_error_np):.3f} km")
print(f"SGP4 Max 3D Error   : {np.max(sgp4_error_np):.3f} km")
print(f"TDNN Mean 3D Error  : {np.mean(tdnn_error_np):.3f} km")
print(f"TDNN Max 3D Error   : {np.max(tdnn_error_np):.3f} km")
print("=======================================")