"""
verify_sp3_tle_alignment_with_orbits_and_3d.py
-------------------------------------------------
Verifies alignment between SP3 (truth) and TLE (SGP4) orbit data.

✅ Shows SP3 and SGP4 start/end times
✅ Marks TLE epoch transitions on X/Y/Z plots
✅ Includes 3D orbit plot (X vs Y vs Z)
✅ Plots 3D position error magnitude over time
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
from astropy.coordinates import ITRS, TEME
from astropy.time import Time
import astropy.units as u
import glob, os


# ==============================================================
# 1. Load SP3 truth data
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
    df = pd.DataFrame({"time_sp3": times, "x_truth": xs, "y_truth": ys, "z_truth": zs})
    return df


# ==============================================================
# 2. Convert SP3 (ECEF/ITRF) → TEME
# ==============================================================
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


# ==============================================================
# 3. Propagate TLE segments (bounded by next epoch)
# ==============================================================
def propagate_tle(filepath, time_list):
    lines = open(filepath).read().strip().splitlines()
    tle_pairs = [(lines[i], lines[i + 1]) for i in range(0, len(lines), 2)]

    # Extract TLE epochs
    tle_epochs = []
    for line1, _ in tle_pairs:
        year = int(line1[18:20])
        year += 2000 if year < 57 else 1900
        day_of_year = float(line1[20:32])
        epoch_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
        tle_epochs.append(epoch_date)

    df_list = []
    for i, (line1, line2) in enumerate(tle_pairs):
        sat = Satrec.twoline2rv(line1, line2)
        start_time = tle_epochs[i]
        end_time = tle_epochs[i + 1] if i + 1 < len(tle_epochs) else time_list[-1]

        # Get only non-overlapping times for this TLE
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

    # Combine all segments, drop duplicate timestamps, sort chronologically
    df = pd.concat(df_list, ignore_index=True)
    df = df.drop_duplicates(subset=["time"]).sort_values(by="time").reset_index(drop=True)
    return df, tle_epochs



# ==============================================================
# 4. Compute SGP4 vs SP3 errors
# ==============================================================
def compute_errors(sp3_df, sgp4_df):
    merged = pd.merge(sp3_df, sgp4_df, left_on="time_sp3", right_on="time", how="inner")
    merged["dx"] = merged["x_truth"] - merged["x_sgp4"]
    merged["dy"] = merged["y_truth"] - merged["y_sgp4"]
    merged["dz"] = merged["z_truth"] - merged["z_sgp4"]
    merged["error_norm_km"] = np.sqrt(merged["dx"]**2 + merged["dy"]**2 + merged["dz"]**2)
    return merged


# ==============================================================
# 5. Run verification on all SP3/TLE pairs
# ==============================================================
sp3_files = sorted(glob.glob("../Data/Lag1_*.sp3"))
tle_files = sorted(glob.glob("../Data/Lag1TLE_*.txt"))

print("\n=== Checking for SP3 and TLE files ===")
print("SP3 files found:", sp3_files)
print("TLE files found:", tle_files)

if not sp3_files or not tle_files:
    print("❌ No matching files found.")
    exit()

for sp3_file, tle_file in zip(sp3_files, tle_files):
    print(f"\n→ {os.path.basename(sp3_file)} with {os.path.basename(tle_file)}")

    sp3_df = load_sp3(sp3_file)
    sp3_df = ecef_to_teme(sp3_df)
    sgp4_df, tle_epochs = propagate_tle(tle_file, list(sp3_df["time_sp3"]))

    print(f"   SP3 range : {sp3_df['time_sp3'].iloc[0]}  →  {sp3_df['time_sp3'].iloc[-1]}")
    if len(tle_epochs) > 0:
        print(f"   TLE first epoch : {tle_epochs[0]}  →  last epoch : {tle_epochs[-1]}")
    print(f"   Total SGP4 points: {len(sgp4_df)}")

    # ===============================
    # PLOT: Raw orbit coordinates (X/Y/Z vs time)
    # ===============================
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    coords = ["x", "y", "z"]
    colors = ["blue", "orange"]

    for i, ax in enumerate(axes):
        coord = coords[i]
        ax.plot(sp3_df["time_sp3"], sp3_df[f"{coord}_truth"], label=f"SP3 {coord.upper()}", color="blue")
        ax.plot(sgp4_df["time"], sgp4_df[f"{coord}_sgp4"], label=f"SGP4 {coord.upper()}", color="orange", linestyle="--")

        # Mark TLE epoch boundaries
        for epoch in tle_epochs:
            if sp3_df["time_sp3"].iloc[0] <= epoch <= sp3_df["time_sp3"].iloc[-1]:
                ax.axvline(epoch, color="gray", linestyle=":", alpha=0.7)
        ax.set_ylabel(f"{coord.upper()} (km)")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time (UTC)")
    plt.suptitle(f"SP3 vs SGP4 Orbit Components — {os.path.basename(sp3_file)}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=False)
    plt.pause(2)

    # ===============================
    # PLOT: 3D Orbit Trajectories
    # ===============================
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(sp3_df["x_truth"], sp3_df["y_truth"], sp3_df["z_truth"], color="blue", label="SP3 (Truth)")
    ax.plot(sgp4_df["x_sgp4"], sgp4_df["y_sgp4"], sgp4_df["z_sgp4"], color="orange", linestyle="--", label="SGP4 (Propagated)")

    ax.set_title(f"3D Orbit Comparison — {os.path.basename(sp3_file)}")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)

    # ===============================
    # Compute and plot errors
    # ===============================
    merged = compute_errors(sp3_df, sgp4_df)
    print(f"   Total aligned points: {len(merged)}")
    print(f"   Mean 3D error: {merged['error_norm_km'].mean():.3f} km")
    print(f"   Max 3D error : {merged['error_norm_km'].max():.3f} km")

    plt.figure(figsize=(10, 4))
    plt.plot(merged["time_sp3"], merged["error_norm_km"], color="red")
    for epoch in tle_epochs:
        if merged["time_sp3"].iloc[0] <= epoch <= merged["time_sp3"].iloc[-1]:
            plt.axvline(epoch, color="gray", linestyle=":", alpha=0.7)
    plt.title(f"SGP4–SP3 3D Error Magnitude — {os.path.basename(sp3_file)}")
    plt.xlabel("Time (UTC)")
    plt.ylabel("3D Error (km)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
