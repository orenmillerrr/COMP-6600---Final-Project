import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F  # <<< NEW
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# ==============================================================
# CONFIGURATION
# ==============================================================

ERROR_FOLDER = os.getcwd() + "\\data"
WINDOW = 180
USE_VELOCITY = True   # turn ON/OFF velocity features
BATCH_SIZE = 128
EPOCHS = 100
LR = 0.0001

MODEL_PATH = "tdnn_model_velocity_LR0001.pth"
LOSS_CSV = "tdnn_loss_curves_velocity_LR0001.csv"
PRED_CSV = "tdnn_predictions_velocity_LR0001.csv"


# ==============================================================
# TDNN MODEL  (Causal, sequence-based)
# ==============================================================

class TDNN(nn.Module):
    """
    Causal Time-Delay Neural Network:
      - Uses Conv1d over time
      - Each layer only looks at current + *past* frames
      - No future information (good for streaming / online)

    Input shape:  (batch, time, feat)
    Output shape: (batch, output_dim)
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dims=(256, 256, 256),
                 context_sizes=(5, 3, 3),
                 dilations=(1, 2, 3),
                 output_dim: int = 3):  # <<< CHANGED default to 3 (x,y,z)
        super().__init__()

        assert len(hidden_dims) == len(context_sizes) == len(dilations)

        self.tdnn_layers = nn.ModuleList()
        self.kernel_sizes = context_sizes
        self.dilations = dilations

        in_channels = input_dim  # features per time step

        for hdim, k, d in zip(hidden_dims, context_sizes, dilations):
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=hdim,
                kernel_size=k,
                dilation=d,
                padding=0  # manual left padding → causal
            )
            self.tdnn_layers.append(conv)
            in_channels = hdim

        self.output_layer = nn.Linear(in_channels, output_dim)

    def forward(self, x):
        """
        x: (batch, time, feat)
        """
        # (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)

        for conv, k, d in zip(self.tdnn_layers, self.kernel_sizes, self.dilations):
            pad_left = (k - 1) * d
            # F.pad pads (left, right) along the last dimension (time)
            x = F.pad(x, (pad_left, 0))   # causal: only pad on the left
            x = F.relu(conv(x))

        # Global mean pooling over time
        x = x.mean(dim=2)  # (B, C)

        return self.output_layer(x)


# ==============================================================
# LOAD MATLAB ERROR FILES
# ==============================================================
def load_error_files(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith("_errors.txt")])
    if not files:
        raise RuntimeError("No _errors.txt files found")

    all_dfs = []
    for f in files:
        df = pd.read_table(os.path.join(folder, f), sep=r"\s+")
        required = {"x_sgp4","y_sgp4","z_sgp4",
                    "vx_sgp4","vy_sgp4","vz_sgp4",
                    "err_x","err_y","err_z"}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns in {f}")

        all_dfs.append(df.reset_index(drop=True))

    return all_dfs, files


# ==============================================================
# FILE-BASED TRAIN/TEST SPLIT
# ==============================================================
def split_by_file(datasets, files, test_fraction=0.2):
    N = len(datasets)
    test_count = max(1, int(N * test_fraction))

    train_sets = datasets[:-test_count]
    test_sets  = datasets[-test_count:]
    train_files= files[:-test_count]
    test_files = files[-test_count:]

    print("\n=== File-Based Split ===")
    print("Train files:")
    for f in train_files: print("  •", f)
    print("\nTest files:")
    for f in test_files: print("  •", f)
    print()

    return train_sets, test_sets


# ==============================================================
# BUILD WINDOWED DATASET
# ==============================================================
def build_windowed_dataset(datasets, window, scaler=None):
    df_all = pd.concat(datasets, ignore_index=True)

    # Extract truth residuals in METERS
    residual = df_all[["err_x","err_y","err_z"]].values.astype(np.float32)

    # Δ residual
    dres = np.diff(residual, axis=0, prepend=residual[0:1])

    # SGP4 state
    if USE_VELOCITY:
        sgp4 = df_all[["x_sgp4","y_sgp4","z_sgp4",
                       "vx_sgp4","vy_sgp4","vz_sgp4"]].values.astype(np.float32)
    else:
        sgp4 = df_all[["x_sgp4","y_sgp4","z_sgp4"]].values.astype(np.float32)

    # ----------------------------
    # NORMALIZE INPUTS (sgp4 only)
    # ----------------------------
    if scaler is None:
        mean = sgp4.mean(axis=0)
        std  = sgp4.std(axis=0)
        std[std == 0] = 1
        scaler = {"mean": mean, "std": std}

    sgp4_norm = (sgp4 - scaler["mean"]) / scaler["std"]

    # --------------------------------
    # WINDOW BUILDING LOOP
    # --------------------------------
    X_list = []
    y_list = []

    for i in range(window, len(df_all)):
        # windowed features
        w_res  = residual[i-window:i]        # (window,3)
        w_dres = dres[i-window:i]            # (window,3)
        w_sgp4 = sgp4_norm[i-window:i]       # (window,6 or 3)

        # CONCAT: residual + Δresidual + sgp4
        w = np.concatenate([w_res, w_dres, w_sgp4], axis=1)   # (window, feat_dim)

        X_list.append(w)
        y_list.append(residual[i])  # target is raw residual in meters

    X = np.array(X_list, dtype=np.float32)         # (N, window, feat_dim)
    y = np.array(y_list, dtype=np.float32)         # (N, 3)

    print(f"Built windowed dataset: {len(X)} samples, window={window}, feat_dim={X.shape[2]}")
    return X, y, scaler


# ==============================================================
# TRAINING LOOP
# ==============================================================
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    criterion = nn.MSELoss()
    optimiz = optim.Adam(model.parameters(), lr=lr)

    train_loss, val_loss = [], []

    for ep in range(epochs):
        # ---------------- TRAIN ----------------
        model.train()
        total = 0
        for xb, yb in train_loader:
            optimiz.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimiz.step()
            total += loss.item()
        tr = total / len(train_loader)

        # ---------------- VAL ----------------
        model.eval()
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                total += loss.item()
        vl = total / len(val_loader)

        train_loss.append(tr)
        val_loss.append(vl)

        print(f"Epoch {ep+1:3d}/{epochs}: Train={tr:.5f}  Val={vl:.5f}")

    return train_loss, val_loss


# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":

    # --------------------------
    # LOAD & SPLIT
    # --------------------------
    datasets, files = load_error_files(ERROR_FOLDER)
    train_sets, test_sets = split_by_file(datasets, files)

    # --------------------------
    # BUILD TRAIN+VAL
    # --------------------------
    X_all, y_all, scaler = build_windowed_dataset(train_sets, WINDOW)
    # X_all: (N, window, feat_dim)

    N = len(X_all)
    idx_val = int(0.85 * N)

    X_train = X_all[:idx_val]
    y_train = y_all[:idx_val]
    X_val   = X_all[idx_val:]
    y_val   = y_all[idx_val:]

    # torch loaders – note: NO FLATTENING HERE  <<< CHANGED
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=BATCH_SIZE, shuffle=False
    )

    # feat_dim per time step
    feat_dim = X_all.shape[2]   # (N, window, feat_dim)
    model = TDNN(
        input_dim=feat_dim,
        hidden_dims=([256]),
        context_sizes=([5]),
        dilations=([1]),
        output_dim=3
    )

    # --------------------------
    # TRAIN
    # --------------------------
    train_curve, val_curve = train_model(model, train_loader, val_loader,
                                         epochs=EPOCHS, lr=LR)

    torch.save(model.state_dict(), MODEL_PATH)
    print("\nSaved model →", MODEL_PATH)

    # save loss curves
    pd.DataFrame({"train":train_curve,"val":val_curve}).to_csv(LOSS_CSV, index=False)

    plt.figure()
    plt.plot(train_curve, label="Train")
    plt.plot(val_curve, label="Val")
    plt.title("TDNN Training (Residual from History)")
    plt.grid(True)
    plt.legend()
    plt.show(block=False)

    # --------------------------
    # TEST SET (HELD-OUT FILES)
    # --------------------------
    X_test, y_test, _ = build_windowed_dataset(test_sets, WINDOW, scaler)
    # X_test: (N_test, window, feat_dim)

    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test)).numpy()   # <<< CHANGED (no flatten)

    # ---------------------------------------
    # COMPUTE CORRECTED ORBITS (DENORMALIZED)
    # ---------------------------------------
    df_test = pd.concat(test_sets, ignore_index=True)

    # raw SGP4 positions
    sgp4_pos = df_test[["x_sgp4","y_sgp4","z_sgp4"]].values.astype(np.float32)

    # align lengths: remove first WINDOW rows
    sgp4_pos = sgp4_pos[WINDOW:]

    # truth pos = sgp4 + true residual
    truth_pos = sgp4_pos + y_test

    # corrected pos = sgp4 + predicted residual
    corrected_pos = sgp4_pos + y_pred   # y_pred is in METERS (correct)

    sgp4_err = np.linalg.norm(truth_pos - sgp4_pos, axis=1)
    tdnn_err = np.linalg.norm(truth_pos - corrected_pos, axis=1)

    plt.figure(figsize=(14,5))
    plt.plot(sgp4_err, alpha=0.6, label="SGP4 Error")
    plt.plot(tdnn_err, alpha=0.6, label="TDNN Corrected Error")
    plt.title("SGP4 vs TDNN-Corrected 3D Error (Held-Out Files)")
    plt.ylabel("3D Error (m)")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    # RMS comparison
    sgp4_rms = np.sqrt(np.mean(sgp4_err**2))
    tdnn_rms = np.sqrt(np.mean(tdnn_err**2))
    improvement = 100*(sgp4_rms-tdnn_rms)/sgp4_rms

    print("\n===== RMS Error Comparison =====")
    print(f"SGP4 RMS   : {sgp4_rms:.3f} m")
    print(f"TDNN RMS   : {tdnn_rms:.3f} m")
    print(f"Improvement: {improvement:+.2f}%")
    print("================================")

    # save predictions
    pd.DataFrame({
        "sgp4_err": sgp4_err,
        "tdnn_err": tdnn_err,
        "true_x": y_test[:,0],
        "true_y": y_test[:,1],
        "true_z": y_test[:,2],
        "pred_x": y_pred[:,0],
        "pred_y": y_pred[:,1],
        "pred_z": y_pred[:,2],
    }).to_csv(PRED_CSV, index=False)
    # ==============================================================
    # PLOTTING SECTION
    # ==============================================================

    print("\n=== Generating Plots ===")

    # 1 — TRAINING & VALIDATION LOSS
    plt.figure(figsize=(10,5))
    plt.plot(train_curve, label="Train Loss")
    plt.plot(val_curve, label="Val Loss")
    plt.title("TDNN Training Loss (Residual Prediction)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (m²)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)


    # 2 — TRUE vs PREDICTED RESIDUAL MAGNITUDE
    true_norm = np.linalg.norm(y_test, axis=1)
    pred_norm = np.linalg.norm(y_pred, axis=1)

    plt.figure(figsize=(14,5))
    plt.plot(true_norm, label="True Residual |SP3 − SGP4|", alpha=0.8)
    plt.plot(pred_norm, label="Predicted Residual", alpha=0.8)
    plt.title("TDNN Residual Prediction on Held-Out Files")
    plt.xlabel("Sample")
    plt.ylabel("3D Residual (m)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)


    # 3 — SGP4 vs TDNN-CORRECTED POSITION ERROR
    plt.figure(figsize=(14,5))
    plt.plot(sgp4_err, label="SGP4 Error", alpha=0.6)
    plt.plot(tdnn_err, label="TDNN-Corrected Error", alpha=0.6)
    plt.title("SGP4 vs TDNN-Corrected 3D Error (Held-Out Files)")
    plt.xlabel("Sample")
    plt.ylabel("3D Error (m)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)


    # 4 — COMPONENT-WISE RESIDUAL PREDICTION
    fig, axs = plt.subplots(3, 1, figsize=(14,12), sharex=True)

    labels = ["X Residual (m)", "Y Residual (m)", "Z Residual (m)"]
    for i, ax in enumerate(axs):
        ax.plot(y_test[:, i], label="True", alpha=0.8)
        ax.plot(y_pred[:, i], label="Pred", alpha=0.8)
        ax.set_ylabel(labels[i])
        ax.grid(True)
        ax.legend()

    axs[-1].set_xlabel("Sample Index")
    fig.suptitle("Component-wise Residual Prediction")
    plt.tight_layout()
    plt.show(block=False)


    # 5 — ERROR HISTOGRAM COMPARISON
    plt.figure(figsize=(8,5))
    plt.hist(sgp4_err, bins=80, alpha=0.5, label="SGP4 Error")
    plt.hist(tdnn_err, bins=80, alpha=0.5, label="TDNN Error")
    plt.title("Distribution of 3D Errors (Held-Out Files)")
    plt.xlabel("3D Error (m)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)


    # 6 — SCATTER OF TRUE vs PREDICTED RESIDUAL COMPONENTS
    fig, axs = plt.subplots(1, 3, figsize=(17,5))

    c_titles = ["X Component", "Y Component", "Z Component"]
    for i in range(3):
        axs[i].scatter(y_test[:, i], y_pred[:, i], s=3, alpha=0.4)
        axs[i].plot([y_test[:, i].min(), y_test[:, i].max()],
                    [y_test[:, i].min(), y_test[:, i].max()],
                    'r--')
        axs[i].set_title(c_titles[i])
        axs[i].set_xlabel("True (m)")
        axs[i].set_ylabel("Predicted (m)")
        axs[i].grid(True)

    fig.suptitle("True vs Predicted Residual Components")
    plt.tight_layout()
    plt.show(block=False)

    print("=== Plotting Complete ===\n")

    print("\nSaved predictions →", PRED_CSV)
    input("Press ENTER to close plots...")
