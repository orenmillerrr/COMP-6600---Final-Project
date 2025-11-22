import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# ==============================================================
# CONFIGURATION
# ==============================================================
PATH = os.getcwd()
WINDOW = 180
USE_VELOCITY = True   # turn ON/OFF velocity features
BATCH_SIZE = 128
EPOCHS = 100
LR = 0.0001

ERROR_PATH = PATH + "\\data"
MODEL_PATH = PATH + "\\save\\models\\tdnn_model.pth"
LOSS_CSV   = PATH + "\\save\\csv\\tdnn_loss_curves.csv"
PRED_CSV   = PATH + "\\save\\csv\\tdnn_predictions.csv"

SAVE_MODEL   = True          # set True if you want to save the trained model
TEST_SAMPLES = 10000           # None = use all test windows, or set e.g. 5000


# ==============================================================
# TDNN (Conv ->Flatten -> Dense MLP)
# ==============================================================

class TDNN(nn.Module):
    """
    TDNN:
      - stack of 1D convs over time
      - flatten (time x channels)
      - dense MLP stack
    
    Input:  (batch, time, feat)   -> residual, dResidual, SGP4 state
    Output: (batch, 3)            -> residual correction (x,y,z in meters)
    """
    def __init__(
        self,
        input_dim: int,          # feat_dim per time step
        window: int,             # number of time steps in the window
        conv_channels=(5, 5),    # out_channels for each conv layer
        context_sizes=(5, 3),    # kernel sizes
        dilations=(1, 2),        # dilations
        fc_dims=(64, 64),        # hidden sizes for dense layers
        output_dim: int = 3      # final output dim (x,y,z residual)
    ):
        super().__init__()

        assert len(conv_channels) == len(context_sizes) == len(dilations), \
            "conv_channels, context_sizes, and dilations must have same length"

        # -------- TDNN (Conv1d) stack --------
        self.kernel_sizes = context_sizes
        self.dilations = dilations
        self.tdnn_layers = nn.ModuleList()

        in_channels = input_dim
        for out_ch, k, d in zip(conv_channels, context_sizes, dilations):
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_ch,
                kernel_size=k,
                dilation=d,
                padding=0 
            )
            self.tdnn_layers.append(conv)
            in_channels = out_ch

        self.window = window
        flattened_dim = in_channels * window

        # -------- Fully-connected (MLP) stack --------
        self.fc_layers = nn.ModuleList()
        in_features = flattened_dim
        for h in fc_dims:
            self.fc_layers.append(nn.Linear(in_features, h))
            in_features = h

        self.output_layer = nn.Linear(in_features, output_dim)

    def forward(self, x):
        """
        x: (batch, time, feat)
        """
        x = x.transpose(1, 2)

        # TDNN conv
        for conv, k, d in zip(self.tdnn_layers, self.kernel_sizes, self.dilations):
            pad_left = (k - 1) * d
            x = F.pad(x, (pad_left, 0))
            x = F.relu(conv(x))

        # Flatten
        x = x.reshape(x.size(0), -1)

        # Dense layers
        for fc in self.fc_layers:
            x = F.relu(fc(x))

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
            raise ValueError(f"Missing columns in " + f)

        all_dfs.append(df.reset_index(drop=True))

    return all_dfs, files


# ==============================================================
# FILE SPLIT
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
    for f in train_files: print(" \u2022", f)
    print("\nTest files:")
    for f in test_files: print("  \u2022", f)
    print()

    return train_sets, test_sets


# ==============================================================
# WINDOW BUILDER: Uses SGP4 + residual history + dresidual
# ==============================================================
def build_windowed_dataset(datasets, window, scaler=None):
    df_all = pd.concat(datasets, ignore_index=True)

    # ------------ TARGET: truth residual ------------
    residual = df_all[["err_x","err_y","err_z"]].values.astype(np.float32)

    # dresidual
    dres = np.diff(residual, axis=0, prepend=residual[0:1])

    # ------------ SGP4 INPUT ------------
    if USE_VELOCITY:
        sgp4 = df_all[["x_sgp4","y_sgp4","z_sgp4",
                       "vx_sgp4","vy_sgp4","vz_sgp4"]].values.astype(np.float32)
    else:
        sgp4 = df_all[["x_sgp4","y_sgp4","z_sgp4"]].values.astype(np.float32)

    # Normalize SGP4
    if scaler is None:
        mean = sgp4.mean(axis=0)
        std  = sgp4.std(axis=0)
        std[std == 0] = 1
        scaler = {"mean": mean, "std": std}

    sgp4_norm = (sgp4 - scaler["mean"]) / scaler["std"]

    # ------------ Build Windows ------------
    X_list, y_list = [], []

    for i in range(window, len(df_all)):
        w_res  = residual[i-window:i]      
        w_dres = dres[i-window:i]       
        w_sgp4 = sgp4_norm[i-window:i]   

        # residual + dResidual + SGP4
        w = np.concatenate([w_res, w_dres, w_sgp4], axis=1)

        X_list.append(w)
        y_list.append(residual[i])    

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    print(f"Dataset: {len(X)} samples | window={window} | feat_dim={X.shape[2]}")
    return X, y, scaler


# ==============================================================
# TRAINING
# ==============================================================
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    criterion = nn.MSELoss()
    optimiz = optim.Adam(model.parameters(), lr=lr)
    train_loss, val_loss = [], []

    for ep in range(epochs):
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

        model.eval()
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                loss = criterion(model(xb), yb)
                total += loss.item()
        vl = total / len(val_loader)

        train_loss.append(tr)
        val_loss.append(vl)
        print(f"Epoch {ep+1:3d}/{epochs}: Train={tr:.6f}  Val={vl:.6f}")

    return train_loss, val_loss

def add_noise_to_raw_residuals(datasets, noise_mean=0.0, noise_std=0.0):
    noisy_sets = []
    for df in datasets:
        df = df.copy()
        if noise_std > 0:
            df[["err_x", "err_y", "err_z"]] += np.random.normal(
                noise_mean, noise_std, df[["err_x", "err_y", "err_z"]].shape
            )
        noisy_sets.append(df)
    return noisy_sets

# ==============================================================
# MAIN EXECUTION
# ==============================================================
if __name__ == "__main__":

    datasets, files = load_error_files(ERROR_PATH)
    train_sets, test_sets = split_by_file(datasets, files, 0.2)

    X_all, y_all, scaler = build_windowed_dataset(train_sets, WINDOW)

    # Split train/validation
    N = len(X_all)
    idx_val = int(0.85 * N)
    X_train, y_train = X_all[:idx_val], y_all[:idx_val]
    X_val,   y_val   = X_all[idx_val:], y_all[idx_val:]

    # Loaders
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
                              batch_size=BATCH_SIZE, shuffle=False)

    feat_dim = X_all.shape[2]

    model = TDNN(input_dim=feat_dim, window=WINDOW,
                conv_channels=(5, 5),
                context_sizes=(5, 3),
                dilations=(1, 2),
                fc_dims=(64, 64),
                output_dim=3)

    # ---------------- Train OR Load Model ----------------
    if not os.path.exists(MODEL_PATH):

        train_curve, val_curve = train_model(model, train_loader, val_loader,
                                             epochs=EPOCHS, lr=LR)

        if SAVE_MODEL:
            torch.save(model.state_dict(), MODEL_PATH)

        pd.DataFrame({"train":train_curve,"val":val_curve}).to_csv(LOSS_CSV, index=False)

        # Plot training loss
        plt.figure(figsize=(10,5))
        plt.plot(train_curve, label="Train")
        plt.plot(val_curve, label="Val")
        plt.title("TDNN Training Loss (Residual + dResidual + SGP4)")
        plt.xlabel("Epoch"); plt.ylabel("MSE (m^2)")
        plt.grid(); plt.legend(); plt.tight_layout()
        plt.show(block=False)

    else:
        
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print("\nLoaded saved model.\n")

    # ---------------- TEST ----------------
    test_sets = add_noise_to_raw_residuals(test_sets, noise_mean=0.0, noise_std=10.0)

    X_test, y_test, _ = build_windowed_dataset(test_sets, WINDOW, scaler)

    if TEST_SAMPLES is not None:
        M = min(TEST_SAMPLES, len(X_test))
        X_test, y_test = X_test[:M], y_test[:M]

    with torch.no_grad():
        y_pred = model(torch.tensor(X_test)).numpy()

    # Compute corrected positions
    df_test = pd.concat(test_sets, ignore_index=True)
    sgp4_pos = df_test[["x_sgp4","y_sgp4","z_sgp4"]].values.astype(np.float32)
    sgp4_pos = sgp4_pos[WINDOW:len(X_test)+WINDOW]

    truth_pos     = sgp4_pos + y_test
    corrected_pos = sgp4_pos + y_pred

    sgp4_err = np.linalg.norm(truth_pos - sgp4_pos, axis=1)
    tdnn_err = np.linalg.norm(truth_pos - corrected_pos, axis=1)

    # RMS improvement
    sgp4_rms = np.sqrt(np.mean(sgp4_err**2))
    tdnn_rms = np.sqrt(np.mean(tdnn_err**2))

    print("\n===== RMS Error =====")
    print(f"SGP4 RMS   : {sgp4_rms:.3f} m")
    print(f"TDNN RMS   : {tdnn_rms:.3f} m")
    print(f"Improvement: {(100*(sgp4_rms-tdnn_rms)/sgp4_rms):+.2f}%")
    print("======================")

    # Save predictions
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

    print("\nSaved predictions →", PRED_CSV)

    print("\n=== Generating Plots ===")

    # TRUE vs PREDICTED RESIDUAL MAGNITUDE
    true_norm = np.linalg.norm(y_test, axis=1)
    pred_norm = np.linalg.norm(y_pred, axis=1)

    plt.figure(figsize=(14,5))
    plt.plot(true_norm, label="True Residual |SP3 - SGP4|", alpha=0.8)
    plt.plot(pred_norm, label="Predicted Residual", alpha=0.8)
    plt.title("TDNN Residual Prediction")
    plt.xlabel("Sample")
    plt.ylabel("3D Residual (m)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)

    # SGP4 vs TDNN-CORRECTED POSITION ERROR
    plt.figure(figsize=(14,5))
    plt.plot(sgp4_err, label="SGP4 Error", alpha=0.6)
    plt.plot(tdnn_err, label="TDNN-Corrected Error", alpha=0.6)
    plt.title("SGP4 vs TDNN-Corrected 3D Error")
    plt.xlabel("Sample")
    plt.ylabel("3D Error (m)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)

    # COMPONENT-WISE RESIDUAL PREDICTION
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

    # ERROR HISTOGRAM COMPARISON
    plt.figure(figsize=(8,5))
    plt.hist(sgp4_err, bins=80, alpha=0.5, label="SGP4 Error")
    plt.hist(tdnn_err, bins=80, alpha=0.5, label="TDNN Error")
    plt.title("Distribution of 3D Errors")
    plt.xlabel("3D Error (m)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

    # SCATTER OF TRUE vs PREDICTED RESIDUAL COMPONENTS
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

    input("\nPress ENTER to exit...")
