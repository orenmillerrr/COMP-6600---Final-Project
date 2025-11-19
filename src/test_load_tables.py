import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
ERROR_FOLDER = "../Data/"
WINDOW = 30              # Time window length (TDNN receptive field)
BATCH_SIZE = 128
EPOCHS = 150
LR = 0.001

MODEL_PATH = "tdnn_model.pth"
LOSS_CSV_PATH = "tdnn_loss_curves.csv"
PRED_PATH = "tdnn_test_predictions.csv"


# ---------------------------------------------------------------
# TDNN ARCHITECTURE (Like Kassas et al.)
# ---------------------------------------------------------------
class TDNN(nn.Module):
    def __init__(self, in_dim=6, out_dim=3, hidden=64):
        super().__init__()

        # Input shape: (batch, W, 6)

        self.tdnn = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=32, kernel_size=3, dilation=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, dilation=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, dilation=4),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)    # collapse time dimension → (batch, 32)
        )

        self.fc = nn.Sequential(
            nn.Linear(32, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        # x: (batch, window, 6)
        x = x.transpose(1, 2)  # → (batch, 6, window)
        x = self.tdnn(x)        # → (batch, 32, 1)
        x = x.squeeze(-1)       # → (batch, 32)
        return self.fc(x)


# ---------------------------------------------------------------
# LOAD ALL *_errors.txt FILES
# ---------------------------------------------------------------
def load_error_files(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith("_errors.txt")])
    if not files:
        raise RuntimeError("No error files found.")

    datasets = []
    print(f"Found {len(files)} SP3 error files.\n")

    for f in files:
        print("Loading", f)
        df = pd.read_table(os.path.join(folder, f), sep=r"\s+")

        required = {
            "x_sgp4","y_sgp4","z_sgp4",
            "vx_sgp4","vy_sgp4","vz_sgp4",
            "err_x","err_y","err_z"
        }
        if not required.issubset(df.columns):
            raise ValueError(f"{f} missing required cols")

        datasets.append(df.reset_index(drop=True))

    return datasets, files


# ---------------------------------------------------------------
# FILE-BASED SPLITTING (TRAIN EARLY FILES → TEST LATER FILES)
# ---------------------------------------------------------------
def split_by_file(datasets, files, test_frac=0.2):
    N = len(datasets)
    k_test = max(1, int(N * test_frac))

    train_sets = datasets[:-k_test]
    test_sets  = datasets[-k_test:]

    print("\n=== File-Based Split ===")
    print("Training Files:")
    for name in files[:-k_test]:
        print("  •", name)
    print("\nTesting Files:")
    for name in files[-k_test:]:
        print("  •", name)
    print()

    return train_sets, test_sets


# ---------------------------------------------------------------
# BUILD SEQUENCE DATASET WITH TIME WINDOW
# ---------------------------------------------------------------
def build_window_dataset(datasets, scaler=None):
    df_all = pd.concat(datasets, ignore_index=True)

    X_raw = df_all[[
        "x_sgp4","y_sgp4","z_sgp4",
        "vx_sgp4","vy_sgp4","vz_sgp4"
    ]].values.astype(np.float32)

    y = df_all[["err_x","err_y","err_z"]].values.astype(np.float32)

    # Normalize the 6D SGP4 state
    if scaler is None:
        mean = X_raw.mean(axis=0)
        std = X_raw.std(axis=0)
        std[std == 0] = 1.0
        scaler = {"mean": mean, "std": std}

    X = (X_raw - scaler["mean"]) / scaler["std"]

    # Build windowed samples
    Xw, yw = [], []
    for i in range(len(X) - WINDOW):
        Xw.append(X[i:i+WINDOW])
        yw.append(y[i+WINDOW])

    Xw = np.array(Xw, dtype=np.float32)
    yw = np.array(yw, dtype=np.float32)

    print(f"Built windowed dataset: {len(Xw)} samples, input={Xw.shape}")
    return Xw, yw, scaler


# ---------------------------------------------------------------
# TRAIN LOOP
# ---------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs, lr):
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    train_losses, val_losses = [], []

    for e in range(epochs):
        model.train()
        tl = 0
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            tl += loss.item()

        model.eval()
        vl = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                vl += crit(model(xb), yb).item()

        tl /= len(train_loader)
        vl /= len(val_loader)
        train_losses.append(tl)
        val_losses.append(vl)

        print(f"Epoch {e+1}/{epochs} -- Train={tl:.5f}, Val={vl:.5f}")

    return train_losses, val_losses


# ---------------------------------------------------------------
# MAIN SCRIPT
# ---------------------------------------------------------------
if __name__ == "__main__":

    datasets, files = load_error_files(ERROR_FOLDER)
    train_sets, test_sets = split_by_file(datasets, files)

    # Build windowed training data
    Xw_train, y_train, scaler = build_window_dataset(train_sets)

    # Train/validation split
    N = len(Xw_train)
    idx_val = int(0.85 * N)

    Xtr, Xval = Xw_train[:idx_val], Xw_train[idx_val:]
    ytr, yval = y_train[:idx_val], y_train[idx_val:]

    train_loader = DataLoader(TensorDataset(
        torch.tensor(Xtr), torch.tensor(ytr)), batch_size=BATCH_SIZE, shuffle=True)

    val_loader = DataLoader(TensorDataset(
        torch.tensor(Xval), torch.tensor(yval)), batch_size=BATCH_SIZE)

    # Build & train TDNN
    model = TDNN(in_dim=6, out_dim=3, hidden=64)

    print("\n=== Training TDNN ===")
    train_losses, val_losses = train_model(model, train_loader, val_loader,
                                           epochs=EPOCHS, lr=LR)

    # Plot training curves
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.grid(True)
    plt.legend()
    plt.title("TDNN Loss Curves")
    plt.show(block=False)

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)

    # Build test dataset
    Xw_test, y_test, _ = build_window_dataset(test_sets, scaler=scaler)

    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(Xw_test)).numpy()

    # Compute performance
    truth = y_test
    prediction = y_pred

    true_norm = np.linalg.norm(truth, axis=1)
    pred_norm = np.linalg.norm(prediction, axis=1)

    plt.figure()
    plt.plot(true_norm, label="True Residual Norm")
    plt.plot(pred_norm, label="Predicted Residual Norm")
    plt.legend()
    plt.grid(True)
    plt.title("Residual Magnitudes — TDNN")
    plt.show(block=False)

    # Save predictions
    pd.DataFrame({
        "err_true_x": truth[:,0],
        "err_true_y": truth[:,1],
        "err_true_z": truth[:,2],
        "err_pred_x": prediction[:,0],
        "err_pred_y": prediction[:,1],
        "err_pred_z": prediction[:,2],
    }).to_csv(PRED_PATH, index=False)

    print("\n=== DONE ===")
    input("Press ENTER to close...")
