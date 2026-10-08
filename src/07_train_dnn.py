import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib

# ============================================================
# CICIDS2017 - DNN TRAINING
# ============================================================

BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data" / "ml"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMBALANCED_FILE = DATA_DIR / "imbalanced_train.csv"
BALANCED_FILE = DATA_DIR / "balanced_train.csv"
TEST_FILE = DATA_DIR / "test.csv"

CHUNK_SIZE = 10_000
BATCH_SIZE = 256

# The paper uses 50 epochs and Adam with learning rate 0.001
EPOCHS = 50
LEARNING_RATE = 0.001

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("CICIDS2017 - DNN TRAINING")
print("=" * 70)

print(f"\nDevice: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available.")
    print("Training will run on CPU.")

# ============================================================
# DNN MODEL
# ============================================================

class DNN(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# FIND NUMBER OF FEATURES
# ============================================================

sample = pd.read_csv(
    IMBALANCED_FILE,
    nrows=5
)

sample.columns = sample.columns.str.strip()

FEATURE_COLUMNS = [
    column for column in sample.columns
    if column != "Label"
]

NUM_FEATURES = len(FEATURE_COLUMNS)

print(f"\nNumber of input features: {NUM_FEATURES}")

# ============================================================
# FIT SCALER
# ============================================================

def fit_scaler(file_path):

    print("\n" + "=" * 70)
    print(f"FITTING SCALER")
    print("=" * 70)

    scaler = StandardScaler()

    total_rows = 0

    for chunk in pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE
    ):

        chunk.columns = chunk.columns.str.strip()

        X = chunk[FEATURE_COLUMNS].astype(np.float32)

        scaler.partial_fit(X)

        total_rows += len(chunk)

        print(
            f"Scaler processed: {total_rows:,} rows",
            end="\r"
        )

    print(
        f"\nScaler fitted using {total_rows:,} rows."
    )

    return scaler


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    train_file,
    model_name
):

    print("\n")
    print("=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Fit scaler
    # --------------------------------------------------------

    scaler = fit_scaler(train_file)

    scaler_file = (
        RESULTS_DIR /
        f"{model_name}_scaler.pkl"
    )

    joblib.dump(
        scaler,
        scaler_file
    )

    print(
        f"Scaler saved to:\n{scaler_file}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = DNN(NUM_FEATURES)

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("\nModel:")
    print(model)

    print(f"\nEpochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0
        total_samples = 0

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        for chunk in pd.read_csv(
            train_file,
            chunksize=CHUNK_SIZE
        ):

            chunk.columns = chunk.columns.str.strip()

            X = chunk[FEATURE_COLUMNS].astype(
                np.float32
            )

            y = chunk["Label"].astype(
                np.float32
            )

            # Scale features
            X = scaler.transform(X)

            # Convert to numpy arrays
            X = np.asarray(
                X,
                dtype=np.float32
            )

            y = y.to_numpy(
                dtype=np.float32
            )

            # ------------------------------------------------
            # Mini-batches
            # ------------------------------------------------

            for start in range(
                0,
                len(X),
                BATCH_SIZE
            ):

                end = min(
                    start + BATCH_SIZE,
                    len(X)
                )

                X_batch = torch.tensor(
                    X[start:end],
                    dtype=torch.float32,
                    device=device
                )

                y_batch = torch.tensor(
                    y[start:end],
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(1)

                # Forward pass
                outputs = model(
                    X_batch
                )

                loss = criterion(
                    outputs,
                    y_batch
                )

                # Backpropagation
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                total_loss += (
                    loss.item()
                    * len(X_batch)
                )

                total_samples += len(
                    X_batch
                )

            del X
            del y

        average_loss = (
            total_loss /
            total_samples
        )

        print(
            f"Average loss: {average_loss:.6f}"
        )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_file = (
        RESULTS_DIR /
        f"{model_name}.pth"
    )

    torch.save(
        model.state_dict(),
        model_file
    )

    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"\nModel saved to:\n{model_file}"
    )

    return model, scaler


# ============================================================
# EXPERIMENT A
# ============================================================

model_a, scaler_a = train_model(
    IMBALANCED_FILE,
    "dnn_imbalanced"
)


# ============================================================
# EXPERIMENT B
# ============================================================

model_b, scaler_b = train_model(
    BALANCED_FILE,
    "dnn_balanced"
)


# ============================================================
# COMPLETED
# ============================================================

print("\n")
print("=" * 70)
print("ALL DNN TRAINING COMPLETED")
print("=" * 70)

print("\nCreated models:")

print(
    RESULTS_DIR /
    "dnn_imbalanced.pth"
)

print(
    RESULTS_DIR /
    "dnn_balanced.pth"
)

print("\nNext step:")
print("Evaluate both models on the SAME test set.")