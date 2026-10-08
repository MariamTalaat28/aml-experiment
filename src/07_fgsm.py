import pandas as pd
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import joblib


# ============================================================
# CICIDS2017 - FGSM ATTACK
# ============================================================

BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data" / "ml"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = RESULTS_DIR / "dnn_balanced.pth"
SCALER_FILE = RESULTS_DIR / "dnn_balanced_scaler.pkl"
TEST_FILE = DATA_DIR / "test.csv"

CHUNK_SIZE = 10_000
BATCH_SIZE = 256

# FGSM perturbation levels from the paper
EPSILONS = [0.01, 0.05, 0.10]


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("CICIDS2017 - FGSM ADVERSARIAL ATTACK")
print("=" * 70)

print(f"\nDevice: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available.")
    print("FGSM will run on CPU.")


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
# LOAD DATA INFORMATION
# ============================================================

sample = pd.read_csv(
    TEST_FILE,
    nrows=5
)

sample.columns = sample.columns.str.strip()

FEATURE_COLUMNS = [
    column
    for column in sample.columns
    if column != "Label"
]

NUM_FEATURES = len(FEATURE_COLUMNS)

print(f"\nNumber of input features: {NUM_FEATURES}")


# ============================================================
# LOAD SCALER
# ============================================================

print("\nLoading scaler...")

scaler = joblib.load(
    SCALER_FILE
)

print("[OK] Scaler loaded.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading balanced DNN...")

model = DNN(NUM_FEATURES)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("[OK] Balanced DNN loaded.")


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.BCEWithLogitsLoss()


# ============================================================
# METRIC STORAGE
# ============================================================

results = []


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    return (
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# CLEAN BASELINE
# ============================================================

print("\n" + "=" * 70)
print("CLEAN BASELINE")
print("=" * 70)

clean_true = []
clean_pred = []

total_test_rows = 0

with torch.no_grad():

    for chunk in pd.read_csv(
        TEST_FILE,
        chunksize=CHUNK_SIZE
    ):

        chunk.columns = chunk.columns.str.strip()

        X = chunk[
            FEATURE_COLUMNS
        ].astype(np.float32)

        y = chunk[
            "Label"
        ].astype(np.float32)

        X = scaler.transform(X)

        X = np.asarray(
            X,
            dtype=np.float32
        )

        y = y.to_numpy(
            dtype=np.float32
        )

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

            outputs = model(
                X_batch
            )

            probabilities = torch.sigmoid(
                outputs
            )

            predictions = (
                probabilities >= 0.5
            ).int().cpu().numpy().flatten()

            clean_pred.extend(
                predictions
            )

            clean_true.extend(
                y[start:end].astype(int)
            )

        total_test_rows += len(chunk)

        print(
            f"Clean evaluation: "
            f"{total_test_rows:,} rows",
            end="\r"
        )


clean_accuracy, clean_precision, clean_recall, clean_f1 = (
    calculate_metrics(
        clean_true,
        clean_pred
    )
)

print("\n\nClean results:")

print(
    f"Accuracy : {clean_accuracy:.4f} "
    f"({clean_accuracy * 100:.2f}%)"
)

print(
    f"Precision: {clean_precision:.4f} "
    f"({clean_precision * 100:.2f}%)"
)

print(
    f"Recall   : {clean_recall:.4f} "
    f"({clean_recall * 100:.2f}%)"
)

print(
    f"F1-score : {clean_f1:.4f} "
    f"({clean_f1 * 100:.2f}%)"
)

print("\nClean confusion matrix:")

print(
    confusion_matrix(
        clean_true,
        clean_pred
    )
)


# ============================================================
# FGSM ATTACK
# ============================================================

print("\n" + "=" * 70)
print("FGSM ATTACK")
print("=" * 70)

print(
    "\nTesting epsilon values:",
    EPSILONS
)


for epsilon in EPSILONS:

    print("\n" + "-" * 70)

    print(
        f"FGSM epsilon = {epsilon}"
    )

    print("-" * 70)

    y_true_all = []
    y_pred_all = []

    total_rows = 0

    # --------------------------------------------------------
    # Read test data in chunks
    # --------------------------------------------------------

    for chunk in pd.read_csv(
        TEST_FILE,
        chunksize=CHUNK_SIZE
    ):

        chunk.columns = chunk.columns.str.strip()

        X = chunk[
            FEATURE_COLUMNS
        ].astype(np.float32)

        y = chunk[
            "Label"
        ].astype(np.float32)

        # Scale exactly as during training
        X = scaler.transform(X)

        X = np.asarray(
            X,
            dtype=np.float32
        )

        y = y.to_numpy(
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Process mini-batches
        # ----------------------------------------------------

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

            # ------------------------------------------------
            # Enable gradient with respect to input
            # ------------------------------------------------

            X_batch.requires_grad = True

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            outputs = model(
                X_batch
            )

            # ------------------------------------------------
            # Calculate loss
            # ------------------------------------------------

            loss = criterion(
                outputs,
                y_batch
            )

            # ------------------------------------------------
            # Calculate gradient
            # ------------------------------------------------

            model.zero_grad()

            loss.backward()

            gradient = X_batch.grad

            # ------------------------------------------------
            # FGSM
            #
            # x_adv = x + epsilon * sign(gradient)
            # ------------------------------------------------

            X_adversarial = (
                X_batch
                + epsilon * gradient.sign()
            )

            # ------------------------------------------------
            # Make prediction on adversarial data
            # ------------------------------------------------

            with torch.no_grad():

                adversarial_outputs = model(
                    X_adversarial
                )

                probabilities = torch.sigmoid(
                    adversarial_outputs
                )

                predictions = (
                    probabilities >= 0.5
                ).int().cpu().numpy().flatten()

            y_pred_all.extend(
                predictions
            )

            y_true_all.extend(
                y[start:end].astype(int)
            )

        total_rows += len(chunk)

        print(
            f"FGSM epsilon {epsilon}: "
            f"{total_rows:,} rows",
            end="\r"
        )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    accuracy, precision, recall, f1 = (
        calculate_metrics(
            y_true_all,
            y_pred_all
        )
    )

    # --------------------------------------------------------
    # Accuracy drop
    # --------------------------------------------------------

    accuracy_drop = (
        clean_accuracy - accuracy
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n")

    print(
        f"Accuracy : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision: {precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall   : {recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1-score : {f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print(
        f"Accuracy drop: {accuracy_drop:.4f} "
        f"({accuracy_drop * 100:.2f} percentage points)"
    )

    print("\nConfusion matrix:")

    print(
        confusion_matrix(
            y_true_all,
            y_pred_all
        )
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results.append({
        "Attack": "FGSM",
        "Epsilon": epsilon,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy_Drop": accuracy_drop
    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = (
    RESULTS_DIR /
    "fgsm_results.csv"
)

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FGSM EXPERIMENT COMPLETED")
print("=" * 70)

print("\nResults:")

print(
    results_df.to_string(
        index=False
    )
)

print(
    f"\nResults saved to:\n{results_file}"
)

print("\nNext step:")
print("Analyze FGSM results and then implement PGD.")