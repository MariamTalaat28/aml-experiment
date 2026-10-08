import pandas as pd
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import joblib


# ============================================================
# CICIDS2017 - MODEL EVALUATION
# ============================================================

BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data" / "ml"
RESULTS_DIR = BASE_DIR / "results"

IMBALANCED_MODEL = RESULTS_DIR / "dnn_imbalanced.pth"
BALANCED_MODEL = RESULTS_DIR / "dnn_balanced.pth"

IMBALANCED_SCALER = RESULTS_DIR / "dnn_imbalanced_scaler.pkl"
BALANCED_SCALER = RESULTS_DIR / "dnn_balanced_scaler.pkl"

TEST_FILE = DATA_DIR / "test.csv"

CHUNK_SIZE = 10_000

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("CICIDS2017 - MODEL EVALUATION")
print("=" * 70)

print(f"\nDevice: {device}")


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
# FIND FEATURES
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

print(
    f"\nNumber of input features: {NUM_FEATURES}"
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_file,
    scaler_file,
    model_name
):

    print("\n")
    print("=" * 70)
    print(f"EVALUATING: {model_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load scaler
    # --------------------------------------------------------

    scaler = joblib.load(
        scaler_file
    )

    print("\nScaler loaded.")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = DNN(NUM_FEATURES)

    model.load_state_dict(
        torch.load(
            model_file,
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    print("Model loaded.")

    # --------------------------------------------------------
    # Evaluation variables
    # --------------------------------------------------------

    all_predictions = []
    all_labels = []

    total_rows = 0

    # --------------------------------------------------------
    # Read test data in chunks
    # --------------------------------------------------------

    print("\nEvaluating test set...")

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

        # ----------------------------------------------------
        # Scale using the training scaler
        # ----------------------------------------------------

        X = scaler.transform(X)

        X = np.asarray(
            X,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Convert to PyTorch tensor
        # ----------------------------------------------------

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32,
            device=device
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(
                X_tensor
            )

            probabilities = torch.sigmoid(
                outputs
            )

            predictions = (
                probabilities >= 0.5
            ).int()

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        all_predictions.extend(
            predictions.cpu().numpy().flatten()
        )

        all_labels.extend(
            y.to_numpy()
        )

        total_rows += len(chunk)

        print(
            f"Evaluated: {total_rows:,} rows",
            end="\r"
        )

    print(
        f"\nTotal test rows evaluated: {total_rows:,}"
    )

    # ========================================================
    # METRICS
    # ========================================================

    y_true = np.array(
        all_labels,
        dtype=int
    )

    y_pred = np.array(
        all_predictions,
        dtype=int
    )

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

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n")
    print("-" * 70)
    print("RESULTS")
    print("-" * 70)

    print(
        f"Accuracy  : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision : {precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall    : {recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1-score  : {f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print("\n")
    print("-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)

    print(
        "\n                 Predicted"
    )

    print(
        "                 BENIGN   ATTACK"
    )

    print(
        f"Actual BENIGN    {cm[0,0]:8d} {cm[0,1]:8d}"
    )

    print(
        f"Actual ATTACK    {cm[1,0]:8d} {cm[1,1]:8d}"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\n")
    print("-" * 70)
    print("CLASSIFICATION REPORT")
    print("-" * 70)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "BENIGN",
                "ATTACK"
            ],
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }


# ============================================================
# EXPERIMENT A
# ============================================================

results_imbalanced = evaluate_model(
    IMBALANCED_MODEL,
    IMBALANCED_SCALER,
    "DNN - IMBALANCED TRAINING"
)


# ============================================================
# EXPERIMENT B
# ============================================================

results_balanced = evaluate_model(
    BALANCED_MODEL,
    BALANCED_SCALER,
    "DNN - BALANCED TRAINING"
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(
    f"\n{'Metric':<15}"
    f"{'Imbalanced':>15}"
    f"{'Balanced':>15}"
)

print("-" * 45)

print(
    f"{'Accuracy':<15}"
    f"{results_imbalanced['accuracy'] * 100:>14.2f}%"
    f"{results_balanced['accuracy'] * 100:>14.2f}%"
)

print(
    f"{'Precision':<15}"
    f"{results_imbalanced['precision'] * 100:>14.2f}%"
    f"{results_balanced['precision'] * 100:>14.2f}%"
)

print(
    f"{'Recall':<15}"
    f"{results_imbalanced['recall'] * 100:>14.2f}%"
    f"{results_balanced['recall'] * 100:>14.2f}%"
)

print(
    f"{'F1-score':<15}"
    f"{results_imbalanced['f1'] * 100:>14.2f}%"
    f"{results_balanced['f1'] * 100:>14.2f}%"
)

print("\n")
print("=" * 70)
print("MODEL EVALUATION COMPLETED")
print("=" * 70)

print("\nNext step:")
print("Apply adversarial attacks to the trained DNN.")