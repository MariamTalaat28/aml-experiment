import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import joblib


# ============================================================
# Configuration
# ============================================================

BASE_DIR = "/home/ubuntu/aml-experiment"

TEST_FILE = os.path.join(
    BASE_DIR,
    "data/ml/test.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_balanced.pth"
)

SCALER_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_balanced_scaler.pkl"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "results/cw_results.csv"
)

BATCH_SIZE = 256
CHUNK_SIZE = 10000

# C&W parameters
CW_STEPS = 30
CW_LR = 0.01
CW_C = 1.0
CW_KAPPA = 0.0

# Clean accuracy of our balanced DNN
CLEAN_ACCURACY = 0.9956


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cpu":
    print(
        "WARNING: C&W is running on CPU. "
        "This may take a long time."
    )


# ============================================================
# DNN Model
# ============================================================

class DNN(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, 128),
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
# Load scaler
# ============================================================

print("\nLoading scaler...")

scaler = joblib.load(
    SCALER_FILE
)

print("Scaler loaded.")


# ============================================================
# Load model
# ============================================================

print("\nLoading balanced DNN model...")

sample_df = pd.read_csv(
    TEST_FILE,
    nrows=1
)

feature_columns = [
    col
    for col in sample_df.columns
    if col != "Label"
]

input_dim = len(feature_columns)

print(
    "Number of input features:",
    input_dim
)

model = DNN(input_dim)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device
    )
)

model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# C&W Loss
# ============================================================

def cw_loss(
    model,
    x_original,
    delta,
    y,
    c,
    kappa
):

    x_adv = x_original + delta

    logits = model(
        x_adv
    ).squeeze(1)

    # Binary classification:
    #
    # y = 0 -> BENIGN
    # y = 1 -> ATTACK
    #
    # We want the adversarial example
    # to be classified as the opposite class.

    direction = (
        2.0 * y - 1.0
    )

    classification_term = torch.clamp(
        direction * logits + kappa,
        min=0.0
    )

    # L2 perturbation
    l2_term = torch.sum(
        delta ** 2,
        dim=1
    )

    # C&W objective
    loss = (
        l2_term
        + c * classification_term
    )

    return loss.mean()


# ============================================================
# C&W Attack
# ============================================================

def cw_attack(
    model,
    x,
    y
):

    # Start with zero perturbation
    delta = torch.zeros_like(
        x,
        requires_grad=True
    )

    optimizer = optim.Adam(
        [delta],
        lr=CW_LR
    )

    # Store best successful adversarial examples
    best_adv = x.detach().clone()

    best_l2 = torch.full(
        (x.shape[0],),
        float("inf"),
        device=x.device
    )

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    for step in range(CW_STEPS):

        optimizer.zero_grad()

        x_adv = x + delta

        logits = model(
            x_adv
        ).squeeze(1)

        direction = (
            2.0 * y - 1.0
        )

        classification_term = torch.clamp(
            direction * logits + CW_KAPPA,
            min=0.0
        )

        l2_per_sample = torch.sum(
            delta ** 2,
            dim=1
        )

        loss = (
            l2_per_sample
            + CW_C * classification_term
        ).mean()

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Find successful attacks
        # ----------------------------------------------------

        with torch.no_grad():

            predictions = (
                torch.sigmoid(logits)
                >= 0.5
            ).float()

            successful = (
                predictions != y
            )

            improved = (
                successful
                &
                (l2_per_sample < best_l2)
            )

            best_l2[improved] = (
                l2_per_sample[improved]
            )

            best_adv[improved] = (
                x_adv[improved]
            )

    return best_adv


# ============================================================
# Evaluate C&W
# ============================================================

def evaluate_cw():

    print("\n" + "=" * 60)
    print("Running C&W Attack")
    print("=" * 60)

    all_predictions = []
    all_targets = []

    total_rows = 0

    # Process the test dataset in chunks
    for chunk_number, chunk in enumerate(
        pd.read_csv(
            TEST_FILE,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        X = chunk[
            feature_columns
        ].values.astype(
            np.float32
        )

        y = chunk[
            "Label"
        ].values.astype(
            np.float32
        )

        # Standardize using the training scaler
        X_scaled = scaler.transform(
            X
        )

        X_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32
        )

        y_tensor = torch.tensor(
            y,
            dtype=torch.float32
        )

        # Process mini-batches
        for start in range(
            0,
            len(X_tensor),
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                len(X_tensor)
            )

            x_batch = X_tensor[
                start:end
            ].to(device)

            y_batch = y_tensor[
                start:end
            ].to(device)

            # Generate adversarial examples
            x_adv = cw_attack(
                model,
                x_batch,
                y_batch
            )

            # Final prediction
            with torch.no_grad():

                outputs = model(
                    x_adv
                ).squeeze(1)

                probabilities = torch.sigmoid(
                    outputs
                )

                predictions = (
                    probabilities >= 0.5
                ).long()

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                y_batch.cpu()
                .numpy()
                .astype(int)
            )

        total_rows += len(chunk)

        if chunk_number % 5 == 0:

            print(
                f"Processed approximately "
                f"{total_rows:,} rows..."
            )

    # ========================================================
    # Calculate metrics
    # ========================================================

    y_true = np.array(
        all_targets
    )

    y_pred = np.array(
        all_predictions
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

    accuracy_drop = (
        CLEAN_ACCURACY - accuracy
    )

    # ========================================================
    # Display results
    # ========================================================

    print("\nC&W Results")
    print("-" * 40)

    print(
        f"Optimization steps: {CW_STEPS}"
    )

    print(
        f"Learning rate:      {CW_LR}"
    )

    print(
        f"C:                  {CW_C}"
    )

    print(
        f"Kappa:              {CW_KAPPA}"
    )

    print(
        f"Accuracy:            {accuracy:.6f}"
    )

    print(
        f"Precision:           {precision:.6f}"
    )

    print(
        f"Recall:              {recall:.6f}"
    )

    print(
        f"F1 Score:            {f1:.6f}"
    )

    print(
        f"Accuracy drop:      {accuracy_drop:.6f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    return {
        "attack": "C&W",
        "steps": CW_STEPS,
        "learning_rate": CW_LR,
        "c": CW_C,
        "kappa": CW_KAPPA,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy_drop": accuracy_drop
    }


# ============================================================
# Run experiment
# ============================================================

result = evaluate_cw()


# ============================================================
# Save results
# ============================================================

results_df = pd.DataFrame(
    [result]
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("C&W experiment completed.")
print("=" * 60)

print("\nResults:")

print(
    results_df
)

print(
    f"\nResults saved to:\n{OUTPUT_FILE}"
)