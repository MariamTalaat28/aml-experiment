import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib


# =========================
# Configuration
# =========================
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
    "results/pgd_results.csv"
)

BATCH_SIZE = 256
CHUNK_SIZE = 10000

EPSILONS = [0.01, 0.05, 0.10]

# PGD implementation choice
PGD_STEPS = 10


# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if device.type == "cpu":
    print("WARNING: Running PGD on CPU. This may take a long time.")


# =========================
# DNN Model
# =========================
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


# =========================
# Load scaler
# =========================
print("\nLoading scaler...")

scaler = joblib.load(SCALER_FILE)

print("Scaler loaded.")


# =========================
# Load model
# =========================
print("\nLoading balanced DNN model...")

# Determine number of features
sample_df = pd.read_csv(TEST_FILE, nrows=1)

feature_columns = [
    col for col in sample_df.columns
    if col != "Label"
]

input_dim = len(feature_columns)

print("Number of input features:", input_dim)

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


# =========================
# PGD Attack
# =========================
def pgd_attack(
    model,
    x,
    y,
    epsilon,
    alpha,
    steps
):
    """
    Projected Gradient Descent attack.

    x       : standardized input features
    y       : true labels
    epsilon : maximum perturbation
    alpha   : step size
    steps   : number of attack iterations
    """

    # Start from the original input
    x_original = x.detach().clone()

    # Random initialization inside epsilon-ball
    x_adv = x_original + torch.empty_like(x_original).uniform_(
        -epsilon,
        epsilon
    )

    x_adv = x_adv.detach()

    for _ in range(steps):

        x_adv.requires_grad = True

        outputs = model(x_adv).squeeze(1)

        loss = nn.functional.binary_cross_entropy_with_logits(
            outputs,
            y
        )

        model.zero_grad()

        loss.backward()

        gradient = x_adv.grad.detach()

        # Move in direction that maximizes the loss
        x_adv = x_adv.detach() + alpha * gradient.sign()

        # Project perturbation back into epsilon-ball
        perturbation = torch.clamp(
            x_adv - x_original,
            min=-epsilon,
            max=epsilon
        )

        x_adv = x_original + perturbation

        x_adv = x_adv.detach()

    return x_adv


# =========================
# Evaluate PGD
# =========================
def evaluate_pgd(epsilon):

    print("\n" + "=" * 60)
    print(f"Running PGD with epsilon = {epsilon}")
    print("=" * 60)

    alpha = epsilon / 10

    all_predictions = []
    all_targets = []

    total_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            TEST_FILE,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        X = chunk[feature_columns].values.astype(np.float32)
        y = chunk["Label"].values.astype(np.float32)

        # Scale features using training scaler
        X_scaled = scaler.transform(X)

        X_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32
        )

        y_tensor = torch.tensor(
            y,
            dtype=torch.float32
        )

        # Process in mini-batches
        for start in range(
            0,
            len(X_tensor),
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                len(X_tensor)
            )

            x_batch = X_tensor[start:end].to(device)
            y_batch = y_tensor[start:end].to(device)

            # Generate adversarial examples
            x_adv = pgd_attack(
                model,
                x_batch,
                y_batch,
                epsilon,
                alpha,
                PGD_STEPS
            )

            # Evaluate adversarial examples
            with torch.no_grad():

                outputs = model(x_adv).squeeze(1)

                probabilities = torch.sigmoid(outputs)

                predictions = (
                    probabilities >= 0.5
                ).long()

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                y_batch.cpu().numpy().astype(int)
            )

        total_rows += len(chunk)

        if chunk_number % 5 == 0:
            print(
                f"Processed approximately "
                f"{total_rows:,} rows..."
            )

    # Convert to arrays
    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)

    # Metrics
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

    print("\nPGD Results")
    print("-" * 40)

    print(f"Epsilon:       {epsilon}")
    print(f"PGD steps:     {PGD_STEPS}")
    print(f"Alpha:         {alpha}")
    print(f"Accuracy:      {accuracy:.6f}")
    print(f"Precision:     {precision:.6f}")
    print(f"Recall:        {recall:.6f}")
    print(f"F1 Score:      {f1:.6f}")

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "attack": "PGD",
        "epsilon": epsilon,
        "steps": PGD_STEPS,
        "alpha": alpha,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy_drop": 0.9956 - accuracy
    }


# =========================
# Run all epsilon values
# =========================
results = []

for epsilon in EPSILONS:

    result = evaluate_pgd(epsilon)

    results.append(result)


# =========================
# Save results
# =========================
results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("PGD experiment completed.")
print("=" * 60)

print("\nResults:")
print(results_df)

print(
    f"\nResults saved to:\n{OUTPUT_FILE}"
)