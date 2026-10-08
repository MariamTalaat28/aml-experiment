import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "/home/ubuntu/aml-experiment"

TEST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "ml",
    "test.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "results",
    "dnn_balanced.pth"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "results",
    "dnn_balanced_scaler.pkl"
)

RESULTS_PATH = os.path.join(
    BASE_DIR,
    "results",
    "deepfool_results.csv"
)


# ============================================================
# DEEPFOOL PARAMETERS
# ============================================================

MAX_ITER = 10
OVERSHOOT = 0.02

BATCH_SIZE = 256
CHUNK_SIZE = 10000

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DNN MODEL
# ============================================================

class DNN(nn.Module):

    def __init__(self, input_dim=78):

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
# DEEPFOOL ATTACK
# ============================================================

def deepfool_binary(
    model,
    x,
    max_iter=10,
    overshoot=0.02
):
    """
    Binary DeepFool attack for the tabular DNN.

    The attack iteratively estimates the distance to the
    decision boundary using the gradient of the model logit.

    Input:
        x - standardized feature vectors

    Returns:
        x_adv
        perturbation L2 norm
        attack success flag
        original predictions
    """

    model.eval()

    # --------------------------------------------------------
    # Original input
    # --------------------------------------------------------

    x_original = x.detach().clone()

    x_adv = x.detach().clone()

    # --------------------------------------------------------
    # Original predictions
    # --------------------------------------------------------

    with torch.no_grad():

        original_logits = model(
            x_original
        ).squeeze(1)

        original_predictions = (
            original_logits >= 0
        ).long()

    # --------------------------------------------------------
    # Track successful attacks
    # --------------------------------------------------------

    finished = torch.zeros(
        x.shape[0],
        dtype=torch.bool,
        device=x.device
    )

    # ========================================================
    # ITERATIVE DEEPFOOL
    # ========================================================

    for iteration in range(max_iter):

        # Stop if every sample has already crossed
        # the decision boundary.
        if finished.all():
            break

        x_adv.requires_grad_(True)

        # ----------------------------------------------------
        # Calculate logits
        # ----------------------------------------------------

        logits = model(
            x_adv
        ).squeeze(1)

        # ----------------------------------------------------
        # Calculate gradient
        # ----------------------------------------------------

        gradients = torch.autograd.grad(
            outputs=logits.sum(),
            inputs=x_adv,
            create_graph=False,
            retain_graph=False
        )[0]

        # ----------------------------------------------------
        # Gradient magnitude squared
        # ----------------------------------------------------

        grad_norm_sq = torch.sum(
            gradients ** 2,
            dim=1
        ) + 1e-12

        # ----------------------------------------------------
        # Calculate perturbation toward boundary
        # ----------------------------------------------------

        perturbation = (
            -logits / grad_norm_sq
        ).unsqueeze(1) * gradients

        # ----------------------------------------------------
        # Apply overshoot
        # ----------------------------------------------------

        perturbation = (
            perturbation *
            (1.0 + overshoot)
        )

        # ----------------------------------------------------
        # Update only unsuccessful samples
        # ----------------------------------------------------

        active = ~finished

        x_new = x_adv.detach().clone()

        x_new[active] = (
            x_adv.detach()[active]
            + perturbation.detach()[active]
        )

        x_adv = x_new

        # ----------------------------------------------------
        # Check attack success
        # ----------------------------------------------------

        with torch.no_grad():

            new_logits = model(
                x_adv
            ).squeeze(1)

            new_predictions = (
                new_logits >= 0
            ).long()

            changed = (
                new_predictions !=
                original_predictions
            )

            finished = (
                finished | changed
            )

    # ========================================================
    # FINAL PERTURBATION
    # ========================================================

    final_perturbation = (
        x_adv - x_original
    )

    perturbation_norm = torch.norm(
        final_perturbation,
        p=2,
        dim=1
    )

    return (
        x_adv.detach(),
        perturbation_norm.detach(),
        finished.detach(),
        original_predictions.detach()
    )


# ============================================================
# START EXPERIMENT
# ============================================================

print("=" * 70)
print("Loading DeepFool experiment")
print("=" * 70)

print(f"Device: {DEVICE}")


# ============================================================
# LOAD MODEL
# ============================================================

model = DNN(input_dim=78)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# Support both direct state_dict and checkpoint dictionary

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model.to(DEVICE)

model.eval()

print("Model loaded successfully.")


# ============================================================
# LOAD SCALER
# ============================================================

scaler = joblib.load(
    SCALER_PATH
)

print("Scaler loaded successfully.")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

df = pd.read_csv(
    TEST_PATH
)

print(
    f"Test samples: {len(df):,}"
)


# ============================================================
# SEPARATE FEATURES AND LABEL
# ============================================================

# IMPORTANT:
# The actual target column in test.csv is "Label"
# with a capital L.

X = df.drop(
    columns=["Label"]
).values.astype(
    np.float32
)

y = df["Label"].values.astype(
    np.int64
)

print(
    f"Number of features: {X.shape[1]}"
)


# ============================================================
# SCALE FEATURES
# ============================================================

print("\nScaling test data...")

X_scaled = scaler.transform(
    X
).astype(
    np.float32
)

print("Scaling completed.")


# ============================================================
# CLEAN MODEL PREDICTIONS
# ============================================================

print("\nCalculating clean predictions...")

clean_predictions = []


with torch.no_grad():

    for start in range(
        0,
        len(X_scaled),
        CHUNK_SIZE
    ):

        end = min(
            start + CHUNK_SIZE,
            len(X_scaled)
        )

        X_chunk = torch.tensor(
            X_scaled[start:end],
            dtype=torch.float32,
            device=DEVICE
        )

        logits = model(
            X_chunk
        ).squeeze(1)

        predictions = (
            logits >= 0
        ).long()

        clean_predictions.append(
            predictions.cpu().numpy()
        )


clean_predictions = np.concatenate(
    clean_predictions
)


clean_accuracy = accuracy_score(
    y,
    clean_predictions
)


print(
    f"Clean accuracy: "
    f"{clean_accuracy:.6f} "
    f"({clean_accuracy * 100:.2f}%)"
)


# ============================================================
# START DEEPFOOL
# ============================================================

print("\n" + "=" * 70)
print("Starting DeepFool attack")
print("=" * 70)

print(
    f"Maximum iterations: {MAX_ITER}"
)

print(
    f"Overshoot: {OVERSHOOT}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Chunk size: {CHUNK_SIZE}"
)

print("=" * 70)


# ============================================================
# STORAGE FOR RESULTS
# ============================================================

adv_predictions_all = []

successful_all = []

perturbation_norms_all = []

original_predictions_all = []


total_samples = len(
    X_scaled
)


# ============================================================
# PROCESS DATA IN CHUNKS
# ============================================================

for chunk_start in range(
    0,
    total_samples,
    CHUNK_SIZE
):

    chunk_end = min(
        chunk_start + CHUNK_SIZE,
        total_samples
    )

    print(
        f"\nProcessing samples "
        f"{chunk_start:,} - "
        f"{chunk_end:,} "
        f"of {total_samples:,}"
    )

    X_chunk = X_scaled[
        chunk_start:chunk_end
    ]

    chunk_adv_predictions = []

    chunk_successful = []

    chunk_norms = []

    chunk_original_predictions = []


    # ========================================================
    # PROCESS CHUNK IN BATCHES
    # ========================================================

    for batch_start in range(
        0,
        len(X_chunk),
        BATCH_SIZE
    ):

        batch_end = min(
            batch_start + BATCH_SIZE,
            len(X_chunk)
        )

        X_batch = torch.tensor(
            X_chunk[
                batch_start:batch_end
            ],
            dtype=torch.float32,
            device=DEVICE
        )


        # ----------------------------------------------------
        # Run DeepFool
        # ----------------------------------------------------

        (
            X_adv,
            norms,
            successful,
            original_preds
        ) = deepfool_binary(
            model=model,
            x=X_batch,
            max_iter=MAX_ITER,
            overshoot=OVERSHOOT
        )


        # ----------------------------------------------------
        # Adversarial predictions
        # ----------------------------------------------------

        with torch.no_grad():

            adv_logits = model(
                X_adv
            ).squeeze(1)

            adv_predictions = (
                adv_logits >= 0
            ).long()


        # ----------------------------------------------------
        # Store batch results
        # ----------------------------------------------------

        chunk_adv_predictions.append(
            adv_predictions.cpu().numpy()
        )

        chunk_successful.append(
            successful.cpu().numpy()
        )

        chunk_norms.append(
            norms.cpu().numpy()
        )

        chunk_original_predictions.append(
            original_preds.cpu().numpy()
        )


    # ========================================================
    # COMBINE CHUNK RESULTS
    # ========================================================

    adv_predictions_all.append(
        np.concatenate(
            chunk_adv_predictions
        )
    )

    successful_all.append(
        np.concatenate(
            chunk_successful
        )
    )

    perturbation_norms_all.append(
        np.concatenate(
            chunk_norms
        )
    )

    original_predictions_all.append(
        np.concatenate(
            chunk_original_predictions
        )
    )

    print(
        "Chunk completed."
    )


# ============================================================
# COMBINE ALL RESULTS
# ============================================================

adv_predictions = np.concatenate(
    adv_predictions_all
)

successful = np.concatenate(
    successful_all
)

perturbation_norms = np.concatenate(
    perturbation_norms_all
)

original_predictions = np.concatenate(
    original_predictions_all
)


# ============================================================
# ADVERSARIAL METRICS
# ============================================================

adv_accuracy = accuracy_score(
    y,
    adv_predictions
)


adv_precision = precision_score(
    y,
    adv_predictions,
    zero_division=0
)


adv_recall = recall_score(
    y,
    adv_predictions,
    zero_division=0
)


adv_f1 = f1_score(
    y,
    adv_predictions,
    zero_division=0
)


accuracy_drop = (
    clean_accuracy -
    adv_accuracy
)


# ============================================================
# EVASION RATE
# ============================================================

# Only samples that were originally classified correctly
# are considered for the evasion-rate calculation.

clean_correct = (
    original_predictions == y
)


successful_evasion = (
    clean_correct &
    (adv_predictions != y)
)


num_clean_correct = np.sum(
    clean_correct
)


num_successful_evasion = np.sum(
    successful_evasion
)


if num_clean_correct > 0:

    evasion_rate = (
        num_successful_evasion /
        num_clean_correct
    )

else:

    evasion_rate = 0.0


# ============================================================
# PERTURBATION STATISTICS
# ============================================================

successful_norms = (
    perturbation_norms[
        successful
    ]
)


if len(successful_norms) > 0:

    mean_l2 = np.mean(
        successful_norms
    )

    median_l2 = np.median(
        successful_norms
    )

else:

    mean_l2 = 0.0

    median_l2 = 0.0


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y,
    adv_predictions
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")

print("=" * 70)

print("DeepFool RESULTS")

print("=" * 70)


print(
    f"Clean Accuracy:        "
    f"{clean_accuracy:.6f} "
    f"({clean_accuracy * 100:.2f}%)"
)


print(
    f"Adversarial Accuracy:  "
    f"{adv_accuracy:.6f} "
    f"({adv_accuracy * 100:.2f}%)"
)


print(
    f"Accuracy Drop:         "
    f"{accuracy_drop:.6f} "
    f"({accuracy_drop * 100:.2f} percentage points)"
)


print(
    f"Precision:             "
    f"{adv_precision:.6f}"
)


print(
    f"Recall:                "
    f"{adv_recall:.6f}"
)


print(
    f"F1 Score:              "
    f"{adv_f1:.6f}"
)


print(
    f"Evasion Rate:          "
    f"{evasion_rate:.6f} "
    f"({evasion_rate * 100:.2f}%)"
)


print(
    f"Successful attacks:    "
    f"{np.sum(successful):,} / "
    f"{len(successful):,}"
)


print(
    f"Mean L2 perturbation:  "
    f"{mean_l2:.6f}"
)


print(
    f"Median L2 perturbation:"
    f" {median_l2:.6f}"
)


print("\nConfusion Matrix:")

print(cm)


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame({

    "attack": [
        "DeepFool"
    ],

    "max_iterations": [
        MAX_ITER
    ],

    "overshoot": [
        OVERSHOOT
    ],

    "clean_accuracy": [
        clean_accuracy
    ],

    "accuracy": [
        adv_accuracy
    ],

    "precision": [
        adv_precision
    ],

    "recall": [
        adv_recall
    ],

    "f1": [
        adv_f1
    ],

    "accuracy_drop": [
        accuracy_drop
    ],

    "evasion_rate": [
        evasion_rate
    ],

    "successful_attacks": [
        int(np.sum(successful))
    ],

    "total_samples": [
        len(successful)
    ],

    "mean_l2": [
        mean_l2
    ],

    "median_l2": [
        median_l2
    ]

})


results.to_csv(
    RESULTS_PATH,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)

print("DeepFool experiment completed.")

print("=" * 70)


print("\nResults:")

print(
    results.to_string(
        index=False
    )
)


print("\nResults saved to:")

print(
    RESULTS_PATH
)