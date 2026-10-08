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
# CONFIGURATION
# ============================================================

BASE_DIR = "/home/ubuntu/aml-experiment"

TEST_FILE = BASE_DIR + "/data/ml/test.csv"
TRAIN_FILE = BASE_DIR + "/data/ml/balanced_train.csv"

MODEL_FILE = BASE_DIR + "/results/dnn_balanced.pth"
SCALER_FILE = BASE_DIR + "/results/dnn_balanced_scaler.pkl"

OUTPUT_FILE = BASE_DIR + "/results/cw_final_projected_1000.csv"
DETAILS_FILE = BASE_DIR + "/results/cw_final_projected_1000_details.csv"

N_SAMPLES = 1000
RANDOM_SEED = 42

BATCH_SIZE = 256

# C&W parameters
CW_STEPS = 100
CW_LR = 0.005
CW_C = 1.0
CW_KAPPA = 0.0

# Numerical tolerance for validation
VALIDATION_TOL = 1e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# FIXED DISCRETE FEATURES
# ============================================================

FIXED_FEATURES = [
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio"
]


# ============================================================
# DNN MODEL
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
# LOAD DATA
# ============================================================

print("=" * 70)
print("C&W VALIDATED ATTACK - EXACT RAW DOMAIN PROJECTION")
print("=" * 70)

print("\nDevice:", DEVICE)

print("\nLoading test dataset...")

test_df = pd.read_csv(TEST_FILE)

print("Test rows:", len(test_df))
print("Test columns:", len(test_df.columns))


# ------------------------------------------------------------
# Separate features and labels
# ------------------------------------------------------------

LABEL_COLUMN = "Label"

feature_columns = [
    c for c in test_df.columns
    if c != LABEL_COLUMN
]

X_all_raw = test_df[feature_columns].values.astype(np.float64)
y_all = test_df[LABEL_COLUMN].values.astype(np.int64)


# ============================================================
# LOAD SCALER
# ============================================================

print("\nLoading scaler...")

scaler = joblib.load(SCALER_FILE)

print("Scaler loaded.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = DNN(input_dim=len(feature_columns))

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(checkpoint)


model.to(DEVICE)
model.eval()

print("Model loaded.")


# ============================================================
# SAMPLE FIXED 1000-SAMPLE SUBSET
# ============================================================

print("\nSelecting evaluation subset...")

rng = np.random.default_rng(RANDOM_SEED)

n_available = len(X_all_raw)

if N_SAMPLES > n_available:
    raise ValueError(
        f"N_SAMPLES={N_SAMPLES} is larger than "
        f"dataset size={n_available}"
    )

sample_indices = rng.choice(
    n_available,
    size=N_SAMPLES,
    replace=False
)

X_raw = X_all_raw[sample_indices]
y = y_all[sample_indices]

print("Samples selected:", len(X_raw))

print(
    "BENIGN:",
    np.sum(y == 0)
)

print(
    "ATTACK:",
    np.sum(y == 1)
)


# ============================================================
# STANDARDIZE DATA
# ============================================================

X_scaled = scaler.transform(X_raw)

X = torch.tensor(
    X_scaled.astype(np.float32),
    dtype=torch.float32,
    device=DEVICE
)

y_tensor = torch.tensor(
    y.astype(np.float32),
    dtype=torch.float32,
    device=DEVICE
)


# ============================================================
# FIND FIXED FEATURE INDICES
# ============================================================

fixed_indices = []

print("\nFixed features:")

for feature in FIXED_FEATURES:

    if feature not in feature_columns:

        raise ValueError(
            f"Fixed feature not found: {feature}"
        )

    idx = feature_columns.index(feature)

    fixed_indices.append(idx)

    print(
        f"{feature:<30} index={idx}"
    )

fixed_indices = np.array(
    fixed_indices,
    dtype=np.int64
)


# ============================================================
# RAW-DOMAIN TRAINING BOUNDS
# ============================================================

print("\nCalculating raw-domain feature bounds...")

train_df = pd.read_csv(TRAIN_FILE)

X_train_raw = train_df[feature_columns].values.astype(np.float64)

raw_lower_np = np.nanmin(
    X_train_raw,
    axis=0
)

raw_upper_np = np.nanmax(
    X_train_raw,
    axis=0
)

print("Raw-domain bounds calculated.")


# ============================================================
# CONVERT RAW BOUNDS TO STANDARDIZED BOUNDS
# ============================================================

scaled_lower_np = (
    raw_lower_np - scaler.mean_
) / scaler.scale_

scaled_upper_np = (
    raw_upper_np - scaler.mean_
) / scaler.scale_


# ============================================================
# ENSURE LOWER <= UPPER
# ============================================================

scaled_lower_np, scaled_upper_np = np.minimum(
    scaled_lower_np,
    scaled_upper_np
), np.maximum(
    scaled_lower_np,
    scaled_upper_np
)


# ============================================================
# CONVERT BOUNDS TO TORCH
# ============================================================

scaled_lower = torch.tensor(
    scaled_lower_np.astype(np.float32),
    dtype=torch.float32,
    device=DEVICE
)

scaled_upper = torch.tensor(
    scaled_upper_np.astype(np.float32),
    dtype=torch.float32,
    device=DEVICE
)


# ============================================================
# CLEAN PREDICTIONS
# ============================================================

print("\nEvaluating clean samples...")

with torch.no_grad():

    clean_logits = model(X).squeeze(1)

    clean_probabilities = torch.sigmoid(
        clean_logits
    )

    clean_predictions = (
        clean_probabilities >= 0.5
    ).long().cpu().numpy()


clean_accuracy = accuracy_score(
    y,
    clean_predictions
)

clean_correct = (
    clean_predictions == y
)

n_clean_correct = np.sum(
    clean_correct
)

print(
    f"Clean Accuracy: "
    f"{clean_accuracy * 100:.4f}%"
)

print(
    f"Initially Correct: "
    f"{n_clean_correct}/{N_SAMPLES}"
)


# ============================================================
# ONLY ATTACK INITIALLY CORRECT SAMPLES
# ============================================================

attack_indices = np.where(
    clean_correct
)[0]

X_attack = X[
    attack_indices
]

y_attack = y_tensor[
    attack_indices
]

X_attack_raw = X_raw[
    attack_indices
]

n_attack_samples = len(
    attack_indices
)

print(
    "\nSamples used for attack:",
    n_attack_samples
)


# ============================================================
# STORAGE
# ============================================================

all_adv_scaled = []

all_original_scaled = []

all_original_raw = []

all_labels = []

all_attack_indices = []


# ============================================================
# C&W ATTACK FUNCTION
# ============================================================

def cw_attack(
    model,
    x_original,
    y_true,
    lower_bound,
    upper_bound,
    fixed_indices
):

    """
    Differentiable binary C&W-style untargeted attack.

    Optimization happens entirely in PyTorch.

    The final exact raw-domain projection is performed
    AFTER this function returns.
    """

    x_original = x_original.detach()

    # --------------------------------------------------------
    # Initialize perturbation
    # --------------------------------------------------------

    delta = torch.zeros_like(
        x_original,
        requires_grad=True
    )

    optimizer = optim.Adam(
        [delta],
        lr=CW_LR
    )

    best_adv = x_original.clone()

    best_l2 = torch.full(
        (x_original.size(0),),
        float("inf"),
        device=DEVICE
    )

    best_success = torch.zeros(
        x_original.size(0),
        dtype=torch.bool,
        device=DEVICE
    )

    for step in range(CW_STEPS):

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Create adversarial sample
        # ----------------------------------------------------

        x_adv = (
            x_original + delta
        )

        # ----------------------------------------------------
        # Apply differentiable standardized bounds
        # ----------------------------------------------------

        x_adv = torch.max(
            torch.min(
                x_adv,
                upper_bound
            ),
            lower_bound
        )

        # ----------------------------------------------------
        # Keep discrete features fixed
        # ----------------------------------------------------

        x_adv_fixed = x_adv.clone()

        x_adv_fixed[:, fixed_indices] = (
            x_original[:, fixed_indices]
        )

        x_adv = x_adv_fixed

        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        logits = model(
            x_adv
        ).squeeze(1)

        # ----------------------------------------------------
        # Binary C&W classification term
        #
        # y=1:
        #   direction=+1
        #   minimize max(logit, 0)
        #   pushes logit below 0
        #
        # y=0:
        #   direction=-1
        #   minimize max(-logit, 0)
        #   pushes logit above 0
        # ----------------------------------------------------

        direction = (
            2.0 * y_true - 1.0
        )

        classification_term = torch.clamp(
            direction * logits + CW_KAPPA,
            min=0.0
        )

        # ----------------------------------------------------
        # L2 perturbation
        # ----------------------------------------------------

        l2_term = torch.sum(
            (x_adv - x_original) ** 2,
            dim=1
        )

        # ----------------------------------------------------
        # Total C&W loss
        # ----------------------------------------------------

        loss = (
            l2_term
            + CW_C * classification_term
        ).mean()

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Evaluate current attack
        # ----------------------------------------------------

        with torch.no_grad():

            current_logits = model(
                x_adv
            ).squeeze(1)

            current_predictions = (
                torch.sigmoid(
                    current_logits
                ) >= 0.5
            ).long()

            successful = (
                current_predictions
                != y_true.long()
            )

            current_l2 = torch.sum(
                (x_adv - x_original) ** 2,
                dim=1
            )

            # ------------------------------------------------
            # Store successful attacks with smallest L2
            # ------------------------------------------------

            improved = (
                successful
                & (
                    current_l2
                    < best_l2
                )
            )

            if improved.any():

                best_adv[improved] = (
                    x_adv[improved].detach()
                )

                best_l2[improved] = (
                    current_l2[improved]
                )

                best_success[improved] = True

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (step + 1) % 20 == 0
            or step == 0
            or step == CW_STEPS - 1
        ):

            successful_count = int(
                best_success.sum().item()
            )

            print(
                f"      Step "
                f"{step + 1:3d}/{CW_STEPS} | "
                f"Successful: "
                f"{successful_count}/{len(x_original)} | "
                f"Loss: "
                f"{loss.item():.6f}"
            )

    return best_adv.detach()


# ============================================================
# RUN C&W IN BATCHES
# ============================================================

print("\n" + "=" * 70)
print("STARTING C&W OPTIMIZATION")
print("=" * 70)

for start in range(
    0,
    n_attack_samples,
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        n_attack_samples
    )

    print(
        f"\nBatch "
        f"{start}:{end}"
    )

    x_batch = X_attack[
        start:end
    ]

    y_batch = y_attack[
        start:end
    ]

    adv_batch = cw_attack(
        model=model,
        x_original=x_batch,
        y_true=y_batch,
        lower_bound=scaled_lower,
        upper_bound=scaled_upper,
        fixed_indices=fixed_indices
    )

    all_adv_scaled.append(
        adv_batch.cpu().numpy()
    )

    all_original_scaled.append(
        x_batch.detach().cpu().numpy()
    )

    all_original_raw.append(
        X_attack_raw[start:end]
    )

    all_labels.append(
        y_batch.detach().cpu().numpy()
    )

    all_attack_indices.append(
        attack_indices[start:end]
    )


# ============================================================
# COMBINE ATTACK RESULTS
# ============================================================

X_adv_np = np.vstack(
    all_adv_scaled
)

X_original_np = np.vstack(
    all_original_scaled
)

X_original_raw_np = np.vstack(
    all_original_raw
)

y_attack_np = np.concatenate(
    all_labels
)

attack_indices_np = np.concatenate(
    all_attack_indices
)


# ============================================================
# INVERSE TRANSFORM TO RAW DOMAIN
# ============================================================

print("\nApplying exact raw-domain projection...")

X_adv_raw = (
    X_adv_np * scaler.scale_
    + scaler.mean_
)


# ============================================================
# EXACT RAW DOMAIN CLIPPING
# ============================================================

X_adv_raw = np.clip(
    X_adv_raw,
    raw_lower_np,
    raw_upper_np
)


# ============================================================
# RESTORE ALL FIXED DISCRETE FEATURES
# ============================================================

X_adv_raw[:, fixed_indices] = (
    X_original_raw_np[:, fixed_indices]
)


# ============================================================
# RE-STANDARDIZE AFTER RAW PROJECTION
# ============================================================

X_adv_np = (
    X_adv_raw - scaler.mean_
) / scaler.scale_


# ============================================================
# FLOAT32 CONVERSION
# ============================================================

X_adv_np = X_adv_np.astype(
    np.float32
)


# ============================================================
# FINAL TENSOR
# ============================================================

X_adv = torch.tensor(
    X_adv_np,
    dtype=torch.float32,
    device=DEVICE
)


# ============================================================
# FINAL RAW VALIDATION
# ============================================================

print("\nValidating raw-domain constraints...")

raw_below = (
    X_adv_raw
    < (
        raw_lower_np
        - VALIDATION_TOL
    )
)

raw_above = (
    X_adv_raw
    > (
        raw_upper_np
        + VALIDATION_TOL
    )
)

raw_violation_count = np.sum(
    raw_below | raw_above
)

print(
    "Total raw-domain violations:",
    raw_violation_count
)


# ============================================================
# VALIDATE FIXED FEATURES
# ============================================================

fixed_violations = 0

for idx in fixed_indices:

    original_values = (
        X_original_raw_np[:, idx]
    )

    adversarial_values = (
        X_adv_raw[:, idx]
    )

    violations = np.sum(
        np.abs(
            original_values
            - adversarial_values
        ) > VALIDATION_TOL
    )

    if violations > 0:

        print(
            f"Fixed feature violation "
            f"at index {idx}: "
            f"{violations}"
        )

        fixed_violations += violations


print(
    "Total fixed-feature violations:",
    fixed_violations
)


# ============================================================
# FINAL MODEL EVALUATION
# ============================================================

print("\nEvaluating final projected adversarial examples...")

with torch.no_grad():

    adv_logits = model(
        X_adv
    ).squeeze(1)

    adv_probabilities = torch.sigmoid(
        adv_logits
    )

    adv_predictions = (
        adv_probabilities >= 0.5
    ).long().cpu().numpy()


# ============================================================
# FINAL METRICS
# ============================================================

clean_predictions_attack_subset = (
    clean_predictions[
        attack_indices_np
    ]
)

adv_accuracy = accuracy_score(
    y_attack_np,
    adv_predictions
)

adv_precision = precision_score(
    y_attack_np,
    adv_predictions,
    zero_division=0
)

adv_recall = recall_score(
    y_attack_np,
    adv_predictions,
    zero_division=0
)

adv_f1 = f1_score(
    y_attack_np,
    adv_predictions,
    zero_division=0
)


# ============================================================
# ATTACK SUCCESS
# ============================================================

attack_success_mask = (
    adv_predictions
    != y_attack_np
)

successful_attacks = np.sum(
    attack_success_mask
)

attack_success_rate = (
    successful_attacks
    / n_attack_samples
)


# ============================================================
# ACCURACY DROP
# ============================================================

accuracy_drop = (
    clean_accuracy
    - adv_accuracy
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_attack_np,
    adv_predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


# ============================================================
# PERTURBATION METRICS
# ============================================================

perturbation = (
    X_adv_np.astype(np.float64)
    - X_original_np.astype(np.float64)
)

l2_values = np.linalg.norm(
    perturbation,
    axis=1
)

linf_values = np.max(
    np.abs(perturbation),
    axis=1
)


mean_l2 = np.mean(
    l2_values
)

median_l2 = np.median(
    l2_values
)

max_l2 = np.max(
    l2_values
)

mean_linf = np.mean(
    linf_values
)

median_linf = np.median(
    linf_values
)

max_linf = np.max(
    linf_values
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL C&W RESULTS")
print("=" * 70)

print(
    f"Clean Accuracy:        "
    f"{clean_accuracy * 100:.4f}%"
)

print(
    f"C&W Accuracy:          "
    f"{adv_accuracy * 100:.4f}%"
)

print(
    f"Accuracy Drop:         "
    f"{accuracy_drop * 100:.4f} pp"
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
    f"Attack Success Rate:   "
    f"{attack_success_rate * 100:.4f}%"
)

print(
    f"Successful Attacks:    "
    f"{successful_attacks}/{n_attack_samples}"
)

print(
    f"Mean L2:               "
    f"{mean_l2:.6f}"
)

print(
    f"Median L2:             "
    f"{median_l2:.6f}"
)

print(
    f"Max L2:                "
    f"{max_l2:.6f}"
)

print(
    f"Mean L-infinity:       "
    f"{mean_linf:.6f}"
)

print(
    f"Median L-infinity:     "
    f"{median_linf:.6f}"
)

print(
    f"Max L-infinity:        "
    f"{max_linf:.6f}"
)

print("\nConfusion Matrix:")

print(
    f"TN = {tn}"
)

print(
    f"FP = {fp}"
)

print(
    f"FN = {fn}"
)

print(
    f"TP = {tp}"
)

print("\nRaw-domain violations:", raw_violation_count)

print(
    "Fixed-feature violations:",
    fixed_violations
)


# ============================================================
# SAVE SUMMARY RESULTS
# ============================================================

results = pd.DataFrame({

    "Attack": [
        "C&W"
    ],

    "Clean Accuracy": [
        clean_accuracy
    ],

    "Adversarial Accuracy": [
        adv_accuracy
    ],

    "Accuracy Drop": [
        accuracy_drop
    ],

    "Precision": [
        adv_precision
    ],

    "Recall": [
        adv_recall
    ],

    "F1": [
        adv_f1
    ],

    "Attack Success Rate": [
        attack_success_rate
    ],

    "Successful Attacks": [
        successful_attacks
    ],

    "Attack Samples": [
        n_attack_samples
    ],

    "Mean L2": [
        mean_l2
    ],

    "Median L2": [
        median_l2
    ],

    "Max L2": [
        max_l2
    ],

    "Mean Linf": [
        mean_linf
    ],

    "Median Linf": [
        median_linf
    ],

    "Max Linf": [
        max_linf
    ],

    "TN": [
        tn
    ],

    "FP": [
        fp
    ],

    "FN": [
        fn
    ],

    "TP": [
        tp
    ],

    "Raw Domain Violations": [
        raw_violation_count
    ],

    "Fixed Feature Violations": [
        fixed_violations
    ]

})


results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

details = pd.DataFrame({

    "Original_Index": attack_indices_np,

    "Label": y_attack_np,

    "Clean_Prediction":
        clean_predictions_attack_subset,

    "Adversarial_Prediction":
        adv_predictions,

    "Clean_Probability":
        clean_probabilities[
            attack_indices_np
        ].detach().cpu().numpy(),

    "Adversarial_Probability":
        adv_probabilities.detach().cpu().numpy(),

    "Attack_Success":
        attack_success_mask,

    "L2":
        l2_values,

    "Linf":
        linf_values

})


details.to_csv(
    DETAILS_FILE,
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    OUTPUT_FILE
)

print(
    DETAILS_FILE
)

print("\nC&W evaluation completed.")
print("=" * 70)