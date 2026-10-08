import os
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

import joblib


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "/home/ubuntu/aml-experiment"

TEST_FILE = BASE_DIR + "/data/ml/test.csv"
TRAIN_FILE = BASE_DIR + "/data/ml/balanced_train.csv"

MODEL_FILE = BASE_DIR + "/results/dnn_balanced.pth"
SCALER_FILE = BASE_DIR + "/results/dnn_balanced_scaler.pkl"

OUTPUT_FILE = (
    BASE_DIR
    + "/results/deepfool_corrected_projected_1000.csv"
)

DETAILS_FILE = (
    BASE_DIR
    + "/results/deepfool_corrected_projected_1000_details.csv"
)

N_SAMPLES = 1000
RANDOM_SEED = 42

BATCH_SIZE = 64

# DeepFool parameters
DEEPFOOL_STEPS = 100
DEEPFOOL_OVERSHOOT = 0.10

VALIDATION_TOL = 1e-5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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
print(
    "DEEPFOOL CORRECTED - "
    "BINARY DECISION BOUNDARY"
)
print("=" * 70)

print("\nDevice:", DEVICE)

print("\nLoading test dataset...")

test_df = pd.read_csv(
    TEST_FILE
)

print(
    "Test rows:",
    len(test_df)
)

print(
    "Test columns:",
    len(test_df.columns)
)


LABEL_COLUMN = "Label"

feature_columns = [
    c
    for c in test_df.columns
    if c != LABEL_COLUMN
]

X_all_raw = test_df[
    feature_columns
].values.astype(np.float64)

y_all = test_df[
    LABEL_COLUMN
].values.astype(np.int64)


# ============================================================
# LOAD SCALER
# ============================================================

print("\nLoading scaler...")

scaler = joblib.load(
    SCALER_FILE
)

print("Scaler loaded.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = DNN(
    input_dim=len(feature_columns)
)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)

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

print("Model loaded.")


# ============================================================
# SELECT SAME 1000-SAMPLE SUBSET
# ============================================================

print("\nSelecting evaluation subset...")

rng = np.random.default_rng(
    RANDOM_SEED
)

sample_indices = rng.choice(
    len(X_all_raw),
    size=N_SAMPLES,
    replace=False
)

X_raw = X_all_raw[
    sample_indices
]

y = y_all[
    sample_indices
]

print(
    "Samples selected:",
    len(X_raw)
)

print(
    "BENIGN:",
    np.sum(y == 0)
)

print(
    "ATTACK:",
    np.sum(y == 1)
)


# ============================================================
# STANDARDIZE
# ============================================================

X_scaled = scaler.transform(
    X_raw
)

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
# FIXED FEATURE INDICES
# ============================================================

fixed_indices = []

print("\nFixed features:")

for feature in FIXED_FEATURES:

    if feature not in feature_columns:

        raise ValueError(
            f"Fixed feature not found: {feature}"
        )

    idx = feature_columns.index(
        feature
    )

    fixed_indices.append(
        idx
    )

    print(
        f"{feature:<30} index={idx}"
    )

fixed_indices = np.array(
    fixed_indices,
    dtype=np.int64
)


# ============================================================
# RAW DOMAIN BOUNDS
# ============================================================

print(
    "\nCalculating raw-domain feature bounds..."
)

train_df = pd.read_csv(
    TRAIN_FILE
)

X_train_raw = train_df[
    feature_columns
].values.astype(np.float64)

raw_lower_np = np.nanmin(
    X_train_raw,
    axis=0
)

raw_upper_np = np.nanmax(
    X_train_raw,
    axis=0
)

print(
    "Raw-domain bounds calculated."
)


# ============================================================
# STANDARDIZED DOMAIN BOUNDS
# ============================================================

scaled_lower_np = (
    raw_lower_np - scaler.mean_
) / scaler.scale_

scaled_upper_np = (
    raw_upper_np - scaler.mean_
) / scaler.scale_

original_lower = scaled_lower_np.copy()
original_upper = scaled_upper_np.copy()

scaled_lower_np = np.minimum(
    original_lower,
    original_upper
)

scaled_upper_np = np.maximum(
    original_lower,
    original_upper
)

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
# CLEAN EVALUATION
# ============================================================

print(
    "\nEvaluating clean samples..."
)

with torch.no_grad():

    clean_logits = model(
        X
    ).squeeze(1)

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
# ATTACK ONLY INITIALLY CORRECT SAMPLES
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
# CORRECTED BINARY DEEPFOOL
# ============================================================

def deepfool_attack(
    model,
    x_original,
    y_true,
    lower_bound,
    upper_bound,
    fixed_indices
):

    """
    Binary DeepFool-style attack.

    The DNN has a single logit f(x):

        f(x) >= 0 -> ATTACK
        f(x) <  0 -> BENIGN

    For each sample, DeepFool linearizes the decision
    boundary around the current point.

    For y=1, we need the logit to become negative.

    For y=0, we need the logit to become positive.

    The first-order boundary step is:

        r = -f(x) / ||grad f(x)||^2 * grad f(x)

    with the sign selected according to the true class.
    """

    x_original = x_original.detach()

    x_adv = x_original.clone()

    batch_size = x_original.size(0)

    successful = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=DEVICE
    )

    for step in range(
        DEEPFOOL_STEPS
    ):

        # ----------------------------------------------------
        # Current point
        # ----------------------------------------------------

        x_current = (
            x_adv.detach()
            .clone()
            .requires_grad_(True)
        )

        # ----------------------------------------------------
        # Current logits
        # ----------------------------------------------------

        logits = model(
            x_current
        ).squeeze(1)

        predictions = (
            logits >= 0
        ).long()

        # ----------------------------------------------------
        # Successful attacks
        # ----------------------------------------------------

        current_success = (
            predictions
            != y_true.long()
        )

        successful = (
            successful
            | current_success
        )

        active = (
            ~successful
        )

        if not active.any():

            break

        # ----------------------------------------------------
        # Calculate gradient of the logit
        # ----------------------------------------------------

        gradient = torch.autograd.grad(
            logits.sum(),
            x_current,
            retain_graph=False,
            create_graph=False
        )[0]

        # ----------------------------------------------------
        # For the binary classifier:
        #
        # y = 1:
        #   need f(x) < 0
        #
        # y = 0:
        #   need f(x) > 0
        #
        # We therefore define:
        #
        # signed_logit = (2y - 1) * f(x)
        #
        # Successful boundary crossing occurs when
        # signed_logit <= 0.
        # ----------------------------------------------------

        direction = (
            2.0 * y_true - 1.0
        )

        signed_logit = (
            direction * logits
        )

        signed_gradient = (
            direction.unsqueeze(1)
            * gradient
        )

        # ----------------------------------------------------
        # ||gradient||^2
        # ----------------------------------------------------

        gradient_norm_squared = torch.sum(
            signed_gradient ** 2,
            dim=1
        )

        gradient_norm_squared = torch.clamp(
            gradient_norm_squared,
            min=1e-12
        )

        # ----------------------------------------------------
        # First-order minimum perturbation
        #
        # Move opposite to the signed gradient.
        # ----------------------------------------------------

        distance = (
            torch.abs(signed_logit)
            / gradient_norm_squared
        )

        perturbation = (
            -distance.unsqueeze(1)
            * signed_gradient
        )

        # ----------------------------------------------------
        # DeepFool overshoot
        # ----------------------------------------------------

        perturbation = (
            1.0 + DEEPFOOL_OVERSHOOT
        ) * perturbation

        # ----------------------------------------------------
        # Only modify active samples
        # ----------------------------------------------------

        active_mask = active.unsqueeze(1)

        x_new = torch.where(
            active_mask,
            x_current + perturbation,
            x_current
        )

        # ----------------------------------------------------
        # Standardized feature bounds
        # ----------------------------------------------------

        x_new = torch.max(
            torch.min(
                x_new,
                upper_bound
            ),
            lower_bound
        )

        # ----------------------------------------------------
        # Restore fixed discrete features
        # ----------------------------------------------------

        x_new[:, fixed_indices] = (
            x_original[:, fixed_indices]
        )

        x_adv = x_new.detach()

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        with torch.no_grad():

            check_logits = model(
                x_adv
            ).squeeze(1)

            check_predictions = (
                check_logits >= 0
            ).long()

            check_success = (
                check_predictions
                != y_true.long()
            )

            successful_count = int(
                check_success.sum().item()
            )

        if (
            step == 0
            or (step + 1) % 10 == 0
            or step == DEEPFOOL_STEPS - 1
        ):

            print(
                f"      Step "
                f"{step + 1:2d}/{DEEPFOOL_STEPS} | "
                f"Successful: "
                f"{successful_count}/{batch_size}"
            )

    return x_adv.detach()


# ============================================================
# RUN DEEPFOOL IN BATCHES
# ============================================================

print("\n" + "=" * 70)
print(
    "STARTING CORRECTED DEEPFOOL OPTIMIZATION"
)
print("=" * 70)

all_adv_scaled = []

all_original_scaled = []

all_original_raw = []

all_labels = []

all_attack_indices = []


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
        f"\nBatch {start}:{end}"
    )

    x_batch = X_attack[
        start:end
    ]

    y_batch = y_attack[
        start:end
    ]

    adv_batch = deepfool_attack(
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
# COMBINE RESULTS
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
# RAW DOMAIN PROJECTION
# ============================================================

print(
    "\nApplying exact raw-domain projection..."
)

X_adv_raw = (
    X_adv_np * scaler.scale_
    + scaler.mean_
)

X_adv_raw = np.clip(
    X_adv_raw,
    raw_lower_np,
    raw_upper_np
)


# ============================================================
# RESTORE FIXED FEATURES
# ============================================================

X_adv_raw[:, fixed_indices] = (
    X_original_raw_np[:, fixed_indices]
)


# ============================================================
# STANDARDIZE AGAIN
# ============================================================

X_adv_np = (
    X_adv_raw - scaler.mean_
) / scaler.scale_

X_adv_np = X_adv_np.astype(
    np.float32
)

X_adv = torch.tensor(
    X_adv_np,
    dtype=torch.float32,
    device=DEVICE
)


# ============================================================
# RAW DOMAIN VALIDATION
# ============================================================

print(
    "\nValidating raw-domain constraints..."
)

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
# FIXED FEATURE VALIDATION
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
# FINAL ADVERSARIAL EVALUATION
# ============================================================

print(
    "\nEvaluating final projected "
    "adversarial examples..."
)

with torch.no_grad():

    adv_logits = model(
        X_adv
    ).squeeze(1)

    adv_probabilities = torch.sigmoid(
        adv_logits
    )

    adv_predictions = (
        adv_logits >= 0
    ).long().cpu().numpy()


# ============================================================
# FINAL METRICS
# ============================================================

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
print("FINAL CORRECTED DEEPFOOL RESULTS")
print("=" * 70)

print(
    f"Clean Accuracy:        "
    f"{clean_accuracy * 100:.4f}%"
)

print(
    f"DeepFool Accuracy:     "
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


# ============================================================
# CONFUSION MATRIX
# ============================================================

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

print(
    "\nRaw-domain violations:",
    raw_violation_count
)

print(
    "Fixed-feature violations:",
    fixed_violations
)


# ============================================================
# SAVE SUMMARY
# ============================================================

results = pd.DataFrame({

    "Attack": [
        "DeepFool"
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
# SAVE DETAILS
# ============================================================

clean_predictions_attack_subset = (
    clean_predictions[
        attack_indices_np
    ]
)

details = pd.DataFrame({

    "Original_Index":
        attack_indices_np,

    "Label":
        y_attack_np,

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

print(
    "\nCorrected DeepFool evaluation completed."
)

print("=" * 70)