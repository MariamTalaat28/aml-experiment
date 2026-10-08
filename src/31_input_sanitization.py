import os
import warnings
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

warnings.filterwarnings("ignore", message="X does not have valid feature names")


# ============================================================
# Configuration
# ============================================================

BASE_DIR = "/home/ubuntu/aml-experiment"

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "data/ml/balanced_train.csv"
)

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
    "results/input_sanitization_final_attack_evaluation.csv"
)
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/

data/
*.pem
.env
DETAILS_FILE = os.path.join(
    BASE_DIR,
    "results/input_sanitization_final_attack_details.csv"
)


# ============================================================
# Evaluation settings
# ============================================================

N_SAMPLES = 1000
RANDOM_SEED = 42
BATCH_SIZE = 256

# ------------------------------------------------------------
# Feature Squeezing
# ------------------------------------------------------------

SQUEEZE_BITS = 8
SQUEEZE_LEVELS = 2 ** SQUEEZE_BITS

# Use the 1st and 99th percentiles of the training data
# to define the squeezing range.
SQUEEZE_LOWER_PERCENTILE = 1
SQUEEZE_UPPER_PERCENTILE = 99


# ------------------------------------------------------------
# FGSM
# ------------------------------------------------------------

FGSM_EPSILON = 0.10


# ------------------------------------------------------------
# PGD
# ------------------------------------------------------------

PGD_EPSILON = 0.10
PGD_STEPS = 10
PGD_ALPHA = PGD_EPSILON / PGD_STEPS


# ------------------------------------------------------------
# C&W
# ------------------------------------------------------------

CW_STEPS = 100
CW_LR = 0.005
CW_C = 1.0
CW_KAPPA = 0.0


# ------------------------------------------------------------
# DeepFool
# ------------------------------------------------------------

DEEPFOOL_STEPS = 100
DEEPFOOL_OVERSHOOT = 0.10


# ============================================================
# Fixed features
# ============================================================
#
# These features should not be modified by the attacks.
#
# We restore them to their exact original values after every
# attack step.
# ============================================================

FIXED_FEATURE_NAMES = [
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
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("INPUT SANITIZATION / FEATURE SQUEEZING")
print("=" * 70)

print(f"Device: {DEVICE}")
print()


# ============================================================
# Model
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

        return self.network(x).view(-1)


# ============================================================
# Utility functions
# ============================================================

def get_predictions(model, x):

    model.eval()

    with torch.no_grad():

        logits = model(x)

        predictions = (
            torch.sigmoid(logits) >= 0.5
        ).long()

    return predictions


def calculate_metrics(y_true, y_pred):

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

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


# ============================================================
# Fixed-feature restoration
# ============================================================

def restore_fixed_features(
    x_current,
    x_original,
    fixed_indices
):

    if len(fixed_indices) > 0:

        x_current[:, fixed_indices] = (
            x_original[:, fixed_indices]
        )

    return x_current


# ============================================================
# FGSM
# ============================================================

def fgsm_attack(
    model,
    x,
    y,
    epsilon,
    fixed_indices
):

    model.eval()

    x_adv = x.clone().detach()
    x_adv.requires_grad = True

    logits = model(x_adv)

    loss = nn.BCEWithLogitsLoss()(
        logits,
        y.float()
    )

    model.zero_grad()

    loss.backward()

    gradient = x_adv.grad.sign()

    x_adv = (
        x_adv.detach()
        + epsilon * gradient
    )

    x_adv = restore_fixed_features(
        x_adv,
        x,
        fixed_indices
    )

    return x_adv.detach()


# ============================================================
# PGD
# ============================================================

def pgd_attack(
    model,
    x,
    y,
    epsilon,
    alpha,
    steps,
    fixed_indices
):

    model.eval()

    x_original = x.clone().detach()

    # Random start inside epsilon ball
    x_adv = (
        x_original
        + torch.empty_like(x_original)
        .uniform_(-epsilon, epsilon)
    )

    x_adv = restore_fixed_features(
        x_adv,
        x_original,
        fixed_indices
    )

    for _ in range(steps):

        x_adv.requires_grad = True

        logits = model(x_adv)

        loss = nn.BCEWithLogitsLoss()(
            logits,
            y.float()
        )

        model.zero_grad()

        loss.backward()

        gradient = x_adv.grad.sign()

        x_adv = (
            x_adv.detach()
            + alpha * gradient
        )

        # Project into epsilon ball
        delta = x_adv - x_original

        delta = torch.clamp(
            delta,
            -epsilon,
            epsilon
        )

        x_adv = (
            x_original
            + delta
        )

        x_adv = restore_fixed_features(
            x_adv,
            x_original,
            fixed_indices
        )

        x_adv = x_adv.detach()

    return x_adv


# ============================================================
# C&W-style attack
# ============================================================
#
# Important:
# No raw training min/max projection is used here.
#
# Only the fixed CICIDS features are restored.
# ============================================================

def cw_attack(
    model,
    x,
    y,
    steps,
    learning_rate,
    c_value,
    kappa,
    fixed_indices
):

    model.eval()

    x_original = x.clone().detach()

    # We optimize delta rather than x directly.
    delta = torch.zeros_like(
        x_original,
        requires_grad=True
    )

    optimizer = torch.optim.Adam(
        [delta],
        lr=learning_rate
    )

    y_float = y.float()

    for step in range(steps):

        x_adv = (
            x_original
            + delta
        )

        x_adv = restore_fixed_features(
            x_adv,
            x_original,
            fixed_indices
        )

        logits = model(x_adv)

        # Binary C&W-style objective
        real = (
            y_float * logits
            + (1.0 - y_float) * (-logits)
        )

        other = -real

        f = torch.clamp(
            real - other,
            min=-kappa
        )

        attack_loss = torch.clamp(
            f,
            min=0.0
        )

        l2_loss = torch.sum(
            (x_adv - x_original) ** 2,
            dim=1
        )

        loss = (
            l2_loss
            + c_value * attack_loss
        ).mean()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # Restore fixed features in delta
        with torch.no_grad():

            if len(fixed_indices) > 0:

                delta[:, fixed_indices] = 0.0

        if (
            step == 0
            or (step + 1) % 25 == 0
            or step == steps - 1
        ):

            with torch.no_grad():

                candidate = (
                    x_original
                    + delta
                )

                candidate = restore_fixed_features(
                    candidate,
                    x_original,
                    fixed_indices
                )

                candidate_pred = (
                    torch.sigmoid(
                        model(candidate)
                    ) >= 0.5
                ).long()

                successful = (
                    candidate_pred != y
                ).sum().item()

            print(
                f"  C&W step {step + 1:3d}/{steps} "
                f"| successful: "
                f"{successful}/{len(y)}"
            )

    with torch.no_grad():

        x_adv = (
            x_original
            + delta
        )

        x_adv = restore_fixed_features(
            x_adv,
            x_original,
            fixed_indices
        )

    return x_adv.detach()


# ============================================================
# DeepFool-style binary attack
# ============================================================

def deepfool_attack(
    model,
    x,
    y,
    steps,
    overshoot,
    fixed_indices
):

    model.eval()

    x_original = x.clone().detach()

    x_current = x_original.clone().detach()

    y_float = y.float()

    active = torch.ones(
        len(x_current),
        dtype=torch.bool,
        device=x_current.device
    )

    for step in range(steps):

        if not active.any():

            break

        x_active = (
            x_current[active]
            .clone()
            .detach()
            .requires_grad_(True)
        )

        y_active = y_float[active]

        logits = model(x_active)

        signed_logit = (
            (2.0 * y_active - 1.0)
            * logits
        )

        gradients = []

        for i in range(len(x_active)):

            model.zero_grad()

            if x_active.grad is not None:

                x_active.grad.zero_()

            signed_logit[i].backward(
                retain_graph=True
            )

            gradients.append(
                x_active.grad[i].detach().clone()
            )

        gradients = torch.stack(
            gradients
        )

        grad_norm_squared = (
            gradients ** 2
        ).sum(dim=1)

        grad_norm_squared = torch.clamp(
            grad_norm_squared,
            min=1e-12
        )

        distance = (
            torch.abs(signed_logit.detach())
            / grad_norm_squared
        )

        perturbation = (
            -distance.unsqueeze(1)
            * gradients
        )

        perturbation *= (
            1.0 + overshoot
        )

        new_active = (
            x_active.detach()
            + perturbation
        )

        # Restore fixed features
        new_active = restore_fixed_features(
            new_active,
            x_original[active],
            fixed_indices
        )

        x_current[active] = new_active

        # Check success
        with torch.no_grad():

            current_logits = model(
                x_current[active]
            )

            current_predictions = (
                torch.sigmoid(
                    current_logits
                ) >= 0.5
            ).long()

            current_success = (
                current_predictions
                != y[active]
            )

        active_indices = torch.where(
            active
        )[0]

        active[active_indices[
            current_success
        ]] = False

        if (
            step == 0
            or (step + 1) % 25 == 0
            or step == steps - 1
        ):

            total_success = (
                (~active).sum().item()
            )

            remaining = active.sum().item()

            print(
                f"  DeepFool step "
                f"{step + 1:3d}/{steps} "
                f"| successful: "
                f"{total_success}/{len(y)} "
                f"| remaining: {remaining}"
            )

    return x_current.detach()


# ============================================================
# Feature squeezing
# ============================================================

def feature_squeeze(
    x_raw,
    lower_percentiles,
    upper_percentiles,
    levels
):

    x = x_raw.clone()

    lower = lower_percentiles.to(
        x.device,
        dtype=x.dtype
    )

    upper = upper_percentiles.to(
        x.device,
        dtype=x.dtype
    )

    # Avoid zero ranges
    value_range = upper - lower

    value_range = torch.where(
        value_range > 0,
        value_range,
        torch.ones_like(value_range)
    )

    # Normalize to [0, 1]
    normalized = (
        x - lower
    ) / value_range

    normalized = torch.clamp(
        normalized,
        0.0,
        1.0
    )

    # Quantize
    squeezed = torch.round(
        normalized * (levels - 1)
    ) / (levels - 1)

    # Convert back to original feature scale
    squeezed = (
        squeezed * value_range
        + lower
    )

    return squeezed


# ============================================================
# Fixed-feature validation
# ============================================================

def count_fixed_feature_violations(
    original_raw,
    adversarial_raw,
    fixed_indices
):

    if len(fixed_indices) == 0:

        return 0

    difference = torch.abs(
        adversarial_raw[:, fixed_indices]
        - original_raw[:, fixed_indices]
    )

    violations = (
        difference > 1e-6
    ).any(dim=1)

    return violations.sum().item()


# ============================================================
# Load data
# ============================================================

print("Loading training data...")

train_df = pd.read_csv(
    TRAIN_FILE
)

print(
    f"Training rows: {len(train_df):,}"
)

print()

print("Loading test data...")

test_df = pd.read_csv(
    TEST_FILE
)

print(
    f"Test rows: {len(test_df):,}"
)

print()


# ============================================================
# Feature columns
# ============================================================

feature_columns = [
    c for c in train_df.columns
    if c != "Label"
]

print(
    f"Number of features: "
    f"{len(feature_columns)}"
)

print()


# ============================================================
# Feature indices
# ============================================================

fixed_indices = []

for feature_name in FIXED_FEATURE_NAMES:

    if feature_name not in feature_columns:

        raise ValueError(
            f"Fixed feature not found: "
            f"{feature_name}"
        )

    fixed_indices.append(
        feature_columns.index(feature_name)
    )

print(
    f"Fixed features: "
    f"{len(fixed_indices)}"
)

for name in FIXED_FEATURE_NAMES:

    print(
        f"  - {name}"
    )

print()


# ============================================================
# Prepare train/test arrays
# ============================================================

X_train_raw = train_df[
    feature_columns
].values.astype(
    np.float32
)

X_test_raw = test_df[
    feature_columns
].values.astype(
    np.float32
)

y_test = test_df[
    "Label"
].values.astype(
    np.int64
)


# ============================================================
# Load scaler
# ============================================================

print("Loading scaler...")

scaler = joblib.load(
    SCALER_FILE
)

print("Scaler loaded.")

print()


# ============================================================
# Calculate feature-squeezing percentile ranges
# ============================================================

print(
    "Calculating feature-squeezing "
    "percentile ranges..."
)

lower_percentiles = np.percentile(
    X_train_raw,
    SQUEEZE_LOWER_PERCENTILE,
    axis=0
).astype(
    np.float32
)

upper_percentiles = np.percentile(
    X_train_raw,
    SQUEEZE_UPPER_PERCENTILE,
    axis=0
).astype(
    np.float32
)

print(
    f"Lower percentile: "
    f"{SQUEEZE_LOWER_PERCENTILE}%"
)

print(
    f"Upper percentile: "
    f"{SQUEEZE_UPPER_PERCENTILE}%"
)

print(
    f"Squeezing levels: "
    f"{SQUEEZE_LEVELS}"
)

print()


# ============================================================
# Select fixed 1000-sample subset
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

sample_indices = rng.choice(
    len(X_test_raw),
    size=N_SAMPLES,
    replace=False
)

X_test_subset_raw = (
    X_test_raw[sample_indices]
)

y_test_subset = (
    y_test[sample_indices]
)

print(
    f"Evaluation samples: "
    f"{len(X_test_subset_raw)}"
)

print(
    f"BENIGN: "
    f"{(y_test_subset == 0).sum()}"
)

print(
    f"ATTACK: "
    f"{(y_test_subset == 1).sum()}"
)

print()


# ============================================================
# Scale evaluation data
# ============================================================

X_test_subset_scaled = scaler.transform(
    X_test_subset_raw
).astype(
    np.float32
)


# ============================================================
# Convert to tensors
# ============================================================

X_test = torch.tensor(
    X_test_subset_scaled,
    dtype=torch.float32,
    device=DEVICE
)

y_test_tensor = torch.tensor(
    y_test_subset,
    dtype=torch.long,
    device=DEVICE
)

X_test_raw_tensor = torch.tensor(
    X_test_subset_raw,
    dtype=torch.float32,
    device=DEVICE
)

lower_percentiles_tensor = torch.tensor(
    lower_percentiles,
    dtype=torch.float32,
    device=DEVICE
)

upper_percentiles_tensor = torch.tensor(
    upper_percentiles,
    dtype=torch.float32,
    device=DEVICE
)


# ============================================================
# Load model
# ============================================================

print("Loading DNN model...")

model = DNN(
    input_dim=len(feature_columns)
)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    elif "state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

else:

    model.load_state_dict(
        checkpoint
    )

model.to(DEVICE)
model.eval()

print("Model loaded.")

print()


# ============================================================
# Clean evaluation
# ============================================================

print("=" * 70)
print("CLEAN EVALUATION")
print("=" * 70)

clean_predictions = get_predictions(
    model,
    X_test
)

clean_metrics = calculate_metrics(
    y_test_subset,
    clean_predictions.cpu().numpy()
)

print(
    f"Clean Accuracy: "
    f"{clean_metrics['accuracy'] * 100:.2f}%"
)

print(
    f"Precision: "
    f"{clean_metrics['precision'] * 100:.2f}%"
)

print(
    f"Recall: "
    f"{clean_metrics['recall'] * 100:.2f}%"
)

print(
    f"F1: "
    f"{clean_metrics['f1'] * 100:.2f}%"
)

print(
    "Confusion Matrix:"
)

print(
    f"TN={clean_metrics['tn']} "
    f"FP={clean_metrics['fp']} "
    f"FN={clean_metrics['fn']} "
    f"TP={clean_metrics['tp']}"
)

print()


# ============================================================
# Clean feature squeezing
# ============================================================

print("=" * 70)
print("CLEAN FEATURE SQUEEZING")
print("=" * 70)

X_test_squeezed_raw = feature_squeeze(
    X_test_raw_tensor,
    lower_percentiles_tensor,
    upper_percentiles_tensor,
    SQUEEZE_LEVELS
)

X_test_squeezed_scaled = torch.tensor(
    scaler.transform(
        X_test_squeezed_raw.cpu().numpy()
    ).astype(np.float32),
    device=DEVICE
)

squeezed_predictions = get_predictions(
    model,
    X_test_squeezed_scaled
)

squeezed_metrics = calculate_metrics(
    y_test_subset,
    squeezed_predictions.cpu().numpy()
)

print(
    f"Squeezed Clean Accuracy: "
    f"{squeezed_metrics['accuracy'] * 100:.2f}%"
)

print(
    f"Precision: "
    f"{squeezed_metrics['precision'] * 100:.2f}%"
)

print(
    f"Recall: "
    f"{squeezed_metrics['recall'] * 100:.2f}%"
)

print(
    f"F1: "
    f"{squeezed_metrics['f1'] * 100:.2f}%"
)

print(
    "Confusion Matrix:"
)

print(
    f"TN={squeezed_metrics['tn']} "
    f"FP={squeezed_metrics['fp']} "
    f"FN={squeezed_metrics['fn']} "
    f"TP={squeezed_metrics['tp']}"
)

print()


# ============================================================
# Attack evaluation helper
# ============================================================

results = []

details = []


def evaluate_attack(
    attack_name,
    x_adv_scaled
):

    print()
    print("=" * 70)
    print(
        f"{attack_name.upper()} "
        "EVALUATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Original adversarial accuracy
    # --------------------------------------------------------

    original_predictions = get_predictions(
        model,
        x_adv_scaled
    )

    original_metrics = calculate_metrics(
        y_test_subset,
        original_predictions.cpu().numpy()
    )

    # --------------------------------------------------------
    # Convert adversarial data back to raw domain
    # --------------------------------------------------------

    x_adv_raw = torch.tensor(
        scaler.inverse_transform(
            x_adv_scaled.cpu().numpy()
        ).astype(np.float32),
        device=DEVICE
    )

    # --------------------------------------------------------
    # Validate fixed features
    # --------------------------------------------------------

    fixed_violations = (
        count_fixed_feature_violations(
            X_test_raw_tensor,
            x_adv_raw,
            fixed_indices
        )
    )

    # --------------------------------------------------------
    # Feature squeezing
    # --------------------------------------------------------

    x_squeezed_raw = feature_squeeze(
        x_adv_raw,
        lower_percentiles_tensor,
        upper_percentiles_tensor,
        SQUEEZE_LEVELS
    )

    # --------------------------------------------------------
    # Restore fixed features AFTER squeezing
    #
    # This ensures that feature squeezing itself does not
    # alter the fixed/discrete CICIDS features.
    # --------------------------------------------------------

    x_squeezed_raw = restore_fixed_features(
        x_squeezed_raw,
        X_test_raw_tensor,
        fixed_indices
    )

    # --------------------------------------------------------
    # Scale sanitized data
    # --------------------------------------------------------

    x_squeezed_scaled = torch.tensor(
        scaler.transform(
            x_squeezed_raw.cpu().numpy()
        ).astype(np.float32),
        device=DEVICE
    )

    # --------------------------------------------------------
    # Sanitized predictions
    # --------------------------------------------------------

    sanitized_predictions = get_predictions(
        model,
        x_squeezed_scaled
    )

    sanitized_metrics = calculate_metrics(
        y_test_subset,
        sanitized_predictions.cpu().numpy()
    )

    # --------------------------------------------------------
    # Accuracy improvement
    # --------------------------------------------------------

    accuracy_improvement = (
        sanitized_metrics["accuracy"]
        - original_metrics["accuracy"]
    ) * 100.0

    # --------------------------------------------------------
    # Accuracy drop from squeezed clean
    # --------------------------------------------------------

    accuracy_drop = (
        squeezed_metrics["accuracy"]
        - sanitized_metrics["accuracy"]
    ) * 100.0

    # --------------------------------------------------------
    # Evasion success
    #
    # Compare against clean squeezed model predictions for
    # samples that were initially correctly classified.
    # --------------------------------------------------------

    clean_correct = (
        squeezed_predictions
        == y_test_tensor
    )

    attack_predictions = (
        sanitized_predictions
    )

    successful_evasions = (
        clean_correct
        & (
            attack_predictions
            != y_test_tensor
        )
    )

    initially_correct_count = (
        clean_correct.sum().item()
    )

    successful_evasion_count = (
        successful_evasions.sum().item()
    )

    if initially_correct_count > 0:

        evasion_success_rate = (
            successful_evasion_count
            / initially_correct_count
        ) * 100.0

    else:

        evasion_success_rate = 0.0

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"Original Attack Accuracy: "
        f"{original_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Sanitized Accuracy: "
        f"{sanitized_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Accuracy Improvement: "
        f"{accuracy_improvement:+.2f} pp"
    )

    print(
        f"Accuracy Drop from Clean Squeezed: "
        f"{accuracy_drop:.2f} pp"
    )

    print(
        f"Precision: "
        f"{sanitized_metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall: "
        f"{sanitized_metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1: "
        f"{sanitized_metrics['f1'] * 100:.2f}%"
    )

    print(
        f"Evasion Success Rate: "
        f"{evasion_success_rate:.2f}%"
    )

    print(
        f"Fixed Feature Violations: "
        f"{fixed_violations}"
    )

    print(
        "Confusion Matrix:"
    )

    print(
        f"TN={sanitized_metrics['tn']} "
        f"FP={sanitized_metrics['fp']} "
        f"FN={sanitized_metrics['fn']} "
        f"TP={sanitized_metrics['tp']}"
    )

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append({

        "attack": attack_name,

        "original_attack_accuracy":
            original_metrics["accuracy"] * 100,

        "sanitized_accuracy":
            sanitized_metrics["accuracy"] * 100,

        "accuracy_improvement":
            accuracy_improvement,

        "precision":
            sanitized_metrics["precision"] * 100,

        "recall":
            sanitized_metrics["recall"] * 100,

        "f1":
            sanitized_metrics["f1"] * 100,

        "accuracy_drop_from_clean_squeezed":
            accuracy_drop,

        "evasion_success_rate":
            evasion_success_rate,

        "tn":
            sanitized_metrics["tn"],

        "fp":
            sanitized_metrics["fp"],

        "fn":
            sanitized_metrics["fn"],

        "tp":
            sanitized_metrics["tp"],

        "fixed_feature_violations":
            fixed_violations
    })

    return (
        x_adv_raw,
        x_squeezed_raw
    )


# ============================================================
# FGSM
# ============================================================

print()
print("=" * 70)
print("GENERATING FGSM ADVERSARIAL EXAMPLES")
print("=" * 70)

X_fgsm = fgsm_attack(
    model,
    X_test,
    y_test_tensor,
    FGSM_EPSILON,
    fixed_indices
)

X_fgsm_raw, X_fgsm_squeezed_raw = (
    evaluate_attack(
        "FGSM",
        X_fgsm
    )
)


# ============================================================
# PGD
# ============================================================

print()
print("=" * 70)
print("GENERATING PGD ADVERSARIAL EXAMPLES")
print("=" * 70)

X_pgd = pgd_attack(
    model,
    X_test,
    y_test_tensor,
    PGD_EPSILON,
    PGD_ALPHA,
    PGD_STEPS,
    fixed_indices
)

X_pgd_raw, X_pgd_squeezed_raw = (
    evaluate_attack(
        "PGD",
        X_pgd
    )
)


# ============================================================
# C&W
# ============================================================

print()
print("=" * 70)
print("GENERATING C&W ADVERSARIAL EXAMPLES")
print("=" * 70)

# Attack only samples that the clean model initially
# classifies correctly.

clean_correct_mask = (
    clean_predictions
    == y_test_tensor
)

correct_indices = torch.where(
    clean_correct_mask
)[0]

print(
    f"Initially correctly classified: "
    f"{len(correct_indices)}/{N_SAMPLES}"
)

X_cw = X_test.clone()

if len(correct_indices) > 0:

    X_cw_attack = cw_attack(
        model,
        X_test[correct_indices],
        y_test_tensor[correct_indices],
        CW_STEPS,
        CW_LR,
        CW_C,
        CW_KAPPA,
        fixed_indices
    )

    X_cw[correct_indices] = (
        X_cw_attack
    )

X_cw_raw, X_cw_squeezed_raw = (
    evaluate_attack(
        "C&W",
        X_cw
    )
)


# ============================================================
# DeepFool
# ============================================================

print()
print("=" * 70)
print("GENERATING DEEPFOOL ADVERSARIAL EXAMPLES")
print("=" * 70)

X_deepfool = X_test.clone()

if len(correct_indices) > 0:

    X_deepfool_attack = deepfool_attack(
        model,
        X_test[correct_indices],
        y_test_tensor[correct_indices],
        DEEPFOOL_STEPS,
        DEEPFOOL_OVERSHOOT,
        fixed_indices
    )

    X_deepfool[correct_indices] = (
        X_deepfool_attack
    )

X_deepfool_raw, X_deepfool_squeezed_raw = (
    evaluate_attack(
        "DeepFool",
        X_deepfool
    )
)


# ============================================================
# Clean result row
# ============================================================

results.insert(
    0,
    {
        "attack": "Clean",

        "original_attack_accuracy":
            clean_metrics["accuracy"] * 100,

        "sanitized_accuracy":
            squeezed_metrics["accuracy"] * 100,

        "accuracy_improvement":
            (
                squeezed_metrics["accuracy"]
                - clean_metrics["accuracy"]
            ) * 100,

        "precision":
            squeezed_metrics["precision"] * 100,

        "recall":
            squeezed_metrics["recall"] * 100,

        "f1":
            squeezed_metrics["f1"] * 100,

        "accuracy_drop_from_clean_squeezed":
            0.0,

        "evasion_success_rate":
            0.0,

        "tn":
            squeezed_metrics["tn"],

        "fp":
            squeezed_metrics["fp"],

        "fn":
            squeezed_metrics["fn"],

        "tp":
            squeezed_metrics["tp"],

        "fixed_feature_violations":
            0
    }
)


# ============================================================
# Save results
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Attack details
# ============================================================

def calculate_perturbation_details(
    attack_name,
    x_original_raw,
    x_adv_raw,
    fixed_violations
):

    perturbation = (
        x_adv_raw
        - x_original_raw
    )

    l2 = torch.sqrt(
        torch.sum(
            perturbation ** 2,
            dim=1
        )
    )

    linf = torch.max(
        torch.abs(perturbation),
        dim=1
    ).values

    return {

        "attack":
            attack_name,

        "mean_l2":
            l2.mean().item(),

        "median_l2":
            l2.median().item(),

        "max_l2":
            l2.max().item(),

        "mean_linf":
            linf.mean().item(),

        "median_linf":
            linf.median().item(),

        "max_linf":
            linf.max().item(),

        "fixed_feature_violations":
            fixed_violations
    }


details.append(
    calculate_perturbation_details(
        "C&W",
        X_test_raw_tensor,
        X_cw_raw,
        results_df.loc[
            results_df["attack"] == "C&W",
            "fixed_feature_violations"
        ].iloc[0]
    )
)

details.append(
    calculate_perturbation_details(
        "DeepFool",
        X_test_raw_tensor,
        X_deepfool_raw,
        results_df.loc[
            results_df["attack"] == "DeepFool",
            "fixed_feature_violations"
        ].iloc[0]
    )
)

details_df = pd.DataFrame(
    details
)

details_df.to_csv(
    DETAILS_FILE,
    index=False
)


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 70)
print("FINAL FEATURE SQUEEZING RESULTS")
print("=" * 70)

display_columns = [
    "attack",
    "original_attack_accuracy",
    "sanitized_accuracy",
    "accuracy_improvement",
    "precision",
    "recall",
    "f1",
    "accuracy_drop_from_clean_squeezed",
    "evasion_success_rate",
    "fixed_feature_violations"
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print()
print(
    "Results saved to:"
)

print(
    OUTPUT_FILE
)

print()

print(
    "Attack details saved to:"
)

print(
    DETAILS_FILE
)

print()

print("=" * 70)
print(
    "INPUT SANITIZATION / FEATURE SQUEEZING "
    "EVALUATION COMPLETED"
)
print("=" * 70)