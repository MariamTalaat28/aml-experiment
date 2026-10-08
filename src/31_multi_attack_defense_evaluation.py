# ============================================================
# 31_multi_attack_defense_evaluation.py
# ============================================================
# FINAL MULTI-ATTACK ADVERSARIAL TRAINING EVALUATION
#
# Evaluates the final multi-attack adversarially trained model
# against:
#
#   1. Clean
#   2. FGSM
#   3. PGD
#   4. Final C&W
#   5. Final DeepFool
#
# IMPORTANT:
# - Attacks are generated against the FINAL defended model.
# - Uses the final C&W configuration:
#       steps = 100
#       learning rate = 0.005
#       C = 1.0
#       kappa = 0.0
#
# - Uses the final DeepFool configuration:
#       steps = 100
#       overshoot = 0.10
#
# - Uses the same 1000-sample test subset:
#       np.random.default_rng(42)
#
# - Uses the same fixed features and raw-domain constraints
#   used in the final attack implementations.
# ============================================================

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
# 1. CONFIGURATION
# ============================================================

BASE_DIR = "/home/ubuntu/aml-experiment"

TEST_FILE = os.path.join(
    BASE_DIR,
    "data/ml/test.csv"
)

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "data/ml/balanced_train.csv"
)

SCALER_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_balanced_scaler.pkl"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_multi_attack_final_adversarial_trained.pth"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "results/multi_attack_final_defense_evaluation.csv"
)

DETAILS_FILE = os.path.join(
    BASE_DIR,
    "results/multi_attack_final_defense_evaluation_details.csv"
)

# Evaluation subset
N_SAMPLES = 1000
RANDOM_SEED = 42

# FGSM
FGSM_EPSILON = 0.10

# PGD
PGD_EPSILON = 0.10
PGD_STEPS = 10
PGD_ALPHA = PGD_EPSILON / PGD_STEPS

# Final C&W
CW_STEPS = 100
CW_LR = 0.005
CW_C = 1.0
CW_KAPPA = 0.0

# Final DeepFool
DEEPFOOL_STEPS = 100
DEEPFOOL_OVERSHOOT = 0.10

# Batch size is not currently needed because the attacks
# operate on the selected 1000-sample evaluation subset.
BATCH_SIZE = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. HEADER
# ============================================================

print("=" * 70)
print("FINAL MULTI-ATTACK DEFENSE EVALUATION")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Test file: {TEST_FILE}")
print(f"Model: {MODEL_FILE}")


# ============================================================
# 3. MODEL DEFINITION
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
# 4. LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

test_df = pd.read_csv(TEST_FILE)

print(
    f"Total test samples: {len(test_df)}"
)

print(
    f"Total columns: {len(test_df.columns)}"
)

LABEL_COLUMN = "Label"

if LABEL_COLUMN not in test_df.columns:

    raise ValueError(
        f"'{LABEL_COLUMN}' column not found in test dataset."
    )


# Separate features and labels
X_test_df = test_df.drop(
    columns=[LABEL_COLUMN]
)

y_test = test_df[
    LABEL_COLUMN
].values.astype(np.float32)


# Convert features to numeric
X_test_df = X_test_df.apply(
    pd.to_numeric,
    errors="coerce"
)

if X_test_df.isna().any().any():

    print(
        "Warning: NaN values detected. "
        "Replacing NaN with 0."
    )

    X_test_df = X_test_df.fillna(0)


X_test_raw = X_test_df.values.astype(
    np.float32
)

print(
    f"Feature shape: {X_test_raw.shape}"
)


# ============================================================
# 5. LOAD SCALER
# ============================================================

print("\nLoading scaler...")

scaler = joblib.load(
    SCALER_FILE
)

X_test_scaled = scaler.transform(
    X_test_df
).astype(np.float32)


# ============================================================
# 6. SELECT SAME 1000-SAMPLE SUBSET
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

indices = rng.choice(
    len(X_test_scaled),
    size=N_SAMPLES,
    replace=False
)

X_subset_raw = X_test_raw[
    indices
]

X_subset = X_test_scaled[
    indices
]

y_subset = y_test[
    indices
]


print("\nEvaluation subset:")
print(
    f"Samples: {len(X_subset)}"
)

print(
    f"BENIGN: {(y_subset == 0).sum()}"
)

print(
    f"ATTACK: {(y_subset == 1).sum()}"
)


# ============================================================
# 7. LOAD FINAL MULTI-ATTACK MODEL
# ============================================================

print(
    "\nLoading final multi-attack-trained model..."
)

input_dim = X_subset.shape[1]

model = DNN(
    input_dim
).to(DEVICE)


checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=True
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


model.eval()

print(
    "Model loaded successfully."
)


# ============================================================
# 8. CREATE TENSORS
# ============================================================

X_tensor = torch.tensor(
    X_subset,
    dtype=torch.float32,
    device=DEVICE
)

y_tensor = torch.tensor(
    y_subset,
    dtype=torch.float32,
    device=DEVICE
).view(-1, 1)


# ============================================================
# 9. FIXED FEATURES
# ============================================================

FIXED_FEATURES = [
    30,  # Fwd PSH Flags
    31,  # Bwd PSH Flags
    32,  # Fwd URG Flags
    33,  # Bwd URG Flags
    43,  # FIN Flag Count
    44,  # SYN Flag Count
    45,  # RST Flag Count
    46,  # PSH Flag Count
    47,  # ACK Flag Count
    48,  # URG Flag Count
    49,  # CWE Flag Count
    50,  # ECE Flag Count
    51   # Down/Up Ratio
]

print(
    f"\nFixed features: {len(FIXED_FEATURES)}"
)


# ============================================================
# 10. LOAD TRAINING DATA FOR RAW-DOMAIN BOUNDS
# ============================================================

print(
    "\nLoading training data for feature bounds..."
)

train_df = pd.read_csv(
    TRAIN_FILE
)

X_train_df = train_df.drop(
    columns=[LABEL_COLUMN]
)

X_train_df = X_train_df.apply(
    pd.to_numeric,
    errors="coerce"
)

X_train_df = X_train_df.fillna(0)

X_train_raw = X_train_df.values.astype(
    np.float32
)


# Raw minimum and maximum values
raw_min = np.min(
    X_train_raw,
    axis=0
)

raw_max = np.max(
    X_train_raw,
    axis=0
)


# Transform bounds into standardized space
scaled_min = scaler.transform(
    pd.DataFrame(
        raw_min.reshape(1, -1),
        columns=X_train_df.columns
    )
)[0]

scaled_max = scaler.transform(
    pd.DataFrame(
        raw_max.reshape(1, -1),
        columns=X_train_df.columns
    )
)[0]


lower_bounds = torch.tensor(
    np.minimum(
        scaled_min,
        scaled_max
    ),
    dtype=torch.float32,
    device=DEVICE
)

upper_bounds = torch.tensor(
    np.maximum(
        scaled_min,
        scaled_max
    ),
    dtype=torch.float32,
    device=DEVICE
)


# ============================================================
# 11. HELPER: PREDICTIONS
# ============================================================

def predict_labels(
    model,
    X
):

    model.eval()

    with torch.no_grad():

        logits = model(X)

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= 0.5
        ).float()

    return predictions.cpu().numpy().reshape(-1)


# ============================================================
# 12. HELPER: METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    name
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

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "Evaluation": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


# ============================================================
# 13. CLEAN EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("CLEAN EVALUATION")
print("=" * 70)

clean_preds = predict_labels(
    model,
    X_tensor
)

clean_result = calculate_metrics(
    y_subset,
    clean_preds,
    "Clean"
)

print(
    f"Clean Accuracy: "
    f"{clean_result['Accuracy'] * 100:.4f}%"
)

print(
    f"Precision: "
    f"{clean_result['Precision']:.6f}"
)

print(
    f"Recall: "
    f"{clean_result['Recall']:.6f}"
)

print(
    f"F1: "
    f"{clean_result['F1']:.6f}"
)


# ============================================================
# 14. FGSM
# ============================================================

def generate_fgsm(
    model,
    X,
    y,
    epsilon
):

    X_adv = X.clone().detach()

    X_adv.requires_grad = True

    model.zero_grad()

    logits = model(
        X_adv
    )

    loss_fn = nn.BCEWithLogitsLoss()

    loss = loss_fn(
        logits,
        y
    )

    loss.backward()

    gradient = X_adv.grad.sign()

    X_adv = (
        X_adv.detach()
        + epsilon * gradient
    )

    # Apply feature bounds
    X_adv = torch.max(
        torch.min(
            X_adv,
            upper_bounds
        ),
        lower_bounds
    )

    # Restore fixed features
    X_adv[:, FIXED_FEATURES] = (
        X[:, FIXED_FEATURES]
    )

    return X_adv.detach()


print("\n" + "=" * 70)
print("FGSM")
print("=" * 70)

X_fgsm = generate_fgsm(
    model,
    X_tensor,
    y_tensor,
    FGSM_EPSILON
)

fgsm_preds = predict_labels(
    model,
    X_fgsm
)

fgsm_result = calculate_metrics(
    y_subset,
    fgsm_preds,
    "FGSM"
)

print(
    f"FGSM Accuracy: "
    f"{fgsm_result['Accuracy'] * 100:.4f}%"
)


# ============================================================
# 15. PGD
# ============================================================

def generate_pgd(
    model,
    X,
    y,
    epsilon,
    alpha,
    steps
):

    X_original = X.clone().detach()

    # Random start
    random_noise = torch.empty_like(
        X_original
    ).uniform_(
        -epsilon,
        epsilon
    )

    X_adv = (
        X_original
        + random_noise
    )

    X_adv = torch.max(
        torch.min(
            X_adv,
            upper_bounds
        ),
        lower_bounds
    )

    # Restore fixed features
    X_adv[:, FIXED_FEATURES] = (
        X_original[:, FIXED_FEATURES]
    )

    loss_fn = nn.BCEWithLogitsLoss()

    for step in range(steps):

        X_adv.requires_grad = True

        model.zero_grad()

        logits = model(
            X_adv
        )

        loss = loss_fn(
            logits,
            y
        )

        loss.backward()

        gradient = X_adv.grad.sign()

        X_adv = (
            X_adv.detach()
            + alpha * gradient
        )

        # Project into epsilon ball
        delta = torch.clamp(
            X_adv - X_original,
            -epsilon,
            epsilon
        )

        X_adv = (
            X_original
            + delta
        )

        # Apply feature bounds
        X_adv = torch.max(
            torch.min(
                X_adv,
                upper_bounds
            ),
            lower_bounds
        )

        # Restore fixed features
        X_adv[:, FIXED_FEATURES] = (
            X_original[:, FIXED_FEATURES]
        )

    return X_adv.detach()


print("\n" + "=" * 70)
print("PGD")
print("=" * 70)

X_pgd = generate_pgd(
    model,
    X_tensor,
    y_tensor,
    PGD_EPSILON,
    PGD_ALPHA,
    PGD_STEPS
)

pgd_preds = predict_labels(
    model,
    X_pgd
)

pgd_result = calculate_metrics(
    y_subset,
    pgd_preds,
    "PGD"
)

print(
    f"PGD Accuracy: "
    f"{pgd_result['Accuracy'] * 100:.4f}%"
)


# ============================================================
# 16. FINAL C&W
# ============================================================

def generate_cw(
    model,
    X,
    y,
    steps=100,
    lr=0.005,
    c=1.0,
    kappa=0.0
):

    X_original = X.clone().detach()

    X_result = X_original.clone().detach()

    # --------------------------------------------------------
    # Determine initially correctly classified samples
    # --------------------------------------------------------

    with torch.no_grad():

        original_logits = (
            model(
                X_original
            ).view(-1)
        )

        original_preds = (
            torch.sigmoid(
                original_logits
            ) >= 0.5
        ).float()

    y_flat = y.view(-1)

    attack_mask = (
        original_preds == y_flat
    )

    print(
        f"C&W attacking "
        f"{attack_mask.sum().item()} "
        f"/ {len(y_flat)} initially correct samples"
    )

    if attack_mask.sum().item() == 0:

        return X_result


    # --------------------------------------------------------
    # Select initially correct samples
    # --------------------------------------------------------

    X_attack = (
        X_original[
            attack_mask
        ]
        .clone()
        .detach()
    )

    y_attack = (
        y_flat[
            attack_mask
        ]
        .clone()
        .detach()
        .view(-1)
    )


    # --------------------------------------------------------
    # Optimize perturbation
    # --------------------------------------------------------

    delta = torch.zeros_like(
        X_attack,
        requires_grad=True
    )

    optimizer = torch.optim.Adam(
        [delta],
        lr=lr
    )


    best_adv = (
        X_attack
        .clone()
        .detach()
    )

    best_norm = torch.full(
        (X_attack.shape[0],),
        float("inf"),
        dtype=torch.float32,
        device=DEVICE
    )


    # --------------------------------------------------------
    # C&W optimization
    # --------------------------------------------------------

    for step in range(steps):

        optimizer.zero_grad()

        candidate = (
            X_attack
            + delta
        )

        # Differentiable standardized bounds
        candidate = torch.max(
            torch.min(
                candidate,
                upper_bounds
            ),
            lower_bounds
        )

        # Restore fixed features
        candidate[:, FIXED_FEATURES] = (
            X_attack[:, FIXED_FEATURES]
        )

        candidate_logits = (
            model(
                candidate
            ).view(-1)
        )

        # Make sure labels are [N]
        y_signed = (
            2.0 * y_attack
            - 1.0
        ).view(-1)

        # C&W decision function
        f = (
            y_signed
            * candidate_logits
        )

        attack_loss = torch.clamp(
            f + kappa,
            min=0.0
        )

        l2 = torch.sum(
            delta ** 2,
            dim=1
        )

        total_loss = torch.mean(
            l2
            + c * attack_loss
        )

        total_loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # Evaluate current candidate
        # ----------------------------------------------------

        with torch.no_grad():

            candidate = (
                X_attack
                + delta
            )

            candidate = torch.max(
                torch.min(
                    candidate,
                    upper_bounds
                ),
                lower_bounds
            )

            candidate[:, FIXED_FEATURES] = (
                X_attack[:, FIXED_FEATURES]
            )

            candidate_logits = (
                model(
                    candidate
                ).view(-1)
            )

            candidate_preds = (
                torch.sigmoid(
                    candidate_logits
                ) >= 0.5
            ).float().view(-1)

            # IMPORTANT:
            # Both tensors are explicitly [N]
            successful = (
                candidate_preds
                != y_attack
            ).view(-1)

            current_norm = torch.sum(
                (
                    candidate
                    - X_attack
                ) ** 2,
                dim=1
            ).view(-1)

            improved = (
                successful
                & (
                    current_norm
                    < best_norm
                )
            )

            best_adv[improved] = (
                candidate[improved]
            )

            best_norm[improved] = (
                current_norm[improved]
            )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            step == 0
            or (step + 1) % 20 == 0
            or step == steps - 1
        ):

            success_count = (
                torch.isfinite(
                    best_norm
                )
                .sum()
                .item()
            )

            print(
                f"  C&W step "
                f"{step + 1}/{steps} | "
                f"successful: "
                f"{success_count}"
            )


    # --------------------------------------------------------
    # Put optimized samples back
    # --------------------------------------------------------

    X_result[
        attack_mask
    ] = best_adv


    # --------------------------------------------------------
    # Exact raw-domain projection
    # --------------------------------------------------------

    attack_indices = torch.where(
        attack_mask
    )[0].cpu().numpy()

    best_adv_raw = (
        scaler.inverse_transform(
            best_adv.cpu().numpy()
        )
    )

    original_raw = (
        X_subset_raw[
            attack_indices
        ]
    )

    # Clip to observed raw-domain limits
    best_adv_raw = np.minimum(
        np.maximum(
            best_adv_raw,
            raw_min
        ),
        raw_max
    )

    # Restore fixed raw features
    best_adv_raw[
        :,
        FIXED_FEATURES
    ] = (
        original_raw[
            :,
            FIXED_FEATURES
        ]
    )

    # Transform back to standardized space
    best_adv_scaled = scaler.transform(
        pd.DataFrame(
            best_adv_raw,
            columns=X_test_df.columns
        )
    ).astype(np.float32)

    best_adv_tensor = torch.tensor(
        best_adv_scaled,
        dtype=torch.float32,
        device=DEVICE
    )

    X_result[
        attack_mask
    ] = best_adv_tensor

    return X_result.detach()


print("\n" + "=" * 70)
print("FINAL C&W")
print("=" * 70)

X_cw = generate_cw(
    model,
    X_tensor,
    y_tensor,
    steps=CW_STEPS,
    lr=CW_LR,
    c=CW_C,
    kappa=CW_KAPPA
)

cw_preds = predict_labels(
    model,
    X_cw
)

cw_result = calculate_metrics(
    y_subset,
    cw_preds,
    "C&W"
)

print(
    f"C&W Accuracy: "
    f"{cw_result['Accuracy'] * 100:.4f}%"
)

print(
    f"C&W Precision: "
    f"{cw_result['Precision']:.6f}"
)

print(
    f"C&W Recall: "
    f"{cw_result['Recall']:.6f}"
)

print(
    f"C&W F1: "
    f"{cw_result['F1']:.6f}"
)


# ============================================================
# 17. FINAL DEEPFOOL
# ============================================================

def generate_deepfool(
    model,
    X,
    y,
    steps=100,
    overshoot=0.10
):

    X_original = (
        X.clone().detach()
    )

    X_result = (
        X_original.clone().detach()
    )

    y_flat = y.view(-1)

    # --------------------------------------------------------
    # Determine initially correct samples
    # --------------------------------------------------------

    with torch.no_grad():

        original_logits = (
            model(
                X_original
            ).view(-1)
        )

        original_preds = (
            torch.sigmoid(
                original_logits
            ) >= 0.5
        ).float()

    attack_mask = (
        original_preds
        == y_flat
    )

    attack_indices = torch.where(
        attack_mask
    )[0]

    print(
        f"DeepFool attacking "
        f"{attack_indices.numel()} "
        f"/ {len(y_flat)} initially correct samples"
    )

    if attack_indices.numel() == 0:

        return X_result


    # --------------------------------------------------------
    # Attack each initially correct sample
    # --------------------------------------------------------

    for count, idx in enumerate(
        attack_indices
    ):

        x_current = (
            X_original[idx]
            .clone()
            .detach()
        )

        true_label = int(
            y_flat[idx].item()
        )


        for step in range(steps):

            x_current.requires_grad = True

            model.zero_grad()

            logit = (
                model(
                    x_current.unsqueeze(0)
                )
                .view(-1)[0]
            )

            # ------------------------------------------------
            # Binary decision-boundary formulation
            # ------------------------------------------------

            sign = (
                2 * true_label
                - 1
            )

            signed_logit = (
                sign
                * logit
            )

            signed_gradient = (
                torch.autograd.grad(
                    signed_logit,
                    x_current,
                    retain_graph=False,
                    create_graph=False
                )[0]
            )

            grad_norm_sq = torch.sum(
                signed_gradient ** 2
            )

            if (
                grad_norm_sq.item()
                < 1e-12
            ):

                break


            distance = (
                torch.abs(
                    signed_logit
                )
                / grad_norm_sq
            )

            perturbation = (
                -distance
                * signed_gradient
            )

            perturbation = (
                (1.0 + overshoot)
                * perturbation
            )


            # ------------------------------------------------
            # Update point
            # ------------------------------------------------

            with torch.no_grad():

                x_current = (
                    x_current.detach()
                    + perturbation.detach()
                )

                # Apply standardized bounds
                x_current = torch.max(
                    torch.min(
                        x_current,
                        upper_bounds
                    ),
                    lower_bounds
                )

                # Restore fixed features
                x_current[
                    FIXED_FEATURES
                ] = (
                    X_original[
                        idx,
                        FIXED_FEATURES
                    ]
                )


                # Check current prediction
                current_logit = (
                    model(
                        x_current.unsqueeze(0)
                    )
                    .view(-1)[0]
                )

                current_pred = int(
                    (
                        torch.sigmoid(
                            current_logit
                        )
                        >= 0.5
                    ).item()
                )


            # Stop after successful crossing
            if (
                current_pred
                != true_label
            ):

                break


        X_result[
            idx
        ] = x_current.detach()


        # Progress every 100 samples
        if (
            (count + 1) % 100 == 0
            or count == attack_indices.numel() - 1
        ):

            print(
                f"  DeepFool progress: "
                f"{count + 1}/"
                f"{attack_indices.numel()}"
            )


    # --------------------------------------------------------
    # Exact raw-domain projection
    # --------------------------------------------------------

    attack_indices_np = (
        attack_indices
        .cpu()
        .numpy()
    )

    X_adv_raw = (
        scaler.inverse_transform(
            X_result[
                attack_indices
            ]
            .cpu()
            .numpy()
        )
    )

    original_raw = (
        X_subset_raw[
            attack_indices_np
        ]
    )

    # Raw bounds
    X_adv_raw = np.minimum(
        np.maximum(
            X_adv_raw,
            raw_min
        ),
        raw_max
    )

    # Restore fixed features
    X_adv_raw[
        :,
        FIXED_FEATURES
    ] = (
        original_raw[
            :,
            FIXED_FEATURES
        ]
    )

    # Transform back to standardized space
    X_adv_scaled = scaler.transform(
        pd.DataFrame(
            X_adv_raw,
            columns=X_test_df.columns
        )
    ).astype(np.float32)

    X_result[
        attack_indices
    ] = torch.tensor(
        X_adv_scaled,
        dtype=torch.float32,
        device=DEVICE
    )

    return X_result.detach()


print("\n" + "=" * 70)
print("FINAL DEEPFOOL")
print("=" * 70)

X_deepfool = generate_deepfool(
    model,
    X_tensor,
    y_tensor,
    steps=DEEPFOOL_STEPS,
    overshoot=DEEPFOOL_OVERSHOOT
)

deepfool_preds = predict_labels(
    model,
    X_deepfool
)

deepfool_result = calculate_metrics(
    y_subset,
    deepfool_preds,
    "DeepFool"
)

print(
    f"DeepFool Accuracy: "
    f"{deepfool_result['Accuracy'] * 100:.4f}%"
)

print(
    f"DeepFool Precision: "
    f"{deepfool_result['Precision']:.6f}"
)

print(
    f"DeepFool Recall: "
    f"{deepfool_result['Recall']:.6f}"
)

print(
    f"DeepFool F1: "
    f"{deepfool_result['F1']:.6f}"
)


# ============================================================
# 18. CALCULATE ACCURACY DROP
# ============================================================

clean_accuracy = (
    clean_result["Accuracy"]
)

all_results = [
    clean_result,
    fgsm_result,
    pgd_result,
    cw_result,
    deepfool_result
]

for result in all_results:

    result["Accuracy_Drop_pp"] = (
        clean_accuracy
        - result["Accuracy"]
    ) * 100


# ============================================================
# 19. FINAL RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    all_results
)

print("\n" + "=" * 70)
print("FINAL MULTI-ATTACK DEFENSE RESULTS")
print("=" * 70)

display_columns = [
    "Evaluation",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "Accuracy_Drop_pp",
    "TN",
    "FP",
    "FN",
    "TP"
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# 20. SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nResults saved to:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# 21. SAVE PREDICTION DETAILS
# ============================================================

details_df = pd.DataFrame({
    "Index": indices,
    "True_Label": y_subset.astype(int),
    "Clean_Prediction": clean_preds.astype(int),
    "FGSM_Prediction": fgsm_preds.astype(int),
    "PGD_Prediction": pgd_preds.astype(int),
    "CW_Prediction": cw_preds.astype(int),
    "DeepFool_Prediction": deepfool_preds.astype(int)
})

details_df.to_csv(
    DETAILS_FILE,
    index=False
)

print(
    f"Prediction details saved to:"
)

print(
    DETAILS_FILE
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

for result in all_results:

    print(
        f"{result['Evaluation']:10s} | "
        f"Accuracy: "
        f"{result['Accuracy'] * 100:8.4f}% | "
        f"Drop: "
        f"{result['Accuracy_Drop_pp']:8.4f} pp"
    )

print("=" * 70)
print(
    "FINAL MULTI-ATTACK DEFENSE EVALUATION COMPLETED"
)
print("=" * 70)