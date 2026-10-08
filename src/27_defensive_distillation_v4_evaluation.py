# ============================================================
# 27_defensive_distillation_v4_evaluation.py
#
# FINAL V4 DEFENSIVE DISTILLATION EVALUATION
#
# Evaluate V4 Defensive Distillation against:
#   1. Clean
#   2. FGSM
#   3. PGD
#   4. Final C&W
#   5. Final DeepFool
#
# The V4 distilled model is NOT retrained.
#
# Final C&W:
#   Steps = 100
#   LR = 0.005
#   C = 1.0
#   Kappa = 0.0
#
# Final DeepFool:
#   Steps = 100
#   Overshoot = 0.10
#
# Uses:
#   - Same 1000-sample test subset
#   - Random seed 42
#   - default_rng(42)
#   - Same V4 scaler
#   - Same V4 student model
#   - Same 13 fixed features
#   - Raw-domain feature constraints
#
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import joblib

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

MODEL_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_distilled_student_v4.pth"
)

SCALER_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_distilled_student_scaler_v4.pkl"
)

# Do NOT overwrite the previous evaluation.
OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "results/defensive_distillation_v4_final_attack_evaluation.csv"
)

DETAILS_FILE = os.path.join(
    BASE_DIR,
    "results/defensive_distillation_v4_final_attack_evaluation_details.csv"
)


# ============================================================
# Evaluation settings
# ============================================================

N_SAMPLES = 1000

RANDOM_SEED = 42

BATCH_SIZE = 256


# ============================================================
# FGSM
# ============================================================

FGSM_EPSILON = 0.10


# ============================================================
# PGD
# ============================================================

PGD_EPSILON = 0.10

PGD_STEPS = 10

PGD_ALPHA = PGD_EPSILON / PGD_STEPS


# ============================================================
# FINAL C&W
# ============================================================

CW_STEPS = 100

CW_LR = 0.005

CW_C = 1.0

CW_KAPPA = 0.0


# ============================================================
# FINAL DEEPFOOL
# ============================================================

DEEPFOOL_STEPS = 100

DEEPFOOL_OVERSHOOT = 0.10


# ============================================================
# Fixed CICIDS-2017 features
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
# 2. REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)


# ============================================================
# 3. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 80)
print("FINAL V4 DEFENSIVE DISTILLATION EVALUATION")
print("=" * 80)

print(f"Device: {DEVICE}")
print(f"Number of test samples: {N_SAMPLES}")
print()


# ============================================================
# 4. MODEL
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
# 5. LOAD TEST DATA
# ============================================================

print("=" * 80)
print("LOADING TEST DATA")
print("=" * 80)

df = pd.read_csv(TEST_FILE)

print(
    f"Full test dataset: "
    f"{len(df):,} rows"
)

print(
    f"Total columns: "
    f"{len(df.columns)}"
)

if "Label" not in df.columns:

    raise ValueError(
        "ERROR: Label column not found."
    )


# ============================================================
# 6. FIX LABELS
# ============================================================

labels = df["Label"]

if labels.dtype == object:

    labels = (
        labels
        .astype(str)
        .str.strip()
        .str.upper()
        .map({
            "BENIGN": 0,
            "ATTACK": 1
        })
    )

labels = pd.to_numeric(
    labels,
    errors="coerce"
)

if labels.isna().any():

    raise ValueError(
        "ERROR: Some labels could not be converted."
    )

labels = labels.astype(np.int64)


# ============================================================
# 7. FIXED RANDOM TEST SUBSET
# ============================================================

print("=" * 80)
print("SELECTING FINAL FIXED TEST SUBSET")
print("=" * 80)

if len(df) < N_SAMPLES:

    raise ValueError(
        "Test dataset contains fewer than "
        f"{N_SAMPLES} samples."
    )


# IMPORTANT:
# Same method used in final multi-attack evaluation.

rng = np.random.default_rng(
    RANDOM_SEED
)

indices = rng.choice(
    len(df),
    size=N_SAMPLES,
    replace=False
)

indices = np.sort(indices)


X_test_df = (
    df.iloc[indices]
    .drop(columns=["Label"])
)

y_test = (
    labels.iloc[indices]
    .to_numpy(dtype=np.int64)
)


print(
    f"Selected samples: "
    f"{len(X_test_df)}"
)

print(
    f"BENIGN: "
    f"{(y_test == 0).sum()}"
)

print(
    f"ATTACK: "
    f"{(y_test == 1).sum()}"
)

print()


# ============================================================
# 8. LOAD V4 SCALER
# ============================================================

print("=" * 80)
print("LOADING V4 SCALER")
print("=" * 80)

scaler = joblib.load(
    SCALER_FILE
)

print(
    f"Scaler loaded:"
)

print(
    SCALER_FILE
)

print()


# ============================================================
# 9. LOAD BALANCED TRAINING DATA
# ============================================================
# Used only to obtain raw-domain feature bounds.
# ============================================================

print("=" * 80)
print("LOADING BALANCED TRAINING DATA")
print("=" * 80)

train_df = pd.read_csv(
    TRAIN_FILE
)

if "Label" not in train_df.columns:

    raise ValueError(
        "ERROR: Label column not found "
        "in balanced training data."
    )

X_train_df = train_df.drop(
    columns=["Label"]
)


print(
    f"Training rows: "
    f"{len(X_train_df):,}"
)

print(
    f"Training features: "
    f"{X_train_df.shape[1]}"
)


# ============================================================
# 10. VERIFY FEATURE ORDER
# ============================================================

if list(X_train_df.columns) != list(
    X_test_df.columns
):

    raise ValueError(
        "ERROR: Training and test feature "
        "columns/order do not match."
    )


# ============================================================
# 11. RAW-DOMAIN FEATURE BOUNDS
# ============================================================

print()
print("=" * 80)
print("CALCULATING RAW-DOMAIN FEATURE BOUNDS")
print("=" * 80)


raw_feature_min = (
    X_train_df
    .min(axis=0)
    .to_numpy(dtype=np.float32)
)

raw_feature_max = (
    X_train_df
    .max(axis=0)
    .to_numpy(dtype=np.float32)
)


# ============================================================
# 12. CONVERT RAW BOUNDS TO SCALED SPACE
# ============================================================

scale = np.asarray(
    scaler.scale_,
    dtype=np.float32
)

mean = np.asarray(
    scaler.mean_,
    dtype=np.float32
)


safe_scale = np.where(
    np.abs(scale) < 1e-12,
    1.0,
    scale
)


scaled_feature_min = (
    raw_feature_min - mean
) / safe_scale


scaled_feature_max = (
    raw_feature_max - mean
) / safe_scale


scaled_lower = np.minimum(
    scaled_feature_min,
    scaled_feature_max
)

scaled_upper = np.maximum(
    scaled_feature_min,
    scaled_feature_max
)


SCALED_LOWER = torch.tensor(
    scaled_lower,
    dtype=torch.float32,
    device=DEVICE
).view(1, -1)


SCALED_UPPER = torch.tensor(
    scaled_upper,
    dtype=torch.float32,
    device=DEVICE
).view(1, -1)


print(
    "Raw-domain bounds loaded successfully."
)

print()


# ============================================================
# 13. FIXED FEATURE INDICES
# ============================================================

print("=" * 80)
print("CHECKING FIXED FEATURES")
print("=" * 80)


feature_names = list(
    X_test_df.columns
)

fixed_indices = []


for feature_name in FIXED_FEATURE_NAMES:

    if feature_name not in feature_names:

        raise ValueError(
            f"ERROR: Fixed feature "
            f"'{feature_name}' not found."
        )

    fixed_indices.append(
        feature_names.index(feature_name)
    )


for name, idx in zip(
    FIXED_FEATURE_NAMES,
    fixed_indices
):

    print(
        f"{idx:>2}: {name}"
    )


print()


FIXED_INDICES = torch.tensor(
    fixed_indices,
    dtype=torch.long,
    device=DEVICE
)


# ============================================================
# 14. SCALE TEST DATA
# ============================================================

print("=" * 80)
print("SCALING TEST DATA")
print("=" * 80)

X_test_scaled = scaler.transform(
    X_test_df
)

X_test_scaled = np.asarray(
    X_test_scaled,
    dtype=np.float32
)


print(
    f"Scaled test shape: "
    f"{X_test_scaled.shape}"
)

print()


# ============================================================
# 15. TENSORS
# ============================================================

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32,
    device=DEVICE
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32,
    device=DEVICE
).view(-1, 1)


INPUT_DIM = X_test_scaled.shape[1]


# ============================================================
# 16. LOAD V4 DISTILLED STUDENT
# ============================================================

print("=" * 80)
print("LOADING V4 DISTILLED STUDENT")
print("=" * 80)

model = DNN(
    INPUT_DIM
).to(DEVICE)


model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()


print(
    "Model loaded:"
)

print(
    MODEL_FILE
)

print()


# ============================================================
# 17. HELPER FUNCTIONS
# ============================================================

def predict_labels(x):

    model.eval()

    predictions = []

    with torch.no_grad():

        for start in range(
            0,
            len(x),
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                len(x)
            )

            logits = model(
                x[start:end]
            )

            probs = torch.sigmoid(
                logits
            )

            preds = (
                probs >= 0.5
            ).long()

            predictions.append(
                preds.cpu()
            )


    return torch.cat(
        predictions
    ).numpy().reshape(-1)


# ============================================================
# Metrics
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

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


# ============================================================
# 18. ADVERSARIAL CONSTRAINT FUNCTION
# ============================================================

def constrain_adv(
    adv,
    original
):

    # --------------------------------------------------------
    # Apply standardized-domain bounds.
    # --------------------------------------------------------

    adv = torch.maximum(
        adv,
        SCALED_LOWER
    )

    adv = torch.minimum(
        adv,
        SCALED_UPPER
    )


    # --------------------------------------------------------
    # Restore fixed features.
    # --------------------------------------------------------

    adv = adv.clone()

    adv[:, FIXED_INDICES] = (
        original[:, FIXED_INDICES]
    )


    return adv


# ============================================================
# 19. CLEAN EVALUATION
# ============================================================

print("=" * 80)
print("CLEAN EVALUATION")
print("=" * 80)


clean_pred = predict_labels(
    X_test_tensor
)

clean_metrics = calculate_metrics(
    y_test,
    clean_pred
)

clean_accuracy = (
    clean_metrics["accuracy"]
)


print(
    f"Clean Accuracy: "
    f"{clean_accuracy * 100:.2f}%"
)

print(
    f"Clean Precision: "
    f"{clean_metrics['precision']:.6f}"
)

print(
    f"Clean Recall: "
    f"{clean_metrics['recall']:.6f}"
)

print(
    f"Clean F1: "
    f"{clean_metrics['f1']:.6f}"
)

print()


# ============================================================
# 20. FGSM
# ============================================================

def fgsm_attack(
    x,
    y,
    epsilon
):

    model.eval()

    x_adv = (
        x.clone()
        .detach()
    )

    x_adv.requires_grad_(True)

    logits = model(
        x_adv
    )

    loss = nn.BCEWithLogitsLoss()(
        logits,
        y
    )

    model.zero_grad()

    loss.backward()

    gradient = (
        x_adv.grad.sign()
    )

    x_adv = (
        x_adv
        +
        epsilon * gradient
    )

    x_adv = constrain_adv(
        x_adv.detach(),
        x
    )

    return x_adv.detach()


print("=" * 80)
print("FGSM ATTACK")
print("=" * 80)


X_fgsm = fgsm_attack(
    X_test_tensor,
    y_test_tensor,
    FGSM_EPSILON
)


fgsm_pred = predict_labels(
    X_fgsm
)

fgsm_metrics = calculate_metrics(
    y_test,
    fgsm_pred
)

fgsm_accuracy = (
    fgsm_metrics["accuracy"]
)


print(
    f"FGSM epsilon = "
    f"{FGSM_EPSILON:.2f}"
)

print(
    f"Accuracy: "
    f"{fgsm_accuracy * 100:.2f}%"
)

print()


# ============================================================
# 21. PGD
# ============================================================

def pgd_attack(
    x,
    y,
    epsilon,
    alpha,
    steps
):

    model.eval()

    original = (
        x.clone()
        .detach()
    )


    # Random start

    x_adv = (
        original
        +
        torch.empty_like(
            original
        ).uniform_(
            -epsilon,
            epsilon
        )
    )


    x_adv = constrain_adv(
        x_adv.detach(),
        original
    )


    for _ in range(steps):

        x_adv.requires_grad_(True)

        logits = model(
            x_adv
        )

        loss = nn.BCEWithLogitsLoss()(
            logits,
            y
        )

        model.zero_grad()

        loss.backward()

        gradient = (
            x_adv.grad.sign()
        )


        x_adv = (
            x_adv
            +
            alpha * gradient
        )


        perturbation = (
            x_adv - original
        )


        perturbation = torch.clamp(
            perturbation,
            -epsilon,
            epsilon
        )


        x_adv = (
            original
            +
            perturbation
        ).detach()


        x_adv = constrain_adv(
            x_adv,
            original
        )


    return x_adv.detach()


print("=" * 80)
print("PGD ATTACK")
print("=" * 80)


X_pgd = pgd_attack(
    X_test_tensor,
    y_test_tensor,
    PGD_EPSILON,
    PGD_ALPHA,
    PGD_STEPS
)


pgd_pred = predict_labels(
    X_pgd
)

pgd_metrics = calculate_metrics(
    y_test,
    pgd_pred
)

pgd_accuracy = (
    pgd_metrics["accuracy"]
)


print(
    f"PGD epsilon = "
    f"{PGD_EPSILON:.2f}"
)

print(
    f"Steps: {PGD_STEPS}"
)

print(
    f"Alpha: {PGD_ALPHA:.4f}"
)

print(
    f"Accuracy: "
    f"{pgd_accuracy * 100:.2f}%"
)

print()


# ============================================================
# 22. FINAL C&W
# ============================================================

def cw_attack(
    x,
    y,
    steps=100,
    lr=0.005,
    c=1.0,
    kappa=0.0
):

    model.eval()

    original = (
        x.clone()
        .detach()
    )


    # --------------------------------------------------------
    # Determine initially correctly classified samples.
    # --------------------------------------------------------

    with torch.no_grad():

        clean_logits = model(
            original
        )

        clean_pred = (
            torch.sigmoid(
                clean_logits
            ) >= 0.5
        ).long().view(-1)

        original_target = (
            y.long().view(-1)
        )

        attack_mask = (
            clean_pred
            ==
            original_target
        )


    attack_indices = torch.where(
        attack_mask
    )[0]


    n_attack = len(
        attack_indices
    )


    print(
        f"Initially correctly "
        f"classified samples: "
        f"{n_attack}/{len(x)}"
    )


    # If none are correctly classified,
    # there is nothing to attack.

    if n_attack == 0:

        return original.detach()


    # --------------------------------------------------------
    # IMPORTANT FIX:
    #
    # Work only with the 995 (or whatever number) attackable
    # samples.
    #
    # Do NOT compare them against the original 1000 labels.
    # --------------------------------------------------------

    attack_x = (
        original[attack_indices]
        .clone()
        .detach()
    )

    attack_y = (
        y[attack_indices]
        .clone()
        .detach()
        .view(-1)
    )


    # --------------------------------------------------------
    # Optimization variable
    # --------------------------------------------------------

    delta = torch.zeros_like(
        attack_x,
        requires_grad=True
    )


    optimizer = torch.optim.Adam(
        [delta],
        lr=lr
    )


    best_attack = (
        attack_x.clone()
    )


    best_l2 = torch.full(
        (n_attack,),
        float("inf"),
        device=DEVICE
    )


    # ========================================================
    # C&W OPTIMIZATION
    # ========================================================

    for step in range(steps):

        candidate = (
            attack_x
            +
            delta
        )


        # Apply domain and fixed-feature constraints.

        candidate = constrain_adv(
            candidate,
            attack_x
        )


        logits = model(
            candidate
        )


        # L2 perturbation.

        l2 = torch.sum(
            (
                candidate
                -
                attack_x
            ) ** 2,
            dim=1
        )


        # ----------------------------------------------------
        # Binary C&W objective.
        #
        # y=0:
        #   target is ATTACK
        #
        # y=1:
        #   target is BENIGN
        # ----------------------------------------------------

        target_direction = (
            2.0 * attack_y
            - 1.0
        )


        f = (
            target_direction
            *
            logits.view(-1)
        )


        attack_loss = torch.clamp(
            f + kappa,
            min=0.0
        )


        total_loss = (
            l2
            +
            c * attack_loss
        ).mean()


        optimizer.zero_grad()

        total_loss.backward()

        optimizer.step()


        # ====================================================
        # Check successful attacks
        # ====================================================

        with torch.no_grad():

            candidate = (
                attack_x
                +
                delta
            )


            candidate = constrain_adv(
                candidate,
                attack_x
            )


            candidate_logits = model(
                candidate
            )


            candidate_pred = (
                torch.sigmoid(
                    candidate_logits
                ) >= 0.5
            ).long().view(-1)


            # IMPORTANT:
            # candidate_pred has n_attack elements.
            # target also has n_attack elements.

            target = (
                attack_y.long()
                .view(-1)
            )


            successful = (
                candidate_pred
                !=
                target
            )


            current_l2 = torch.sum(
                (
                    candidate
                    -
                    attack_x
                ) ** 2,
                dim=1
            )


            improved = (
                successful
                &
                (
                    current_l2
                    <
                    best_l2
                )
            )


            if improved.any():

                best_attack[
                    improved
                ] = candidate[
                    improved
                ]


                best_l2[
                    improved
                ] = current_l2[
                    improved
                ]


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            step == 0
            or
            (step + 1) % 25 == 0
            or
            step == steps - 1
        ):

            with torch.no_grad():

                progress_pred = (
                    torch.sigmoid(
                        model(candidate)
                    ) >= 0.5
                ).long().view(-1)


                progress_success = (
                    progress_pred
                    !=
                    target
                ).sum().item()


            print(
                f"  C&W step "
                f"{step + 1:>3}/{steps} | "
                f"successful: "
                f"{progress_success}/{n_attack}"
            )


    # ========================================================
    # Restore the complete 1000-sample tensor.
    #
    # Initially incorrect samples remain unchanged.
    # ========================================================

    result = (
        original.clone()
    )


    result[
        attack_indices
    ] = best_attack


    return result.detach()


print("=" * 80)
print("FINAL C&W ATTACK")
print("=" * 80)

print(
    f"Steps: {CW_STEPS}"
)

print(
    f"Learning rate: {CW_LR}"
)

print(
    f"C: {CW_C}"
)

print(
    f"Kappa: {CW_KAPPA}"
)

print(
    "Attacking initially correctly "
    "classified samples only."
)

print()


X_cw = cw_attack(
    X_test_tensor,
    y_test_tensor,
    steps=CW_STEPS,
    lr=CW_LR,
    c=CW_C,
    kappa=CW_KAPPA
)


cw_pred = predict_labels(
    X_cw
)


cw_metrics = calculate_metrics(
    y_test,
    cw_pred
)


cw_accuracy = (
    cw_metrics["accuracy"]
)


print()

print(
    f"Final C&W Accuracy: "
    f"{cw_accuracy * 100:.2f}%"
)

print()


# ============================================================
# 23. FINAL DEEPFOOL
# ============================================================

def deepfool_attack(
    x,
    y,
    steps=100,
    overshoot=0.10
):

    model.eval()

    x_adv = (
        x.clone()
        .detach()
    )


    # --------------------------------------------------------
    # Determine initially correctly classified samples.
    # --------------------------------------------------------

    with torch.no_grad():

        clean_logits = model(
            x_adv
        )

        clean_pred = (
            torch.sigmoid(
                clean_logits
            ) >= 0.5
        ).long().view(-1)

        target = (
            y.long().view(-1)
        )


    attack_mask = (
        clean_pred
        ==
        target
    )


    attack_indices = torch.where(
        attack_mask
    )[0]


    print(
        f"Initially correctly "
        f"classified samples: "
        f"{len(attack_indices)}/{len(x)}"
    )


    # ========================================================
    # Attack each initially correctly classified sample.
    # ========================================================

    for count, idx in enumerate(
        attack_indices
    ):

        original_sample = (
            x[idx:idx + 1]
            .clone()
            .detach()
        )


        sample = (
            original_sample.clone()
        )


        original_label = int(
            target[idx].item()
        )


        for step in range(
            steps
        ):

            sample.requires_grad_(True)


            logit = (
                model(sample)
                .view(-1)[0]
            )


            current_prob = (
                torch.sigmoid(
                    logit
                ).item()
            )


            current_pred = int(
                current_prob >= 0.5
            )


            # Stop once the decision changes.

            if (
                current_pred
                !=
                original_label
            ):

                break


            gradient = torch.autograd.grad(
                logit,
                sample,
                retain_graph=False,
                create_graph=False
            )[0]


            # ------------------------------------------------
            # Binary decision-boundary formulation.
            # ------------------------------------------------

            y_sign = (
                2.0
                *
                float(original_label)
                -
                1.0
            )


            signed_logit = (
                y_sign
                *
                logit
            )


            signed_gradient = (
                y_sign
                *
                gradient
            )


            gradient_norm_sq = (
                torch.sum(
                    signed_gradient ** 2
                )
                +
                1e-12
            )


            distance = (
                torch.abs(
                    signed_logit
                )
                /
                gradient_norm_sq
            )


            perturbation = (
                -distance
                *
                signed_gradient
            )


            perturbation = (
                (1.0 + overshoot)
                *
                perturbation
            )


            # ------------------------------------------------
            # Fixed features cannot be changed.
            # ------------------------------------------------

            perturbation = (
                perturbation.clone()
            )


            perturbation[
                :,
                FIXED_INDICES
            ] = 0.0


            # ------------------------------------------------
            # Update.
            # ------------------------------------------------

            sample = (
                sample.detach()
                +
                perturbation.detach()
            )


            # ------------------------------------------------
            # Apply raw-domain/fixed-feature constraints.
            # ------------------------------------------------

            sample = constrain_adv(
                sample,
                original_sample
            )


        # ----------------------------------------------------
        # Save adversarial sample.
        # ----------------------------------------------------

        x_adv[idx] = (
            sample.squeeze(0)
        )


        # ----------------------------------------------------
        # Progress.
        # ----------------------------------------------------

        if (
            count == 0
            or
            (count + 1) % 100 == 0
            or
            count + 1 == len(
                attack_indices
            )
        ):

            print(
                f"  DeepFool progress: "
                f"{count + 1}/"
                f"{len(attack_indices)}"
            )


    return x_adv.detach()


print("=" * 80)
print("FINAL DEEPFOOL ATTACK")
print("=" * 80)

print(
    f"Steps: {DEEPFOOL_STEPS}"
)

print(
    f"Overshoot: "
    f"{DEEPFOOL_OVERSHOOT}"
)

print(
    "Binary decision-boundary formulation."
)

print(
    "Attacking initially correctly "
    "classified samples only."
)

print()


X_deepfool = deepfool_attack(
    X_test_tensor,
    y_test_tensor,
    steps=DEEPFOOL_STEPS,
    overshoot=DEEPFOOL_OVERSHOOT
)


deepfool_pred = predict_labels(
    X_deepfool
)


deepfool_metrics = calculate_metrics(
    y_test,
    deepfool_pred
)


deepfool_accuracy = (
    deepfool_metrics["accuracy"]
)


print()

print(
    f"Final DeepFool Accuracy: "
    f"{deepfool_accuracy * 100:.2f}%"
)

print()


# ============================================================
# 24. ATTACK SUCCESS
# ============================================================

def calculate_attack_success(
    clean_pred,
    adv_pred
):

    changed = (
        clean_pred
        !=
        adv_pred
    )

    return (
        100.0
        *
        np.mean(changed)
    )


def calculate_evasion_success(
    y_true,
    clean_pred,
    adv_pred
):

    clean_correct = (
        clean_pred
        ==
        y_true
    )


    successful_evasion = (
        clean_correct
        &
        (
            adv_pred
            !=
            y_true
        )
    )


    if np.sum(clean_correct) == 0:

        return 0.0


    return (
        100.0
        *
        np.sum(successful_evasion)
        /
        np.sum(clean_correct)
    )


# ============================================================
# Attack success
# ============================================================

fgsm_attack_success = (
    calculate_attack_success(
        clean_pred,
        fgsm_pred
    )
)


pgd_attack_success = (
    calculate_attack_success(
        clean_pred,
        pgd_pred
    )
)


cw_attack_success = (
    calculate_attack_success(
        clean_pred,
        cw_pred
    )
)


deepfool_attack_success = (
    calculate_attack_success(
        clean_pred,
        deepfool_pred
    )
)


# ============================================================
# Evasion success
# ============================================================

fgsm_evasion = (
    calculate_evasion_success(
        y_test,
        clean_pred,
        fgsm_pred
    )
)


pgd_evasion = (
    calculate_evasion_success(
        y_test,
        clean_pred,
        pgd_pred
    )
)


cw_evasion = (
    calculate_evasion_success(
        y_test,
        clean_pred,
        cw_pred
    )
)


deepfool_evasion = (
    calculate_evasion_success(
        y_test,
        clean_pred,
        deepfool_pred
    )
)


# ============================================================
# 25. ACCURACY DROPS
# ============================================================

fgsm_drop = (
    clean_accuracy
    -
    fgsm_accuracy
) * 100


pgd_drop = (
    clean_accuracy
    -
    pgd_accuracy
) * 100


cw_drop = (
    clean_accuracy
    -
    cw_accuracy
) * 100


deepfool_drop = (
    clean_accuracy
    -
    deepfool_accuracy
) * 100


# ============================================================
# 26. RESULTS TABLE
# ============================================================

results = [

    {
        "Attack": "Clean",
        "Accuracy":
            clean_metrics["accuracy"] * 100,
        "Precision":
            clean_metrics["precision"],
        "Recall":
            clean_metrics["recall"],
        "F1":
            clean_metrics["f1"],
        "Accuracy Drop":
            0.0,
        "Attack Success":
            np.nan,
        "Evasion Success":
            np.nan,
        "TN":
            clean_metrics["tn"],
        "FP":
            clean_metrics["fp"],
        "FN":
            clean_metrics["fn"],
        "TP":
            clean_metrics["tp"]
    },


    {
        "Attack": "FGSM epsilon=0.10",
        "Accuracy":
            fgsm_metrics["accuracy"] * 100,
        "Precision":
            fgsm_metrics["precision"],
        "Recall":
            fgsm_metrics["recall"],
        "F1":
            fgsm_metrics["f1"],
        "Accuracy Drop":
            fgsm_drop,
        "Attack Success":
            fgsm_attack_success,
        "Evasion Success":
            fgsm_evasion,
        "TN":
            fgsm_metrics["tn"],
        "FP":
            fgsm_metrics["fp"],
        "FN":
            fgsm_metrics["fn"],
        "TP":
            fgsm_metrics["tp"]
    },


    {
        "Attack": "PGD epsilon=0.10",
        "Accuracy":
            pgd_metrics["accuracy"] * 100,
        "Precision":
            pgd_metrics["precision"],
        "Recall":
            pgd_metrics["recall"],
        "F1":
            pgd_metrics["f1"],
        "Accuracy Drop":
            pgd_drop,
        "Attack Success":
            pgd_attack_success,
        "Evasion Success":
            pgd_evasion,
        "TN":
            pgd_metrics["tn"],
        "FP":
            pgd_metrics["fp"],
        "FN":
            pgd_metrics["fn"],
        "TP":
            pgd_metrics["tp"]
    },


    {
        "Attack": "C&W",
        "Accuracy":
            cw_metrics["accuracy"] * 100,
        "Precision":
            cw_metrics["precision"],
        "Recall":
            cw_metrics["recall"],
        "F1":
            cw_metrics["f1"],
        "Accuracy Drop":
            cw_drop,
        "Attack Success":
            cw_attack_success,
        "Evasion Success":
            cw_evasion,
        "TN":
            cw_metrics["tn"],
        "FP":
            cw_metrics["fp"],
        "FN":
            cw_metrics["fn"],
        "TP":
            cw_metrics["tp"]
    },


    {
        "Attack": "DeepFool",
        "Accuracy":
            deepfool_metrics["accuracy"] * 100,
        "Precision":
            deepfool_metrics["precision"],
        "Recall":
            deepfool_metrics["recall"],
        "F1":
            deepfool_metrics["f1"],
        "Accuracy Drop":
            deepfool_drop,
        "Attack Success":
            deepfool_attack_success,
        "Evasion Success":
            deepfool_evasion,
        "TN":
            deepfool_metrics["tn"],
        "FP":
            deepfool_metrics["fp"],
        "FN":
            deepfool_metrics["fn"],
        "TP":
            deepfool_metrics["tp"]
    }
]


results_df = pd.DataFrame(
    results
)


# ============================================================
# 27. PRINT RESULTS
# ============================================================

print()
print("=" * 110)
print("FINAL V4 DEFENSIVE DISTILLATION RESULTS")
print("=" * 110)


print(
    results_df[
        [
            "Attack",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Accuracy Drop",
            "Attack Success",
            "Evasion Success"
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.2f}"
    )
)


# ============================================================
# 28. CONFUSION MATRICES
# ============================================================

print()
print("=" * 80)
print("CONFUSION MATRICES")
print("=" * 80)


for _, row in results_df.iterrows():

    print()
    print(
        row["Attack"]
    )

    print(
        f"TN = {int(row['TN'])}"
    )

    print(
        f"FP = {int(row['FP'])}"
    )

    print(
        f"FN = {int(row['FN'])}"
    )

    print(
        f"TP = {int(row['TP'])}"
    )


# ============================================================
# 29. SAVE SUMMARY RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 30. SAVE SAMPLE-LEVEL DETAILS
# ============================================================

details_df = pd.DataFrame({

    "Original_Index":
        indices,

    "True_Label":
        y_test,

    "Clean_Prediction":
        clean_pred,

    "FGSM_Prediction":
        fgsm_pred,

    "PGD_Prediction":
        pgd_pred,

    "CW_Prediction":
        cw_pred,

    "DeepFool_Prediction":
        deepfool_pred
})


details_df.to_csv(
    DETAILS_FILE,
    index=False
)


# ============================================================
# 31. FINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("RESULTS SAVED")
print("=" * 80)

print(
    "Summary:"
)

print(
    OUTPUT_FILE
)

print()

print(
    "Details:"
)

print(
    DETAILS_FILE
)

print()

print("=" * 80)
print("FINAL ATTACK SETTINGS")
print("=" * 80)

print(
    f"C&W steps: {CW_STEPS}"
)

print(
    f"C&W learning rate: {CW_LR}"
)

print(
    f"C&W C: {CW_C}"
)

print(
    f"C&W kappa: {CW_KAPPA}"
)

print()

print(
    f"DeepFool steps: "
    f"{DEEPFOOL_STEPS}"
)

print(
    f"DeepFool overshoot: "
    f"{DEEPFOOL_OVERSHOOT}"
)

print()

print("=" * 80)
print("V4 FINAL ATTACK EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 80)