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
from sklearn.model_selection import train_test_split


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
    "attack_validation_results.csv"
)

# Fixed validation sample
VALIDATION_SIZE = 1000
RANDOM_STATE = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# ATTACK PARAMETERS
# ============================================================

# FGSM
FGSM_EPSILONS = [0.01, 0.05, 0.10]

# PGD
PGD_EPSILONS = [0.01, 0.05, 0.10]
PGD_STEPS = 10

# C&W
CW_STEPS = 30
CW_LR = 0.01
CW_C = 1.0
CW_KAPPA = 0.0

# DeepFool
DEEPFOOL_MAX_ITER = 10
DEEPFOOL_OVERSHOOT = 0.02


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
# LOAD MODEL
# ============================================================

print("=" * 70)
print("ATTACK VALIDATION")
print("=" * 70)

print(f"Device: {DEVICE}")

print("\nLoading model...")

model = DNN(input_dim=78)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
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

print("Model loaded successfully.")


# ============================================================
# LOAD SCALER
# ============================================================

print("\nLoading scaler...")

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
    f"Full test set: {len(df):,} samples"
)


# ============================================================
# FEATURES / LABEL
# ============================================================

X = df.drop(
    columns=["Label"]
).values.astype(
    np.float32
)

y = df["Label"].values.astype(
    np.int64
)


# ============================================================
# CREATE FIXED STRATIFIED VALIDATION SUBSET
# ============================================================

print(
    f"\nSelecting {VALIDATION_SIZE} samples "
    f"for validation..."
)

X_validation, _, y_validation, _ = train_test_split(
    X,
    y,
    train_size=VALIDATION_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print(
    f"Validation samples: "
    f"{len(X_validation):,}"
)

print(
    f"Validation BENIGN: "
    f"{np.sum(y_validation == 0):,}"
)

print(
    f"Validation ATTACK: "
    f"{np.sum(y_validation == 1):,}"
)


# ============================================================
# SCALE VALIDATION DATA
# ============================================================

X_validation = scaler.transform(
    X_validation
).astype(
    np.float32
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def predict(model, x):

    model.eval()

    with torch.no_grad():

        logits = model(x).squeeze(1)

        predictions = (
            logits >= 0
        ).long()

    return predictions


def calculate_metrics(
    y_true,
    predictions
):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
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

X_clean = torch.tensor(
    X_validation,
    dtype=torch.float32,
    device=DEVICE
)

y_clean = torch.tensor(
    y_validation,
    dtype=torch.long,
    device=DEVICE
)

clean_predictions = predict(
    model,
    X_clean
)

clean_predictions_np = (
    clean_predictions.cpu().numpy()
)

clean_accuracy = accuracy_score(
    y_validation,
    clean_predictions_np
)

print(
    f"Clean accuracy: "
    f"{clean_accuracy * 100:.4f}%"
)


# ============================================================
# RESULTS STORAGE
# ============================================================

results = []


# ============================================================
# FGSM
# ============================================================

def fgsm_attack(
    model,
    x,
    y,
    epsilon
):

    x_adv = x.detach().clone()

    x_adv.requires_grad_(True)

    logits = model(
        x_adv
    ).squeeze(1)

    loss = nn.BCEWithLogitsLoss()(
        logits,
        y.float()
    )

    model.zero_grad()

    loss.backward()

    gradient = x_adv.grad.data

    x_adv = (
        x_adv.detach()
        + epsilon * gradient.sign()
    )

    return x_adv.detach()


print("\n" + "=" * 70)
print("FGSM VALIDATION")
print("=" * 70)


for epsilon in FGSM_EPSILONS:

    print(
        f"\nTesting FGSM epsilon = {epsilon}"
    )

    X_adv = fgsm_attack(
        model,
        X_clean,
        y_clean,
        epsilon
    )

    predictions = predict(
        model,
        X_adv
    )

    predictions_np = (
        predictions.cpu().numpy()
    )

    accuracy, precision, recall, f1 = (
        calculate_metrics(
            y_validation,
            predictions_np
        )
    )

    originally_correct = (
        clean_predictions_np ==
        y_validation
    )

    successful_evasion = (
        originally_correct &
        (predictions_np != y_validation)
    )

    clean_correct_count = np.sum(
        originally_correct
    )

    evasion_rate = (
        np.sum(successful_evasion)
        / clean_correct_count
        if clean_correct_count > 0
        else 0
    )

    perturbation = (
        X_adv - X_clean
    )

    mean_l2 = torch.norm(
        perturbation,
        p=2,
        dim=1
    ).mean().item()

    results.append({

        "attack": "FGSM",

        "parameter": f"epsilon={epsilon}",

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "accuracy_drop":
            clean_accuracy - accuracy,

        "evasion_rate":
            evasion_rate,

        "successful_attacks":
            int(np.sum(successful_evasion)),

        "clean_correct":
            int(clean_correct_count),

        "mean_l2":
            mean_l2
    })

    print(
        f"Accuracy: "
        f"{accuracy * 100:.4f}%"
    )

    print(
        f"Evasion rate: "
        f"{evasion_rate * 100:.4f}%"
    )


# ============================================================
# PGD
# ============================================================

def pgd_attack(
    model,
    x,
    y,
    epsilon,
    steps=10
):

    alpha = epsilon / steps

    # Random start inside epsilon ball
    x_adv = (
        x.detach()
        + torch.empty_like(x).uniform_(
            -epsilon,
            epsilon
        )
    )

    x_adv = torch.max(
        torch.min(
            x_adv,
            x + epsilon
        ),
        x - epsilon
    )

    for _ in range(steps):

        x_adv.requires_grad_(True)

        logits = model(
            x_adv
        ).squeeze(1)

        loss = nn.BCEWithLogitsLoss()(
            logits,
            y.float()
        )

        model.zero_grad()

        loss.backward()

        gradient = x_adv.grad.data

        x_adv = (
            x_adv.detach()
            + alpha * gradient.sign()
        )

        # Project back into epsilon ball
        perturbation = (
            x_adv - x
        )

        perturbation = torch.clamp(
            perturbation,
            -epsilon,
            epsilon
        )

        x_adv = (
            x + perturbation
        ).detach()

    return x_adv


print("\n" + "=" * 70)
print("PGD VALIDATION")
print("=" * 70)


for epsilon in PGD_EPSILONS:

    print(
        f"\nTesting PGD epsilon = {epsilon}"
    )

    X_adv = pgd_attack(
        model,
        X_clean,
        y_clean,
        epsilon,
        PGD_STEPS
    )

    predictions = predict(
        model,
        X_adv
    )

    predictions_np = (
        predictions.cpu().numpy()
    )

    accuracy, precision, recall, f1 = (
        calculate_metrics(
            y_validation,
            predictions_np
        )
    )

    originally_correct = (
        clean_predictions_np ==
        y_validation
    )

    successful_evasion = (
        originally_correct &
        (predictions_np != y_validation)
    )

    clean_correct_count = np.sum(
        originally_correct
    )

    evasion_rate = (
        np.sum(successful_evasion)
        / clean_correct_count
        if clean_correct_count > 0
        else 0
    )

    perturbation = (
        X_adv - X_clean
    )

    mean_l2 = torch.norm(
        perturbation,
        p=2,
        dim=1
    ).mean().item()

    results.append({

        "attack": "PGD",

        "parameter":
            f"epsilon={epsilon},steps={PGD_STEPS}",

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "accuracy_drop":
            clean_accuracy - accuracy,

        "evasion_rate":
            evasion_rate,

        "successful_attacks":
            int(np.sum(successful_evasion)),

        "clean_correct":
            int(clean_correct_count),

        "mean_l2":
            mean_l2
    })

    print(
        f"Accuracy: "
        f"{accuracy * 100:.4f}%"
    )

    print(
        f"Evasion rate: "
        f"{evasion_rate * 100:.4f}%"
    )


# ============================================================
# C&W
# ============================================================

def cw_attack(
    model,
    x,
    y,
    steps=30,
    learning_rate=0.01,
    c=1.0,
    kappa=0.0
):

    model.eval()

    delta = torch.zeros_like(
        x,
        requires_grad=True
    )

    optimizer = torch.optim.Adam(
        [delta],
        lr=learning_rate
    )

    best_adv = x.detach().clone()

    best_l2 = torch.full(
        (x.shape[0],),
        float("inf"),
        device=x.device
    )

    for _ in range(steps):

        adv = x + delta

        logits = model(
            adv
        ).squeeze(1)

        # C&W binary classification loss
        #
        # y = 1:
        #   force logit <= -kappa
        #
        # y = 0:
        #   force logit >= +kappa

        signed_logits = (
            (2 * y.float() - 1)
            * logits
        )

        classification_loss = torch.clamp(
            signed_logits + kappa,
            min=0
        )

        l2 = torch.sum(
            delta ** 2,
            dim=1
        )

        loss = (
            l2
            + c * classification_loss
        ).mean()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Check successful attacks
        # ----------------------------------------------------

        with torch.no_grad():

            current_logits = model(
                x + delta
            ).squeeze(1)

            current_predictions = (
                current_logits >= 0
            ).long()

            successful = (
                current_predictions != y
            )

            current_l2 = torch.sum(
                delta ** 2,
                dim=1
            )

            better = (
                successful &
                (current_l2 < best_l2)
            )

            if better.any():

                best_adv[better] = (
                    x + delta
                )[better].detach()

                best_l2[better] = (
                    current_l2[better]
                )

    return best_adv.detach()


print("\n" + "=" * 70)
print("C&W VALIDATION")
print("=" * 70)

print(
    "\nTesting C&W with "
    f"steps={CW_STEPS}, "
    f"lr={CW_LR}, "
    f"C={CW_C}, "
    f"kappa={CW_KAPPA}"
)

X_adv = cw_attack(
    model,
    X_clean,
    y_clean,
    steps=CW_STEPS,
    learning_rate=CW_LR,
    c=CW_C,
    kappa=CW_KAPPA
)

predictions = predict(
    model,
    X_adv
)

predictions_np = (
    predictions.cpu().numpy()
)

accuracy, precision, recall, f1 = (
    calculate_metrics(
        y_validation,
        predictions_np
    )
)

originally_correct = (
    clean_predictions_np ==
    y_validation
)

successful_evasion = (
    originally_correct &
    (predictions_np != y_validation)
)

clean_correct_count = np.sum(
    originally_correct
)

evasion_rate = (
    np.sum(successful_evasion)
    / clean_correct_count
    if clean_correct_count > 0
    else 0
)

perturbation = (
    X_adv - X_clean
)

mean_l2 = torch.norm(
    perturbation,
    p=2,
    dim=1
).mean().item()

results.append({

    "attack": "C&W",

    "parameter":
        f"steps={CW_STEPS},lr={CW_LR},C={CW_C},kappa={CW_KAPPA}",

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "f1": f1,

    "accuracy_drop":
        clean_accuracy - accuracy,

    "evasion_rate":
        evasion_rate,

    "successful_attacks":
        int(np.sum(successful_evasion)),

    "clean_correct":
        int(clean_correct_count),

    "mean_l2":
        mean_l2
})

print(
    f"Accuracy: "
    f"{accuracy * 100:.4f}%"
)

print(
    f"Evasion rate: "
    f"{evasion_rate * 100:.4f}%"
)


# ============================================================
# DEEPFOOL
# ============================================================

def deepfool_attack(
    model,
    x,
    max_iter=10,
    overshoot=0.02
):

    x_original = x.detach().clone()

    x_adv = x.detach().clone()

    with torch.no_grad():

        logits = model(
            x_original
        ).squeeze(1)

        original_predictions = (
            logits >= 0
        ).long()

    finished = torch.zeros(
        x.shape[0],
        dtype=torch.bool,
        device=x.device
    )

    for _ in range(max_iter):

        if finished.all():
            break

        x_adv.requires_grad_(True)

        logits = model(
            x_adv
        ).squeeze(1)

        gradients = torch.autograd.grad(
            outputs=logits.sum(),
            inputs=x_adv,
            create_graph=False,
            retain_graph=False
        )[0]

        grad_norm_sq = torch.sum(
            gradients ** 2,
            dim=1
        ) + 1e-12

        perturbation = (
            -logits / grad_norm_sq
        ).unsqueeze(1) * gradients

        perturbation = (
            perturbation *
            (1.0 + overshoot)
        )

        active = ~finished

        x_new = (
            x_adv.detach().clone()
        )

        x_new[active] = (
            x_adv.detach()[active]
            + perturbation.detach()[active]
        )

        x_adv = x_new

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

    return (
        x_adv.detach(),
        finished.detach()
    )


print("\n" + "=" * 70)
print("DEEPFOOL VALIDATION")
print("=" * 70)

print(
    "\nTesting DeepFool with "
    f"max_iter={DEEPFOOL_MAX_ITER}, "
    f"overshoot={DEEPFOOL_OVERSHOOT}"
)

X_adv, deepfool_success = deepfool_attack(
    model,
    X_clean,
    max_iter=DEEPFOOL_MAX_ITER,
    overshoot=DEEPFOOL_OVERSHOOT
)

predictions = predict(
    model,
    X_adv
)

predictions_np = (
    predictions.cpu().numpy()
)

accuracy, precision, recall, f1 = (
    calculate_metrics(
        y_validation,
        predictions_np
    )
)

originally_correct = (
    clean_predictions_np ==
    y_validation
)

successful_evasion = (
    originally_correct &
    (predictions_np != y_validation)
)

clean_correct_count = np.sum(
    originally_correct
)

evasion_rate = (
    np.sum(successful_evasion)
    / clean_correct_count
    if clean_correct_count > 0
    else 0
)

perturbation = (
    X_adv - X_clean
)

mean_l2 = torch.norm(
    perturbation,
    p=2,
    dim=1
).mean().item()

results.append({

    "attack": "DeepFool",

    "parameter":
        f"max_iter={DEEPFOOL_MAX_ITER},overshoot={DEEPFOOL_OVERSHOOT}",

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "f1": f1,

    "accuracy_drop":
        clean_accuracy - accuracy,

    "evasion_rate":
        evasion_rate,

    "successful_attacks":
        int(np.sum(successful_evasion)),

    "clean_correct":
        int(clean_correct_count),

    "mean_l2":
        mean_l2
})

print(
    f"Accuracy: "
    f"{accuracy * 100:.4f}%"
)

print(
    f"Evasion rate: "
    f"{evasion_rate * 100:.4f}%"
)


# ============================================================
# FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    RESULTS_PATH,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

print(
    f"\nClean accuracy on validation set: "
    f"{clean_accuracy * 100:.4f}%"
)

print(
    f"Validation samples: "
    f"{VALIDATION_SIZE}"
)

print(
    f"Random state: "
    f"{RANDOM_STATE}"
)

print("\nResults:")

print(
    results_df.to_string(
        index=False
    )
)


print("\nResults saved to:")

print(
    RESULTS_PATH
)

print("\n" + "=" * 70)
print("Attack validation completed.")
print("=" * 70)