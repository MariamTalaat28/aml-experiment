import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "/home/ubuntu/aml-experiment"

TEST_FILE = os.path.join(
    BASE_DIR,
    "data/ml/test.csv"
)

SCALER_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_balanced_scaler.pkl"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_deepfool_adversarial_trained.pth"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "results/deepfool_defense_evaluation.csv"
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

N_SAMPLES = 1000
RANDOM_SEED = 42

BATCH_SIZE = 256


# FGSM
FGSM_EPSILON = 0.10


# PGD
PGD_EPSILON = 0.10
PGD_STEPS = 10
PGD_ALPHA = PGD_EPSILON / PGD_STEPS


# C&W
CW_STEPS = 30
CW_LR = 0.01
CW_C = 1.0
CW_KAPPA = 0.0


# DeepFool
DEEPFOOL_STEPS = 10
DEEPFOOL_OVERSHOOT = 0.02


# ============================================================
# SEED
# ============================================================

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("\n")
print("=" * 90)
print("       DEEPFOOL-TRAINED MODEL - ALL ATTACK EVALUATION")
print("=" * 90)

print(f"Device: {device}")
print(f"Samples: {N_SAMPLES}")


# ============================================================
# DNN
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
# LOAD MODEL
# ============================================================

print("\nLoading DeepFool-trained model...")

model = DNN(78)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device
    )
)

model.to(device)
model.eval()

print(
    "DeepFool-trained model loaded successfully."
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

df = pd.read_csv(
    TEST_FILE
)

feature_columns = [
    col
    for col in df.columns
    if col != "Label"
]

X_all = df[
    feature_columns
].values.astype(
    np.float32
)


# ============================================================
# LABELS
# ============================================================

if df["Label"].dtype == object:

    y_all = np.array([
        1
        if str(label).upper() == "ATTACK"
        else 0
        for label in df["Label"]
    ])

else:

    y_all = df[
        "Label"
    ].values.astype(int)


# ============================================================
# FIXED SUBSET
# ============================================================

rng = np.random.RandomState(
    RANDOM_SEED
)

indices = rng.choice(
    len(X_all),
    size=N_SAMPLES,
    replace=False
)

X = X_all[
    indices
]

y = y_all[
    indices
]


print(
    f"Full test set: {len(X_all):,}"
)

print(
    f"Selected samples: {len(X):,}"
)

print(
    f"BENIGN: {int((y == 0).sum())}"
)

print(
    f"ATTACK: {int((y == 1).sum())}"
)


# ============================================================
# SCALE
# ============================================================

print("\nLoading scaler...")

scaler = joblib.load(
    SCALER_FILE
)

X_scaled = scaler.transform(
    X
).astype(
    np.float32
)

print("Data standardized.")


# ============================================================
# PREDICTION
# ============================================================

def predict_batch(
    model,
    x
):

    with torch.no_grad():

        logits = model(x).view(-1)

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= 0.5
        ).long()

    return predictions


# ============================================================
# FGSM
# ============================================================

def fgsm_attack(
    model,
    x,
    y,
    epsilon
):

    x_adv = x.clone().detach()
    x_adv.requires_grad_(True)

    logits = model(
        x_adv
    ).view(-1)

    loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        y.float().view(-1)
    )

    model.zero_grad()

    gradient = torch.autograd.grad(
        loss,
        x_adv
    )[0]

    x_adv = (
        x_adv
        +
        epsilon * gradient.sign()
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
    steps
):

    x_original = x.clone().detach()

    x_adv = (
        x_original
        +
        torch.empty_like(x_original).uniform_(
            -epsilon,
            epsilon
        )
    )

    for _ in range(steps):

        x_adv.requires_grad_(True)

        logits = model(
            x_adv
        ).view(-1)

        loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            y.float().view(-1)
        )

        gradient = torch.autograd.grad(
            loss,
            x_adv
        )[0]

        x_adv = (
            x_adv
            +
            alpha * gradient.sign()
        ).detach()

        perturbation = (
            x_adv
            -
            x_original
        )

        perturbation = torch.clamp(
            perturbation,
            -epsilon,
            epsilon
        )

        x_adv = (
            x_original
            +
            perturbation
        ).detach()

    return x_adv


# ============================================================
# C&W
# ============================================================

def cw_attack(
    model,
    x,
    y,
    steps,
    lr,
    c,
    kappa
):

    x_original = x.clone().detach()

    delta = torch.zeros_like(
        x_original,
        requires_grad=True
    )

    optimizer = torch.optim.Adam(
        [delta],
        lr=lr
    )

    best_adv = x_original.clone()

    best_l2 = torch.full(
        (x.size(0),),
        float("inf"),
        device=x.device
    )

    target = y.long().view(-1)

    for _ in range(steps):

        adv = x_original + delta

        logits = model(
            adv
        ).view(-1)

        true_logit = torch.where(
            target == 1,
            logits,
            -logits
        )

        other_logit = torch.where(
            target == 1,
            torch.zeros_like(logits),
            torch.zeros_like(logits)
        )

        f = torch.clamp(
            true_logit - other_logit + kappa,
            min=0
        )

        l2 = torch.sum(
            delta ** 2,
            dim=1
        )

        loss = torch.sum(
            l2 + c * f
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        with torch.no_grad():

            current_predictions = (
                torch.sigmoid(
                    model(
                        x_original + delta
                    ).view(-1)
                )
                >= 0.5
            ).long()

            successful = (
                current_predictions != target
            )

            improved = (
                successful
                &
                (l2 < best_l2)
            )

            best_adv[improved] = (
                x_original + delta
            )[improved]

            best_l2[improved] = (
                l2[improved]
            )

    return best_adv.detach()


# ============================================================
# DEEPFOOL
# ============================================================

def deepfool_attack(
    model,
    x,
    y
):

    x_adv = x.clone().detach()

    original_labels = y.long().view(-1)

    for _ in range(DEEPFOOL_STEPS):

        x_current = (
            x_adv
            .clone()
            .detach()
            .requires_grad_(True)
        )

        logits = model(
            x_current
        ).view(-1)

        predictions = (
            torch.sigmoid(logits)
            >= 0.5
        ).long()

        active = (
            predictions
            ==
            original_labels
        )

        if not active.any():
            break

        gradient = torch.autograd.grad(
            logits.sum(),
            x_current,
            retain_graph=False
        )[0]

        gradient_norm_squared = (
            torch.sum(
                gradient ** 2,
                dim=1
            )
            +
            1e-12
        )

        # ----------------------------------------------------
        # Logit-boundary DeepFool update
        # ----------------------------------------------------

        perturbation = (
            -logits
            /
            gradient_norm_squared
        ).unsqueeze(1) * gradient

        perturbation = (
            1.0 + DEEPFOOL_OVERSHOOT
        ) * perturbation

        # ----------------------------------------------------
        # Update active samples
        # ----------------------------------------------------

        x_adv = torch.where(
            active.unsqueeze(1),
            x_adv + perturbation.detach(),
            x_adv
        )

        x_adv = x_adv.detach()

    return x_adv


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    predictions,
    clean_predictions
):

    y_true = np.asarray(
        y_true
    )

    predictions = np.asarray(
        predictions
    )

    clean_predictions = np.asarray(
        clean_predictions
    )

    accuracy = (
        predictions == y_true
    ).mean()

    tp = int(
        (
            (predictions == 1)
            &
            (y_true == 1)
        ).sum()
    )

    tn = int(
        (
            (predictions == 0)
            &
            (y_true == 0)
        ).sum()
    )

    fp = int(
        (
            (predictions == 1)
            &
            (y_true == 0)
        ).sum()
    )

    fn = int(
        (
            (predictions == 0)
            &
            (y_true == 1)
        ).sum()
    )

    precision = (
        tp /
        (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp /
        (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    clean_accuracy = (
        clean_predictions == y_true
    ).mean()

    accuracy_drop = (
        clean_accuracy
        -
        accuracy
    )

    attack_success = (
        predictions != y_true
    ).mean()

    clean_correct = (
        clean_predictions == y_true
    )

    evasion_success = (
        clean_correct
        &
        (predictions != y_true)
    )

    if clean_correct.sum() > 0:

        evasion_rate = (
            evasion_success.sum()
            /
            clean_correct.sum()
        )

    else:

        evasion_rate = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy_drop": accuracy_drop,
        "attack_success_rate": attack_success,
        "evasion_success_rate": evasion_rate,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


# ============================================================
# DATA TENSOR
# ============================================================

X_tensor = torch.tensor(
    X_scaled,
    dtype=torch.float32,
    device=device
)

y_tensor = torch.tensor(
    y,
    dtype=torch.float32,
    device=device
)


# ============================================================
# CLEAN EVALUATION
# ============================================================

print("\n")
print("=" * 90)
print("                         CLEAN EVALUATION")
print("=" * 90)

clean_predictions = (
    predict_batch(
        model,
        X_tensor
    )
    .cpu()
    .numpy()
)

clean_metrics = calculate_metrics(
    y,
    clean_predictions,
    clean_predictions
)

print(
    f"Clean accuracy: "
    f"{clean_metrics['accuracy'] * 100:.2f}%"
)


# ============================================================
# ATTACK EVALUATION
# ============================================================

results = []


def evaluate_attack(
    attack_name,
    x_adv
):

    predictions = (
        predict_batch(
            model,
            x_adv
        )
        .cpu()
        .numpy()
    )

    metrics = calculate_metrics(
        y,
        predictions,
        clean_predictions
    )

    metrics["attack"] = attack_name

    return metrics


# ============================================================
# CLEAN RESULT
# ============================================================

clean_result = clean_metrics.copy()

clean_result["attack"] = "Clean"

results.append(
    clean_result
)


# ============================================================
# FGSM
# ============================================================

print("\n")
print("=" * 90)
print("                         FGSM ε = 0.10")
print("=" * 90)

x_adv = fgsm_attack(
    model,
    X_tensor,
    y_tensor,
    FGSM_EPSILON
)

fgsm_result = evaluate_attack(
    "FGSM ε=0.10",
    x_adv
)

results.append(
    fgsm_result
)

print(
    f"Accuracy: "
    f"{fgsm_result['accuracy'] * 100:.2f}%"
)


# ============================================================
# PGD
# ============================================================

print("\n")
print("=" * 90)
print("                         PGD ε = 0.10")
print("=" * 90)

x_adv = pgd_attack(
    model,
    X_tensor,
    y_tensor,
    PGD_EPSILON,
    PGD_ALPHA,
    PGD_STEPS
)

pgd_result = evaluate_attack(
    "PGD ε=0.10",
    x_adv
)

results.append(
    pgd_result
)

print(
    f"Accuracy: "
    f"{pgd_result['accuracy'] * 100:.2f}%"
)


# ============================================================
# C&W
# ============================================================

print("\n")
print("=" * 90)
print("                         C&W")
print("=" * 90)

x_adv = cw_attack(
    model,
    X_tensor,
    y_tensor,
    CW_STEPS,
    CW_LR,
    CW_C,
    CW_KAPPA
)

cw_result = evaluate_attack(
    "C&W",
    x_adv
)

results.append(
    cw_result
)

print(
    f"Accuracy: "
    f"{cw_result['accuracy'] * 100:.2f}%"
)


# ============================================================
# DEEPFOOL
# ============================================================

print("\n")
print("=" * 90)
print("                         DEEPFOOL")
print("=" * 90)

x_adv = deepfool_attack(
    model,
    X_tensor,
    y_tensor
)

deepfool_result = evaluate_attack(
    "DeepFool",
    x_adv
)

results.append(
    deepfool_result
)

print(
    f"Accuracy: "
    f"{deepfool_result['accuracy'] * 100:.2f}%"
)


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df[
    [
        "attack",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "accuracy_drop",
        "attack_success_rate",
        "evasion_success_rate",
        "tn",
        "fp",
        "fn",
        "tp"
    ]
]


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# PRINT FINAL TABLE
# ============================================================

print("\n")
print("=" * 90)
print("                  FINAL DEFENSE EVALUATION")
print("=" * 90)

print(
    results_df.to_string(
        index=False
    )
)

print("\n")
print(
    f"Results saved to:\n{OUTPUT_FILE}"
)

print("\n")
print("=" * 90)
print("                  EVALUATION COMPLETE")
print("=" * 90)