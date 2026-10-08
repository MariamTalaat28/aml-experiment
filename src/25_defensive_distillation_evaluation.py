import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

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

TEST_FILE = os.path.join(
    BASE_DIR,
    "data/ml/test.csv"
)

# ============================================================
# V3 DISTILLED STUDENT
# ============================================================

STUDENT_MODEL_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_distilled_student_v3.pth"
)

# ============================================================
# V3 SCALER
# ============================================================

SCALER_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_distilled_student_scaler_v3.pkl"
)

# ============================================================
# V3 OUTPUT
# ============================================================

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "results/defensive_distillation_v3_evaluation.csv"
)

RANDOM_SEED = 42
N_SAMPLES = 1000

BATCH_SIZE = 256


# ============================================================
# ATTACK PARAMETERS
# ============================================================

FGSM_EPSILON = 0.10

PGD_EPSILON = 0.10
PGD_ALPHA = 0.01
PGD_STEPS = 10

CW_STEPS = 30
CW_LR = 0.01
CW_C = 1.0
CW_KAPPA = 0.0

DEEPFOOL_STEPS = 50
DEEPFOOL_OVERSHOOT = 0.02
DEEPFOOL_STEP_SIZE = 0.01

TEMPERATURE = 20.0


# ============================================================
# SEED / DEVICE
# ============================================================

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("\n")

print(
    "=" * 110
)

print(
    "             DEFENSIVE DISTILLATION V3 - ATTACK EVALUATION"
)

print(
    "=" * 110
)

print(
    f"\nDevice: {device}"
)

print(
    "Model: Distilled Student V3"
)

print(
    f"Temperature: {TEMPERATURE}"
)

print(
    f"Test samples: {N_SAMPLES}"
)

print(
    f"Random seed: {RANDOM_SEED}"
)


# ============================================================
# DNN MODEL
# ============================================================

class DNN(nn.Module):

    def __init__(
        self,
        input_dim
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                1
            )
        )

    def forward(
        self,
        x
    ):

        return self.network(x)


# ============================================================
# LOAD TEST DATA
# ============================================================

print(
    "\nLoading test data..."
)

if not os.path.exists(
    TEST_FILE
):

    raise FileNotFoundError(
        f"Test file not found:\n{TEST_FILE}"
    )

df = pd.read_csv(
    TEST_FILE
)


feature_columns = [
    col
    for col in df.columns
    if col != "Label"
]


X = df[
    feature_columns
].values.astype(
    np.float32
)


# ============================================================
# LABEL CONVERSION
# ============================================================

if df["Label"].dtype == object:

    y = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("ATTACK")
        .astype(np.int64)
        .values
    )

else:

    y = df[
        "Label"
    ].values.astype(
        np.int64
    )


# ============================================================
# FIXED 1000-SAMPLE SUBSET
# ============================================================

rng = np.random.RandomState(
    RANDOM_SEED
)

indices = rng.choice(
    len(X),
    size=N_SAMPLES,
    replace=False
)

X = X[
    indices
]

y = y[
    indices
]


print(
    f"Selected {len(X):,} samples."
)

print(
    f"BENIGN: {(y == 0).sum():,}"
)

print(
    f"ATTACK: {(y == 1).sum():,}"
)


# ============================================================
# LOAD V3 SCALER
# ============================================================

print(
    "\nLoading V3 scaler..."
)

if not os.path.exists(
    SCALER_FILE
):

    raise FileNotFoundError(
        f"Scaler not found:\n{SCALER_FILE}"
    )


scaler = joblib.load(
    SCALER_FILE
)


X_scaled = scaler.transform(
    X
).astype(
    np.float32
)

print(
    "Test data standardized."
)


# ============================================================
# LOAD V3 STUDENT
# ============================================================

input_dim = X_scaled.shape[1]

print(
    "\nLoading distilled student V3..."
)

if not os.path.exists(
    STUDENT_MODEL_FILE
):

    raise FileNotFoundError(
        f"Student model not found:\n{STUDENT_MODEL_FILE}"
    )


student = DNN(
    input_dim
).to(device)


student.load_state_dict(
    torch.load(
        STUDENT_MODEL_FILE,
        map_location=device,
        weights_only=True
    )
)

student.eval()


print(
    "Distilled Student V3 loaded successfully."
)

print(
    f"Input features: {input_dim}"
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(
    model,
    X_data
):

    model.eval()

    predictions = []

    with torch.no_grad():

        for start in range(
            0,
            len(X_data),
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                len(X_data)
            )

            batch = torch.tensor(
                X_data[start:end],
                dtype=torch.float32,
                device=device
            )

            logits = model(
                batch
            ).squeeze(1)

            probabilities = torch.sigmoid(
                logits
            )

            preds = (
                probabilities >= 0.5
            ).long()

            predictions.append(
                preds.cpu().numpy()
            )

    return np.concatenate(
        predictions
    )


# ============================================================
# CLEAN PERFORMANCE
# ============================================================

print(
    "\nEvaluating clean data..."
)

clean_predictions = predict(
    student,
    X_scaled
)


clean_accuracy = accuracy_score(
    y,
    clean_predictions
)

clean_precision = precision_score(
    y,
    clean_predictions,
    zero_division=0
)

clean_recall = recall_score(
    y,
    clean_predictions,
    zero_division=0
)

clean_f1 = f1_score(
    y,
    clean_predictions,
    zero_division=0
)

clean_cm = confusion_matrix(
    y,
    clean_predictions
)


print(
    f"Clean accuracy: "
    f"{clean_accuracy * 100:.2f}%"
)


# ============================================================
# FGSM ATTACK
# ============================================================

def fgsm_attack(
    model,
    X_input,
    y_true,
    epsilon
):

    x = torch.tensor(
        X_input,
        dtype=torch.float32,
        device=device,
        requires_grad=True
    )

    labels = torch.tensor(
        y_true,
        dtype=torch.float32,
        device=device
    )

    logits = model(
        x
    ).squeeze(1)

    loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels
    )

    model.zero_grad()

    loss.backward()

    gradient = x.grad.sign()

    x_adv = (
        x
        +
        epsilon * gradient
    )

    return (
        x_adv
        .detach()
        .cpu()
        .numpy()
    )


# ============================================================
# PGD ATTACK
# ============================================================

def pgd_attack(
    model,
    X_input,
    y_true,
    epsilon,
    alpha,
    steps
):

    x_original = torch.tensor(
        X_input,
        dtype=torch.float32,
        device=device
    )

    # Random start
    x_adv = (
        x_original
        +
        torch.empty_like(
            x_original
        ).uniform_(
            -epsilon,
            epsilon
        )
    )

    labels = torch.tensor(
        y_true,
        dtype=torch.float32,
        device=device
    )

    for _ in range(
        steps
    ):

        x_adv.requires_grad_(True)

        logits = model(
            x_adv
        ).squeeze(1)

        loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels
        )

        model.zero_grad()

        loss.backward()

        gradient = x_adv.grad.sign()

        x_adv = (
            x_adv.detach()
            +
            alpha * gradient
        )

        perturbation = (
            x_adv
            -
            x_original
        ).clamp(
            -epsilon,
            epsilon
        )

        x_adv = (
            x_original
            +
            perturbation
        ).detach()

    return (
        x_adv
        .cpu()
        .numpy()
    )


# ============================================================
# C&W ATTACK
# ============================================================

def cw_attack_single(
    model,
    x_input,
    y_true,
    steps=CW_STEPS,
    lr=CW_LR,
    c=CW_C,
    kappa=CW_KAPPA
):

    x = torch.tensor(
        x_input,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    target = int(
        y_true
    )

    delta = torch.zeros_like(
        x,
        requires_grad=True
    )

    optimizer = torch.optim.Adam(
        [delta],
        lr=lr
    )

    best_adv = (
        x.detach()
        .clone()
    )

    best_l2 = float(
        "inf"
    )

    for _ in range(
        steps
    ):

        adv = (
            x + delta
        )

        logits = model(
            adv
        ).squeeze()

        probability = torch.sigmoid(
            logits
        )

        if target == 1:

            f = torch.clamp(
                logits - kappa,
                min=0
            )

        else:

            f = torch.clamp(
                -logits - kappa,
                min=0
            )

        l2 = torch.sum(
            delta ** 2
        )

        loss = (
            l2
            +
            c * f
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        with torch.no_grad():

            prediction = int(
                probability.item()
                >= 0.5
            )

            successful = (
                prediction != target
            )

            current_l2 = (
                torch.sum(
                    delta ** 2
                ).item()
            )

            if (
                successful
                and
                current_l2 < best_l2
            ):

                best_l2 = current_l2

                best_adv = (
                    x + delta
                ).detach().clone()

    return (
        best_adv
        .squeeze(0)
        .cpu()
        .numpy()
    )


def cw_attack(
    model,
    X_input,
    y_true
):

    adversarial_samples = []

    for i in range(
        len(X_input)
    ):

        adv = cw_attack_single(
            model,
            X_input[i],
            y_true[i]
        )

        adversarial_samples.append(
            adv
        )

        if (
            (i + 1) % 100 == 0
            or
            i == len(X_input) - 1
        ):

            print(
                f"  C&W: "
                f"{i + 1}/{len(X_input)}"
            )

    return np.array(
        adversarial_samples,
        dtype=np.float32
    )


# ============================================================
# DEEPFOOL ATTACK
# ============================================================

def deepfool_single(
    model,
    x_input,
    y_true,
    steps=DEEPFOOL_STEPS,
    overshoot=DEEPFOOL_OVERSHOOT,
    step_size=DEEPFOOL_STEP_SIZE
):

    x_original = torch.tensor(
        x_input,
        dtype=torch.float32,
        device=device
    )

    x_adv = (
        x_original.clone()
    )

    target = int(
        y_true
    )

    for _ in range(
        steps
    ):

        x_adv = (
            x_adv
            .detach()
            .clone()
        )

        x_adv.requires_grad_(True)

        logit = model(
            x_adv.unsqueeze(0)
        ).squeeze()

        prediction = int(
            torch.sigmoid(
                logit
            ).item()
            >= 0.5
        )

        if prediction != target:

            break

        model.zero_grad()

        logit.backward()

        gradient = (
            x_adv.grad.detach()
        )

        gradient_norm_sq = torch.sum(
            gradient ** 2
        ).item()

        if gradient_norm_sq < 1e-12:

            break

        direction = (
            -1.0
            if target == 1
            else 1.0
        )

        perturbation = (
            direction
            *
            (
                -logit.detach().item()
            )
            /
            gradient_norm_sq
            *
            gradient
        )

        perturbation_norm = torch.norm(
            perturbation
        ).item()

        if (
            perturbation_norm
            <
            step_size
        ):

            normalized_gradient = (
                gradient
                /
                (
                    torch.norm(
                        gradient
                    )
                    +
                    1e-12
                )
            )

            perturbation = (
                direction
                *
                step_size
                *
                normalized_gradient
            )

        x_adv = (
            x_adv.detach()
            +
            (1.0 + overshoot)
            *
            perturbation
        )

    return (
        x_adv
        .detach()
        .cpu()
        .numpy()
    )


def deepfool_attack(
    model,
    X_input,
    y_true
):

    adversarial_samples = []

    for i in range(
        len(X_input)
    ):

        adv = deepfool_single(
            model,
            X_input[i],
            y_true[i]
        )

        adversarial_samples.append(
            adv
        )

        if (
            (i + 1) % 100 == 0
            or
            i == len(X_input) - 1
        ):

            print(
                f"  DeepFool: "
                f"{i + 1}/{len(X_input)}"
            )

    return np.array(
        adversarial_samples,
        dtype=np.float32
    )


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_attack(
    name,
    X_adv,
    clean_predictions
):

    predictions = predict(
        student,
        X_adv
    )

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    # Percentage of all samples whose
    # prediction changed.
    changed = (
        predictions
        !=
        clean_predictions
    )

    attack_success = (
        changed.mean()
    )

    # Evasion success among samples
    # that were originally classified correctly.
    clean_correct = (
        clean_predictions
        ==
        y
    )

    successful_evasion = (
        clean_correct
        &
        (predictions != y)
    )

    evasion_success = (
        successful_evasion.sum()
        /
        max(
            clean_correct.sum(),
            1
        )
    )

    return {
        "Attack": name,
        "Accuracy": accuracy * 100,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Attack Success": attack_success * 100,
        "Evasion Success": evasion_success * 100,
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1])
    }


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# CLEAN
# ============================================================

results.append({

    "Attack": "Clean",

    "Accuracy":
        clean_accuracy * 100,

    "Precision":
        clean_precision,

    "Recall":
        clean_recall,

    "F1 Score":
        clean_f1,

    "Attack Success":
        np.nan,

    "Evasion Success":
        np.nan,

    "TN":
        int(clean_cm[0, 0]),

    "FP":
        int(clean_cm[0, 1]),

    "FN":
        int(clean_cm[1, 0]),

    "TP":
        int(clean_cm[1, 1])
})


# ============================================================
# FGSM
# ============================================================

print("\n")

print(
    "=" * 110
)

print(
    "Running FGSM..."
)

print(
    "=" * 110
)

X_fgsm = fgsm_attack(
    student,
    X_scaled,
    y,
    FGSM_EPSILON
)

results.append(
    evaluate_attack(
        "FGSM ε=0.10",
        X_fgsm,
        clean_predictions
    )
)


# ============================================================
# PGD
# ============================================================

print("\n")

print(
    "=" * 110
)

print(
    "Running PGD..."
)

print(
    "=" * 110
)

X_pgd = pgd_attack(
    student,
    X_scaled,
    y,
    PGD_EPSILON,
    PGD_ALPHA,
    PGD_STEPS
)

results.append(
    evaluate_attack(
        "PGD ε=0.10",
        X_pgd,
        clean_predictions
    )
)


# ============================================================
# C&W
# ============================================================

print("\n")

print(
    "=" * 110
)

print(
    "Running C&W..."
)

print(
    "=" * 110
)

X_cw = cw_attack(
    student,
    X_scaled,
    y
)

results.append(
    evaluate_attack(
        "C&W",
        X_cw,
        clean_predictions
    )
)


# ============================================================
# DEEPFOOL
# ============================================================

print("\n")

print(
    "=" * 110
)

print(
    "Running DeepFool..."
)

print(
    "=" * 110
)

X_deepfool = deepfool_attack(
    student,
    X_scaled,
    y
)

results.append(
    evaluate_attack(
        "DeepFool",
        X_deepfool,
        clean_predictions
    )
)


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY MAIN TABLE
# ============================================================

print("\n\n")

print(
    "=" * 120
)

print(
    "                 DEFENSIVE DISTILLATION V3 RESULTS"
)

print(
    "=" * 120
)


display_df = results_df[
    [
        "Attack",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Attack Success",
        "Evasion Success"
    ]
].copy()


display_df["Accuracy"] = (
    display_df["Accuracy"]
    .map(
        lambda x:
        "-"
        if pd.isna(x)
        else f"{x:.2f}%"
    )
)


display_df["Precision"] = (
    display_df["Precision"]
    .map(
        lambda x:
        "-"
        if pd.isna(x)
        else f"{x:.4f}"
    )
)


display_df["Recall"] = (
    display_df["Recall"]
    .map(
        lambda x:
        "-"
        if pd.isna(x)
        else f"{x:.4f}"
    )
)


display_df["F1 Score"] = (
    display_df["F1 Score"]
    .map(
        lambda x:
        "-"
        if pd.isna(x)
        else f"{x:.4f}"
    )
)


display_df["Attack Success"] = (
    display_df["Attack Success"]
    .map(
        lambda x:
        "-"
        if pd.isna(x)
        else f"{x:.2f}%"
    )
)


display_df["Evasion Success"] = (
    display_df["Evasion Success"]
    .map(
        lambda x:
        "-"
        if pd.isna(x)
        else f"{x:.2f}%"
    )
)


print(
    display_df.to_string(
        index=False
    )
)


# ============================================================
# CONFUSION MATRICES
# ============================================================

print("\n")

print(
    "=" * 110
)

print(
    "                         CONFUSION MATRICES"
)

print(
    "=" * 110
)


for _, row in results_df.iterrows():

    print(
        f"\n{row['Attack']}:"
    )

    print(
        f"TN={int(row['TN'])}, "
        f"FP={int(row['FP'])}, "
        f"FN={int(row['FN'])}, "
        f"TP={int(row['TP'])}"
    )


# ============================================================
# ACCURACY DROP
# ============================================================

print("\n")

print(
    "=" * 110
)

print(
    "                         ACCURACY DROP"
)

print(
    "=" * 110
)


for _, row in results_df.iterrows():

    if row["Attack"] == "Clean":

        continue

    drop = (
        clean_accuracy * 100
        -
        row["Accuracy"]
    )

    print(
        f"{row['Attack']:15s} "
        f"Accuracy drop: "
        f"{drop:.2f} percentage points"
    )


# ============================================================
# FINAL
# ============================================================

print("\n")

print(
    "=" * 110
)

print(
    "                         EVALUATION COMPLETE"
)

print(
    "=" * 110
)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_FILE
)

print("\n")