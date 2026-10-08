import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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

SCALER_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_balanced_scaler.pkl"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_cw_adversarial_trained.pth"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "results/cw_defense_evaluation.csv"
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
DEEPFOOL_STEPS = 50
DEEPFOOL_OVERSHOOT = 0.02
DEEPFOOL_MIN_STEP = 1e-6


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
print("=" * 100)
print("          C&W-TRAINED MODEL - CROSS-ATTACK EVALUATION")
print("=" * 100)

print(f"\nDevice: {device}")
print(f"Validation samples: {N_SAMPLES}")

print("\nAttack settings:")
print(f"FGSM epsilon: {FGSM_EPSILON}")

print(
    f"PGD epsilon: {PGD_EPSILON} | "
    f"steps: {PGD_STEPS} | "
    f"alpha: {PGD_ALPHA}"
)

print(
    f"C&W steps: {CW_STEPS} | "
    f"learning rate: {CW_LR} | "
    f"C: {CW_C}"
)

print(
    f"DeepFool steps: {DEEPFOOL_STEPS} | "
    f"overshoot: {DEEPFOOL_OVERSHOOT}"
)


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

def load_model(
    model_file,
    input_dim
):

    print("\nLoading C&W-trained model...")

    if not os.path.exists(model_file):

        raise FileNotFoundError(
            f"\nModel file not found:\n{model_file}"
        )

    model = DNN(
        input_dim
    )

    model.load_state_dict(
        torch.load(
            model_file,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    print(
        "C&W-trained model loaded successfully."
    )

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict(
    model,
    X
):

    predictions = []

    with torch.no_grad():

        for start in range(
            0,
            len(X),
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                len(X)
            )

            x_batch = torch.tensor(
                X[start:end],
                dtype=torch.float32,
                device=device
            )

            logits = model(
                x_batch
            ).squeeze(1)

            probabilities = torch.sigmoid(
                logits
            )

            pred = (
                probabilities >= 0.5
            ).long()

            predictions.extend(
                pred.cpu().numpy()
            )

    return np.array(
        predictions
    )


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

    x_adv.requires_grad = True

    logits = model(
        x_adv
    ).squeeze(1)

    loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        y
    )

    model.zero_grad()

    loss.backward()

    gradient = x_adv.grad.sign()

    x_adv = (
        x_adv
        +
        epsilon * gradient
    )

    return x_adv.detach()


def generate_fgsm(
    model,
    X,
    y
):

    print("\nGenerating FGSM examples...")

    adv_examples = []

    for start in range(
        0,
        len(X),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(X)
        )

        print(
            f"  FGSM: "
            f"{start + 1}-{end}/{len(X)}",
            end="\r"
        )

        x = torch.tensor(
            X[start:end],
            dtype=torch.float32,
            device=device
        )

        labels = torch.tensor(
            y[start:end],
            dtype=torch.float32,
            device=device
        )

        adv = fgsm_attack(
            model,
            x,
            labels,
            FGSM_EPSILON
        )

        adv_examples.append(
            adv.cpu().numpy()
        )

    print()

    return np.concatenate(
        adv_examples,
        axis=0
    )


# ============================================================
# PGD
# ============================================================

def pgd_attack(
    model,
    x,
    y
):

    delta = torch.empty_like(
        x
    ).uniform_(
        -PGD_EPSILON,
        PGD_EPSILON
    )

    delta = delta.detach()

    for _ in range(
        PGD_STEPS
    ):

        delta.requires_grad = True

        logits = model(
            x + delta
        ).squeeze(1)

        loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            y
        )

        gradient = torch.autograd.grad(
            loss,
            delta
        )[0]

        delta = (
            delta
            +
            PGD_ALPHA
            *
            gradient.sign()
        )

        delta = torch.clamp(
            delta,
            -PGD_EPSILON,
            PGD_EPSILON
        ).detach()

    return (
        x + delta
    ).detach()


def generate_pgd(
    model,
    X,
    y
):

    print("\nGenerating PGD examples...")

    adv_examples = []

    for start in range(
        0,
        len(X),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(X)
        )

        print(
            f"  PGD: "
            f"{start + 1}-{end}/{len(X)}",
            end="\r"
        )

        x = torch.tensor(
            X[start:end],
            dtype=torch.float32,
            device=device
        )

        labels = torch.tensor(
            y[start:end],
            dtype=torch.float32,
            device=device
        )

        adv = pgd_attack(
            model,
            x,
            labels
        )

        adv_examples.append(
            adv.cpu().numpy()
        )

    print()

    return np.concatenate(
        adv_examples,
        axis=0
    )


# ============================================================
# C&W
# ============================================================

def cw_attack(
    model,
    x,
    y
):

    delta = torch.zeros_like(
        x,
        requires_grad=True
    )

    optimizer = optim.Adam(
        [delta],
        lr=CW_LR
    )

    best_adv = x.detach().clone()

    best_l2 = torch.full(
        (x.shape[0],),
        float("inf"),
        device=x.device
    )

    for _ in range(
        CW_STEPS
    ):

        optimizer.zero_grad()

        x_adv = x + delta

        logits = model(
            x_adv
        ).squeeze(1)

        direction = (
            2.0 * y - 1.0
        )

        classification_term = torch.clamp(
            direction * logits
            +
            CW_KAPPA,
            min=0.0
        )

        l2 = torch.sum(
            delta ** 2,
            dim=1
        )

        loss = (
            l2
            +
            CW_C * classification_term
        ).mean()

        loss.backward()

        optimizer.step()

        with torch.no_grad():

            pred = (
                torch.sigmoid(logits)
                >= 0.5
            ).float()

            successful = (
                pred != y
            )

            improved = (
                successful
                &
                (l2 < best_l2)
            )

            best_l2[improved] = (
                l2[improved]
            )

            best_adv[improved] = (
                x_adv[improved]
            )

    return (
        best_adv.detach(),
        best_l2.detach()
    )


def generate_cw(
    model,
    X,
    y
):

    print("\nGenerating C&W examples...")

    adv_examples = []
    l2_values = []

    for start in range(
        0,
        len(X),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(X)
        )

        print(
            f"  C&W: "
            f"{start + 1}-{end}/{len(X)}",
            end="\r"
        )

        x = torch.tensor(
            X[start:end],
            dtype=torch.float32,
            device=device
        )

        labels = torch.tensor(
            y[start:end],
            dtype=torch.float32,
            device=device
        )

        adv, l2 = cw_attack(
            model,
            x,
            labels
        )

        adv_examples.append(
            adv.cpu().numpy()
        )

        l2_values.append(
            l2.cpu().numpy()
        )

    print()

    return (
        np.concatenate(
            adv_examples,
            axis=0
        ),
        np.concatenate(
            l2_values,
            axis=0
        )
    )


# ============================================================
# DEEPFOOL
# ============================================================

def deepfool_single(
    model,
    x,
    original_label
):

    x_adv = x.detach().clone()

    original_label = int(
        original_label
    )

    for _ in range(
        DEEPFOOL_STEPS
    ):

        x_adv = x_adv.detach()

        x_adv.requires_grad_(True)

        logit = model(
            x_adv.unsqueeze(0)
        ).squeeze()

        current_label = int(
            (
                torch.sigmoid(logit)
                >= 0.5
            ).item()
        )

        # Attack succeeded
        if current_label != original_label:

            return x_adv.detach()

        gradient = torch.autograd.grad(
            logit,
            x_adv,
            retain_graph=False,
            create_graph=False
        )[0]

        gradient_norm_squared = torch.sum(
            gradient ** 2
        )

        if gradient_norm_squared.item() < 1e-12:

            return x_adv.detach()

        # Move toward decision boundary
        delta = (
            -logit
            /
            (
                gradient_norm_squared
                +
                1e-12
            )
        ) * gradient

        delta = (
            1.0 + DEEPFOOL_OVERSHOOT
        ) * delta

        # Minimum movement
        delta_norm = torch.norm(
            delta
        )

        if delta_norm.item() < DEEPFOOL_MIN_STEP:

            gradient_direction = (
                gradient
                /
                (
                    torch.norm(gradient)
                    +
                    1e-12
                )
            )

            delta = (
                DEEPFOOL_MIN_STEP
                *
                gradient_direction
            )

        x_adv = (
            x_adv
            +
            delta
        ).detach()

    return x_adv.detach()


def generate_deepfool(
    model,
    X,
    y
):

    print("\nGenerating DeepFool examples...")

    adv_examples = []

    total = len(X)

    for i in range(
        total
    ):

        if (
            i % 50 == 0
            or
            i == total - 1
        ):

            print(
                f"  DeepFool: "
                f"{i + 1}/{total}",
                end="\r"
            )

        x = torch.tensor(
            X[i],
            dtype=torch.float32,
            device=device
        )

        adv = deepfool_single(
            model,
            x,
            y[i]
        )

        adv_examples.append(
            adv.cpu().numpy()
        )

    print()

    return np.stack(
        adv_examples,
        axis=0
    )


# ============================================================
# EVALUATE ATTACK
# ============================================================

def evaluate_attack(
    model,
    attack_name,
    X_adv,
    X_reference,
    y,
    clean_pred,
    l2_values=None
):

    adv_pred = predict(
        model,
        X_adv
    )

    accuracy = accuracy_score(
        y,
        adv_pred
    )

    precision = precision_score(
        y,
        adv_pred,
        zero_division=0
    )

    recall = recall_score(
        y,
        adv_pred,
        zero_division=0
    )

    f1 = f1_score(
        y,
        adv_pred,
        zero_division=0
    )

    # --------------------------------------------------------
    # Successful attacks
    # --------------------------------------------------------

    successful = (
        adv_pred != y
    )

    successful_count = int(
        successful.sum()
    )

    attack_success_rate = (
        successful_count
        /
        len(y)
    )

    # --------------------------------------------------------
    # Evasion success
    # --------------------------------------------------------

    clean_correct = (
        clean_pred == y
    )

    evasion_success = (
        clean_correct
        &
        (adv_pred != y)
    )

    clean_correct_count = int(
        clean_correct.sum()
    )

    evasion_count = int(
        evasion_success.sum()
    )

    if clean_correct_count > 0:

        evasion_rate = (
            evasion_count
            /
            clean_correct_count
        )

    else:

        evasion_rate = 0.0

    # --------------------------------------------------------
    # L2
    # --------------------------------------------------------

    if l2_values is not None:

        successful_l2 = l2_values[
            successful
        ]

        successful_l2 = successful_l2[
            np.isfinite(successful_l2)
        ]

        if len(successful_l2) > 0:

            mean_l2 = float(
                np.mean(successful_l2)
            )

        else:

            mean_l2 = np.nan

    else:

        perturbation = (
            X_adv
            -
            X_reference
        )

        successful_perturbations = (
            perturbation[
                successful
            ]
        )

        if len(
            successful_perturbations
        ) > 0:

            successful_l2 = np.linalg.norm(
                successful_perturbations,
                axis=1
            )

            mean_l2 = float(
                np.mean(successful_l2)
            )

        else:

            mean_l2 = np.nan

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y,
        adv_pred,
        labels=[0, 1]
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"\n{attack_name} Results"
    )

    print(
        f"  Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"  Precision: "
        f"{precision:.4f}"
    )

    print(
        f"  Recall: "
        f"{recall:.4f}"
    )

    print(
        f"  F1: "
        f"{f1:.4f}"
    )

    print(
        f"  Successful attacks: "
        f"{successful_count}/{len(y)}"
    )

    print(
        f"  Attack success rate: "
        f"{attack_success_rate * 100:.2f}%"
    )

    print(
        f"  Evasion success rate: "
        f"{evasion_rate * 100:.2f}%"
    )

    print(
        f"  Mean L2: "
        f"{mean_l2:.6f}"
    )

    print(
        "  Confusion Matrix:"
    )

    print(cm)

    return {

        "Attack": attack_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "Successful Attacks":
            successful_count,

        "Attack Success Rate":
            attack_success_rate,

        "Evasion Success Rate":
            evasion_rate,

        "Mean L2":
            mean_l2,

        "TN":
            int(cm[0, 0]),

        "FP":
            int(cm[0, 1]),

        "FN":
            int(cm[1, 0]),

        "TP":
            int(cm[1, 1])
    }


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading test data...")

if not os.path.exists(TEST_FILE):

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
# FIXED 1000-SAMPLE SUBSET
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
    f"\nUsing fixed "
    f"{N_SAMPLES}-sample subset."
)

print(
    f"BENIGN: "
    f"{int((y == 0).sum())}"
)

print(
    f"ATTACK: "
    f"{int((y == 1).sum())}"
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

print(
    "Data standardized."
)


# ============================================================
# LOAD C&W-TRAINED MODEL
# ============================================================

input_dim = X_scaled.shape[1]

model = load_model(
    MODEL_FILE,
    input_dim
)


# ============================================================
# CLEAN PERFORMANCE
# ============================================================

print("\n")
print("=" * 100)
print("                         CLEAN PERFORMANCE")
print("=" * 100)

clean_pred = predict(
    model,
    X_scaled
)

clean_accuracy = accuracy_score(
    y,
    clean_pred
)

clean_correct = int(
    (clean_pred == y).sum()
)

print(
    f"\nClean accuracy: "
    f"{clean_accuracy * 100:.2f}%"
)

print(
    f"Clean correct: "
    f"{clean_correct}/{len(y)}"
)


# ============================================================
# GENERATE AND EVALUATE ALL ATTACKS
# ============================================================

results = []


# ------------------------------------------------------------
# FGSM
# ------------------------------------------------------------

X_fgsm = generate_fgsm(
    model,
    X_scaled,
    y
)

results.append(
    evaluate_attack(
        model,
        "FGSM",
        X_fgsm,
        X_scaled,
        y,
        clean_pred
    )
)


# ------------------------------------------------------------
# PGD
# ------------------------------------------------------------

X_pgd = generate_pgd(
    model,
    X_scaled,
    y
)

results.append(
    evaluate_attack(
        model,
        "PGD",
        X_pgd,
        X_scaled,
        y,
        clean_pred
    )
)


# ------------------------------------------------------------
# C&W
# ------------------------------------------------------------

X_cw, cw_l2 = generate_cw(
    model,
    X_scaled,
    y
)

results.append(
    evaluate_attack(
        model,
        "C&W",
        X_cw,
        X_scaled,
        y,
        clean_pred,
        cw_l2
    )
)


# ------------------------------------------------------------
# DeepFool
# ------------------------------------------------------------

X_deepfool = generate_deepfool(
    model,
    X_scaled,
    y
)

results.append(
    evaluate_attack(
        model,
        "DeepFool",
        X_deepfool,
        X_scaled,
        y,
        clean_pred
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL CROSS-ATTACK TABLE
# ============================================================

print("\n\n")

print("=" * 115)
print("             C&W-TRAINED MODEL - CROSS-ATTACK RESULTS")
print("=" * 115)

print()

print(
    f"{'Attack':<15}"
    f"{'Accuracy':>14}"
    f"{'Precision':>14}"
    f"{'Recall':>14}"
    f"{'F1':>12}"
    f"{'Attack Success':>18}"
    f"{'Evasion Success':>19}"
)

print("-" * 115)

for _, row in results_df.iterrows():

    print(
        f"{row['Attack']:<15}"
        f"{row['Accuracy'] * 100:>13.2f}%"
        f"{row['Precision']:>14.4f}"
        f"{row['Recall']:>14.4f}"
        f"{row['F1']:>12.4f}"
        f"{row['Attack Success Rate'] * 100:>17.2f}%"
        f"{row['Evasion Success Rate'] * 100:>18.2f}%"
    )


# ============================================================
# CLEAN VS ATTACK ACCURACY
# ============================================================

print("\n")

print("=" * 85)
print("                    CLEAN VS ATTACK ACCURACY")
print("=" * 85)

print()

print(
    f"{'Condition':<20}"
    f"{'Accuracy':>15}"
    f"{'Accuracy Drop':>20}"
)

print("-" * 60)

print(
    f"{'Clean':<20}"
    f"{clean_accuracy * 100:>14.2f}%"
    f"{'-':>20}"
)

for _, row in results_df.iterrows():

    drop = (
        clean_accuracy
        -
        row["Accuracy"]
    )

    print(
        f"{row['Attack']:<20}"
        f"{row['Accuracy'] * 100:>14.2f}%"
        f"{drop * 100:>19.2f} pp"
    )


# ============================================================
# CONFUSION MATRICES
# ============================================================

print("\n")

print("=" * 85)
print("                         CONFUSION MATRICES")
print("=" * 85)

for _, row in results_df.iterrows():

    print(
        f"\n{row['Attack']}:"
    )

    print(
        f"[[{row['TN']} {row['FP']}]"
    )

    print(
        f" [{row['FN']} {row['TP']}]]"
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\n")

print("=" * 85)
print("                         MODEL INFORMATION")
print("=" * 85)

print(
    "\nModel:"
)

print(
    MODEL_FILE
)

print(
    "\nTraining method:"
)

print(
    "C&W adversarial training"
)

print(
    "\nEvaluation subset:"
)

print(
    f"{N_SAMPLES} samples"
)

print(
    "\nRandom seed:"
)

print(
    RANDOM_SEED
)


# ============================================================
# OUTPUT FILE
# ============================================================

print("\n")

print("=" * 85)
print("                         RESULTS SAVED")
print("=" * 85)

print(
    f"\n{OUTPUT_FILE}"
)

print("\nExperiment completed successfully.\n")