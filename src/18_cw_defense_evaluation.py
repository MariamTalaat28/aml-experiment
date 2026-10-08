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

# Original model
ORIGINAL_MODEL_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_balanced.pth"
)

# Defended model
DEFENDED_MODEL_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_adversarial_trained.pth"
)

# Results file
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

# Same C&W settings used in the original C&W experiment
CW_STEPS = 30
CW_LR = 0.01
CW_C = 1.0
CW_KAPPA = 0.0


# ============================================================
# RANDOM SEEDS
# ============================================================

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("\n")
print("=" * 90)
print("                         C&W DEFENSE EVALUATION")
print("=" * 90)

print(f"Device: {device}")
print(f"Validation samples: {N_SAMPLES}")
print(f"C&W steps: {CW_STEPS}")
print(f"C&W learning rate: {CW_LR}")
print(f"C&W C: {CW_C}")
print(f"C&W kappa: {CW_KAPPA}")

if device.type == "cpu":
    print("\nNote: C&W is running on CPU.")


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
# LOAD MODEL
# ============================================================

def load_model(
    model_file,
    input_dim,
    model_name
):

    print(f"\nLoading {model_name}...")

    model = DNN(input_dim)

    model.load_state_dict(
        torch.load(
            model_file,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    print(f"{model_name} loaded successfully.")

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict(model, X):

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

            batch_predictions = (
                probabilities >= 0.5
            ).long()

            predictions.extend(
                batch_predictions
                .cpu()
                .numpy()
            )

    return np.array(predictions)


# ============================================================
# C&W ATTACK
# ============================================================

def cw_attack(
    model,
    x,
    y
):

    # Start with zero perturbation
    delta = torch.zeros_like(
        x,
        requires_grad=True
    )

    optimizer = optim.Adam(
        [delta],
        lr=CW_LR
    )

    # Store the best successful adversarial examples
    best_adv = x.detach().clone()

    best_l2 = torch.full(
        (x.shape[0],),
        float("inf"),
        device=x.device
    )

    # --------------------------------------------------------
    # C&W optimization
    # --------------------------------------------------------

    for step in range(CW_STEPS):

        optimizer.zero_grad()

        x_adv = x + delta

        logits = model(
            x_adv
        ).squeeze(1)

        # Binary classification:
        #
        # y = 0 -> BENIGN
        # y = 1 -> ATTACK
        #
        # The attack tries to change the prediction.

        direction = (
            2.0 * y - 1.0
        )

        classification_term = torch.clamp(
            direction * logits + CW_KAPPA,
            min=0.0
        )

        # L2 perturbation
        l2_per_sample = torch.sum(
            delta ** 2,
            dim=1
        )

        # C&W objective
        loss = (
            l2_per_sample
            + CW_C * classification_term
        ).mean()

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Check attack success
        # ----------------------------------------------------

        with torch.no_grad():

            predictions = (
                torch.sigmoid(logits)
                >= 0.5
            ).float()

            successful = (
                predictions != y
            )

            improved = (
                successful
                &
                (l2_per_sample < best_l2)
            )

            best_l2[improved] = (
                l2_per_sample[improved]
            )

            best_adv[improved] = (
                x_adv[improved]
            )

    return (
        best_adv.detach(),
        best_l2.detach()
    )


# ============================================================
# GENERATE C&W EXAMPLES
# ============================================================

def generate_cw_examples(
    model,
    X,
    y
):

    all_adv = []
    all_l2 = []

    print(
        "\nGenerating C&W adversarial examples..."
    )

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
            f"  Processing "
            f"{start + 1}-{end} "
            f"/ {len(X)}"
        )

        x_batch = torch.tensor(
            X[start:end],
            dtype=torch.float32,
            device=device
        )

        y_batch = torch.tensor(
            y[start:end],
            dtype=torch.float32,
            device=device
        )

        adv_batch, l2_batch = cw_attack(
            model,
            x_batch,
            y_batch
        )

        all_adv.append(
            adv_batch.cpu().numpy()
        )

        all_l2.append(
            l2_batch.cpu().numpy()
        )

    X_adv = np.concatenate(
        all_adv,
        axis=0
    )

    l2_values = np.concatenate(
        all_l2,
        axis=0
    )

    return X_adv, l2_values


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(
    model_name,
    model,
    X,
    y
):

    print("\n")
    print("-" * 90)
    print(f"Testing: {model_name}")
    print("-" * 90)

    # ========================================================
    # Clean performance
    # ========================================================

    clean_pred = predict(
        model,
        X
    )

    clean_accuracy = accuracy_score(
        y,
        clean_pred
    )

    clean_correct = (
        clean_pred == y
    )

    clean_correct_count = int(
        clean_correct.sum()
    )

    # ========================================================
    # Generate C&W examples
    # ========================================================

    X_adv, l2_values = generate_cw_examples(
        model,
        X,
        y
    )

    # ========================================================
    # Adversarial predictions
    # ========================================================

    adv_pred = predict(
        model,
        X_adv
    )

    adv_accuracy = accuracy_score(
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

    # ========================================================
    # Successful attacks
    # ========================================================

    successful_attacks = (
        adv_pred != y
    )

    successful_count = int(
        successful_attacks.sum()
    )

    attack_success_rate = (
        successful_count / len(y)
    )

    # ========================================================
    # Evasion success
    #
    # Only samples classified correctly before attack.
    # ========================================================

    evasion_success = (
        clean_correct
        &
        (adv_pred != y)
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

    # ========================================================
    # L2 statistics
    # ========================================================

    successful_l2 = l2_values[
        successful_attacks
    ]

    if len(successful_l2) > 0:

        mean_l2 = float(
            np.mean(successful_l2)
        )

        median_l2 = float(
            np.median(successful_l2)
        )

    else:

        mean_l2 = np.nan
        median_l2 = np.nan

    # ========================================================
    # Confusion matrix
    # ========================================================

    cm = confusion_matrix(
        y,
        adv_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    # ========================================================
    # Return results
    # ========================================================

    return {

        "Model": model_name,

        "Clean Accuracy": clean_accuracy,

        "CW Accuracy": adv_accuracy,

        "Accuracy Drop": (
            clean_accuracy
            -
            adv_accuracy
        ),

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "Successful Attacks":
            successful_count,

        "Attack Success Rate":
            attack_success_rate,

        "Clean Correct":
            clean_correct_count,

        "Evasion Successes":
            evasion_count,

        "Evasion Success Rate":
            evasion_rate,

        "Mean L2 Successful":
            mean_l2,

        "Median L2 Successful":
            median_l2,

        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


# ============================================================
# MAIN
# ============================================================

print("\nLoading test data...")

df = pd.read_csv(
    TEST_FILE
)

print(
    f"Total test rows: {len(df):,}"
)


# ============================================================
# FEATURES AND LABELS
# ============================================================

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

# Convert labels to:
# BENIGN = 0
# ATTACK = 1

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
# SAME 1000-SAMPLE VALIDATION SUBSET
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
    f"{N_SAMPLES}-sample validation subset."
)

print(
    f"BENIGN samples: "
    f"{int((y == 0).sum())}"
)

print(
    f"ATTACK samples: "
    f"{int((y == 1).sum())}"
)


# ============================================================
# LOAD SCALER
# ============================================================

print("\nLoading scaler...")

scaler = joblib.load(
    SCALER_FILE
)

print("Scaler loaded.")


# ============================================================
# STANDARDIZE DATA
# ============================================================

print("\nStandardizing validation data...")

X_scaled = scaler.transform(
    X
).astype(
    np.float32
)

print("Standardization complete.")


# ============================================================
# LOAD MODELS
# ============================================================

input_dim = X_scaled.shape[1]

print(
    f"\nNumber of input features: "
    f"{input_dim}"
)

original_model = load_model(
    ORIGINAL_MODEL_FILE,
    input_dim,
    "Original DNN"
)

defended_model = load_model(
    DEFENDED_MODEL_FILE,
    input_dim,
    "FGSM-Adversarially-Trained DNN"
)


# ============================================================
# EVALUATE ORIGINAL MODEL
# ============================================================

original_results = evaluate_model(
    "Original DNN",
    original_model,
    X_scaled,
    y
)


# ============================================================
# EVALUATE DEFENDED MODEL
# ============================================================

defended_results = evaluate_model(
    "FGSM-Adversarially-Trained DNN",
    defended_model,
    X_scaled,
    y
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame([
    original_results,
    defended_results
])

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# CLEAN TABLE OUTPUT
# ============================================================

print("\n\n")
print("=" * 110)
print("                         C&W DEFENSE EVALUATION RESULTS")
print("=" * 110)

print()

print(
    f"{'Model':<35}"
    f"{'Clean Acc.':>12}"
    f"{'C&W Acc.':>12}"
    f"{'Attack Success':>16}"
    f"{'Evasion Success':>17}"
    f"{'Mean L2':>12}"
)

print("-" * 110)

for _, row in results.iterrows():

    print(
        f"{row['Model']:<35}"
        f"{row['Clean Accuracy'] * 100:>11.2f}%"
        f"{row['CW Accuracy'] * 100:>11.2f}%"
        f"{row['Attack Success Rate'] * 100:>15.2f}%"
        f"{row['Evasion Success Rate'] * 100:>16.2f}%"
        f"{row['Mean L2 Successful']:>12.4f}"
    )


# ============================================================
# DEFENSE EFFECT
# ============================================================

original = results.iloc[0]
defended = results.iloc[1]

attack_success_reduction = (
    original["Attack Success Rate"]
    -
    defended["Attack Success Rate"]
)

cw_accuracy_improvement = (
    defended["CW Accuracy"]
    -
    original["CW Accuracy"]
)

clean_accuracy_change = (
    defended["Clean Accuracy"]
    -
    original["Clean Accuracy"]
)


print("\n")
print("=" * 80)
print("                           DEFENSE EFFECT")
print("=" * 80)

print()

print(
    f"{'Metric':<35}"
    f"{'Original':>15}"
    f"{'Defended':>15}"
)

print("-" * 70)

print(
    f"{'C&W Accuracy':<35}"
    f"{original['CW Accuracy'] * 100:>14.2f}%"
    f"{defended['CW Accuracy'] * 100:>14.2f}%"
)

print(
    f"{'Attack Success Rate':<35}"
    f"{original['Attack Success Rate'] * 100:>14.2f}%"
    f"{defended['Attack Success Rate'] * 100:>14.2f}%"
)

print(
    f"{'Evasion Success Rate':<35}"
    f"{original['Evasion Success Rate'] * 100:>14.2f}%"
    f"{defended['Evasion Success Rate'] * 100:>14.2f}%"
)

print(
    f"{'Clean Accuracy':<35}"
    f"{original['Clean Accuracy'] * 100:>14.2f}%"
    f"{defended['Clean Accuracy'] * 100:>14.2f}%"
)

print(
    f"{'Mean L2':<35}"
    f"{original['Mean L2 Successful']:>15.4f}"
    f"{defended['Mean L2 Successful']:>15.4f}"
)


print("\n")
print(
    f"Attack success reduction: "
    f"{attack_success_reduction * 100:.2f} percentage points"
)

print(
    f"C&W accuracy improvement: "
    f"{cw_accuracy_improvement * 100:.2f} percentage points"
)

print(
    f"Clean accuracy change: "
    f"{clean_accuracy_change * 100:.2f} percentage points"
)


# ============================================================
# CONFUSION MATRICES
# ============================================================

print("\n")
print("=" * 80)
print("                         CONFUSION MATRICES")
print("=" * 80)

print("\nOriginal DNN:")
print(
    np.array([
        [original["TN"], original["FP"]],
        [original["FN"], original["TP"]]
    ]).astype(int)
)

print("\nFGSM-Adversarially-Trained DNN:")
print(
    np.array([
        [defended["TN"], defended["FP"]],
        [defended["FN"], defended["TP"]]
    ]).astype(int)
)


# ============================================================
# FILE LOCATION
# ============================================================

print("\n")
print("=" * 80)
print("Results saved to:")
print(OUTPUT_FILE)
print("=" * 80)

print("\nExperiment completed successfully.\n")