# ============================================================
# ART PGD EXPERIMENT
# Adversarial Robustness Toolbox (ART)
# CICIDS2017 - Balanced DNN
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

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

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "results",
    "art_pgd_results.csv"
)

SEED = 42

N_SAMPLES = 1000

EPSILONS = [
    0.01,
    0.05,
    0.10
]

PGD_ITERATIONS = 10

DEVICE = "cpu"


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)

torch.manual_seed(SEED)


# ============================================================
# DNN MODEL
#
# Same architecture used in the original experiment:
#
# 78 -> 128 -> 64 -> 32 -> 1
# ============================================================

class DNN(nn.Module):

    def __init__(self, input_dim=78):

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

    def forward(self, x):

        return self.network(x)


# ============================================================
# ART MODEL WRAPPER
#
# Original model produces one binary logit.
#
# ART uses two-class logits:
#
#     [-z, z]
#
# This preserves the same binary decision boundary.
# ============================================================

class ARTModel(nn.Module):

    def __init__(self, base_model):

        super().__init__()

        self.base_model = base_model

    def forward(self, x):

        logit = self.base_model(x)

        return torch.cat(
            [-logit, logit],
            dim=1
        )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true
    ).astype(int)

    y_pred = np.asarray(
        y_pred
    ).astype(int)

    tp = np.sum(
        (y_true == 1) &
        (y_pred == 1)
    )

    tn = np.sum(
        (y_true == 0) &
        (y_pred == 0)
    )

    fp = np.sum(
        (y_true == 0) &
        (y_pred == 1)
    )

    fn = np.sum(
        (y_true == 1) &
        (y_pred == 0)
    )

    total = len(y_true)

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return (
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("ART PGD EXPERIMENT")
print("=" * 70)

print()

print("Configuration:")

print(
    f"Test file       : {TEST_PATH}"
)

print(
    f"Model           : {MODEL_PATH}"
)

print(
    f"Scaler          : {SCALER_PATH}"
)

print(
    f"Samples         : {N_SAMPLES}"
)

print(
    f"Epsilons        : {EPSILONS}"
)

print(
    f"PGD iterations  : {PGD_ITERATIONS}"
)

print(
    f"Device          : {DEVICE}"
)

print()


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

base_model = DNN(
    input_dim=78
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

base_model.load_state_dict(
    state_dict
)

base_model.eval()

print(
    "Model loaded successfully."
)


# ============================================================
# LOAD SCALER
# ============================================================

print()

print("Loading scaler...")

scaler = joblib.load(
    SCALER_PATH
)

print(
    "Scaler loaded successfully."
)


# ============================================================
# LOAD TEST DATASET
# ============================================================

print()

print("Loading test dataset...")

test_df = pd.read_csv(
    TEST_PATH
)

print(
    f"Full test dataset shape: "
    f"{test_df.shape}"
)


# ============================================================
# PREPARE TEST DATA
# ============================================================

print()

print("Preparing test data...")

if "Label" not in test_df.columns:

    raise ValueError(
        "The test dataset does not contain "
        "a 'Label' column."
    )


# ------------------------------------------------------------
# Separate features and labels
# ------------------------------------------------------------

X_df = test_df.drop(
    columns=["Label"]
)

y_series = test_df["Label"]


# ------------------------------------------------------------
# Convert labels to binary
#
# Supports:
#
# BENIGN / ATTACK
#
# and:
#
# 0 / 1
# ------------------------------------------------------------

y = (
    y_series
    .astype(str)
    .str.strip()
    .str.upper()
    .map({
        "BENIGN": 0,
        "ATTACK": 1,
        "0": 0,
        "1": 1
    })
)


# ------------------------------------------------------------
# Check for unknown labels
# ------------------------------------------------------------

if y.isna().any():

    unknown_labels = (
        y_series[
            y.isna()
        ]
        .astype(str)
        .unique()
    )

    raise ValueError(
        f"Unknown labels found: "
        f"{unknown_labels}"
    )


y = y.to_numpy(
    dtype=np.int64
)


# ============================================================
# CONVERT FEATURES TO NUMERIC
# ============================================================

X = X_df.apply(
    pd.to_numeric,
    errors="coerce"
).to_numpy(
    dtype=np.float32
)


# ============================================================
# CHECK FOR INVALID VALUES
# ============================================================

if not np.isfinite(X).all():

    print(
        "Warning: NaN or infinite values detected."
    )

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


print(
    f"Features shape: {X.shape}"
)

print(
    f"Labels shape  : {y.shape}"
)


# ============================================================
# SCALE FEATURES
# ============================================================

print()

print("Applying StandardScaler...")

X_scaled = scaler.transform(
    X
)

X_scaled = X_scaled.astype(
    np.float32
)


print(
    f"Scaled feature range: "
    f"{X_scaled.min():.4f} "
    f"to "
    f"{X_scaled.max():.4f}"
)


# ============================================================
# SELECT FIXED 1,000-SAMPLE SUBSET
#
# Same random state used in the validation experiments.
# ============================================================

print()

print(
    "Selecting validation subset..."
)

if len(X_scaled) < N_SAMPLES:

    raise ValueError(
        f"Test dataset contains only "
        f"{len(X_scaled)} samples."
    )


rng = np.random.RandomState(
    SEED
)

indices = rng.choice(
    len(X_scaled),
    size=N_SAMPLES,
    replace=False
)

X_test = X_scaled[
    indices
]

y_test = y[
    indices
]


print(
    f"Selected {len(X_test)} samples."
)

print(
    f"BENIGN samples: "
    f"{np.sum(y_test == 0)}"
)

print(
    f"ATTACK samples: "
    f"{np.sum(y_test == 1)}"
)


# ============================================================
# CREATE ART MODEL
# ============================================================

print()

print(
    "Creating ART classifier..."
)

art_model = ARTModel(
    base_model
)

art_model.to(
    DEVICE
)

art_model.eval()


optimizer = torch.optim.Adam(
    art_model.parameters(),
    lr=0.001
)


classifier = PyTorchClassifier(
    model=art_model,
    loss=nn.CrossEntropyLoss(),
    input_shape=(78,),
    nb_classes=2,
    optimizer=optimizer,
    device_type=DEVICE
)


print(
    "ART classifier created successfully."
)


# ============================================================
# CLEAN PREDICTIONS
# ============================================================

print()

print(
    "Evaluating clean samples..."
)

clean_probabilities = classifier.predict(
    X_test,
    batch_size=256
)

clean_predictions = np.argmax(
    clean_probabilities,
    axis=1
)


(
    clean_accuracy,
    clean_precision,
    clean_recall,
    clean_f1
) = calculate_metrics(
    y_test,
    clean_predictions
)


clean_correct = (
    clean_predictions == y_test
)


print()

print(
    "Clean Results"
)

print(
    "-" * 50
)

print(
    f"Accuracy  : "
    f"{clean_accuracy * 100:.4f}%"
)

print(
    f"Precision : "
    f"{clean_precision:.6f}"
)

print(
    f"Recall    : "
    f"{clean_recall:.6f}"
)

print(
    f"F1        : "
    f"{clean_f1:.6f}"
)

print(
    f"Correct   : "
    f"{clean_correct.sum()}/{len(y_test)}"
)


# ============================================================
# PGD EXPERIMENT
# ============================================================

results = []


for epsilon in EPSILONS:

    print()

    print(
        "=" * 70
    )

    print(
        f"ART PGD - epsilon = "
        f"{epsilon}"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # PGD step size
    #
    # Same configuration as custom PyTorch PGD:
    #
    # alpha = epsilon / 10
    # --------------------------------------------------------

    eps_step = (
        epsilon /
        PGD_ITERATIONS
    )


    print(
        f"PGD step size     : "
        f"{eps_step}"
    )

    print(
        f"PGD iterations    : "
        f"{PGD_ITERATIONS}"
    )

    print(
        "Random start      : True"
    )


    # --------------------------------------------------------
    # Create ART PGD attack
    # --------------------------------------------------------

    attack = ProjectedGradientDescent(
        estimator=classifier,

        eps=epsilon,

        eps_step=eps_step,

        max_iter=PGD_ITERATIONS,

        targeted=False,

        num_random_init=1,

        batch_size=256
    )


    # --------------------------------------------------------
    # ART expects one-hot labels
    # --------------------------------------------------------

    y_one_hot = np.eye(
        2,
        dtype=np.float32
    )[y_test]


    # --------------------------------------------------------
    # Generate adversarial examples
    # --------------------------------------------------------

    print()

    print(
        "Generating adversarial examples..."
    )

    X_adv = attack.generate(
        x=X_test,
        y=y_one_hot
    )

    X_adv = X_adv.astype(
        np.float32
    )

    print(
        "Adversarial examples generated."
    )


    # --------------------------------------------------------
    # Evaluate adversarial examples
    # --------------------------------------------------------

    print(
        "Evaluating adversarial examples..."
    )

    adv_probabilities = classifier.predict(
        X_adv,
        batch_size=256
    )

    adv_predictions = np.argmax(
        adv_probabilities,
        axis=1
    )


    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    (
        accuracy,
        precision,
        recall,
        f1
    ) = calculate_metrics(
        y_test,
        adv_predictions
    )


    accuracy_drop = (
        clean_accuracy -
        accuracy
    )


    # --------------------------------------------------------
    # Evasion rate
    #
    # Only samples correctly classified before
    # the attack are considered.
    # --------------------------------------------------------

    successful_evasions = np.sum(
        clean_correct &
        (
            adv_predictions != y_test
        )
    )

    clean_correct_count = np.sum(
        clean_correct
    )

    evasion_rate = (
        successful_evasions /
        clean_correct_count
        if clean_correct_count > 0
        else 0.0
    )


    # --------------------------------------------------------
    # L2 perturbation
    # --------------------------------------------------------

    perturbations = (
        X_adv -
        X_test
    )

    l2_distances = np.linalg.norm(
        perturbations,
        axis=1
    )

    mean_l2 = np.mean(
        l2_distances
    )


    # --------------------------------------------------------
    # Maximum L-infinity perturbation
    #
    # Useful for verifying that ART respected epsilon.
    # --------------------------------------------------------

    linf_distances = np.max(
        np.abs(perturbations),
        axis=1
    )

    max_linf = np.max(
        linf_distances
    )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    tn = np.sum(
        (y_test == 0) &
        (adv_predictions == 0)
    )

    fp = np.sum(
        (y_test == 0) &
        (adv_predictions == 1)
    )

    fn = np.sum(
        (y_test == 1) &
        (adv_predictions == 0)
    )

    tp = np.sum(
        (y_test == 1) &
        (adv_predictions == 1)
    )


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    result = {

        "attack": "ART_PGD",

        "epsilon": epsilon,

        "pgd_iterations":
            PGD_ITERATIONS,

        "eps_step":
            eps_step,

        "n_samples":
            len(y_test),

        "clean_accuracy":
            clean_accuracy,

        "accuracy":
            accuracy,

        "accuracy_drop":
            accuracy_drop,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "evasion_rate":
            evasion_rate,

        "successful_evasions":
            successful_evasions,

        "clean_correct":
            clean_correct_count,

        "mean_l2":
            mean_l2,

        "max_linf":
            max_linf,

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "tp":
            tp
    }

    results.append(
        result
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()

    print(
        f"Epsilon             : "
        f"{epsilon}"
    )

    print(
        f"Accuracy            : "
        f"{accuracy * 100:.4f}%"
    )

    print(
        f"Accuracy Drop       : "
        f"{accuracy_drop * 100:.4f} pp"
    )

    print(
        f"Precision           : "
        f"{precision:.6f}"
    )

    print(
        f"Recall              : "
        f"{recall:.6f}"
    )

    print(
        f"F1                  : "
        f"{f1:.6f}"
    )

    print(
        f"Evasion Rate        : "
        f"{evasion_rate * 100:.4f}%"
    )

    print(
        f"Successful Evasions : "
        f"{successful_evasions}"
    )

    print(
        f"Mean L2             : "
        f"{mean_l2:.6f}"
    )

    print(
        f"Max L-infinity      : "
        f"{max_linf:.6f}"
    )

    print()

    print(
        "Confusion Matrix"
    )

    print(
        f"TN={tn}, "
        f"FP={fp}, "
        f"FN={fn}, "
        f"TP={tp}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()

print(
    "=" * 70
)

print(
    "ART PGD EXPERIMENT COMPLETED"
)

print(
    "=" * 70
)

print()

print(
    results_df.to_string(
        index=False
    )
)

print()

print(
    f"Results saved to:\n"
    f"{OUTPUT_PATH}"
)

print()

print(
    "=" * 70
)