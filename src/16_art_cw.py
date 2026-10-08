# ============================================================
# ART C&W EXPERIMENT - CORRECTED VERSION
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
from art.attacks.evasion import CarliniL2Method


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
    "art_cw_results_corrected.csv"
)

SEED = 42

N_SAMPLES = 1000

# C&W parameters
CW_MAX_ITER = 30
CW_LEARNING_RATE = 0.01
CW_BINARY_SEARCH_STEPS = 1
CW_INITIAL_CONST = 1.0
CW_CONFIDENCE = 0.0

# CPU settings
ATTACK_BATCH_SIZE = 32

DEVICE = "cpu"

# ------------------------------------------------------------
# IMPORTANT:
# ART requires an input domain for this C&W configuration.
#
# The model operates in StandardScaler feature space.
# We use a fixed standardized domain [-5, +5].
#
# This is an ART-specific bounded implementation.
# It is NOT claimed to be identical to the original
# unconstrained custom C&W implementation.
# ------------------------------------------------------------

ART_CLIP_MIN = -5.0
ART_CLIP_MAX = 5.0


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)

torch.manual_seed(SEED)


# ============================================================
# DNN MODEL
#
# Original architecture:
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
# Original model:
#       one binary logit
#
# ART:
#       two class logits
#
#       class 0 = -logit
#       class 1 = +logit
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

print(
    "ART C&W EXPERIMENT - CORRECTED"
)

print("=" * 70)

print()

print("Configuration:")

print(
    f"Test file              : {TEST_PATH}"
)

print(
    f"Model                  : {MODEL_PATH}"
)

print(
    f"Scaler                 : {SCALER_PATH}"
)

print(
    f"Samples                : {N_SAMPLES}"
)

print(
    f"C&W max iterations     : {CW_MAX_ITER}"
)

print(
    f"C&W learning rate      : {CW_LEARNING_RATE}"
)

print(
    f"Binary search steps    : "
    f"{CW_BINARY_SEARCH_STEPS}"
)

print(
    f"Initial constant       : "
    f"{CW_INITIAL_CONST}"
)

print(
    f"Confidence (kappa)     : "
    f"{CW_CONFIDENCE}"
)

print(
    f"Attack batch size      : "
    f"{ATTACK_BATCH_SIZE}"
)

print(
    f"ART feature minimum    : "
    f"{ART_CLIP_MIN}"
)

print(
    f"ART feature maximum    : "
    f"{ART_CLIP_MAX}"
)

print(
    f"Device                 : "
    f"{DEVICE}"
)

print()


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "Loading model..."
)

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

print(
    "Loading scaler..."
)

scaler = joblib.load(
    SCALER_PATH
)

print(
    "Scaler loaded successfully."
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print()

print(
    "Loading test dataset..."
)

test_df = pd.read_csv(
    TEST_PATH
)

print(
    f"Full test dataset shape: "
    f"{test_df.shape}"
)


# ============================================================
# PREPARE FEATURES AND LABELS
# ============================================================

print()

print(
    "Preparing test data..."
)

if "Label" not in test_df.columns:

    raise ValueError(
        "The test dataset does not contain "
        "a 'Label' column."
    )


X_df = test_df.drop(
    columns=["Label"]
)

y_series = test_df["Label"]


# ============================================================
# CONVERT LABELS
# ============================================================

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
# HANDLE INVALID VALUES
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
    f"Features shape: "
    f"{X.shape}"
)

print(
    f"Labels shape  : "
    f"{y.shape}"
)


# ============================================================
# STANDARDIZE
# ============================================================

print()

print(
    "Applying StandardScaler..."
)

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
# SELECT SAME 1,000-SAMPLE SUBSET
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
# CHECK WHETHER VALIDATION SAMPLES ARE INSIDE ART DOMAIN
# ============================================================

outside_domain = np.logical_or(
    X_test < ART_CLIP_MIN,
    X_test > ART_CLIP_MAX
)

outside_count = np.sum(
    outside_domain
)

print()

print(
    f"Values outside ART domain: "
    f"{outside_count}"
)


# ------------------------------------------------------------
# We do NOT silently change the original test samples.
#
# ART will constrain adversarial examples to the defined
# domain. The clean model is still evaluated on the original
# standardized samples.
# ------------------------------------------------------------


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


# ============================================================
# CREATE ART CLASSIFIER
# ============================================================

classifier = PyTorchClassifier(

    model=art_model,

    loss=nn.CrossEntropyLoss(),

    input_shape=(78,),

    nb_classes=2,

    optimizer=optimizer,

    device_type=DEVICE,

    clip_values=(
        ART_CLIP_MIN,
        ART_CLIP_MAX
    )
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
# CREATE C&W ATTACK
# ============================================================

print()

print(
    "=" * 70
)

print(
    "STARTING ART C&W ATTACK"
)

print(
    "=" * 70
)

print()

print(
    "C&W is computationally expensive on CPU."
)

print(
    "Please allow the experiment to finish."
)

print()


attack = CarliniL2Method(

    classifier=classifier,

    confidence=CW_CONFIDENCE,

    targeted=False,

    learning_rate=CW_LEARNING_RATE,

    binary_search_steps=CW_BINARY_SEARCH_STEPS,

    max_iter=CW_MAX_ITER,

    initial_const=CW_INITIAL_CONST,

    batch_size=ATTACK_BATCH_SIZE,

    verbose=True
)


# ============================================================
# ONE-HOT LABELS
# ============================================================

y_one_hot = np.eye(
    2,
    dtype=np.float32
)[y_test]


# ============================================================
# GENERATE ADVERSARIAL EXAMPLES
# ============================================================

print()

print(
    "Generating C&W adversarial examples..."
)

X_adv = attack.generate(
    x=X_test,
    y=y_one_hot
)

X_adv = X_adv.astype(
    np.float32
)


print()

print(
    "C&W adversarial examples generated."
)


# ============================================================
# VERIFY ART CLIPPING
# ============================================================

print()

print(
    "Checking adversarial feature range..."
)

print(
    f"Adversarial minimum: "
    f"{X_adv.min():.6f}"
)

print(
    f"Adversarial maximum: "
    f"{X_adv.max():.6f}"
)


# ============================================================
# EVALUATE ADVERSARIAL EXAMPLES
# ============================================================

print()

print(
    "Evaluating C&W adversarial examples..."
)

adv_probabilities = classifier.predict(
    X_adv,
    batch_size=256
)

adv_predictions = np.argmax(
    adv_probabilities,
    axis=1
)


# ============================================================
# CALCULATE METRICS
# ============================================================

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


# ============================================================
# EVASION RATE
# ============================================================

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


# ============================================================
# L2 DISTANCES
# ============================================================

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

median_l2 = np.median(
    l2_distances
)

max_l2 = np.max(
    l2_distances
)


# ============================================================
# L-INFINITY DISTANCES
# ============================================================

linf_distances = np.max(
    np.abs(
        perturbations
    ),
    axis=1
)

mean_linf = np.mean(
    linf_distances
)

max_linf = np.max(
    linf_distances
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

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


# ============================================================
# RESULTS DICTIONARY
# ============================================================

result = {

    "attack":
        "ART_CW",

    "n_samples":
        len(y_test),

    "cw_max_iter":
        CW_MAX_ITER,

    "cw_learning_rate":
        CW_LEARNING_RATE,

    "cw_binary_search_steps":
        CW_BINARY_SEARCH_STEPS,

    "cw_initial_const":
        CW_INITIAL_CONST,

    "cw_confidence":
        CW_CONFIDENCE,

    "art_clip_min":
        ART_CLIP_MIN,

    "art_clip_max":
        ART_CLIP_MAX,

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

    "median_l2":
        median_l2,

    "max_l2":
        max_l2,

    "mean_linf":
        mean_linf,

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


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    [result]
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()

print(
    "=" * 70
)

print(
    "ART C&W RESULTS - CORRECTED"
)

print(
    "=" * 70
)

print()

print(
    f"Clean Accuracy       : "
    f"{clean_accuracy * 100:.4f}%"
)

print(
    f"Accuracy             : "
    f"{accuracy * 100:.4f}%"
)

print(
    f"Accuracy Drop        : "
    f"{accuracy_drop * 100:.4f} pp"
)

print(
    f"Precision            : "
    f"{precision:.6f}"
)

print(
    f"Recall               : "
    f"{recall:.6f}"
)

print(
    f"F1                   : "
    f"{f1:.6f}"
)

print(
    f"Evasion Rate         : "
    f"{evasion_rate * 100:.4f}%"
)

print(
    f"Successful Evasions  : "
    f"{successful_evasions}"
)

print(
    f"Mean L2              : "
    f"{mean_l2:.6f}"
)

print(
    f"Median L2            : "
    f"{median_l2:.6f}"
)

print(
    f"Max L2               : "
    f"{max_l2:.6f}"
)

print(
    f"Mean L-infinity      : "
    f"{mean_linf:.6f}"
)

print(
    f"Max L-infinity       : "
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

print()

print(
    "=" * 70
)

print(
    "ART C&W EXPERIMENT COMPLETED"
)

print(
    "=" * 70
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