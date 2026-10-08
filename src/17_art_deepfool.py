# ============================================================
# 17_art_deepfool.py
# ART DeepFool Attack - Corrected Version
# ============================================================

import os
import random
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

from joblib import load

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import DeepFool


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BASE_DIR = "/home/ubuntu/aml-experiment"

TEST_PATH = os.path.join(
    BASE_DIR,
    "data/ml/test.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "results/dnn_balanced.pth"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "results/dnn_balanced_scaler.pkl"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "results/art_deepfool_results_corrected.csv"
)


# ============================================================
# VALIDATION SET
# ============================================================

N_SAMPLES = 1000


# ============================================================
# ART DEEPFOOL PARAMETERS
# ============================================================

DEEPFOOL_MAX_ITER = 50

# IMPORTANT:
# In ART DeepFool, epsilon is the overshoot parameter.
# It is NOT an L-infinity perturbation limit.
DEEPFOOL_EPSILON = 0.02

DEEPFOOL_BATCH_SIZE = 32


# ============================================================
# ART INPUT DOMAIN
# ============================================================
#
# We use a standardized feature-space domain of [-5, 5].
#
# IMPORTANT:
# This is an experimental implementation choice.
# It is NOT specified by the paper.
#
# We will ONLY select samples that already fall inside
# this domain. We will NOT clip the original clean samples.
#
# ============================================================

ART_CLIP_MIN = -5.0
ART_CLIP_MAX = 5.0


# ============================================================
# 2. DNN MODEL
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
# 3. ART BINARY WRAPPER
# ============================================================
#
# Original DNN:
#
#       one output logit
#
# ART:
#
#       class 0 = BENIGN
#       class 1 = ATTACK
#
# We therefore convert:
#
#       logit
#
# into:
#
#       [-logit, +logit]
#
# ============================================================

class ARTBinaryWrapper(nn.Module):

    def __init__(self, original_model):

        super().__init__()

        self.original_model = original_model

    def forward(self, x):

        logit = self.original_model(x)

        if logit.dim() > 1:

            logit = logit.squeeze(1)

        benign_score = -logit
        attack_score = logit

        return torch.stack(
            [benign_score, attack_score],
            dim=1
        )


# ============================================================
# 4. LOAD TEST DATA
# ============================================================

print("=" * 70)
print("ART DEEPFOOL ATTACK - CORRECTED VERSION")
print("=" * 70)

print("\nDevice:", DEVICE)

print("\nLoading test data...")

df = pd.read_csv(TEST_PATH)

print(
    "Total test rows:",
    len(df)
)

print(
    "Total columns:",
    len(df.columns)
)


# ============================================================
# 5. SEPARATE FEATURES AND LABEL
# ============================================================

if "Label" not in df.columns:

    raise ValueError(
        "The test dataset does not contain a 'Label' column."
    )

X_df = df.drop(
    columns=["Label"]
)

y_raw = df["Label"]


# ============================================================
# 6. LABEL CONVERSION
# ============================================================

print("\nOriginal label values:")

print(
    y_raw.astype(str)
         .value_counts()
         .head(20)
)


def convert_labels(labels):

    labels_str = (
        labels
        .astype(str)
        .str.strip()
    )

    unique_values = set(
        labels_str.unique()
    )

    # --------------------------------------------------------
    # Case 1: 0 / 1
    # --------------------------------------------------------

    if unique_values.issubset(
        {"0", "1"}
    ):

        return (
            labels_str
            .astype(int)
            .values
        )

    # --------------------------------------------------------
    # Case 2: BENIGN / ATTACK
    # --------------------------------------------------------

    if unique_values.issubset(
        {"BENIGN", "ATTACK"}
    ):

        return (
            labels_str
            .map({
                "BENIGN": 0,
                "ATTACK": 1
            })
            .astype(int)
            .values
        )

    # --------------------------------------------------------
    # Case 3: CICIDS labels
    #
    # BENIGN = 0
    # Everything else = ATTACK
    # --------------------------------------------------------

    return (
        labels_str
        .apply(
            lambda x:
                0
                if x.upper() == "BENIGN"
                else 1
        )
        .astype(int)
        .values
    )


y = convert_labels(
    y_raw
)


print("\nConverted labels:")

print(
    "BENIGN:",
    np.sum(y == 0)
)

print(
    "ATTACK:",
    np.sum(y == 1)
)


# ============================================================
# 7. LOAD SCALER
# ============================================================

print("\nLoading scaler...")

scaler = load(
    SCALER_PATH
)

print(
    "Scaler loaded successfully."
)


# ============================================================
# 8. CONVERT FEATURES TO NUMERIC
# ============================================================

X_df = X_df.apply(
    pd.to_numeric,
    errors="coerce"
)

X_df = X_df.replace(
    [np.inf, -np.inf],
    np.nan
)

nan_count = (
    X_df.isna()
        .sum()
        .sum()
)

if nan_count > 0:

    print(
        "\nWARNING:",
        nan_count,
        "NaN values found."
    )

    X_df = X_df.fillna(0)


X = X_df.values.astype(
    np.float32
)


# ============================================================
# 9. SCALE FEATURES
# ============================================================

print("\nScaling features...")

X_scaled = scaler.transform(
    X
).astype(
    np.float32
)


print(
    "Scaled data shape:",
    X_scaled.shape
)

print(
    "Global scaled minimum:",
    X_scaled.min()
)

print(
    "Global scaled maximum:",
    X_scaled.max()
)


# ============================================================
# 10. FIND SAMPLES INSIDE ART DOMAIN
# ============================================================
#
# IMPORTANT:
#
# We DO NOT clip the data.
#
# Instead, we identify samples where EVERY feature already
# satisfies:
#
#       -5 <= feature <= +5
#
# This guarantees that the clean samples are valid inputs
# for the ART classifier without modifying them.
#
# ============================================================

print("\n" + "=" * 70)
print("CHECKING ART INPUT DOMAIN")
print("=" * 70)

inside_domain_mask = np.all(
    (X_scaled >= ART_CLIP_MIN) &
    (X_scaled <= ART_CLIP_MAX),
    axis=1
)

num_inside_domain = np.sum(
    inside_domain_mask
)

num_outside_domain = (
    len(X_scaled) -
    num_inside_domain
)


print(
    f"Samples already inside "
    f"[{ART_CLIP_MIN}, {ART_CLIP_MAX}]: "
    f"{num_inside_domain}"
)

print(
    "Samples outside ART domain:",
    num_outside_domain
)


if num_inside_domain < N_SAMPLES:

    raise RuntimeError(
        "\nNot enough samples inside the "
        f"[{ART_CLIP_MIN}, {ART_CLIP_MAX}] domain.\n"
        f"Required: {N_SAMPLES}\n"
        f"Available: {num_inside_domain}"
    )


# ============================================================
# 11. SELECT FIXED VALIDATION SUBSET
# ============================================================
#
# We randomly select 1000 samples ONLY from the valid domain.
#
# The random seed is still 42.
#
# ============================================================

valid_indices = np.where(
    inside_domain_mask
)[0]


rng = np.random.RandomState(
    SEED
)

selected_indices = rng.choice(
    valid_indices,
    size=N_SAMPLES,
    replace=False
)


X_test = X_scaled[
    selected_indices
]

y_test = y[
    selected_indices
]


print("\nValidation subset:")

print(
    "Samples:",
    len(X_test)
)

print(
    "BENIGN:",
    np.sum(y_test == 0)
)

print(
    "ATTACK:",
    np.sum(y_test == 1)
)


# ============================================================
# 12. VERIFY CLEAN DATA IS REALLY INSIDE DOMAIN
# ============================================================

clean_min = X_test.min()
clean_max = X_test.max()

clean_abs_max = np.max(
    np.abs(X_test)
)


print("\nClean validation data range:")

print(
    "Minimum:",
    clean_min
)

print(
    "Maximum:",
    clean_max
)

print(
    "Maximum absolute value:",
    clean_abs_max
)


if clean_min < ART_CLIP_MIN:

    raise RuntimeError(
        "ERROR: A clean value is below ART_CLIP_MIN."
    )


if clean_max > ART_CLIP_MAX:

    raise RuntimeError(
        "ERROR: A clean value is above ART_CLIP_MAX."
    )


print(
    "\n✓ All clean validation samples are "
    "inside the ART domain."
)


# ============================================================
# 13. CREATE ORIGINAL DNN
# ============================================================

print("\nLoading original DNN...")

input_dim = X_test.shape[1]

print(
    "Input features:",
    input_dim
)


original_model = DNN(
    input_dim=input_dim
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# ============================================================
# HANDLE CHECKPOINT FORMAT
# ============================================================

if isinstance(
    checkpoint,
    dict
):

    if "model_state_dict" in checkpoint:

        original_model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    elif "state_dict" in checkpoint:

        original_model.load_state_dict(
            checkpoint[
                "state_dict"
            ]
        )

    else:

        original_model.load_state_dict(
            checkpoint
        )

else:

    original_model.load_state_dict(
        checkpoint
    )


original_model = (
    original_model
    .to(DEVICE)
)

original_model.eval()


# ============================================================
# 14. ORIGINAL MODEL PREDICTION FUNCTION
# ============================================================

def original_predictions(
    X_input
):

    tensor_x = torch.tensor(
        X_input,
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():

        logits = original_model(
            tensor_x
        )

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= 0.5
        ).long()

    return (
        predictions
        .cpu()
        .numpy()
        .reshape(-1)
    )


# ============================================================
# 15. CLEAN PREDICTIONS
# ============================================================

clean_predictions = (
    original_predictions(
        X_test
    )
)


clean_accuracy = accuracy_score(
    y_test,
    clean_predictions
)


print("\n" + "=" * 70)

print(
    f"Clean Accuracy: "
    f"{clean_accuracy * 100:.4f}%"
)

print(
    "Clean Correct:",
    np.sum(
        clean_predictions == y_test
    ),
    "/",
    len(y_test)
)

print("=" * 70)


# ============================================================
# 16. CLEAN ACCURACY SAFETY CHECK
# ============================================================

if clean_accuracy < 0.90:

    raise RuntimeError(
        "\nClean accuracy is unexpectedly low.\n"
        "The model, scaler, or labels may be incorrect.\n"
        f"Current accuracy: "
        f"{clean_accuracy * 100:.2f}%"
    )


# ============================================================
# 17. CREATE ART MODEL
# ============================================================

print(
    "\nCreating ART binary wrapper..."
)


art_model = ARTBinaryWrapper(
    original_model
)


art_model = (
    art_model
    .to(DEVICE)
)

art_model.eval()


# ============================================================
# 18. LOSS FUNCTION
# ============================================================

loss_fn = nn.CrossEntropyLoss()


# ============================================================
# 19. CREATE ART CLASSIFIER
# ============================================================

print(
    "\nCreating ART PyTorchClassifier..."
)

print(
    f"ART clip range: "
    f"[{ART_CLIP_MIN}, {ART_CLIP_MAX}]"
)


classifier = PyTorchClassifier(

    model=art_model,

    loss=loss_fn,

    input_shape=(
        input_dim,
    ),

    nb_classes=2,

    optimizer=torch.optim.Adam(
        art_model.parameters(),
        lr=0.001
    ),

    clip_values=(
        ART_CLIP_MIN,
        ART_CLIP_MAX
    ),

    device_type=(
        "gpu"
        if torch.cuda.is_available()
        else "cpu"
    )
)


# ============================================================
# 20. VERIFY ART PREDICTIONS
# ============================================================

print(
    "\nChecking ART predictions "
    "against original model..."
)


art_clean_scores = classifier.predict(
    X_test,
    batch_size=DEEPFOOL_BATCH_SIZE
)


art_clean_predictions = np.argmax(
    art_clean_scores,
    axis=1
)


art_prediction_match = np.mean(
    art_clean_predictions ==
    clean_predictions
)


art_clean_accuracy = accuracy_score(
    y_test,
    art_clean_predictions
)


print(
    "Original vs ART prediction match:",
    f"{art_prediction_match * 100:.4f}%"
)

print(
    "ART clean accuracy:",
    f"{art_clean_accuracy * 100:.4f}%"
)


# ============================================================
# 21. ART WRAPPER SAFETY CHECK
# ============================================================

if art_prediction_match < 0.99:

    raise RuntimeError(
        "\nART wrapper predictions do not sufficiently "
        "match the original model.\n"
        f"Prediction match: "
        f"{art_prediction_match * 100:.2f}%"
    )


print(
    "\n✓ ART wrapper matches the original model."
)


# ============================================================
# 22. CREATE DEEPFOOL
# ============================================================

print(
    "\nCreating ART DeepFool attack..."
)

print(
    "Max iterations:",
    DEEPFOOL_MAX_ITER
)

print(
    "Epsilon / overshoot:",
    DEEPFOOL_EPSILON
)

print(
    "Batch size:",
    DEEPFOOL_BATCH_SIZE
)


attack = DeepFool(

    classifier=classifier,

    max_iter=DEEPFOOL_MAX_ITER,

    epsilon=DEEPFOOL_EPSILON,

    nb_grads=2,

    batch_size=DEEPFOOL_BATCH_SIZE,

    verbose=True
)


# ============================================================
# 23. GENERATE ADVERSARIAL EXAMPLES
# ============================================================

print("\n" + "=" * 70)

print(
    "GENERATING ART DEEPFOOL "
    "ADVERSARIAL EXAMPLES"
)

print("=" * 70)


X_adv = attack.generate(
    x=X_test
)


print(
    "\nAdversarial examples generated."
)

print(
    "Adversarial shape:",
    X_adv.shape
)


# ============================================================
# 24. VERIFY ADVERSARIAL DOMAIN
# ============================================================

adv_min = X_adv.min()
adv_max = X_adv.max()


print(
    "\nAdversarial data range:"
)

print(
    "Minimum:",
    adv_min
)

print(
    "Maximum:",
    adv_max
)


if adv_min < ART_CLIP_MIN - 1e-5:

    raise RuntimeError(
        "\nAdversarial examples are below "
        "ART_CLIP_MIN."
    )


if adv_max > ART_CLIP_MAX + 1e-5:

    raise RuntimeError(
        "\nAdversarial examples are above "
        "ART_CLIP_MAX."
    )


print(
    "\n✓ Adversarial examples remain "
    "inside the ART domain."
)


# ============================================================
# 25. CALCULATE PERTURBATIONS
# ============================================================

print(
    "\nCalculating perturbations..."
)


perturbations = (
    X_adv - X_test
)


# L2 norm per sample
l2_norms = np.linalg.norm(
    perturbations,
    ord=2,
    axis=1
)


# L-infinity norm per sample
linf_norms = np.linalg.norm(
    perturbations,
    ord=np.inf,
    axis=1
)


print(
    "\nPerturbation statistics:"
)

print(
    f"Mean L2: "
    f"{l2_norms.mean():.6f}"
)

print(
    f"Median L2: "
    f"{np.median(l2_norms):.6f}"
)

print(
    f"Maximum L2: "
    f"{l2_norms.max():.6f}"
)

print(
    f"Mean L-infinity: "
    f"{linf_norms.mean():.6f}"
)

print(
    f"Maximum L-infinity: "
    f"{linf_norms.max():.6f}"
)


# ============================================================
# 26. ADVERSARIAL PREDICTIONS
# ============================================================

print(
    "\nCalculating adversarial predictions..."
)


adv_scores = classifier.predict(
    X_adv,
    batch_size=DEEPFOOL_BATCH_SIZE
)


adv_predictions = np.argmax(
    adv_scores,
    axis=1
)


# ============================================================
# 27. METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    adv_predictions
)


precision = precision_score(
    y_test,
    adv_predictions,
    zero_division=0
)


recall = recall_score(
    y_test,
    adv_predictions,
    zero_division=0
)


f1 = f1_score(
    y_test,
    adv_predictions,
    zero_division=0
)


accuracy_drop = (
    clean_accuracy -
    accuracy
)


# ============================================================
# 28. EVASION RATE
# ============================================================
#
# Successful evasion:
#
#   original model was correct
#   AND
#   adversarial prediction is incorrect
#
# ============================================================

clean_correct_mask = (
    clean_predictions ==
    y_test
)


successful_evasion_mask = (
    clean_correct_mask &
    (
        adv_predictions !=
        y_test
    )
)


successful_evasions = np.sum(
    successful_evasion_mask
)


clean_correct = np.sum(
    clean_correct_mask
)


if clean_correct > 0:

    evasion_rate = (
        successful_evasions /
        clean_correct
    )

else:

    evasion_rate = 0.0


# ============================================================
# 29. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    adv_predictions,
    labels=[0, 1]
)


tn, fp, fn, tp = cm.ravel()


# ============================================================
# 30. PRINT FINAL RESULTS
# ============================================================

print("\n")

print("=" * 70)

print(
    "ART DEEPFOOL RESULTS - CORRECTED"
)

print("=" * 70)


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
    f"Clean Correct        : "
    f"{clean_correct}"
)

print(
    f"Mean L2              : "
    f"{l2_norms.mean():.6f}"
)

print(
    f"Median L2             : "
    f"{np.median(l2_norms):.6f}"
)

print(
    f"Max L2               : "
    f"{l2_norms.max():.6f}"
)

print(
    f"Mean L-infinity      : "
    f"{linf_norms.mean():.6f}"
)

print(
    f"Max L-infinity       : "
    f"{linf_norms.max():.6f}"
)


print(
    "\nConfusion Matrix:"
)

print(
    "TN:",
    tn,
    " FP:",
    fp
)

print(
    "FN:",
    fn,
    " TP:",
    tp
)


print("=" * 70)


# ============================================================
# 31. SAVE RESULTS
# ============================================================

results = pd.DataFrame({

    "Attack": [
        "ART DeepFool"
    ],

    "Clean_Accuracy": [
        clean_accuracy
    ],

    "Accuracy": [
        accuracy
    ],

    "Accuracy_Drop": [
        accuracy_drop
    ],

    "Precision": [
        precision
    ],

    "Recall": [
        recall
    ],

    "F1": [
        f1
    ],

    "Evasion_Rate": [
        evasion_rate
    ],

    "Successful_Evasions": [
        successful_evasions
    ],

    "Clean_Correct": [
        clean_correct
    ],

    "Mean_L2": [
        l2_norms.mean()
    ],

    "Median_L2": [
        np.median(l2_norms)
    ],

    "Max_L2": [
        l2_norms.max()
    ],

    "Mean_Linf": [
        linf_norms.mean()
    ],

    "Max_Linf": [
        linf_norms.max()
    ],

    "TN": [
        tn
    ],

    "FP": [
        fp
    ],

    "FN": [
        fn
    ],

    "TP": [
        tp
    ],

    "Max_Iterations": [
        DEEPFOOL_MAX_ITER
    ],

    "Epsilon_Overshoot": [
        DEEPFOOL_EPSILON
    ],

    "ART_Clip_Min": [
        ART_CLIP_MIN
    ],

    "ART_Clip_Max": [
        ART_CLIP_MAX
    ],

    "Valid_Domain_Samples": [
        num_inside_domain
    ]

})


results.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 32. FINISHED
# ============================================================

print(
    "\nResults saved to:"
)

print(
    OUTPUT_PATH
)

print(
    "\nDone."
)