# ============================================================
# 13_defense_validation.py
#
# Defense Evaluation:
# FGSM Adversarial Training
#
# Compares:
#   Original Model vs Defended Model
#
# Attacks:
#   1. FGSM       epsilon = 0.10
#   2. PGD        epsilon = 0.10
#   3. C&W
#   4. DeepFool
#
# Validation:
#   - 1000 samples
#   - Random seed = 42
#   - Balanced CICIDS2017 training
#   - Same scaler
#
# IMPORTANT:
# The attacks are generated against the DEFENDED model.
# Therefore this is a white-box evaluation of the defense.
# ============================================================

import os
import glob
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

SCALER_PATH = os.path.join(
    BASE_DIR,
    "results/dnn_balanced_scaler.pkl"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


# ============================================================
# VALIDATION
# ============================================================

N_SAMPLES = 1000


# ============================================================
# ATTACK PARAMETERS
# ============================================================

# FGSM
FGSM_EPSILON = 0.10


# PGD
PGD_EPSILON = 0.10
PGD_ALPHA = PGD_EPSILON / 10
PGD_STEPS = 10


# C&W
CW_STEPS = 30
CW_LR = 0.01
CW_C = 1.0
CW_KAPPA = 0.0


# DeepFool
DEEPFOOL_MAX_ITER = 50
DEEPFOOL_EPSILON = 0.02


# CPU batch size
BATCH_SIZE = 256


# ============================================================
# 2. ORIGINAL-MODEL BASELINES
# ============================================================
#
# These are the results from our previous fixed 1000-sample
# validation experiment using the ORIGINAL model.
#
# Same:
#   - test.csv
#   - seed = 42
#   - 1000 validation samples
#
# These values allow us to calculate:
#
#     Defense Gain =
#     Defended Accuracy - Original Accuracy
#
# ============================================================

ORIGINAL_ATTACK_ACCURACY = {

    "FGSM_eps_0.10": 0.3420,

    "PGD_eps_0.10": 0.3640,

    "C&W": 0.0340,

    "DeepFool": 0.0140
}


ORIGINAL_CLEAN_ACCURACY = 0.9920


# ============================================================
# 3. DNN MODEL
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
# 4. FIND DEFENDED MODEL
# ============================================================

def find_defended_model():

    candidate_names = [

        "dnn_adversarial_trained.pth",
        "dnn_adversarial_training.pth",
        "adversarial_trained_dnn.pth",
        "adversarial_training_model.pth",
        "dnn_defended.pth",
        "defended_model.pth"
    ]


    # --------------------------------------------------------
    # Check expected filenames
    # --------------------------------------------------------

    for name in candidate_names:

        path = os.path.join(
            RESULTS_DIR,
            name
        )

        if os.path.exists(path):

            return path


    # --------------------------------------------------------
    # Search for likely files
    # --------------------------------------------------------

    patterns = [

        "*adversarial*.pth",
        "*defen*.pth",
        "*adv_train*.pth"
    ]


    found = []


    for pattern in patterns:

        found.extend(
            glob.glob(
                os.path.join(
                    RESULTS_DIR,
                    pattern
                )
            )
        )


    found = list(
        dict.fromkeys(found)
    )


    if len(found) == 1:

        return found[0]


    if len(found) > 1:

        print(
            "\nPossible defended models found:"
        )

        for i, path in enumerate(found):

            print(
                f"{i + 1}. "
                f"{os.path.basename(path)}"
            )


        for path in found:

            filename = (
                os.path.basename(path)
                .lower()
            )

            if (
                "adversarial" in filename
                or
                "adv_train" in filename
            ):

                return path


    raise FileNotFoundError(

        "\nCould not find the "
        "adversarially trained model.\n\n"

        "Available .pth files:\n"

        +
        "\n".join(

            os.path.basename(x)

            for x in glob.glob(
                os.path.join(
                    RESULTS_DIR,
                    "*.pth"
                )
            )
        )
    )


# ============================================================
# 5. LABEL CONVERSION
# ============================================================

def convert_labels(labels):

    labels_str = (
        labels
        .astype(str)
        .str.strip()
    )


    unique_values = set(
        labels_str.unique()
    )


    # 0 / 1
    if unique_values.issubset(
        {"0", "1"}
    ):

        return (
            labels_str
            .astype(int)
            .values
        )


    # BENIGN / ATTACK
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


    # CICIDS2017 labels
    #
    # BENIGN = 0
    # Everything else = 1

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


# ============================================================
# 6. LOAD DATA
# ============================================================

print("=" * 70)

print(
    "DEFENSE VALIDATION"
)

print("=" * 70)


print(
    "\nDevice:",
    DEVICE
)


print(
    "\nLoading test data..."
)


df = pd.read_csv(
    TEST_PATH
)


print(
    "Total test rows:",
    len(df)
)


print(
    "Total columns:",
    len(df.columns)
)


if "Label" not in df.columns:

    raise ValueError(
        "Label column not found."
    )


X_df = df.drop(
    columns=["Label"]
)

y_raw = df["Label"]


# ============================================================
# 7. LABELS
# ============================================================

y = convert_labels(
    y_raw
)


print(
    "\nDataset labels:"
)


print(
    "BENIGN:",
    np.sum(y == 0)
)


print(
    "ATTACK:",
    np.sum(y == 1)
)


# ============================================================
# 8. NUMERIC FEATURES
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
        "\nNaN values:",
        nan_count
    )

    X_df = X_df.fillna(0)


X = X_df.values.astype(
    np.float32
)


# ============================================================
# 9. LOAD SCALER
# ============================================================

print(
    "\nLoading scaler..."
)


scaler = load(
    SCALER_PATH
)


X_scaled = scaler.transform(
    X
).astype(
    np.float32
)


print(
    "Scaled shape:",
    X_scaled.shape
)


# ============================================================
# 10. FIXED VALIDATION SUBSET
# ============================================================

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
    "\nValidation samples:",
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
# 11. LOAD DEFENDED MODEL
# ============================================================

defended_model_path = (
    find_defended_model()
)


print(
    "\nDefended model:"
)


print(
    defended_model_path
)


input_dim = X_test.shape[1]


model = DNN(
    input_dim=input_dim
)


checkpoint = torch.load(
    defended_model_path,
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

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    elif "state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint[
                "state_dict"
            ]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(
    DEVICE
)


model.eval()


# ============================================================
# 12. PREDICTION FUNCTION
# ============================================================

def predict_numpy(
    model,
    X_input
):

    predictions = []


    for start in range(
        0,
        len(X_input),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(X_input)
        )


        batch = torch.tensor(
            X_input[start:end],
            dtype=torch.float32,
            device=DEVICE
        )


        with torch.no_grad():

            logits = model(
                batch
            )


            probabilities = (
                torch.sigmoid(
                    logits
                )
            )


            batch_predictions = (
                probabilities >= 0.5
            ).long()


        predictions.append(

            batch_predictions
            .cpu()
            .numpy()
            .reshape(-1)
        )


    return np.concatenate(
        predictions
    )


# ============================================================
# 13. METRIC FUNCTION
# ============================================================

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


    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )


    tn, fp, fn, tp = (
        cm.ravel()
    )


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
# 14. CLEAN DEFENDED MODEL
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "CLEAN DEFENDED MODEL"
)

print(
    "=" * 70
)


clean_predictions = (
    predict_numpy(
        model,
        X_test
    )
)


clean_metrics = (
    calculate_metrics(
        y_test,
        clean_predictions
    )
)


clean_accuracy = (
    clean_metrics["accuracy"]
)


print(
    f"\nDefended clean accuracy: "
    f"{clean_accuracy * 100:.4f}%"
)


print(
    "Clean correct:",
    np.sum(
        clean_predictions == y_test
    ),
    "/",
    len(y_test)
)


print(
    f"Clean accuracy change "
    f"from original: "
    f"{(clean_accuracy - ORIGINAL_CLEAN_ACCURACY) * 100:+.4f} pp"
)


# ============================================================
# 15. GENERIC ATTACK EVALUATION
# ============================================================

def evaluate_attack(
    attack_name,
    X_adv
):

    print(
        f"\nEvaluating {attack_name}..."
    )


    predictions = (
        predict_numpy(
            model,
            X_adv
        )
    )


    metrics = (
        calculate_metrics(
            y_test,
            predictions
        )
    )


    accuracy = (
        metrics["accuracy"]
    )


    # --------------------------------------------------------
    # How much the attack reduced accuracy of the DEFENDED
    # model
    # --------------------------------------------------------

    accuracy_drop = (
        clean_accuracy -
        accuracy
    )


    # --------------------------------------------------------
    # Defense gain
    #
    # Defended accuracy under attack
    # minus
    # Original model accuracy under same attack
    # --------------------------------------------------------

    original_accuracy = (
        ORIGINAL_ATTACK_ACCURACY[
            attack_name
        ]
    )


    defense_gain = (
        accuracy -
        original_accuracy
    )


    # --------------------------------------------------------
    # Evasion rate
    # --------------------------------------------------------

    clean_correct_mask = (
        clean_predictions ==
        y_test
    )


    successful_mask = (
        clean_correct_mask &
        (
            predictions !=
            y_test
        )
    )


    successful_evasions = (
        np.sum(
            successful_mask
        )
    )


    clean_correct = (
        np.sum(
            clean_correct_mask
        )
    )


    if clean_correct > 0:

        evasion_rate = (
            successful_evasions /
            clean_correct
        )

    else:

        evasion_rate = 0.0


    # --------------------------------------------------------
    # Perturbation
    # --------------------------------------------------------

    perturbation = (
        X_adv -
        X_test
    )


    l2 = np.linalg.norm(
        perturbation,
        ord=2,
        axis=1
    )


    linf = np.linalg.norm(
        perturbation,
        ord=np.inf,
        axis=1
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"\n{attack_name}"
    )


    print(
        f"Original Accuracy: "
        f"{original_accuracy * 100:.4f}%"
    )


    print(
        f"Defended Accuracy: "
        f"{accuracy * 100:.4f}%"
    )


    print(
        f"Defense Gain: "
        f"{defense_gain * 100:+.4f} pp"
    )


    print(
        f"Defended Accuracy Drop: "
        f"{accuracy_drop * 100:.4f} pp"
    )


    print(
        f"Precision: "
        f"{metrics['precision']:.6f}"
    )


    print(
        f"Recall: "
        f"{metrics['recall']:.6f}"
    )


    print(
        f"F1: "
        f"{metrics['f1']:.6f}"
    )


    print(
        f"Evasion Rate: "
        f"{evasion_rate * 100:.4f}%"
    )


    print(
        f"Successful Evasions: "
        f"{successful_evasions}"
    )


    print(
        f"Mean L2: "
        f"{l2.mean():.6f}"
    )


    print(
        f"Mean L-infinity: "
        f"{linf.mean():.6f}"
    )


    print(
        "Confusion Matrix:"
    )


    print(
        f"TN={metrics['tn']} "
        f"FP={metrics['fp']} "
        f"FN={metrics['fn']} "
        f"TP={metrics['tp']}"
    )


    return {

        "Attack": attack_name,

        "Original_Accuracy":
            original_accuracy,

        "Defended_Accuracy":
            accuracy,

        "Defense_Gain":
            defense_gain,

        "Defended_Accuracy_Drop":
            accuracy_drop,

        "Precision":
            metrics["precision"],

        "Recall":
            metrics["recall"],

        "F1":
            metrics["f1"],

        "Evasion_Rate":
            evasion_rate,

        "Successful_Evasions":
            successful_evasions,

        "Clean_Correct":
            clean_correct,

        "Mean_L2":
            l2.mean(),

        "Median_L2":
            np.median(l2),

        "Max_L2":
            l2.max(),

        "Mean_Linf":
            linf.mean(),

        "Max_Linf":
            linf.max(),

        "TN":
            metrics["tn"],

        "FP":
            metrics["fp"],

        "FN":
            metrics["fn"],

        "TP":
            metrics["tp"]
    }


# ============================================================
# 16. FGSM
# ============================================================

def generate_fgsm(
    model,
    X_input,
    y_input,
    epsilon
):

    X_adv = []


    model.eval()


    for start in range(
        0,
        len(X_input),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(X_input)
        )


        x = torch.tensor(
            X_input[start:end],
            dtype=torch.float32,
            device=DEVICE,
            requires_grad=True
        )


        y_batch = torch.tensor(
            y_input[start:end],
            dtype=torch.float32,
            device=DEVICE
        )


        model.zero_grad()


        logits = (
            model(x)
            .squeeze(1)
        )


        loss = (
            nn.functional
            .binary_cross_entropy_with_logits(
                logits,
                y_batch
            )
        )


        loss.backward()


        gradient = (
            x.grad.sign()
        )


        x_adv = (
            x +
            epsilon * gradient
        )


        X_adv.append(
            x_adv.detach()
            .cpu()
            .numpy()
        )


    return np.vstack(
        X_adv
    )


# ============================================================
# 17. RUN FGSM
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "FGSM ε = 0.10"
)

print(
    "=" * 70
)


X_adv_fgsm = generate_fgsm(
    model,
    X_test,
    y_test,
    FGSM_EPSILON
)


fgsm_result = (
    evaluate_attack(
        "FGSM_eps_0.10",
        X_adv_fgsm
    )
)


# ============================================================
# 18. PGD
# ============================================================

def generate_pgd(
    model,
    X_input,
    y_input,
    epsilon,
    alpha,
    steps
):

    X_adv_all = []


    model.eval()


    for start in range(
        0,
        len(X_input),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(X_input)
        )


        x_original = torch.tensor(
            X_input[start:end],
            dtype=torch.float32,
            device=DEVICE
        )


        y_batch = torch.tensor(
            y_input[start:end],
            dtype=torch.float32,
            device=DEVICE
        )


        # Random start
        x_adv = (
            x_original +
            torch.empty_like(
                x_original
            ).uniform_(
                -epsilon,
                epsilon
            )
        )


        for _ in range(
            steps
        ):

            x_adv.requires_grad_(
                True
            )


            model.zero_grad()


            logits = (
                model(x_adv)
                .squeeze(1)
            )


            loss = (
                nn.functional
                .binary_cross_entropy_with_logits(
                    logits,
                    y_batch
                )
            )


            loss.backward()


            gradient = (
                x_adv.grad.sign()
            )


            x_adv = (
                x_adv.detach() +
                alpha * gradient
            )


            # Project onto epsilon ball
            perturbation = (
                x_adv -
                x_original
            )


            perturbation = torch.clamp(
                perturbation,
                -epsilon,
                epsilon
            )


            x_adv = (
                x_original +
                perturbation
            ).detach()


        X_adv_all.append(
            x_adv.cpu().numpy()
        )


    return np.vstack(
        X_adv_all
    )


# ============================================================
# 19. RUN PGD
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "PGD ε = 0.10"
)

print(
    "=" * 70
)


X_adv_pgd = generate_pgd(
    model,
    X_test,
    y_test,
    PGD_EPSILON,
    PGD_ALPHA,
    PGD_STEPS
)


pgd_result = (
    evaluate_attack(
        "PGD_eps_0.10",
        X_adv_pgd
    )
)


# ============================================================
# 20. C&W
# ============================================================

def generate_cw(
    model,
    X_input,
    y_input,
    steps=CW_STEPS,
    lr=CW_LR,
    c=CW_C,
    kappa=CW_KAPPA
):

    all_adv = []


    model.eval()


    for start in range(
        0,
        len(X_input),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(X_input)
        )


        x_original = torch.tensor(
            X_input[start:end],
            dtype=torch.float32,
            device=DEVICE
        )


        y_batch = torch.tensor(
            y_input[start:end],
            dtype=torch.float32,
            device=DEVICE
        )


        delta = torch.zeros_like(
            x_original,
            requires_grad=True
        )


        optimizer = torch.optim.Adam(
            [delta],
            lr=lr
        )


        best_adv = (
            x_original.clone()
        )


        best_l2 = torch.full(
            (len(x_original),),
            float("inf"),
            device=DEVICE
        )


        for step in range(
            steps
        ):

            optimizer.zero_grad()


            x_adv = (
                x_original +
                delta
            )


            logits = (
                model(x_adv)
                .squeeze(1)
            )


            # Targeted class:
            #
            # BENIGN (0) -> ATTACK (1)
            # ATTACK (1) -> BENIGN (0)

            target = (
                1.0 -
                y_batch
            )


            real = torch.where(
                target == 1,
                logits,
                -logits
            )


            other = torch.where(
                target == 1,
                -logits,
                logits
            )


            f = torch.clamp(
                real -
                other +
                kappa,
                min=0.0
            )


            l2 = torch.sum(
                delta ** 2,
                dim=1
            )


            loss = torch.mean(
                l2 +
                c * f
            )


            loss.backward()


            optimizer.step()


            with torch.no_grad():

                current_predictions = (
                    torch.sigmoid(
                        logits
                    ) >= 0.5
                ).float()


                successful = (
                    current_predictions !=
                    y_batch
                )


                current_l2 = (
                    torch.sum(
                        delta ** 2,
                        dim=1
                    )
                )


                improved = (
                    successful &
                    (
                        current_l2 <
                        best_l2
                    )
                )


                if torch.any(
                    improved
                ):

                    best_adv[
                        improved
                    ] = x_adv[
                        improved
                    ].detach()


                    best_l2[
                        improved
                    ] = current_l2[
                        improved
                    ]


        all_adv.append(
            best_adv.detach()
            .cpu()
            .numpy()
        )


        print(
            f"C&W processed "
            f"{end}/{len(X_input)} samples"
        )


    return np.vstack(
        all_adv
    )


# ============================================================
# 21. RUN C&W
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "C&W"
)

print(
    "=" * 70
)


print(
    "\nParameters:"
)

print(
    "Steps:",
    CW_STEPS
)

print(
    "Learning rate:",
    CW_LR
)

print(
    "C:",
    CW_C
)

print(
    "Kappa:",
    CW_KAPPA
)


X_adv_cw = generate_cw(
    model,
    X_test,
    y_test
)


cw_result = (
    evaluate_attack(
        "C&W",
        X_adv_cw
    )
)


# ============================================================
# 22. DEEPFOOL
# ============================================================

def generate_deepfool(
    model,
    X_input,
    y_input,
    max_iter=50,
    overshoot=0.02
):

    all_adv = []


    model.eval()


    for start in range(
        0,
        len(X_input),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(X_input)
        )


        x_original = torch.tensor(
            X_input[start:end],
            dtype=torch.float32,
            device=DEVICE
        )


        x_adv = (
            x_original.clone()
        )


        original_predictions = (
            predict_numpy(
                model,
                X_input[start:end]
            )
        )


        for _ in range(
            max_iter
        ):

            x_adv.requires_grad_(
                True
            )


            logits = (
                model(x_adv)
                .squeeze(1)
            )


            predictions = (
                torch.sigmoid(
                    logits
                ) >= 0.5
            ).long()


            original_tensor = torch.tensor(
                original_predictions,
                dtype=torch.long,
                device=DEVICE
            )


            still_correct = (
                predictions ==
                original_tensor
            )


            if not torch.any(
                still_correct
            ):

                break


            model.zero_grad()


            gradient = torch.autograd.grad(
                logits.sum(),
                x_adv,
                retain_graph=False,
                create_graph=False
            )[0]


            logit_abs = (
                torch.abs(
                    logits
                )
            )


            gradient_norm_sq = (
                torch.sum(
                    gradient ** 2,
                    dim=1
                )
                + 1e-12
            )


            step_size = (
                logit_abs /
                gradient_norm_sq
            )


            direction = torch.sign(
                logits
            ).unsqueeze(1)


            perturbation = (
                -step_size.unsqueeze(1)
                *
                direction
                *
                gradient
            )


            x_adv = (
                x_adv.detach()
                +
                (1.0 + overshoot)
                *
                perturbation
            )


        all_adv.append(
            x_adv.detach()
            .cpu()
            .numpy()
        )


        print(
            f"DeepFool processed "
            f"{end}/{len(X_input)} samples"
        )


    return np.vstack(
        all_adv
    )


# ============================================================
# 23. RUN DEEPFOOL
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "DeepFool"
)

print(
    "=" * 70
)


print(
    "\nParameters:"
)

print(
    "Max iterations:",
    DEEPFOOL_MAX_ITER
)

print(
    "Overshoot epsilon:",
    DEEPFOOL_EPSILON
)


X_adv_deepfool = (
    generate_deepfool(
        model,
        X_test,
        y_test,
        max_iter=DEEPFOOL_MAX_ITER,
        overshoot=DEEPFOOL_EPSILON
    )
)


deepfool_result = (
    evaluate_attack(
        "DeepFool",
        X_adv_deepfool
    )
)


# ============================================================
# 24. CLEAN RESULT
# ============================================================

clean_result = {

    "Attack": "Clean",

    "Original_Accuracy":
        ORIGINAL_CLEAN_ACCURACY,

    "Defended_Accuracy":
        clean_accuracy,

    "Defense_Gain":
        clean_accuracy -
        ORIGINAL_CLEAN_ACCURACY,

    "Defended_Accuracy_Drop":
        0.0,

    "Precision":
        clean_metrics["precision"],

    "Recall":
        clean_metrics["recall"],

    "F1":
        clean_metrics["f1"],

    "Evasion_Rate":
        0.0,

    "Successful_Evasions":
        0,

    "Clean_Correct":
        np.sum(
            clean_predictions ==
            y_test
        ),

    "Mean_L2":
        0.0,

    "Median_L2":
        0.0,

    "Max_L2":
        0.0,

    "Mean_Linf":
        0.0,

    "Max_Linf":
        0.0,

    "TN":
        clean_metrics["tn"],

    "FP":
        clean_metrics["fp"],

    "FN":
        clean_metrics["fn"],

    "TP":
        clean_metrics["tp"]
}


# ============================================================
# 25. COMBINE RESULTS
# ============================================================

all_results = pd.DataFrame([

    clean_result,

    fgsm_result,

    pgd_result,

    cw_result,

    deepfool_result
])


# ============================================================
# 26. SAVE RESULTS
# ============================================================

output_path = os.path.join(
    RESULTS_DIR,
    "defense_validation_results.csv"
)


all_results.to_csv(
    output_path,
    index=False
)


# ============================================================
# 27. FINAL COMPARISON
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "FINAL DEFENSE COMPARISON"
)

print(
    "=" * 90
)


comparison = all_results[
    [
        "Attack",
        "Original_Accuracy",
        "Defended_Accuracy",
        "Defense_Gain",
        "Defended_Accuracy_Drop",
        "Evasion_Rate"
    ]
].copy()


comparison[
    "Original_Accuracy"
] *= 100


comparison[
    "Defended_Accuracy"
] *= 100


comparison[
    "Defense_Gain"
] *= 100


comparison[
    "Defended_Accuracy_Drop"
] *= 100


comparison[
    "Evasion_Rate"
] *= 100


comparison = comparison.rename(

    columns={

        "Attack":
            "Attack",

        "Original_Accuracy":
            "Original (%)",

        "Defended_Accuracy":
            "Defended (%)",

        "Defense_Gain":
            "Defense Gain (pp)",

        "Defended_Accuracy_Drop":
            "Defended Drop (pp)",

        "Evasion_Rate":
            "Evasion (%)"
    }
)


print(
    comparison.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.2f}"
    )
)


# ============================================================
# 28. INTERPRETATION
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "INTERPRETATION"
)

print(
    "=" * 90
)


print(
    "\nDefense Gain means:"
)

print(
    "Defended Accuracy - Original Accuracy"
)


print(
    "\nPositive Defense Gain = "
    "higher accuracy under the attack after adversarial training."
)


print(
    "\nDefended Accuracy Drop means:"
)

print(
    "Defended Clean Accuracy - Defended Attack Accuracy"
)


# ============================================================
# 29. SAVE COMPARISON TABLE
# ============================================================

comparison_output = os.path.join(
    RESULTS_DIR,
    "defense_comparison.csv"
)


comparison.to_csv(
    comparison_output,
    index=False
)


print(
    "\nDetailed results saved to:"
)

print(
    output_path
)


print(
    "\nComparison table saved to:"
)

print(
    comparison_output
)


print(
    "\nDefense validation completed."
)