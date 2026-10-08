import pandas as pd
import os


BASE_DIR = os.path.expanduser("~/aml-experiment")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "all_experiment_results.csv")


# ============================================================
# Helper functions
# ============================================================

def get_value(row, *names, default=None):
    """Return the first matching column value."""
    for name in names:
        if name in row.index:
            return row[name]
    return default


def normalize_metric(value):
    """Convert percentage metrics such as 99.7 to 0.997."""
    if pd.isna(value):
        return value

    value = float(value)

    if value > 1:
        return value / 100.0

    return value


# ============================================================
# Master rows
# ============================================================

master_rows = []


def add_row(
    experiment,
    implementation,
    model,
    attack,
    parameter,
    accuracy,
    precision=None,
    recall=None,
    f1=None,
    accuracy_drop=None,
    evasion_rate=None,
    successful_attacks=None,
    mean_l2=None,
    mean_linf=None,
    tn=None,
    fp=None,
    fn=None,
    tp=None,
    status="FINAL"
):
    master_rows.append({
        "experiment": experiment,
        "implementation": implementation,
        "model": model,
        "attack": attack,
        "parameter": parameter,
        "accuracy": normalize_metric(accuracy),
        "precision": normalize_metric(precision),
        "recall": normalize_metric(recall),
        "f1": normalize_metric(f1),
        "accuracy_drop": normalize_metric(accuracy_drop),
        "evasion_rate": normalize_metric(evasion_rate),
        "successful_attacks": successful_attacks,
        "mean_l2": mean_l2,
        "mean_linf": mean_linf,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "status": status
    })


# ============================================================
# 1. CUSTOM ATTACK RESULTS
# ============================================================

custom_attack_file = os.path.join(
    RESULTS_DIR, "fgsm_results.csv"
)

if os.path.exists(custom_attack_file):

    df = pd.read_csv(custom_attack_file)

    for _, row in df.iterrows():

        epsilon = get_value(row, "epsilon", "Epsilon")

        add_row(
            experiment="Attack Evaluation",
            implementation="Custom PyTorch",
            model="Original DNN",
            attack="FGSM",
            parameter=f"epsilon={epsilon}",
            accuracy=get_value(row, "accuracy", "Accuracy"),
            precision=get_value(row, "precision", "Precision"),
            recall=get_value(row, "recall", "Recall"),
            f1=get_value(row, "f1", "F1"),
            accuracy_drop=get_value(
                row, "accuracy_drop", "Accuracy_Drop"
            ),
            evasion_rate=get_value(
                row,
                "evasion_rate",
                "Evasion_Rate",
                "attack_success_rate"
            ),
            successful_attacks=get_value(
                row,
                "successful_attacks",
                "Successful_Evasions"
            ),
            tn=get_value(row, "tn", "TN"),
            fp=get_value(row, "fp", "FP"),
            fn=get_value(row, "fn", "FN"),
            tp=get_value(row, "tp", "TP"),
            status="FINAL"
        )


pgd_file = os.path.join(
    RESULTS_DIR, "pgd_results.csv"
)

if os.path.exists(pgd_file):

    df = pd.read_csv(pgd_file)

    for _, row in df.iterrows():

        epsilon = get_value(row, "epsilon", "Epsilon")
        steps = get_value(row, "steps", "Steps", default=10)

        add_row(
            experiment="Attack Evaluation",
            implementation="Custom PyTorch",
            model="Original DNN",
            attack="PGD",
            parameter=f"epsilon={epsilon},steps={steps}",
            accuracy=get_value(row, "accuracy", "Accuracy"),
            precision=get_value(row, "precision", "Precision"),
            recall=get_value(row, "recall", "Recall"),
            f1=get_value(row, "f1", "F1"),
            accuracy_drop=get_value(
                row, "accuracy_drop", "Accuracy_Drop"
            ),
            evasion_rate=get_value(
                row,
                "evasion_rate",
                "Evasion_Rate",
                "attack_success_rate"
            ),
            successful_attacks=get_value(
                row,
                "successful_attacks",
                "Successful_Evasions"
            ),
            tn=get_value(row, "tn", "TN"),
            fp=get_value(row, "fp", "FP"),
            fn=get_value(row, "fn", "FN"),
            tp=get_value(row, "tp", "TP"),
            status="FINAL"
        )


cw_file = os.path.join(
    RESULTS_DIR, "cw_results.csv"
)

if os.path.exists(cw_file):

    df = pd.read_csv(cw_file)

    for _, row in df.iterrows():

        add_row(
            experiment="Attack Evaluation",
            implementation="Custom PyTorch",
            model="Original DNN",
            attack="C&W",
            parameter="steps=30,lr=0.01,C=1.0,kappa=0.0",
            accuracy=get_value(row, "accuracy", "Accuracy"),
            precision=get_value(row, "precision", "Precision"),
            recall=get_value(row, "recall", "Recall"),
            f1=get_value(row, "f1", "F1"),
            accuracy_drop=get_value(
                row, "accuracy_drop", "Accuracy_Drop"
            ),
            evasion_rate=get_value(
                row,
                "evasion_rate",
                "Evasion_Rate",
                "attack_success_rate"
            ),
            successful_attacks=get_value(
                row,
                "successful_attacks",
                "Successful_Evasions"
            ),
            tn=get_value(row, "tn", "TN"),
            fp=get_value(row, "fp", "FP"),
            fn=get_value(row, "fn", "FN"),
            tp=get_value(row, "tp", "TP"),
            status="EXPERIMENTAL"
        )


deepfool_file = os.path.join(
    RESULTS_DIR, "deepfool_results.csv"
)

if os.path.exists(deepfool_file):

    df = pd.read_csv(deepfool_file)

    for _, row in df.iterrows():

        add_row(
            experiment="Attack Evaluation",
            implementation="Custom PyTorch",
            model="Original DNN",
            attack="DeepFool",
            parameter="max_iter=10,overshoot=0.02",
            accuracy=get_value(row, "accuracy", "Accuracy"),
            precision=get_value(row, "precision", "Precision"),
            recall=get_value(row, "recall", "Recall"),
            f1=get_value(row, "f1", "F1"),
            accuracy_drop=get_value(
                row, "accuracy_drop", "Accuracy_Drop"
            ),
            evasion_rate=get_value(
                row,
                "evasion_rate",
                "Evasion_Rate",
                "attack_success_rate"
            ),
            successful_attacks=get_value(
                row,
                "successful_attacks",
                "Successful_Evasions"
            ),
            tn=get_value(row, "tn", "TN"),
            fp=get_value(row, "fp", "FP"),
            fn=get_value(row, "fn", "FN"),
            tp=get_value(row, "tp", "TP"),
            status="EXPERIMENTAL"
        )


# ============================================================
# 2. ART ATTACK RESULTS
# ============================================================

art_files = [
    ("art_fgsm_results.csv", "FGSM"),
    ("art_pgd_results.csv", "PGD"),
    ("art_cw_results.csv", "C&W"),
    ("art_deepfool_results_corrected.csv", "DeepFool")
]


for filename, attack_name in art_files:

    filepath = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(filepath):
        continue

    df = pd.read_csv(filepath)

    for _, row in df.iterrows():

        epsilon = get_value(
            row, "epsilon", "Epsilon", default=None
        )

        if attack_name in ["FGSM", "PGD"]:

            if attack_name == "PGD":
                steps = get_value(
                    row, "steps", "Steps", "max_iter",
                    default=10
                )

                parameter = (
                    f"epsilon={epsilon},steps={steps}"
                )

            else:
                parameter = f"epsilon={epsilon}"

        elif attack_name == "C&W":

            parameter = (
                "steps=30,lr=0.01,C=1.0,kappa=0.0"
            )

        else:

            parameter = (
                "steps=50,overshoot=0.02"
            )

        status = (
            "FINAL"
            if attack_name in ["FGSM", "PGD"]
            else "EXPERIMENTAL"
        )

        add_row(
            experiment="Attack Evaluation",
            implementation="ART",
            model="Original DNN",
            attack=attack_name,
            parameter=parameter,
            accuracy=get_value(row, "accuracy", "Accuracy"),
            precision=get_value(row, "precision", "Precision"),
            recall=get_value(row, "recall", "Recall"),
            f1=get_value(row, "f1", "F1"),
            accuracy_drop=get_value(
                row, "accuracy_drop", "Accuracy_Drop"
            ),
            evasion_rate=get_value(
                row,
                "evasion_rate",
                "Evasion_Rate",
                "attack_success_rate"
            ),
            successful_attacks=get_value(
                row,
                "successful_attacks",
                "Successful_Evasions"
            ),
            mean_l2=get_value(
                row, "mean_l2", "Mean_L2"
            ),
            mean_linf=get_value(
                row, "mean_linf", "Mean_Linf"
            ),
            tn=get_value(row, "tn", "TN"),
            fp=get_value(row, "fp", "FP"),
            fn=get_value(row, "fn", "FN"),
            tp=get_value(row, "tp", "TP"),
            status=status
        )


# ============================================================
# 3. ADVERSARIAL TRAINING
# ============================================================

unified_file = os.path.join(
    RESULTS_DIR,
    "unified_defense_evaluation.csv"
)

if os.path.exists(unified_file):

    df = pd.read_csv(unified_file)

    for _, row in df.iterrows():

        model = get_value(row, "model", "Model")
        attack = get_value(row, "attack", "Attack")

        if attack == "Clean":
            parameter = "Clean"
        else:
            parameter = "epsilon=0.10"

        # C&W and DeepFool depend on experimental attack
        # implementations.
        if attack in ["C&W", "DeepFool"]:
            status = "EXPERIMENTAL"
        else:
            status = "FINAL"

        add_row(
            experiment="Adversarial Training",
            implementation="Custom PyTorch",
            model=model,
            attack=attack,
            parameter=parameter,
            accuracy=get_value(row, "accuracy", "Accuracy"),
            precision=get_value(row, "precision", "Precision"),
            recall=get_value(row, "recall", "Recall"),
            f1=get_value(row, "f1", "F1"),
            accuracy_drop=get_value(
                row, "accuracy_drop", "Accuracy_Drop"
            ),
            evasion_rate=get_value(
                row,
                "evasion_success_rate",
                "evasion_rate",
                "Evasion_Rate"
            ),
            successful_attacks=get_value(
                row,
                "successful_evasions",
                "successful_attacks",
                "Successful_Evasions"
            ),
            tn=get_value(row, "tn", "TN"),
            fp=get_value(row, "fp", "FP"),
            fn=get_value(row, "fn", "FN"),
            tp=get_value(row, "tp", "TP"),
            status=status
        )


# ============================================================
# 4. DEFENSIVE DISTILLATION V4
# ============================================================

distillation_file = os.path.join(
    RESULTS_DIR,
    "defensive_distillation_v4_evaluation.csv"
)

if os.path.exists(distillation_file):

    df = pd.read_csv(distillation_file)

    for _, row in df.iterrows():

        attack = get_value(
            row, "Attack", "attack"
        )

        accuracy = get_value(
            row, "Accuracy", "accuracy"
        )

        precision = get_value(
            row, "Precision", "precision"
        )

        recall = get_value(
            row, "Recall", "recall"
        )

        f1 = get_value(
            row, "F1", "f1"
        )

        tn = get_value(row, "TN", "tn")
        fp = get_value(row, "FP", "fp")
        fn = get_value(row, "FN", "fn")
        tp = get_value(row, "TP", "tp")

        # Convert percentage accuracy to proportion.
        accuracy = normalize_metric(accuracy)
        precision = normalize_metric(precision)
        recall = normalize_metric(recall)
        f1 = normalize_metric(f1)

        # Calculate accuracy drop relative to clean.
        accuracy_drop = None

        if attack == "Clean":
            clean_accuracy = accuracy

        # We calculate after collecting the rows below.
        if attack in ["C&W", "DeepFool"]:
            status = "EXPERIMENTAL"
        else:
            status = "FINAL"

        add_row(
            experiment="Defensive Distillation",
            implementation="V4",
            model="Distilled Student",
            attack=attack,
            parameter=(
                "Clean"
                if attack == "Clean"
                else "epsilon=0.10"
            ),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy_drop=None,
            evasion_rate=None,
            tn=tn,
            fp=fp,
            fn=fn,
            tp=tp,
            status=status
        )


# ============================================================
# Calculate Defensive Distillation accuracy drops
# ============================================================

distillation_rows = [
    r for r in master_rows
    if r["experiment"] == "Defensive Distillation"
]

clean_rows = [
    r for r in distillation_rows
    if r["attack"] == "Clean"
]

if clean_rows:

    clean_accuracy = clean_rows[0]["accuracy"]

    for r in distillation_rows:

        if r["attack"] != "Clean":

            r["accuracy_drop"] = (
                clean_accuracy - r["accuracy"]
            )


# ============================================================
# 5. INPUT SANITIZATION / FEATURE SQUEEZING
# ============================================================

sanitization_file = os.path.join(
    RESULTS_DIR,
    "input_sanitization_evaluation.csv"
)

if os.path.exists(sanitization_file):

    df = pd.read_csv(sanitization_file)

    for _, row in df.iterrows():

        attack = get_value(
            row, "attack", "Attack"
        )

        # C&W and DeepFool depend on experimental
        # attack implementations.
        if attack in ["C&W", "DeepFool"]:
            status = "EXPERIMENTAL"
        else:
            status = "FINAL"

        add_row(
            experiment="Input Sanitization",
            implementation="Feature Squeezing",
            model="Original DNN",
            attack=attack,
            parameter="8-bit feature squeezing",
            accuracy=get_value(
                row,
                "sanitized_accuracy",
                "accuracy",
                "Accuracy"
            ),
            precision=get_value(
                row, "precision", "Precision"
            ),
            recall=get_value(
                row, "recall", "Recall"
            ),
            f1=get_value(
                row, "f1", "F1"
            ),
            accuracy_drop=get_value(
                row,
                "accuracy_drop_from_clean_squeezed",
                "accuracy_drop",
                "Accuracy_Drop"
            ),
            evasion_rate=get_value(
                row,
                "evasion_success_rate",
                "evasion_rate",
                "Evasion_Rate"
            ),
            tn=get_value(row, "tn", "TN"),
            fp=get_value(row, "fp", "FP"),
            fn=get_value(row, "fn", "FN"),
            tp=get_value(row, "tp", "TP"),
            status=status
        )


# ============================================================
# Create DataFrame
# ============================================================

master_df = pd.DataFrame(master_rows)


# ============================================================
# Final column order
# ============================================================

columns = [
    "experiment",
    "implementation",
    "model",
    "attack",
    "parameter",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "accuracy_drop",
    "evasion_rate",
    "successful_attacks",
    "mean_l2",
    "mean_linf",
    "tn",
    "fp",
    "fn",
    "tp",
    "status"
]

master_df = master_df[columns]


# ============================================================
# Save
# ============================================================

master_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Print summary
# ============================================================

print("\n" + "=" * 75)
print("MASTER RESULTS CREATED")
print("=" * 75)

print(f"Total rows: {len(master_df)}")
print(f"Total columns: {len(master_df.columns)}")

print("\nRows by experiment:")
print(
    master_df["experiment"]
    .value_counts()
    .to_string()
)

print("\nRows by implementation:")
print(
    master_df["implementation"]
    .value_counts()
    .to_string()
)

print("\nStatus:")
print(
    master_df["status"]
    .value_counts()
    .to_string()
)

print("\n" + "=" * 75)
print("MASTER TABLE")
print("=" * 75)

print(
    master_df[
        [
            "experiment",
            "implementation",
            "model",
            "attack",
            "parameter",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "accuracy_drop",
            "evasion_rate",
            "status"
        ]
    ].to_string(index=False)
)

print("\n")
print("[COMPLETE]")
print(f"Output: {OUTPUT_FILE}")