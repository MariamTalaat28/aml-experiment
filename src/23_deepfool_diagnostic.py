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
    "results/dnn_cw_adversarial_trained.pth"
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

N_SAMPLES = 1000
RANDOM_SEED = 42

DEEPFOOL_STEPS = 50
DEEPFOOL_OVERSHOOT = 0.02

# Step size used by our previous DeepFool implementation
DEEPFOOL_STEP_SIZE = 0.01


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
print("              DEEPFOOL DIAGNOSTIC - C&W TRAINED MODEL")
print("=" * 90)

print(f"Device: {device}")
print(f"Samples: {N_SAMPLES}")
print(f"DeepFool steps: {DEEPFOOL_STEPS}")
print(f"Overshoot: {DEEPFOOL_OVERSHOOT}")
print(f"Step size: {DEEPFOOL_STEP_SIZE}")


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

print("\nLoading C&W-trained model...")

model = DNN(78)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device
    )
)

model.to(device)
model.eval()

print("C&W-trained model loaded successfully.")


# ============================================================
# LOAD DATA
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


print("\nUsing the same fixed 1000-sample subset.")

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

def predict_single(model, x):

    with torch.no_grad():

        logits = model(
            x.unsqueeze(0)
        ).squeeze()

        probability = torch.sigmoid(
            logits
        )

        prediction = int(
            probability >= 0.5
        )

    return prediction


# ============================================================
# DEEPFOOL SINGLE SAMPLE
# ============================================================

def deepfool_single(
    model,
    x,
    original_label
):

    x_original = x.clone().detach()

    x_adv = x.clone().detach()

    original_label = int(
        original_label
    )

    iterations = 0

    attack_success = False

    for step in range(
        DEEPFOOL_STEPS
    ):

        iterations = step + 1

        x_adv = x_adv.detach()

        x_adv.requires_grad = True

        logit = model(
            x_adv.unsqueeze(0)
        ).squeeze()

        probability = torch.sigmoid(
            logit
        )

        current_label = int(
            probability >= 0.5
        )

        # ----------------------------------------------------
        # Stop once the prediction changes
        # ----------------------------------------------------

        if current_label != original_label:

            attack_success = True

            break

        # ----------------------------------------------------
        # Gradient
        # ----------------------------------------------------

        gradient = torch.autograd.grad(
            logit,
            x_adv,
            retain_graph=False
        )[0]

        gradient_norm = torch.norm(
            gradient,
            p=2
        )

        if gradient_norm.item() < 1e-12:

            break

        # ----------------------------------------------------
        # Direction
        # ----------------------------------------------------

        if original_label == 1:

            direction = -1.0

        else:

            direction = 1.0

        unit_gradient = (
            direction
            *
            gradient
            /
            (gradient_norm + 1e-12)
        )

        # ----------------------------------------------------
        # Perturbation
        # ----------------------------------------------------

        perturbation = (
            (1.0 + DEEPFOOL_OVERSHOOT)
            *
            DEEPFOOL_STEP_SIZE
            *
            unit_gradient
        )

        x_adv = (
            x_adv
            +
            perturbation
        ).detach()

    return (
        x_adv,
        attack_success,
        iterations
    )


# ============================================================
# RUN DIAGNOSTIC
# ============================================================

print("\n")
print("=" * 90)
print("                 RUNNING DEEPFOOL DIAGNOSTIC")
print("=" * 90)

clean_predictions = []

adversarial_predictions = []

l2_values = []

linf_values = []

successful_indices = []

iterations_list = []


# ============================================================
# SAMPLE LOOP
# ============================================================

for i in range(
    N_SAMPLES
):

    if (
        i % 50 == 0
        or
        i == N_SAMPLES - 1
    ):

        print(
            f"Progress: "
            f"{i + 1}/{N_SAMPLES}",
            end="\r"
        )

    x = torch.tensor(
        X_scaled[i],
        dtype=torch.float32,
        device=device
    )

    # --------------------------------------------------------
    # Clean prediction
    # --------------------------------------------------------

    clean_prediction = predict_single(
        model,
        x
    )

    clean_predictions.append(
        clean_prediction
    )

    # --------------------------------------------------------
    # DeepFool
    # --------------------------------------------------------

    x_adv, success, iterations = deepfool_single(
        model,
        x,
        y[i]
    )

    iterations_list.append(
        iterations
    )

    # --------------------------------------------------------
    # Adversarial prediction
    # --------------------------------------------------------

    adversarial_prediction = predict_single(
        model,
        x_adv
    )

    adversarial_predictions.append(
        adversarial_prediction
    )

    # --------------------------------------------------------
    # Perturbation
    # --------------------------------------------------------

    perturbation = (
        x_adv
        -
        x
    )

    l2 = torch.norm(
        perturbation,
        p=2
    ).item()

    linf = torch.norm(
        perturbation,
        p=float("inf")
    ).item()

    l2_values.append(
        l2
    )

    linf_values.append(
        linf
    )

    if success:

        successful_indices.append(
            i
        )


print("\n")


# ============================================================
# ARRAYS
# ============================================================

clean_predictions = np.array(
    clean_predictions
)

adversarial_predictions = np.array(
    adversarial_predictions
)

l2_values = np.array(
    l2_values
)

linf_values = np.array(
    linf_values
)

iterations_list = np.array(
    iterations_list
)


# ============================================================
# CLEAN PERFORMANCE
# ============================================================

clean_correct = (
    clean_predictions == y
)

clean_accuracy = (
    clean_correct.mean()
)


# ============================================================
# ATTACK PERFORMANCE
# ============================================================

attack_success = (
    adversarial_predictions != y
)

attack_success_count = int(
    attack_success.sum()
)

attack_success_rate = (
    attack_success.mean()
)


# ============================================================
# EVASION PERFORMANCE
# ============================================================

evasion_success = (
    clean_correct
    &
    (adversarial_predictions != y)
)

evasion_count = int(
    evasion_success.sum()
)

if clean_correct.sum() > 0:

    evasion_rate = (
        evasion_count
        /
        clean_correct.sum()
    )

else:

    evasion_rate = 0.0


# ============================================================
# PERTURBATION STATISTICS
# ============================================================

successful_l2 = l2_values[
    attack_success
]

successful_linf = linf_values[
    attack_success
]

successful_iterations = iterations_list[
    attack_success
]


if len(successful_l2) > 0:

    mean_l2 = np.mean(
        successful_l2
    )

    median_l2 = np.median(
        successful_l2
    )

    max_l2 = np.max(
        successful_l2
    )

    mean_linf = np.mean(
        successful_linf
    )

    median_linf = np.median(
        successful_linf
    )

    max_linf = np.max(
        successful_linf
    )

else:

    mean_l2 = np.nan
    median_l2 = np.nan
    max_l2 = np.nan

    mean_linf = np.nan
    median_linf = np.nan
    max_linf = np.nan


# ============================================================
# ALL-SAMPLE PERTURBATIONS
# ============================================================

mean_l2_all = np.mean(
    l2_values
)

median_l2_all = np.median(
    l2_values
)

max_l2_all = np.max(
    l2_values
)

mean_linf_all = np.mean(
    linf_values
)

median_linf_all = np.median(
    linf_values
)

max_linf_all = np.max(
    linf_values
)


# ============================================================
# ITERATION STATISTICS
# ============================================================

mean_iterations = np.mean(
    iterations_list
)

max_iterations = np.max(
    iterations_list
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 90)
print("                    DEEPFOOL DIAGNOSTIC RESULTS")
print("=" * 90)


print("\n")
print("MODEL PERFORMANCE")
print("-" * 60)

print(
    f"Clean accuracy: "
    f"{clean_accuracy * 100:.2f}%"
)

print(
    f"Clean correct: "
    f"{int(clean_correct.sum())}/{N_SAMPLES}"
)

print(
    f"DeepFool attack success: "
    f"{attack_success_rate * 100:.2f}%"
)

print(
    f"Successful attacks: "
    f"{attack_success_count}/{N_SAMPLES}"
)

print(
    f"Evasion success rate: "
    f"{evasion_rate * 100:.2f}%"
)

print(
    f"Evasion successes: "
    f"{evasion_count}/{int(clean_correct.sum())}"
)


print("\n")
print("PERTURBATION SIZE - SUCCESSFUL ATTACKS")
print("-" * 60)

print(
    f"Mean L2: "
    f"{mean_l2:.6f}"
)

print(
    f"Median L2: "
    f"{median_l2:.6f}"
)

print(
    f"Maximum L2: "
    f"{max_l2:.6f}"
)

print(
    f"Mean L-infinity: "
    f"{mean_linf:.6f}"
)

print(
    f"Median L-infinity: "
    f"{median_linf:.6f}"
)

print(
    f"Maximum L-infinity: "
    f"{max_linf:.6f}"
)


print("\n")
print("PERTURBATION SIZE - ALL SAMPLES")
print("-" * 60)

print(
    f"Mean L2: "
    f"{mean_l2_all:.6f}"
)

print(
    f"Median L2: "
    f"{median_l2_all:.6f}"
)

print(
    f"Maximum L2: "
    f"{max_l2_all:.6f}"
)

print(
    f"Mean L-infinity: "
    f"{mean_linf_all:.6f}"
)

print(
    f"Median L-infinity: "
    f"{median_linf_all:.6f}"
)

print(
    f"Maximum L-infinity: "
    f"{max_linf_all:.6f}"
)


print("\n")
print("ITERATION STATISTICS")
print("-" * 60)

print(
    f"Mean iterations: "
    f"{mean_iterations:.2f}"
)

print(
    f"Maximum iterations: "
    f"{max_iterations}"
)


# ============================================================
# PERCENTILES
# ============================================================

print("\n")
print("PERTURBATION PERCENTILES")
print("-" * 60)

print(
    f"L2 25th percentile: "
    f"{np.percentile(l2_values, 25):.6f}"
)

print(
    f"L2 50th percentile: "
    f"{np.percentile(l2_values, 50):.6f}"
)

print(
    f"L2 75th percentile: "
    f"{np.percentile(l2_values, 75):.6f}"
)

print(
    f"L2 90th percentile: "
    f"{np.percentile(l2_values, 90):.6f}"
)

print(
    f"L2 95th percentile: "
    f"{np.percentile(l2_values, 95):.6f}"
)

print(
    f"L2 99th percentile: "
    f"{np.percentile(l2_values, 99):.6f}"
)


# ============================================================
# CHECK FOR EXTREME VALUES
# ============================================================

print("\n")
print("=" * 90)
print("                    SANITY CHECK")
print("=" * 90)

print(
    f"\nMaximum absolute perturbation: "
    f"{max_linf_all:.6f}"
)

print(
    f"Maximum L2 perturbation: "
    f"{max_l2_all:.6f}"
)

print(
    f"Average L2 perturbation: "
    f"{mean_l2_all:.6f}"
)


if max_linf_all > 10:

    print("\nWARNING:")
    print(
        "DeepFool is producing very large perturbations."
    )

    print(
        "The 0% accuracy result should NOT yet be "
        "used as a final research result."
    )

elif max_linf_all > 5:

    print("\nCAUTION:")
    print(
        "Some perturbations are relatively large."
    )

    print(
        "We should inspect the result before "
        "accepting the DeepFool accuracy."
    )

else:

    print("\nDeepFool perturbations are within a "
          "reasonable diagnostic range.")

    print(
        "The DeepFool result can be considered "
        "for further evaluation."
    )


# ============================================================
# SAVE DIAGNOSTIC RESULTS
# ============================================================

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "results/deepfool_cw_training_diagnostic.csv"
)

diagnostic_df = pd.DataFrame({

    "Index": np.arange(
        N_SAMPLES
    ),

    "True_Label": y,

    "Clean_Prediction":
        clean_predictions,

    "DeepFool_Prediction":
        adversarial_predictions,

    "Attack_Success":
        attack_success,

    "Evasion_Success":
        evasion_success,

    "L2":
        l2_values,

    "Linf":
        linf_values,

    "Iterations":
        iterations_list
})

diagnostic_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n")
print("=" * 90)
print("Diagnostic results saved to:")
print(OUTPUT_FILE)
print("=" * 90)

print("\nDiagnostic completed.\n")