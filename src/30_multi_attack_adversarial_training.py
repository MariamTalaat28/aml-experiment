import os
import random
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "/home/ubuntu/aml-experiment"

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "data/ml/balanced_train.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_balanced.pth"
)

SCALER_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_balanced_scaler.pkl"
)

# IMPORTANT:
# New output files so the previous experiment is not overwritten.
OUTPUT_MODEL_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_multi_attack_final_adversarial_trained.pth"
)

OUTPUT_HISTORY_FILE = os.path.join(
    BASE_DIR,
    "results/multi_attack_final_adversarial_training_history.csv"
)


# ============================================================
# TRAINING PARAMETERS
# ============================================================

BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.001


# ------------------------------------------------------------
# FGSM
# ------------------------------------------------------------

FGSM_EPSILON = 0.10


# ------------------------------------------------------------
# PGD
# ------------------------------------------------------------

PGD_EPSILON = 0.10
PGD_STEPS = 10
PGD_ALPHA = PGD_EPSILON / PGD_STEPS


# ------------------------------------------------------------
# C&W
# ------------------------------------------------------------

# Final C&W configuration
CW_STEPS = 100
CW_LR = 0.005
CW_C = 1.0
CW_KAPPA = 0.0


# ------------------------------------------------------------
# DeepFool
# ------------------------------------------------------------

# Final DeepFool configuration
DEEPFOOL_STEPS = 100
DEEPFOOL_OVERSHOOT = 0.10


# ------------------------------------------------------------
# Loss weights
# ------------------------------------------------------------

CLEAN_WEIGHT = 0.5
ADV_WEIGHT = 0.5


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 80)
print("FINAL MULTI-ATTACK ADVERSARIAL TRAINING")
print("=" * 80)

print(f"Device: {DEVICE}")

if DEVICE.type == "cpu":
    print("\nWARNING:")
    print("Running multi-attack adversarial training on CPU.")
    print("Final C&W and DeepFool can make training significantly slower.")


# ============================================================
# DNN MODEL
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
# LOAD SCALER
# ============================================================

print("\nLoading scaler...")

scaler = joblib.load(
    SCALER_FILE
)

print("Scaler loaded successfully.")


# ============================================================
# LOAD BALANCED TRAINING DATA
# ============================================================

print("\nLoading balanced training data...")

df = pd.read_csv(
    TRAIN_FILE
)

print(
    f"Training samples: {len(df):,}"
)

print(
    f"Columns: {df.shape[1]}"
)


# ============================================================
# FEATURES AND LABEL
# ============================================================

df.columns = df.columns.str.strip()

FEATURE_COLUMNS = [
    column
    for column in df.columns
    if column != "Label"
]

NUM_FEATURES = len(FEATURE_COLUMNS)

print(
    f"Number of features: {NUM_FEATURES}"
)

X = df[
    FEATURE_COLUMNS
].values.astype(
    np.float32
)

y = df[
    "Label"
].values.astype(
    np.float32
)

print(
    f"BENIGN: {np.sum(y == 0):,}"
)

print(
    f"ATTACK: {np.sum(y == 1):,}"
)


# ============================================================
# SCALE TRAINING DATA
# ============================================================

print("\nScaling training data...")

X = scaler.transform(
    X
).astype(
    np.float32
)

print("Scaling completed.")


# ============================================================
# TORCH DATASET
# ============================================================

X_tensor = torch.tensor(
    X,
    dtype=torch.float32
)

y_tensor = torch.tensor(
    y,
    dtype=torch.float32
)

dataset = TensorDataset(
    X_tensor,
    y_tensor
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

print(
    f"Batches per epoch: {len(dataloader):,}"
)


# ============================================================
# LOAD BALANCED BASELINE MODEL
# ============================================================

print("\nLoading balanced baseline DNN...")

model = DNN(
    input_dim=NUM_FEATURES
)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model.to(DEVICE)

print(
    "Balanced baseline model loaded successfully."
)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.BCEWithLogitsLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# FGSM ATTACK
# ============================================================

def generate_fgsm(
    model,
    x,
    y,
    epsilon
):

    x_adv = x.detach().clone()

    x_adv.requires_grad_(True)

    logits = model(
        x_adv
    ).squeeze(1)

    loss = criterion(
        logits,
        y
    )

    model.zero_grad()

    loss.backward()

    gradient = x_adv.grad.detach()

    x_adv = (
        x_adv.detach()
        + epsilon * gradient.sign()
    )

    return x_adv.detach()


# ============================================================
# PGD ATTACK
# ============================================================

def generate_pgd(
    model,
    x,
    y,
    epsilon,
    alpha,
    steps
):

    x_original = x.detach().clone()

    # Random initialization inside epsilon-ball
    x_adv = (
        x_original
        + torch.empty_like(x_original).uniform_(
            -epsilon,
            epsilon
        )
    )

    x_adv = x_adv.detach()

    for _ in range(steps):

        x_adv.requires_grad_(True)

        outputs = model(
            x_adv
        ).squeeze(1)

        loss = criterion(
            outputs,
            y
        )

        model.zero_grad()

        loss.backward()

        gradient = x_adv.grad.detach()

        x_adv = (
            x_adv.detach()
            + alpha * gradient.sign()
        )

        # Project back into epsilon-ball
        perturbation = torch.clamp(
            x_adv - x_original,
            min=-epsilon,
            max=epsilon
        )

        x_adv = (
            x_original
            + perturbation
        )

        x_adv = x_adv.detach()

    return x_adv


# ============================================================
# RAW-DOMAIN FEATURE CONSTRAINTS FOR C&W / DEEPFOOL
# ============================================================

# These are the discrete / fixed features used in the
# validated final C&W / DeepFool experiments.

FIXED_FEATURE_NAMES = [
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio"
]


def get_fixed_feature_indices():

    indices = []

    for feature_name in FIXED_FEATURE_NAMES:

        if feature_name in FEATURE_COLUMNS:

            indices.append(
                FEATURE_COLUMNS.index(feature_name)
            )

    return indices


FIXED_FEATURE_INDICES = get_fixed_feature_indices()

print("\nFixed feature constraints:")

for index in FIXED_FEATURE_INDICES:

    print(
        f"  {index}: {FEATURE_COLUMNS[index]}"
    )


# ============================================================
# RAW FEATURE MIN/MAX BOUNDS
# ============================================================

# The scaler transforms:
#
# standardized = (raw - mean) / scale
#
# We therefore obtain the observed raw-domain bounds from
# the balanced training data before scaling.

RAW_TRAIN_X = df[
    FEATURE_COLUMNS
].values.astype(
    np.float32
)

RAW_FEATURE_MIN = np.min(
    RAW_TRAIN_X,
    axis=0
)

RAW_FEATURE_MAX = np.max(
    RAW_TRAIN_X,
    axis=0
)

RAW_FEATURE_MIN_TENSOR = torch.tensor(
    RAW_FEATURE_MIN,
    dtype=torch.float32,
    device=DEVICE
)

RAW_FEATURE_MAX_TENSOR = torch.tensor(
    RAW_FEATURE_MAX,
    dtype=torch.float32,
    device=DEVICE
)

SCALER_MEAN = torch.tensor(
    scaler.mean_,
    dtype=torch.float32,
    device=DEVICE
)

SCALER_SCALE = torch.tensor(
    scaler.scale_,
    dtype=torch.float32,
    device=DEVICE
)


def raw_to_scaled_bounds():

    scaled_min = (
        RAW_FEATURE_MIN_TENSOR
        - SCALER_MEAN
    ) / SCALER_SCALE

    scaled_max = (
        RAW_FEATURE_MAX_TENSOR
        - SCALER_MEAN
    ) / SCALER_SCALE

    return scaled_min, scaled_max


SCALED_FEATURE_MIN, SCALED_FEATURE_MAX = raw_to_scaled_bounds()


def project_to_feature_bounds(
    x_adv,
    x_original
):

    # Keep all numerical features inside the observed
    # training-domain range after standardization.

    x_projected = torch.max(
        torch.min(
            x_adv,
            SCALED_FEATURE_MAX
        ),
        SCALED_FEATURE_MIN
    )

    # Restore features that should remain unchanged.
    if len(FIXED_FEATURE_INDICES) > 0:

        x_projected = x_projected.clone()

        x_projected[
            :,
            FIXED_FEATURE_INDICES
        ] = x_original[
            :,
            FIXED_FEATURE_INDICES
        ]

    return x_projected


# ============================================================
# C&W LOSS
# ============================================================

def cw_loss(
    model,
    x_original,
    delta,
    y,
    c,
    kappa
):

    x_adv = x_original + delta

    logits = model(
        x_adv
    ).squeeze(1)

    direction = (
        2.0 * y - 1.0
    )

    classification_term = torch.clamp(
        direction * logits + kappa,
        min=0.0
    )

    l2_term = torch.sum(
        delta ** 2,
        dim=1
    )

    loss = (
        l2_term
        + c * classification_term
    )

    return loss.mean()


# ============================================================
# FINAL C&W-STYLE ATTACK
# ============================================================

def generate_cw(
    model,
    x,
    y
):

    x_original = x.detach().clone()

    delta = torch.zeros_like(
        x_original,
        requires_grad=True
    )

    cw_optimizer = optim.Adam(
        [delta],
        lr=CW_LR
    )

    best_adv = x_original.detach().clone()

    best_l2 = torch.full(
        (x_original.shape[0],),
        float("inf"),
        device=x_original.device
    )

    for step in range(CW_STEPS):

        cw_optimizer.zero_grad()

        x_adv_raw = (
            x_original
            + delta
        )

        # Differentiable domain projection.
        #
        # This keeps optimization inside the observed
        # standardized feature range.
        x_adv = torch.max(
            torch.min(
                x_adv_raw,
                SCALED_FEATURE_MAX
            ),
            SCALED_FEATURE_MIN
        )

        # Fixed features remain unchanged.
        if len(FIXED_FEATURE_INDICES) > 0:

            x_adv = x_adv.clone()

            x_adv[
                :,
                FIXED_FEATURE_INDICES
            ] = x_original[
                :,
                FIXED_FEATURE_INDICES
            ]

        logits = model(
            x_adv
        ).squeeze(1)

        direction = (
            2.0 * y - 1.0
        )

        classification_term = torch.clamp(
            direction * logits + CW_KAPPA,
            min=0.0
        )

        # Use the actual projected perturbation.
        effective_delta = (
            x_adv - x_original
        )

        l2_per_sample = torch.sum(
            effective_delta ** 2,
            dim=1
        )

        loss = (
            l2_per_sample
            + CW_C * classification_term
        ).mean()

        loss.backward()

        cw_optimizer.step()

        # --------------------------------------------------------
        # Check successful attacks
        # --------------------------------------------------------

        with torch.no_grad():

            final_x_adv = torch.max(
                torch.min(
                    x_original + delta,
                    SCALED_FEATURE_MAX
                ),
                SCALED_FEATURE_MIN
            )

            if len(FIXED_FEATURE_INDICES) > 0:

                final_x_adv = final_x_adv.clone()

                final_x_adv[
                    :,
                    FIXED_FEATURE_INDICES
                ] = x_original[
                    :,
                    FIXED_FEATURE_INDICES
                ]

            final_logits = model(
                final_x_adv
            ).squeeze(1)

            predictions = (
                final_logits >= 0
            ).float()

            successful = (
                predictions != y
            )

            final_delta = (
                final_x_adv - x_original
            )

            final_l2 = torch.sum(
                final_delta ** 2,
                dim=1
            )

            improved = (
                successful
                &
                (final_l2 < best_l2)
            )

            best_l2[improved] = (
                final_l2[improved]
            )

            best_adv[improved] = (
                final_x_adv[improved]
            )

    # ------------------------------------------------------------
    # Exact final projection
    # ------------------------------------------------------------

    best_adv = project_to_feature_bounds(
        best_adv,
        x_original
    )

    return best_adv.detach()


# ============================================================
# FINAL DEEPFOOL
# ============================================================

def generate_deepfool(
    model,
    x,
    y,
    max_iter=100,
    overshoot=0.10
):

    model.eval()

    x_original = x.detach().clone()

    x_adv = x.detach().clone()

    # Original predictions
    with torch.no_grad():

        original_logits = model(
            x_original
        ).squeeze(1)

        original_predictions = (
            original_logits >= 0
        ).long()

    finished = torch.zeros(
        x.shape[0],
        dtype=torch.bool,
        device=x.device
    )

    for iteration in range(max_iter):

        if finished.all():
            break

        x_adv = x_adv.detach()

        x_adv.requires_grad_(True)

        logits = model(
            x_adv
        ).squeeze(1)

        # Gradient of scalar binary logit
        gradients = torch.autograd.grad(
            outputs=logits.sum(),
            inputs=x_adv,
            create_graph=False,
            retain_graph=False
        )[0]

        # Binary decision-boundary formulation:
        #
        # signed_logit = (2y - 1) * f(x)
        #
        # signed_gradient = (2y - 1) * grad(f)
        #
        # distance = |f(x)| / ||grad(f)||^2
        #
        # perturbation moves toward the decision boundary.

        direction = (
            2.0 * y - 1.0
        ).unsqueeze(1)

        signed_logit = (
            direction
            * logits.unsqueeze(1)
        )

        signed_gradient = (
            direction
            * gradients
        )

        grad_norm_sq = torch.sum(
            signed_gradient ** 2,
            dim=1,
            keepdim=True
        ) + 1e-12

        distance = (
            torch.abs(signed_logit)
            / grad_norm_sq
        )

        perturbation = (
            -distance
            * signed_gradient
        )

        perturbation = (
            perturbation
            * (1.0 + overshoot)
        )

        active = ~finished

        x_new = x_adv.detach().clone()

        x_new[active] = (
            x_adv.detach()[active]
            + perturbation.detach()[active]
        )

        # Project to valid feature domain and restore
        # fixed features.
        x_new = project_to_feature_bounds(
            x_new,
            x_original
        )

        x_adv = x_new

        # --------------------------------------------------------
        # Check success
        # --------------------------------------------------------

        with torch.no_grad():

            new_logits = model(
                x_adv
            ).squeeze(1)

            new_predictions = (
                new_logits >= 0
            ).long()

            changed = (
                new_predictions
                != original_predictions
            )

            finished = (
                finished
                | changed
            )

    return x_adv.detach()


# ============================================================
# GENERATE ADVERSARIAL EXAMPLE
# ============================================================

def generate_adversarial_example(
    attack_name,
    x,
    y
):

    if attack_name == "FGSM":

        return generate_fgsm(
            model,
            x,
            y,
            FGSM_EPSILON
        )

    elif attack_name == "PGD":

        return generate_pgd(
            model,
            x,
            y,
            PGD_EPSILON,
            PGD_ALPHA,
            PGD_STEPS
        )

    elif attack_name == "C&W":

        return generate_cw(
            model,
            x,
            y
        )

    elif attack_name == "DeepFool":

        return generate_deepfool(
            model,
            x,
            y,
            DEEPFOOL_STEPS,
            DEEPFOOL_OVERSHOOT
        )

    else:

        raise ValueError(
            f"Unknown attack: {attack_name}"
        )


# ============================================================
# ATTACK SCHEDULE
# ============================================================

ATTACKS = [
    "FGSM",
    "PGD",
    "C&W",
    "DeepFool"
]


print("\n" + "=" * 80)
print("FINAL MULTI-ATTACK TRAINING CONFIGURATION")
print("=" * 80)

print(
    "\nAttack schedule:"
)

print(
    "Batch 1 → FGSM"
)

print(
    "Batch 2 → PGD"
)

print(
    "Batch 3 → C&W"
)

print(
    "Batch 4 → DeepFool"
)

print(
    "Then the cycle repeats."
)

print(
    f"\nFGSM epsilon: {FGSM_EPSILON}"
)

print(
    f"PGD epsilon: {PGD_EPSILON}"
)

print(
    f"PGD steps: {PGD_STEPS}"
)

print(
    f"C&W steps: {CW_STEPS}"
)

print(
    f"C&W learning rate: {CW_LR}"
)

print(
    f"C&W C: {CW_C}"
)

print(
    f"DeepFool steps: {DEEPFOOL_STEPS}"
)

print(
    f"DeepFool overshoot: {DEEPFOOL_OVERSHOOT}"
)

print(
    f"Clean loss weight: {CLEAN_WEIGHT}"
)

print(
    f"Adversarial loss weight: {ADV_WEIGHT}"
)


# ============================================================
# TRAINING
# ============================================================

history = []


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    total_clean_loss = 0.0
    total_adv_loss = 0.0

    correct_clean = 0
    correct_adv = 0

    total_samples = 0

    attack_counts = {
        "FGSM": 0,
        "PGD": 0,
        "C&W": 0,
        "DeepFool": 0
    }

    print(
        f"\n{'=' * 80}"
    )

    print(
        f"EPOCH {epoch + 1}/{EPOCHS}"
    )

    print(
        f"{'=' * 80}"
    )


    for batch_index, (
        x_batch,
        y_batch
    ) in enumerate(dataloader):

        x_batch = x_batch.to(
            DEVICE
        )

        y_batch = y_batch.to(
            DEVICE
        )


        # ----------------------------------------------------
        # Select attack
        # ----------------------------------------------------

        attack_name = (
            ATTACKS[
                batch_index
                % len(ATTACKS)
            ]
        )

        attack_counts[
            attack_name
        ] += 1


        # ----------------------------------------------------
        # Generate adversarial examples
        # ----------------------------------------------------

        model.eval()

        x_adv = generate_adversarial_example(
            attack_name,
            x_batch,
            y_batch
        )


        # ----------------------------------------------------
        # Train model
        # ----------------------------------------------------

        model.train()

        optimizer.zero_grad()


        # Clean prediction
        clean_logits = model(
            x_batch
        ).squeeze(1)

        clean_loss = criterion(
            clean_logits,
            y_batch
        )


        # Adversarial prediction
        adv_logits = model(
            x_adv
        ).squeeze(1)

        adv_loss = criterion(
            adv_logits,
            y_batch
        )


        # Combined loss
        loss = (
            CLEAN_WEIGHT * clean_loss
            +
            ADV_WEIGHT * adv_loss
        )


        # Backpropagation
        loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_size = y_batch.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_clean_loss += (
            clean_loss.item()
            * batch_size
        )

        total_adv_loss += (
            adv_loss.item()
            * batch_size
        )


        clean_predictions = (
            clean_logits >= 0
        ).float()

        adv_predictions = (
            adv_logits >= 0
        ).float()


        correct_clean += (
            (
                clean_predictions
                == y_batch
            )
            .sum()
            .item()
        )

        correct_adv += (
            (
                adv_predictions
                == y_batch
            )
            .sum()
            .item()
        )


        total_samples += batch_size


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_index + 1
        ) % 100 == 0:

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Batch {batch_index + 1}/{len(dataloader)} | "
                f"Attack: {attack_name} | "
                f"Loss: {loss.item():.6f}"
            )


    # ========================================================
    # EPOCH METRICS
    # ========================================================

    epoch_loss = (
        total_loss
        / total_samples
    )

    epoch_clean_loss = (
        total_clean_loss
        / total_samples
    )

    epoch_adv_loss = (
        total_adv_loss
        / total_samples
    )

    epoch_clean_accuracy = (
        correct_clean
        / total_samples
    )

    epoch_adv_accuracy = (
        correct_adv
        / total_samples
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history.append({

        "epoch": epoch + 1,

        "loss": epoch_loss,

        "clean_loss":
            epoch_clean_loss,

        "adversarial_loss":
            epoch_adv_loss,

        "clean_accuracy":
            epoch_clean_accuracy,

        "adversarial_accuracy":
            epoch_adv_accuracy,

        "FGSM_batches":
            attack_counts["FGSM"],

        "PGD_batches":
            attack_counts["PGD"],

        "CW_batches":
            attack_counts["C&W"],

        "DeepFool_batches":
            attack_counts["DeepFool"]
    })


    # ========================================================
    # PRINT EPOCH RESULTS
    # ========================================================

    print(
        "\n" + "-" * 80
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} completed"
    )

    print(
        f"Total loss: "
        f"{epoch_loss:.6f}"
    )

    print(
        f"Clean loss: "
        f"{epoch_clean_loss:.6f}"
    )

    print(
        f"Adversarial loss: "
        f"{epoch_adv_loss:.6f}"
    )

    print(
        f"Clean accuracy: "
        f"{epoch_clean_accuracy * 100:.4f}%"
    )

    print(
        f"Adversarial accuracy: "
        f"{epoch_adv_accuracy * 100:.4f}%"
    )

    print(
        "\nAttack batches:"
    )

    print(
        f"FGSM:      {attack_counts['FGSM']}"
    )

    print(
        f"PGD:       {attack_counts['PGD']}"
    )

    print(
        f"C&W:       {attack_counts['C&W']}"
    )

    print(
        f"DeepFool:  {attack_counts['DeepFool']}"
    )

    print(
        "-" * 80
    )


# ============================================================
# SAVE MODEL
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "SAVING FINAL MULTI-ATTACK MODEL"
)

print(
    "=" * 80
)

torch.save(
    model.state_dict(),
    OUTPUT_MODEL_FILE
)


# ============================================================
# SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    OUTPUT_HISTORY_FILE,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\nModel saved to:"
)

print(
    OUTPUT_MODEL_FILE
)

print(
    "\nTraining history saved to:"
)

print(
    OUTPUT_HISTORY_FILE
)

print(
    "\n" + "=" * 80
)

print(
    "FINAL MULTI-ATTACK ADVERSARIAL TRAINING COMPLETED"
)

print(
    "=" * 80
)