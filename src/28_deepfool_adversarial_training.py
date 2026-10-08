import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "/home/ubuntu/aml-experiment"

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "data/ml/balanced_train.csv"
)

SCALER_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_balanced_scaler.pkl"
)

BASE_MODEL_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_balanced.pth"
)

OUTPUT_MODEL_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_deepfool_adversarial_trained.pth"
)

OUTPUT_HISTORY_FILE = os.path.join(
    BASE_DIR,
    "results/deepfool_adversarial_training_history.csv"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.001

RANDOM_SEED = 42

CLEAN_LOSS_WEIGHT = 0.5
ADV_LOSS_WEIGHT = 0.5


# ============================================================
# DEEPFOOL SETTINGS
# ============================================================

DEEPFOOL_STEPS = 10
DEEPFOOL_OVERSHOOT = 0.02

# Small minimum movement to avoid numerical stagnation
DEEPFOOL_MIN_STEP = 0.01


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

print("\n")
print("=" * 90)
print("          DEEPFOOL ADVERSARIAL TRAINING")
print("=" * 90)

print(f"Device: {device}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"DeepFool steps: {DEEPFOOL_STEPS}")
print(f"DeepFool overshoot: {DEEPFOOL_OVERSHOOT}")
print(f"Clean loss weight: {CLEAN_LOSS_WEIGHT}")
print(f"Adversarial loss weight: {ADV_LOSS_WEIGHT}")


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
# LOAD TRAINING DATA
# ============================================================

print("\nLoading balanced training data...")

df = pd.read_csv(
    TRAIN_FILE
)

print(
    f"Training rows: {len(df):,}"
)

print(
    f"Training columns: {len(df.columns)}"
)


# ============================================================
# FEATURES
# ============================================================

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
# LABELS
# ============================================================

if df["Label"].dtype == object:

    y = np.array([
        1
        if str(label).upper() == "ATTACK"
        else 0
        for label in df["Label"]
    ], dtype=np.float32)

else:

    y = df[
        "Label"
    ].values.astype(
        np.float32
    )


print(
    f"BENIGN samples: {int((y == 0).sum()):,}"
)

print(
    f"ATTACK samples: {int((y == 1).sum()):,}"
)


# ============================================================
# LOAD SCALER
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

print("Training data standardized.")


# ============================================================
# CONVERT TO TENSORS
# ============================================================

X_tensor = torch.tensor(
    X_scaled,
    dtype=torch.float32
)

y_tensor = torch.tensor(
    y,
    dtype=torch.float32
).view(-1, 1)


print(
    f"Input features: {X_tensor.shape[1]}"
)


# ============================================================
# LOAD ORIGINAL BALANCED MODEL
# ============================================================

print("\nLoading original balanced model...")

model = DNN(
    X_tensor.shape[1]
)

model.load_state_dict(
    torch.load(
        BASE_MODEL_FILE,
        map_location=device
    )
)

model.to(device)

print(
    "Original balanced model loaded successfully."
)


# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# DEEPFOOL ATTACK
# ============================================================

def deepfool_batch(
    model,
    x,
    y_true
):

    """
    Generate DeepFool-style adversarial examples
    using the binary classification logit boundary.

    The update is based on:

        r = -f(x) / ||grad f(x)||^2 * grad f(x)

    followed by the overshoot factor.

    A minimum step is used only when the calculated
    boundary step becomes numerically too small.
    """

    x_adv = x.clone().detach()

    batch_size = x_adv.size(0)

    # Keep track of which samples have already crossed
    # the decision boundary.
    active = torch.ones(
        batch_size,
        dtype=torch.bool,
        device=x.device
    )

    for _ in range(DEEPFOOL_STEPS):

        if not active.any():
            break

        x_current = x_adv.detach().clone()

        x_current.requires_grad_(True)

        logits = model(
            x_current
        ).view(-1)

        predictions = (
            torch.sigmoid(logits) >= 0.5
        ).long()

        true_labels = y_true.view(-1).long()

        # Samples that still have the original prediction
        still_correct = (
            predictions == true_labels
        )

        active = active & still_correct

        if not active.any():
            break

        # ----------------------------------------------------
        # Calculate gradient for each active sample
        # ----------------------------------------------------

        gradient = torch.autograd.grad(
            logits.sum(),
            x_current,
            retain_graph=False,
            create_graph=False
        )[0]

        gradient_norm_squared = (
            torch.sum(
                gradient ** 2,
                dim=1
            )
            +
            1e-12
        )

        # ----------------------------------------------------
        # Boundary distance in logit space
        # ----------------------------------------------------

        boundary_step = (
            -logits
            /
            gradient_norm_squared
        ).unsqueeze(1)

        perturbation = (
            (1.0 + DEEPFOOL_OVERSHOOT)
            *
            boundary_step
            *
            gradient
        )

        # ----------------------------------------------------
        # Minimum step for numerical stability
        # ----------------------------------------------------

        perturbation_norm = torch.norm(
            perturbation,
            p=2,
            dim=1,
            keepdim=True
        )

        unit_gradient = (
            gradient
            /
            (
                torch.norm(
                    gradient,
                    p=2,
                    dim=1,
                    keepdim=True
                )
                +
                1e-12
            )
        )

        small_step = (
            perturbation_norm
            <
            DEEPFOOL_MIN_STEP
        )

        perturbation = torch.where(
            small_step,
            DEEPFOOL_MIN_STEP * unit_gradient,
            perturbation
        )

        # Only modify samples that are still active
        active_mask = active.unsqueeze(1)

        x_adv = torch.where(
            active_mask,
            x_adv + perturbation.detach(),
            x_adv
        )

        x_adv = x_adv.detach()

    return x_adv


# ============================================================
# DATA LOADER
# ============================================================

dataset = torch.utils.data.TensorDataset(
    X_tensor,
    y_tensor
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False
)


# ============================================================
# TRAINING HISTORY
# ============================================================

history = []


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n")
print("=" * 90)
print("                 STARTING ADVERSARIAL TRAINING")
print("=" * 90)


for epoch in range(EPOCHS):

    model.train()

    total_loss_sum = 0.0
    clean_loss_sum = 0.0
    adv_loss_sum = 0.0

    clean_correct = 0
    adv_correct = 0

    total_samples = 0


    for batch_idx, (
        batch_x,
        batch_y
    ) in enumerate(loader):

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)


        # ----------------------------------------------------
        # Generate DeepFool adversarial examples
        # ----------------------------------------------------

        model.eval()

        batch_x_adv = deepfool_batch(
            model,
            batch_x,
            batch_y
        )

        model.train()


        # ----------------------------------------------------
        # Clean predictions
        # ----------------------------------------------------

        clean_logits = model(
            batch_x
        )

        clean_loss = criterion(
            clean_logits,
            batch_y
        )


        # ----------------------------------------------------
        # Adversarial predictions
        # ----------------------------------------------------

        adv_logits = model(
            batch_x_adv
        )

        adv_loss = criterion(
            adv_logits,
            batch_y
        )


        # ----------------------------------------------------
        # Combined loss
        # ----------------------------------------------------

        total_loss = (
            CLEAN_LOSS_WEIGHT * clean_loss
            +
            ADV_LOSS_WEIGHT * adv_loss
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        total_loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        with torch.no_grad():

            clean_predictions = (
                torch.sigmoid(clean_logits) >= 0.5
            ).float()

            adv_predictions = (
                torch.sigmoid(adv_logits) >= 0.5
            ).float()

            clean_correct += int(
                (
                    clean_predictions == batch_y
                ).sum().item()
            )

            adv_correct += int(
                (
                    adv_predictions == batch_y
                ).sum().item()
            )


        batch_size = batch_x.size(0)

        total_samples += batch_size

        total_loss_sum += (
            total_loss.item()
            *
            batch_size
        )

        clean_loss_sum += (
            clean_loss.item()
            *
            batch_size
        )

        adv_loss_sum += (
            adv_loss.item()
            *
            batch_size
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_idx % 100 == 0
            or
            batch_idx == len(loader) - 1
        ):

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Batch {batch_idx + 1}/{len(loader)}",
                end="\r"
            )


    # ========================================================
    # EPOCH RESULTS
    # ========================================================

    epoch_total_loss = (
        total_loss_sum
        /
        total_samples
    )

    epoch_clean_loss = (
        clean_loss_sum
        /
        total_samples
    )

    epoch_adv_loss = (
        adv_loss_sum
        /
        total_samples
    )

    epoch_clean_accuracy = (
        clean_correct
        /
        total_samples
    )

    epoch_adv_accuracy = (
        adv_correct
        /
        total_samples
    )


    print("\n")

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Total loss: {epoch_total_loss:.6f} | "
        f"Clean loss: {epoch_clean_loss:.6f} | "
        f"Adversarial loss: {epoch_adv_loss:.6f} | "
        f"Clean accuracy: {epoch_clean_accuracy * 100:.4f}% | "
        f"Adversarial accuracy: {epoch_adv_accuracy * 100:.4f}%"
    )


    history.append({

        "epoch": epoch + 1,

        "total_loss":
            epoch_total_loss,

        "clean_loss":
            epoch_clean_loss,

        "adversarial_loss":
            epoch_adv_loss,

        "clean_accuracy":
            epoch_clean_accuracy,

        "adversarial_accuracy":
            epoch_adv_accuracy

    })


# ============================================================
# SAVE MODEL
# ============================================================

print("\n")
print("=" * 90)
print("                    SAVING MODEL")
print("=" * 90)

torch.save(
    model.state_dict(),
    OUTPUT_MODEL_FILE
)

print(
    f"Model saved:\n{OUTPUT_MODEL_FILE}"
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    OUTPUT_HISTORY_FILE,
    index=False
)

print(
    f"Training history saved:\n"
    f"{OUTPUT_HISTORY_FILE}"
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 90)
print("              DEEPFOOL ADVERSARIAL TRAINING COMPLETE")
print("=" * 90)

print(
    "\nNext step:"
)

print(
    "Evaluate the DeepFool-trained model against:"
)

print(
    "Clean, FGSM, PGD, C&W, and DeepFool."
)