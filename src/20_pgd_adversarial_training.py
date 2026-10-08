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

ORIGINAL_MODEL_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_balanced.pth"
)

OUTPUT_MODEL_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_pgd_adversarial_trained.pth"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 0.001

# PGD settings
EPSILON = 0.10
PGD_STEPS = 10
ALPHA = EPSILON / PGD_STEPS

# Loss weights
CLEAN_LOSS_WEIGHT = 0.5
ADV_LOSS_WEIGHT = 0.5

RANDOM_SEED = 42


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

print("=" * 70)
print("PGD ADVERSARIAL TRAINING")
print("=" * 70)

print(f"Device: {device}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"PGD epsilon: {EPSILON}")
print(f"PGD steps: {PGD_STEPS}")
print(f"PGD alpha: {ALPHA}")
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
# LOAD DATA
# ============================================================

print("\nLoading balanced training data...")

df = pd.read_csv(
    TRAIN_FILE
)

print(
    f"Training rows: {len(df):,}"
)

print(
    f"Columns: {len(df.columns)}"
)


# ============================================================
# FEATURES AND LABELS
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

# Convert labels
if df["Label"].dtype == object:

    y = np.array([
        1.0
        if str(label).upper() == "ATTACK"
        else 0.0
        for label in df["Label"]
    ], dtype=np.float32)

else:

    y = df[
        "Label"
    ].values.astype(
        np.float32
    )


print(
    f"Features: {X.shape[1]}"
)

print(
    f"BENIGN: {int((y == 0).sum()):,}"
)

print(
    f"ATTACK: {int((y == 1).sum()):,}"
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
# STANDARDIZE TRAINING DATA
# ============================================================

print("\nStandardizing training data...")

X = scaler.transform(
    X
).astype(
    np.float32
)

print("Standardization complete.")


# ============================================================
# CONVERT TO TENSORS
# ============================================================

X_tensor = torch.tensor(
    X,
    dtype=torch.float32
)

y_tensor = torch.tensor(
    y,
    dtype=torch.float32
)


# ============================================================
# CREATE MODEL
# ============================================================

input_dim = X.shape[1]

model = DNN(
    input_dim
).to(device)


# ============================================================
# LOAD ORIGINAL CLEAN MODEL
# ============================================================

print("\nLoading original DNN...")

checkpoint = torch.load(
    ORIGINAL_MODEL_FILE,
    map_location=device
)

model.load_state_dict(
    checkpoint
)

print(
    "Original DNN loaded successfully."
)

print(
    "PGD adversarial training will start "
    "from the original clean model."
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
# PGD ATTACK FOR TRAINING
# ============================================================

def pgd_attack(
    model,
    x,
    y
):

    # --------------------------------------------------------
    # Random initialization
    # --------------------------------------------------------

    delta = torch.empty_like(
        x
    ).uniform_(
        -EPSILON,
        EPSILON
    )

    delta.requires_grad = True

    # --------------------------------------------------------
    # PGD iterations
    # --------------------------------------------------------

    for _ in range(PGD_STEPS):

        logits = model(
            x + delta
        ).squeeze(1)

        loss = criterion(
            logits,
            y
        )

        gradients = torch.autograd.grad(
            loss,
            delta,
            retain_graph=False,
            create_graph=False
        )[0]

        # Move in direction that maximizes loss
        delta = (
            delta
            +
            ALPHA * gradients.sign()
        )

        # Project perturbation back into
        # the L-infinity epsilon ball
        delta = torch.clamp(
            delta,
            -EPSILON,
            EPSILON
        ).detach()

        delta.requires_grad = True

    return (
        x + delta
    ).detach()


# ============================================================
# TRAINING
# ============================================================

print("\n")
print("=" * 70)
print("STARTING PGD ADVERSARIAL TRAINING")
print("=" * 70)


num_samples = len(X_tensor)

indices = np.arange(
    num_samples
)


for epoch in range(
    EPOCHS
):

    # --------------------------------------------------------
    # Shuffle training data
    # --------------------------------------------------------

    np.random.shuffle(
        indices
    )

    model.train()

    total_loss = 0.0
    total_clean_loss = 0.0
    total_adv_loss = 0.0

    total_clean_correct = 0
    total_adv_correct = 0

    total_samples = 0

    # --------------------------------------------------------
    # Mini-batches
    # --------------------------------------------------------

    for start in range(
        0,
        num_samples,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            num_samples
        )

        batch_indices = indices[
            start:end
        ]

        x_batch = X_tensor[
            batch_indices
        ].to(device)

        y_batch = y_tensor[
            batch_indices
        ].to(device)

        # ----------------------------------------------------
        # Generate PGD adversarial examples
        # ----------------------------------------------------

        model.eval()

        x_adv = pgd_attack(
            model,
            x_batch,
            y_batch
        )

        # ----------------------------------------------------
        # Return to training mode
        # ----------------------------------------------------

        model.train()

        # ----------------------------------------------------
        # Clean prediction
        # ----------------------------------------------------

        clean_logits = model(
            x_batch
        ).squeeze(1)

        clean_loss = criterion(
            clean_logits,
            y_batch
        )

        # ----------------------------------------------------
        # Adversarial prediction
        # ----------------------------------------------------

        adv_logits = model(
            x_adv
        ).squeeze(1)

        adv_loss = criterion(
            adv_logits,
            y_batch
        )

        # ----------------------------------------------------
        # Combined loss
        # ----------------------------------------------------

        loss = (
            CLEAN_LOSS_WEIGHT * clean_loss
            +
            ADV_LOSS_WEIGHT * adv_loss
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_size_actual = (
            end - start
        )

        total_loss += (
            loss.item()
            *
            batch_size_actual
        )

        total_clean_loss += (
            clean_loss.item()
            *
            batch_size_actual
        )

        total_adv_loss += (
            adv_loss.item()
            *
            batch_size_actual
        )

        clean_predictions = (
            torch.sigmoid(
                clean_logits
            ) >= 0.5
        ).float()

        adv_predictions = (
            torch.sigmoid(
                adv_logits
            ) >= 0.5
        ).float()

        total_clean_correct += int(
            (
                clean_predictions
                ==
                y_batch
            ).sum().item()
        )

        total_adv_correct += int(
            (
                adv_predictions
                ==
                y_batch
            ).sum().item()
        )

        total_samples += (
            batch_size_actual
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            start == 0
            or
            end == num_samples
            or
            end % 100000 < BATCH_SIZE
        ):

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Samples: {end:,}/{num_samples:,}",
                end="\r"
            )

    # ========================================================
    # Epoch statistics
    # ========================================================

    avg_loss = (
        total_loss
        /
        total_samples
    )

    avg_clean_loss = (
        total_clean_loss
        /
        total_samples
    )

    avg_adv_loss = (
        total_adv_loss
        /
        total_samples
    )

    clean_accuracy = (
        total_clean_correct
        /
        total_samples
    )

    adv_accuracy = (
        total_adv_correct
        /
        total_samples
    )

    print("\n")

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Total loss: {avg_loss:.6f} | "
        f"Clean loss: {avg_clean_loss:.6f} | "
        f"Adversarial loss: {avg_adv_loss:.6f} | "
        f"Clean acc: {clean_accuracy * 100:.4f}% | "
        f"Adversarial acc: {adv_accuracy * 100:.4f}%"
    )


# ============================================================
# SAVE MODEL
# ============================================================

print("\n")
print("=" * 70)
print("SAVING PGD-TRAINED MODEL")
print("=" * 70)

torch.save(
    model.state_dict(),
    OUTPUT_MODEL_FILE
)

print(
    f"\nPGD-trained model saved to:"
)

print(
    OUTPUT_MODEL_FILE
)


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("PGD ADVERSARIAL TRAINING COMPLETED")
print("=" * 70)