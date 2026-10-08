import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "/home/ubuntu/aml-experiment"

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "data",
    "ml",
    "balanced_train.csv"
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

OUTPUT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "results",
    "dnn_adversarial_trained.pth"
)

OUTPUT_HISTORY_PATH = os.path.join(
    BASE_DIR,
    "results",
    "adversarial_training_history.csv"
)


# ============================================================
# TRAINING PARAMETERS
# ============================================================

BATCH_SIZE = 256

# CPU-friendly starting point
EPOCHS = 10

LEARNING_RATE = 0.001

# FGSM epsilon used for adversarial training
FGSM_EPSILON = 0.10

# Equal contribution from clean and adversarial examples
CLEAN_WEIGHT = 0.5
ADV_WEIGHT = 0.5

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("ADVERSARIAL TRAINING")
print("=" * 70)

print(f"Device: {DEVICE}")

if DEVICE.type == "cpu":
    print(
        "\nWARNING: Running on CPU."
        "\nAdversarial training will take longer."
    )


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
    SCALER_PATH
)

print("Scaler loaded successfully.")


# ============================================================
# LOAD BALANCED TRAINING DATA
# ============================================================

print("\nLoading balanced training data...")

df = pd.read_csv(
    TRAIN_PATH
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

X = df.drop(
    columns=["Label"]
).values.astype(
    np.float32
)

y = df["Label"].values.astype(
    np.float32
)


print(
    f"Feature shape before scaling: {X.shape}"
)

print(
    f"BENIGN: {np.sum(y == 0):,}"
)

print(
    f"ATTACK: {np.sum(y == 1):,}"
)


# ============================================================
# SCALE DATA
# ============================================================

print("\nScaling training data...")

X = scaler.transform(
    X
).astype(
    np.float32
)

print(
    f"Feature shape after scaling: {X.shape}"
)


# ============================================================
# CREATE TORCH DATASET
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
    f"\nBatches per epoch: {len(dataloader):,}"
)


# ============================================================
# LOAD ORIGINAL BALANCED MODEL
# ============================================================

print("\nLoading original balanced DNN...")

model = DNN(
    input_dim=78
)

checkpoint = torch.load(
    MODEL_PATH,
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
    "Original balanced model loaded successfully."
)


# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# FGSM GENERATION
# ============================================================

def generate_fgsm(
    model,
    x,
    y,
    epsilon
):

    """
    Generate FGSM adversarial examples.

    Perturbation is applied in the standardized
    feature space used by the DNN.
    """

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

    gradient = x_adv.grad.data

    x_adv = (
        x_adv.detach()
        + epsilon * gradient.sign()
    )

    return x_adv.detach()


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING ADVERSARIAL TRAINING")
print("=" * 70)

print(
    f"\nFGSM epsilon: {FGSM_EPSILON}"
)

print(
    f"Epochs: {EPOCHS}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Learning rate: {LEARNING_RATE}"
)

print(
    f"Clean loss weight: {CLEAN_WEIGHT}"
)

print(
    f"Adversarial loss weight: {ADV_WEIGHT}"
)


history = []


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    total_clean_loss = 0.0

    total_adv_loss = 0.0

    correct_clean = 0

    correct_adv = 0

    total_samples = 0


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
        # STEP 1: Generate FGSM adversarial examples
        # ----------------------------------------------------

        model.eval()

        x_adv = generate_fgsm(
            model,
            x_batch,
            y_batch,
            FGSM_EPSILON
        )


        # ----------------------------------------------------
        # STEP 2: Train on clean + adversarial examples
        # ----------------------------------------------------

        model.train()

        optimizer.zero_grad()


        # Clean predictions
        clean_logits = model(
            x_batch
        ).squeeze(1)

        clean_loss = criterion(
            clean_logits,
            y_batch
        )


        # Adversarial predictions
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


        loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_size = (
            y_batch.size(0)
        )

        total_loss += (
            loss.item() * batch_size
        )

        total_clean_loss += (
            clean_loss.item() * batch_size
        )

        total_adv_loss += (
            adv_loss.item() * batch_size
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
        ) % 500 == 0:

            print(
                f"Epoch "
                f"{epoch + 1}/{EPOCHS} | "
                f"Batch "
                f"{batch_index + 1}/{len(dataloader)} | "
                f"Loss "
                f"{loss.item():.4f}"
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
            epoch_adv_accuracy
    })


    print("\n" + "-" * 70)

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

    print("-" * 70)


# ============================================================
# SAVE MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING ADVERSARIALLY TRAINED MODEL")
print("=" * 70)


torch.save(
    model.state_dict(),
    OUTPUT_MODEL_PATH
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    OUTPUT_HISTORY_PATH,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\nModel saved to:"
)

print(
    OUTPUT_MODEL_PATH
)

print(
    "\nTraining history saved to:"
)

print(
    OUTPUT_HISTORY_PATH
)

print("\n" + "=" * 70)
print("ADVERSARIAL TRAINING COMPLETED")
print("=" * 70)