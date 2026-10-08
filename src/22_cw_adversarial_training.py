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

OUTPUT_MODEL = os.path.join(
    BASE_DIR,
    "results/dnn_cw_adversarial_trained.pth"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.001

RANDOM_SEED = 42

# C&W settings
CW_STEPS = 10
CW_LR = 0.01
CW_C = 1.0
CW_KAPPA = 0.0

# Loss weights
CLEAN_LOSS_WEIGHT = 0.5
ADVERSARIAL_LOSS_WEIGHT = 0.5


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
print("             C&W ADVERSARIAL TRAINING")
print("=" * 90)

print(f"\nDevice: {device}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")

print("\nC&W settings:")
print(f"Steps: {CW_STEPS}")
print(f"Learning rate: {CW_LR}")
print(f"C: {CW_C}")
print(f"Kappa: {CW_KAPPA}")

print("\nLoss weights:")
print(f"Clean loss: {CLEAN_LOSS_WEIGHT}")
print(f"Adversarial loss: {ADVERSARIAL_LOSS_WEIGHT}")


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
# C&W ATTACK
# ============================================================

def cw_attack(
    model,
    x,
    y
):

    """
    Generate C&W adversarial examples.

    The attack is performed in the standardized feature space.
    """

    delta = torch.zeros_like(
        x,
        requires_grad=True
    )

    optimizer = optim.Adam(
        [delta],
        lr=CW_LR
    )

    best_adv = x.detach().clone()

    best_l2 = torch.full(
        (x.shape[0],),
        float("inf"),
        device=x.device
    )

    for _ in range(CW_STEPS):

        optimizer.zero_grad()

        x_adv = x + delta

        logits = model(
            x_adv
        ).squeeze(1)

        # ----------------------------------------------------
        # C&W classification term
        # ----------------------------------------------------

        direction = (
            2.0 * y - 1.0
        )

        classification_term = torch.clamp(
            direction * logits + CW_KAPPA,
            min=0.0
        )

        # ----------------------------------------------------
        # L2 perturbation
        # ----------------------------------------------------

        l2 = torch.sum(
            delta ** 2,
            dim=1
        )

        loss = (
            l2
            +
            CW_C * classification_term
        ).mean()

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Save best successful adversarial examples
        # ----------------------------------------------------

        with torch.no_grad():

            predictions = (
                torch.sigmoid(logits)
                >= 0.5
            ).float()

            successful = (
                predictions != y
            )

            improved = (
                successful
                &
                (l2 < best_l2)
            )

            best_l2[improved] = (
                l2[improved]
            )

            best_adv[improved] = (
                x_adv[improved]
            )

    # --------------------------------------------------------
    # If some examples were not successfully attacked,
    # keep their original values.
    # --------------------------------------------------------

    return best_adv.detach()


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\n")
print("=" * 90)
print("                    LOADING TRAINING DATA")
print("=" * 90)

print("\nLoading balanced training dataset...")

df = pd.read_csv(
    TRAIN_FILE
)

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

# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

if df["Label"].dtype == object:

    y = np.array([

        1
        if str(label).upper() == "ATTACK"
        else 0

        for label in df["Label"]

    ])

else:

    y = df[
        "Label"
    ].values.astype(
        np.int64
    )


print(
    f"\nTraining samples: {len(X)}"
)

print(
    f"Features: {X.shape[1]}"
)

print(
    f"BENIGN: {int((y == 0).sum())}"
)

print(
    f"ATTACK: {int((y == 1).sum())}"
)


# ============================================================
# LOAD SCALER
# ============================================================

print("\nLoading scaler...")

scaler = joblib.load(
    SCALER_FILE
)

X = scaler.transform(
    X
).astype(
    np.float32
)

print(
    "Training data standardized."
)


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
# MODEL
# ============================================================

input_dim = X.shape[1]

model = DNN(
    input_dim
)

model.to(device)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\n")
print("=" * 90)
print("                    STARTING TRAINING")
print("=" * 90)

print(
    "\nIMPORTANT:"
)

print(
    "C&W adversarial examples are generated "
    "inside every training batch."
)

print(
    "This will be significantly slower than FGSM/PGD training."
)


num_samples = len(
    X_tensor
)


for epoch in range(
    EPOCHS
):

    model.train()

    # --------------------------------------------------------
    # Shuffle training data
    # --------------------------------------------------------

    permutation = torch.randperm(
        num_samples
    )

    epoch_total_loss = 0.0
    epoch_clean_loss = 0.0
    epoch_adv_loss = 0.0

    clean_correct = 0
    adv_correct = 0

    processed_samples = 0

    # --------------------------------------------------------
    # Batches
    # --------------------------------------------------------

    for batch_start in range(
        0,
        num_samples,
        BATCH_SIZE
    ):

        batch_end = min(
            batch_start + BATCH_SIZE,
            num_samples
        )

        batch_indices = permutation[
            batch_start:batch_end
        ]

        x_batch = X_tensor[
            batch_indices
        ].to(device)

        y_batch = y_tensor[
            batch_indices
        ].to(device)

        # ----------------------------------------------------
        # Generate C&W adversarial examples
        #
        # IMPORTANT:
        # Do this using the current model, but do not allow
        # the attack optimization to update the model itself.
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            # We only need x/y here.
            # C&W itself requires gradients with respect
            # to delta, so model parameters are temporarily
            # frozen below.
            pass

        # Freeze model parameters during C&W generation
        requires_grad_state = []

        for parameter in model.parameters():

            requires_grad_state.append(
                parameter.requires_grad
            )

            parameter.requires_grad = False

        model.eval()

        x_adv = cw_attack(
            model,
            x_batch,
            y_batch
        )

        # Restore parameter gradient settings
        for parameter, old_state in zip(
            model.parameters(),
            requires_grad_state
        ):

            parameter.requires_grad = old_state

        # ----------------------------------------------------
        # Train on clean + adversarial examples
        # ----------------------------------------------------

        model.train()

        optimizer.zero_grad()

        # Clean prediction
        clean_logits = model(
            x_batch
        ).squeeze(1)

        clean_loss = nn.functional.binary_cross_entropy_with_logits(
            clean_logits,
            y_batch
        )

        # Adversarial prediction
        adv_logits = model(
            x_adv
        ).squeeze(1)

        adv_loss = nn.functional.binary_cross_entropy_with_logits(
            adv_logits,
            y_batch
        )

        # Combined loss
        total_loss = (
            CLEAN_LOSS_WEIGHT
            *
            clean_loss
            +
            ADVERSARIAL_LOSS_WEIGHT
            *
            adv_loss
        )

        total_loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_size_actual = len(
            x_batch
        )

        epoch_total_loss += (
            total_loss.item()
            *
            batch_size_actual
        )

        epoch_clean_loss += (
            clean_loss.item()
            *
            batch_size_actual
        )

        epoch_adv_loss += (
            adv_loss.item()
            *
            batch_size_actual
        )

        # Clean accuracy
        clean_predictions = (
            torch.sigmoid(
                clean_logits
            )
            >= 0.5
        ).long()

        clean_correct += int(
            (
                clean_predictions
                ==
                y_batch.long()
            ).sum().item()
        )

        # Adversarial accuracy
        adv_predictions = (
            torch.sigmoid(
                adv_logits
            )
            >= 0.5
        ).long()

        adv_correct += int(
            (
                adv_predictions
                ==
                y_batch.long()
            ).sum().item()
        )

        processed_samples += (
            batch_size_actual
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_start == 0
            or
            batch_end == num_samples
            or
            batch_end % 50000 < BATCH_SIZE
        ):

            print(
                f"\rEpoch {epoch + 1}/{EPOCHS} | "
                f"Samples: "
                f"{batch_end:,}/{num_samples:,}",
                end=""
            )

    print()

    # --------------------------------------------------------
    # Epoch averages
    # --------------------------------------------------------

    avg_total_loss = (
        epoch_total_loss
        /
        processed_samples
    )

    avg_clean_loss = (
        epoch_clean_loss
        /
        processed_samples
    )

    avg_adv_loss = (
        epoch_adv_loss
        /
        processed_samples
    )

    clean_accuracy = (
        clean_correct
        /
        processed_samples
    )

    adv_accuracy = (
        adv_correct
        /
        processed_samples
    )

    # --------------------------------------------------------
    # Epoch output
    # --------------------------------------------------------

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"  Total loss: "
        f"{avg_total_loss:.6f}"
    )

    print(
        f"  Clean loss: "
        f"{avg_clean_loss:.6f}"
    )

    print(
        f"  Adversarial loss: "
        f"{avg_adv_loss:.6f}"
    )

    print(
        f"  Clean accuracy: "
        f"{clean_accuracy * 100:.4f}%"
    )

    print(
        f"  Adversarial accuracy: "
        f"{adv_accuracy * 100:.4f}%"
    )


# ============================================================
# SAVE MODEL
# ============================================================

print("\n")
print("=" * 90)
print("                    SAVING C&W-TRAINED MODEL")
print("=" * 90)

torch.save(
    model.state_dict(),
    OUTPUT_MODEL
)

print(
    f"\nC&W-trained model saved to:"
)

print(
    OUTPUT_MODEL
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 90)
print("                  C&W TRAINING COMPLETED")
print("=" * 90)

print(
    "\nNext step:"
)

print(
    "Evaluate this model against:"
)

print(
    "FGSM + PGD + C&W + DeepFool"
)

print("\n")