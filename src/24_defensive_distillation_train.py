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

# Original scaler used for the balanced model
SCALER_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_balanced_scaler.pkl"
)

# IMPORTANT:
# Use the already-trained high-quality balanced model
# as the teacher.
TEACHER_MODEL_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_balanced.pth"
)

# New V3 teacher/student output names
# The teacher is copied from the original balanced model.
V3_TEACHER_MODEL_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_distillation_teacher_v3.pth"
)

STUDENT_MODEL_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_distilled_student_v3.pth"
)

STUDENT_SCALER_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_distilled_student_scaler_v3.pkl"
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

RANDOM_SEED = 42

BATCH_SIZE = 256

STUDENT_EPOCHS = 10

LEARNING_RATE = 0.001

# Defensive distillation temperature
TEMPERATURE = 20.0

# Weight assigned to the soft-target loss
ALPHA = 0.5


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
print("=" * 100)
print("       DEFENSIVE DISTILLATION V3 - PRETRAINED TEACHER")
print("=" * 100)

print(f"\nDevice: {device}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Student epochs: {STUDENT_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Temperature: {TEMPERATURE}")
print(f"Alpha: {ALPHA}")

print("\nTeacher:")
print(TEACHER_MODEL_FILE)


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
# LOAD BALANCED TRAINING DATA
# ============================================================

print("\n")
print("=" * 100)
print("                    LOADING TRAINING DATA")
print("=" * 100)

if not os.path.exists(TRAIN_FILE):

    raise FileNotFoundError(
        f"Training file not found:\n{TRAIN_FILE}"
    )

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

    y = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("ATTACK")
        .astype(np.float32)
        .values
    )

else:

    y = df[
        "Label"
    ].values.astype(
        np.float32
    )


print(
    f"BENIGN samples: "
    f"{int((y == 0).sum()):,}"
)

print(
    f"ATTACK samples: "
    f"{int((y == 1).sum()):,}"
)


# ============================================================
# LOAD SCALER
# ============================================================

print("\nLoading balanced-model scaler...")

if not os.path.exists(SCALER_FILE):

    raise FileNotFoundError(
        f"Scaler not found:\n{SCALER_FILE}"
    )

scaler = joblib.load(
    SCALER_FILE
)

X_scaled = scaler.transform(
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
    X_scaled,
    dtype=torch.float32
)

y_tensor = torch.tensor(
    y,
    dtype=torch.float32
)


# ============================================================
# DATASET
# ============================================================

dataset = torch.utils.data.TensorDataset(
    X_tensor,
    y_tensor
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# ============================================================
# MODEL INFORMATION
# ============================================================

input_dim = X_scaled.shape[1]

print(
    f"\nInput features: {input_dim}"
)


# ============================================================
# LOAD PRETRAINED TEACHER
# ============================================================

print("\n")
print("=" * 100)
print("                 LOADING PRETRAINED TEACHER")
print("=" * 100)

if not os.path.exists(TEACHER_MODEL_FILE):

    raise FileNotFoundError(
        f"Teacher model not found:\n"
        f"{TEACHER_MODEL_FILE}"
    )

teacher = DNN(
    input_dim
).to(device)

teacher.load_state_dict(
    torch.load(
        TEACHER_MODEL_FILE,
        map_location=device,
        weights_only=True
    )
)

teacher.eval()

# Teacher must NOT be updated.
for parameter in teacher.parameters():

    parameter.requires_grad = False


print(
    "Pretrained balanced teacher loaded successfully."
)

print(
    f"Teacher model:\n{TEACHER_MODEL_FILE}"
)


# ============================================================
# VERIFY TEACHER ON TRAINING DATA
# ============================================================

print("\n")
print("=" * 100)
print("                 VERIFYING TEACHER")
print("=" * 100)

teacher_correct = 0
teacher_total = 0

with torch.no_grad():

    for start in range(
        0,
        len(X_scaled),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(X_scaled)
        )

        batch_X = torch.tensor(
            X_scaled[start:end],
            dtype=torch.float32,
            device=device
        )

        batch_y = torch.tensor(
            y[start:end],
            dtype=torch.float32,
            device=device
        )

        logits = teacher(
            batch_X
        ).squeeze(1)

        predictions = (
            torch.sigmoid(
                logits
            )
            >= 0.5
        ).float()

        teacher_correct += int(
            (predictions == batch_y).sum()
        )

        teacher_total += len(
            batch_X
        )


teacher_training_accuracy = (
    teacher_correct /
    teacher_total
)

print(
    f"Teacher training accuracy: "
    f"{teacher_training_accuracy * 100:.4f}%"
)


# ============================================================
# SAVE COPY OF TEACHER
# ============================================================

torch.save(
    teacher.state_dict(),
    V3_TEACHER_MODEL_FILE
)

print(
    "\nV3 teacher copy saved:"
)

print(
    V3_TEACHER_MODEL_FILE
)


# ============================================================
# GENERATE SOFT TARGETS
# ============================================================

print("\n")
print("=" * 100)
print("                    GENERATING SOFT TARGETS")
print("=" * 100)

teacher.eval()

soft_targets = []

with torch.no_grad():

    for start in range(
        0,
        len(X_scaled),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(X_scaled)
        )

        batch_X = torch.tensor(
            X_scaled[start:end],
            dtype=torch.float32,
            device=device
        )

        teacher_logits = teacher(
            batch_X
        ).squeeze(1)

        # Temperature-scaled teacher probabilities
        soft_probabilities = torch.sigmoid(
            teacher_logits /
            TEMPERATURE
        )

        soft_targets.append(
            soft_probabilities.cpu()
        )


soft_targets = torch.cat(
    soft_targets
)

print(
    f"Soft targets generated for "
    f"{len(soft_targets):,} samples."
)

print(
    f"Temperature used: {TEMPERATURE}"
)


# ============================================================
# STUDENT DATASET
# ============================================================

student_dataset = torch.utils.data.TensorDataset(
    X_tensor,
    y_tensor,
    soft_targets
)

student_loader = torch.utils.data.DataLoader(
    student_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# ============================================================
# CREATE STUDENT
# ============================================================

print("\n")
print("=" * 100)
print("                    CREATING STUDENT MODEL")
print("=" * 100)

student = DNN(
    input_dim
).to(device)

student_optimizer = optim.Adam(
    student.parameters(),
    lr=LEARNING_RATE
)

hard_criterion = nn.BCEWithLogitsLoss()


# ============================================================
# TRAIN STUDENT
# ============================================================

print("\n")
print("=" * 100)
print("                    TRAINING DISTILLED STUDENT")
print("=" * 100)

for epoch in range(
    STUDENT_EPOCHS
):

    student.train()

    total_loss = 0.0

    total_soft_loss = 0.0

    total_hard_loss = 0.0

    correct = 0

    total = 0

    for (
        batch_X,
        batch_y,
        batch_soft
    ) in student_loader:

        batch_X = batch_X.to(device)

        batch_y = batch_y.to(device)

        batch_soft = batch_soft.to(device)

        student_optimizer.zero_grad()

        # ----------------------------------------------------
        # Student raw logits
        # ----------------------------------------------------

        student_raw_logits = student(
            batch_X
        ).squeeze(1)

        # ----------------------------------------------------
        # Hard-label loss
        # ----------------------------------------------------

        hard_loss = hard_criterion(
            student_raw_logits,
            batch_y
        )

        # ----------------------------------------------------
        # Temperature-scaled student logits
        # ----------------------------------------------------

        student_logits_T = (
            student_raw_logits /
            TEMPERATURE
        )

        # ----------------------------------------------------
        # Soft-target loss
        # ----------------------------------------------------

        soft_loss = nn.functional.binary_cross_entropy(
            torch.sigmoid(
                student_logits_T
            ),
            batch_soft
        )

        # ----------------------------------------------------
        # T^2 scaling
        #
        # This compensates for the smaller gradients
        # produced by temperature scaling.
        # ----------------------------------------------------

        soft_loss_scaled = (
            TEMPERATURE ** 2
        ) * soft_loss

        # ----------------------------------------------------
        # Combined distillation loss
        # ----------------------------------------------------

        loss = (
            ALPHA * soft_loss_scaled
            +
            (1.0 - ALPHA) * hard_loss
        )

        loss.backward()

        student_optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_loss += (
            loss.item()
            *
            len(batch_X)
        )

        total_soft_loss += (
            soft_loss.item()
            *
            len(batch_X)
        )

        total_hard_loss += (
            hard_loss.item()
            *
            len(batch_X)
        )

        # ----------------------------------------------------
        # Normal inference
        #
        # IMPORTANT:
        # No temperature is used during inference.
        # ----------------------------------------------------

        predictions = (
            torch.sigmoid(
                student_raw_logits
            )
            >= 0.5
        ).float()

        correct += int(
            (predictions == batch_y).sum()
        )

        total += len(
            batch_X
        )

    epoch_loss = (
        total_loss /
        total
    )

    epoch_soft_loss = (
        total_soft_loss /
        total
    )

    epoch_hard_loss = (
        total_hard_loss /
        total
    )

    epoch_accuracy = (
        correct /
        total
    )

    print(
        f"Student Epoch "
        f"{epoch + 1}/{STUDENT_EPOCHS} | "
        f"Total Loss: {epoch_loss:.6f} | "
        f"Soft Loss: {epoch_soft_loss:.6f} | "
        f"Hard Loss: {epoch_hard_loss:.6f} | "
        f"Accuracy: {epoch_accuracy * 100:.4f}%"
    )


# ============================================================
# SAVE STUDENT
# ============================================================

torch.save(
    student.state_dict(),
    STUDENT_MODEL_FILE
)

print(
    "\nStudent model saved:"
)

print(
    STUDENT_MODEL_FILE
)


# ============================================================
# SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    STUDENT_SCALER_FILE
)

print(
    "\nStudent scaler saved:"
)

print(
    STUDENT_SCALER_FILE
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 100)
print("             DEFENSIVE DISTILLATION V3 COMPLETE")
print("=" * 100)

print("\nTeacher:")
print(TEACHER_MODEL_FILE)

print("\nTeacher copy:")
print(V3_TEACHER_MODEL_FILE)

print("\nStudent:")
print(STUDENT_MODEL_FILE)

print("\nStudent scaler:")
print(STUDENT_SCALER_FILE)

print("\nTemperature:")
print(TEMPERATURE)

print("\nAlpha:")
print(ALPHA)

print("\nTraining samples:")
print(f"{len(X_scaled):,}")

print("\nCompleted successfully.\n")