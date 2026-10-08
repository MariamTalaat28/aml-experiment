# ============================================================
# 26_defensive_distillation_v4_train.py
#
# Defensive Distillation V4
#
# Teacher:
#   Pretrained balanced DNN
#
# Student:
#   Same DNN architecture
#
# Distillation:
#   - Temperature T = 20
#   - KL-divergence soft-target loss
#   - T^2 scaling
#   - Hard-label BCE loss
#   - alpha = 0.5
#
# Dataset:
#   Balanced CICIDS2017 training data
#
# IMPORTANT:
#   The paper mentions defensive distillation but does not
#   provide exact temperature/loss implementation details.
#   Therefore T=20 and alpha=0.5 are implementation choices.
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 1. CONFIGURATION
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

TEACHER_MODEL_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_balanced.pth"
)

V4_TEACHER_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_distillation_teacher_v4.pth"
)

STUDENT_MODEL_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_distilled_student_v4.pth"
)

STUDENT_SCALER_FILE = os.path.join(
    BASE_DIR,
    "results/dnn_distilled_student_scaler_v4.pkl"
)


# Training parameters
BATCH_SIZE = 256

STUDENT_EPOCHS = 10

LEARNING_RATE = 0.001

TEMPERATURE = 20.0

ALPHA = 0.5

RANDOM_SEED = 42


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# 3. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("DEFENSIVE DISTILLATION V4")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Student epochs: {STUDENT_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Temperature: {TEMPERATURE}")
print(f"Alpha: {ALPHA}")
print()


# ============================================================
# 4. MODEL ARCHITECTURE
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
# 5. LOAD TRAINING DATA
# ============================================================

print("Loading balanced training data...")

df = pd.read_csv(TRAIN_FILE)

print(f"Training rows: {len(df):,}")
print(f"Training columns: {len(df.columns)}")

if "Label" not in df.columns:

    raise ValueError(
        "ERROR: 'Label' column was not found in training data."
    )


# ------------------------------------------------------------
# Separate features and labels
# ------------------------------------------------------------

X_train = df.drop(columns=["Label"])

y_train = df["Label"]


# Convert labels to numeric
#
# Expected:
# BENIGN = 0
# ATTACK = 1
#
# Handles both numeric and string labels.
# ------------------------------------------------------------

if y_train.dtype == object:

    y_train = (
        y_train
        .astype(str)
        .str.strip()
        .str.upper()
        .map({
            "BENIGN": 0,
            "ATTACK": 1
        })
    )


y_train = pd.to_numeric(
    y_train,
    errors="coerce"
)


if y_train.isna().any():

    raise ValueError(
        "ERROR: Some labels could not be converted to 0/1."
    )


y_train = y_train.astype(np.float32)


print(
    f"BENIGN samples: {(y_train == 0).sum():,}"
)

print(
    f"ATTACK samples: {(y_train == 1).sum():,}"
)

print()


# ============================================================
# 6. LOAD ORIGINAL SCALER
# ============================================================

print("Loading balanced-model scaler...")

scaler = joblib.load(SCALER_FILE)

print(
    f"Scaler loaded from:\n{SCALER_FILE}"
)

print()


# ============================================================
# 7. SCALE TRAINING DATA
# ============================================================

print("Standardizing training data...")

X_train_scaled = scaler.transform(X_train)

X_train_scaled = np.asarray(
    X_train_scaled,
    dtype=np.float32
)

y_train_np = y_train.to_numpy(
    dtype=np.float32
)

print(
    f"Scaled training shape: {X_train_scaled.shape}"
)

print()


# ============================================================
# 8. CONVERT TO PYTORCH TENSORS
# ============================================================

X_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

y_tensor = torch.tensor(
    y_train_np,
    dtype=torch.float32
).view(-1, 1)


# ============================================================
# 9. DATA LOADER
# ============================================================

dataset = TensorDataset(
    X_tensor,
    y_tensor
)

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)


INPUT_DIM = X_train_scaled.shape[1]

print(f"Input features: {INPUT_DIM}")
print()


# ============================================================
# 10. LOAD PRETRAINED TEACHER
# ============================================================

print("=" * 70)
print("LOADING PRETRAINED TEACHER")
print("=" * 70)

teacher = DNN(INPUT_DIM).to(DEVICE)

teacher.load_state_dict(
    torch.load(
        TEACHER_MODEL_FILE,
        map_location=DEVICE,
        weights_only=True
    )
)

teacher.eval()

print("Pretrained balanced teacher loaded successfully.")

print(
    f"Teacher model:\n{TEACHER_MODEL_FILE}"
)

print()


# ============================================================
# 11. VERIFY TEACHER
# ============================================================

print("Verifying teacher...")

correct = 0
total = 0

with torch.no_grad():

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        logits = teacher(X_batch)

        predictions = (
            torch.sigmoid(logits) >= 0.5
        ).float()

        correct += (
            predictions == y_batch
        ).sum().item()

        total += y_batch.numel()


teacher_accuracy = (
    100.0 * correct / total
)

print(
    f"Teacher training accuracy: "
    f"{teacher_accuracy:.4f}%"
)

print()


# ============================================================
# 12. SAVE TEACHER COPY
# ============================================================

torch.save(
    teacher.state_dict(),
    V4_TEACHER_FILE
)

print(
    f"V4 teacher copy saved:\n"
    f"{V4_TEACHER_FILE}"
)

print()


# ============================================================
# 13. GENERATE TEACHER SOFT TARGETS
# ============================================================

print("=" * 70)
print("GENERATING SOFT TARGETS")
print("=" * 70)

teacher_soft_targets = []

teacher_hard_predictions = []

with torch.no_grad():

    for start in range(
        0,
        len(X_tensor),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(X_tensor)
        )

        X_batch = X_tensor[start:end].to(
            DEVICE
        )

        # Raw teacher logits
        teacher_logits = teacher(X_batch)

        # Temperature-scaled logits
        teacher_scaled_logits = (
            teacher_logits / TEMPERATURE
        )

        # Convert to soft probabilities
        soft_probs = torch.sigmoid(
            teacher_scaled_logits
        )

        teacher_soft_targets.append(
            soft_probs.cpu()
        )

        # Normal teacher prediction
        hard_probs = torch.sigmoid(
            teacher_logits
        )

        hard_predictions = (
            hard_probs >= 0.5
        ).float()

        teacher_hard_predictions.append(
            hard_predictions.cpu()
        )


teacher_soft_targets = torch.cat(
    teacher_soft_targets,
    dim=0
)

teacher_hard_predictions = torch.cat(
    teacher_hard_predictions,
    dim=0
)


print(
    f"Soft targets generated for "
    f"{len(teacher_soft_targets):,} samples."
)

print(
    f"Temperature used: {TEMPERATURE}"
)

print()


# ============================================================
# 14. CREATE STUDENT DATASET
# ============================================================

student_dataset = TensorDataset(
    X_tensor,
    y_tensor,
    teacher_soft_targets
)

student_loader = DataLoader(
    student_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)


# ============================================================
# 15. CREATE STUDENT MODEL
# ============================================================

student = DNN(INPUT_DIM).to(DEVICE)


# ============================================================
# 16. LOSS FUNCTIONS
# ============================================================

#
# Hard-label loss
#
hard_loss_function = nn.BCEWithLogitsLoss()


#
# Soft distillation loss
#
# We use KL divergence between:
#
# Teacher:
#   sigmoid(teacher_logits / T)
#
# Student:
#   sigmoid(student_logits / T)
#
# For binary classification, we represent both as
# two-class probability distributions:
#
# [P(BENIGN), P(ATTACK)]
#

def binary_kl_divergence(
    student_logits,
    teacher_soft_targets,
    temperature
):

    # --------------------------------------------------------
    # Student probabilities
    # --------------------------------------------------------

    student_probs_attack = torch.sigmoid(
        student_logits / temperature
    )

    student_probs_benign = (
        1.0 - student_probs_attack
    )

    student_probs = torch.cat(
        [
            student_probs_benign,
            student_probs_attack
        ],
        dim=1
    )


    # --------------------------------------------------------
    # Teacher probabilities
    # --------------------------------------------------------

    teacher_probs_attack = teacher_soft_targets

    teacher_probs_benign = (
        1.0 - teacher_probs_attack
    )

    teacher_probs = torch.cat(
        [
            teacher_probs_benign,
            teacher_probs_attack
        ],
        dim=1
    )


    # --------------------------------------------------------
    # Numerical stability
    # --------------------------------------------------------

    eps = 1e-8

    student_probs = torch.clamp(
        student_probs,
        min=eps,
        max=1.0
    )

    teacher_probs = torch.clamp(
        teacher_probs,
        min=eps,
        max=1.0
    )


    # --------------------------------------------------------
    # KL divergence
    #
    # KL(P_teacher || P_student)
    # --------------------------------------------------------

    kl = torch.sum(
        teacher_probs *
        (
            torch.log(teacher_probs)
            -
            torch.log(student_probs)
        ),
        dim=1
    )

    return kl.mean()


# ============================================================
# 17. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    student.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 18. TRAIN STUDENT
# ============================================================

print("=" * 70)
print("TRAINING DISTILLED STUDENT V4")
print("=" * 70)

print(
    "Loss = "
    "Alpha × T² × KL(teacher || student) "
    "+ "
    "(1 - Alpha) × BCE(hard labels)"
)

print()


for epoch in range(STUDENT_EPOCHS):

    student.train()

    total_loss_sum = 0.0
    soft_loss_sum = 0.0
    hard_loss_sum = 0.0

    correct = 0
    total = 0

    for (
        X_batch,
        y_batch,
        teacher_soft_batch
    ) in student_loader:

        X_batch = X_batch.to(DEVICE)

        y_batch = y_batch.to(DEVICE)

        teacher_soft_batch = (
            teacher_soft_batch.to(DEVICE)
        )


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        student_logits = student(
            X_batch
        )


        # ----------------------------------------------------
        # Soft distillation loss
        # ----------------------------------------------------

        soft_loss = binary_kl_divergence(
            student_logits,
            teacher_soft_batch,
            TEMPERATURE
        )


        # ----------------------------------------------------
        # Hard-label loss
        # ----------------------------------------------------

        hard_loss = hard_loss_function(
            student_logits,
            y_batch
        )


        # ----------------------------------------------------
        # Combined loss
        #
        # T² compensates for the gradient scaling introduced
        # by temperature.
        # ----------------------------------------------------

        total_loss = (
            ALPHA
            * (TEMPERATURE ** 2)
            * soft_loss
            +
            (1.0 - ALPHA)
            * hard_loss
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

        batch_size_actual = (
            y_batch.size(0)
        )

        total_loss_sum += (
            total_loss.item()
            * batch_size_actual
        )

        soft_loss_sum += (
            soft_loss.item()
            * batch_size_actual
        )

        hard_loss_sum += (
            hard_loss.item()
            * batch_size_actual
        )


        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        predictions = (
            torch.sigmoid(student_logits)
            >= 0.5
        ).float()

        correct += (
            predictions == y_batch
        ).sum().item()

        total += y_batch.numel()


    # --------------------------------------------------------
    # Epoch statistics
    # --------------------------------------------------------

    avg_total_loss = (
        total_loss_sum / total
    )

    avg_soft_loss = (
        soft_loss_sum / total
    )

    avg_hard_loss = (
        hard_loss_sum / total
    )

    accuracy = (
        100.0 * correct / total
    )


    print(
        f"Student Epoch "
        f"{epoch + 1}/{STUDENT_EPOCHS} | "
        f"Total Loss: {avg_total_loss:.6f} | "
        f"Soft KL Loss: {avg_soft_loss:.6f} | "
        f"Hard Loss: {avg_hard_loss:.6f} | "
        f"Accuracy: {accuracy:.4f}%"
    )


# ============================================================
# 19. SAVE STUDENT
# ============================================================

torch.save(
    student.state_dict(),
    STUDENT_MODEL_FILE
)

print()
print(
    f"Student model saved:\n"
    f"{STUDENT_MODEL_FILE}"
)


# ============================================================
# 20. SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    STUDENT_SCALER_FILE
)

print(
    f"Student scaler saved:\n"
    f"{STUDENT_SCALER_FILE}"
)


# ============================================================
# 21. FINAL VERIFICATION
# ============================================================

print()
print("=" * 70)
print("V4 TRAINING COMPLETE")
print("=" * 70)

print(
    f"Teacher:\n{V4_TEACHER_FILE}"
)

print(
    f"Student:\n{STUDENT_MODEL_FILE}"
)

print(
    f"Scaler:\n{STUDENT_SCALER_FILE}"
)

print()
print("Configuration:")
print(f"Temperature = {TEMPERATURE}")
print(f"Alpha       = {ALPHA}")
print(f"Epochs      = {STUDENT_EPOCHS}")
print(f"Batch size  = {BATCH_SIZE}")
print(f"Learning rate = {LEARNING_RATE}")
print()
print("The student is evaluated at normal temperature")
print("during inference.")
print("=" * 70)