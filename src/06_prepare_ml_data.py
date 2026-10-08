import pandas as pd
from pathlib import Path
import hashlib

# ============================================================
# CICIDS2017 - Prepare ML Data
# ============================================================
#
# We will create:
#
# Experiment A:
#   Original imbalanced training data
#
# Experiment B:
#   Balanced training data
#
# The TEST set will remain unchanged for both experiments.
#
# ============================================================

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "cicids2017_binary.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "ml"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

CHUNK_SIZE = 50_000

# 80% training / 20% testing
TEST_PERCENTAGE = 20

# ------------------------------------------------------------
# Output files
# ------------------------------------------------------------

IMBALANCED_TRAIN = OUTPUT_DIR / "imbalanced_train.csv"
BALANCED_TRAIN = OUTPUT_DIR / "balanced_train.csv"
TEST_FILE = OUTPUT_DIR / "test.csv"

print("=" * 70)
print("CICIDS2017 - PREPARE ML DATA")
print("=" * 70)

print(f"\nInput file:")
print(INPUT_FILE)

print(f"\nOutput directory:")
print(OUTPUT_DIR)

# ------------------------------------------------------------
# Check input file
# ------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{INPUT_FILE}"
    )

print("\n[OK] Input dataset found.")

# ============================================================
# Remove old output files
# ============================================================

for output_file in [
    IMBALANCED_TRAIN,
    BALANCED_TRAIN,
    TEST_FILE
]:
    if output_file.exists():
        output_file.unlink()

print("\n[OK] Previous output files cleared.")

# ============================================================
# STEP 1
# Create the original train/test split
# ============================================================

print("\n" + "=" * 70)
print("STEP 1 - CREATING TRAIN / TEST SPLIT")
print("=" * 70)

print("\nUsing:")
print("  80% training")
print("  20% testing")

train_rows = 0
test_rows = 0

train_benign = 0
train_attack = 0

test_benign = 0
test_attack = 0

first_train_write = True
first_test_write = True

for chunk_number, chunk in enumerate(
    pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE),
    start=1
):

    # --------------------------------------------------------
    # Create a deterministic hash for each row
    #
    # This lets us make the same train/test decision
    # every time we run the script.
    # --------------------------------------------------------

    row_hash = pd.util.hash_pandas_object(
        chunk,
        index=False
    )

    # Use the hash to assign approximately 20% to test
    test_mask = (
        row_hash % 100 < TEST_PERCENTAGE
    )

    train_chunk = chunk.loc[~test_mask].copy()
    test_chunk = chunk.loc[test_mask].copy()

    # --------------------------------------------------------
    # Count labels
    # --------------------------------------------------------

    train_counts = train_chunk["Label"].value_counts()

    train_benign += train_counts.get(0, 0)
    train_attack += train_counts.get(1, 0)

    test_counts = test_chunk["Label"].value_counts()

    test_benign += test_counts.get(0, 0)
    test_attack += test_counts.get(1, 0)

    # --------------------------------------------------------
    # Save training chunk
    # --------------------------------------------------------

    if len(train_chunk) > 0:

        train_chunk.to_csv(
            IMBALANCED_TRAIN,
            mode="w" if first_train_write else "a",
            header=first_train_write,
            index=False
        )

        first_train_write = False

    # --------------------------------------------------------
    # Save test chunk
    # --------------------------------------------------------

    if len(test_chunk) > 0:

        test_chunk.to_csv(
            TEST_FILE,
            mode="w" if first_test_write else "a",
            header=first_test_write,
            index=False
        )

        first_test_write = False

    train_rows += len(train_chunk)
    test_rows += len(test_chunk)

    print(
        f"Processed chunk {chunk_number} | "
        f"Train: {train_rows:,} | "
        f"Test: {test_rows:,}"
    )

# ============================================================
# STEP 1 RESULTS
# ============================================================

print("\n" + "-" * 70)
print("TRAIN / TEST SPLIT RESULTS")
print("-" * 70)

print(f"\nTraining rows: {train_rows:,}")
print(f"Testing rows : {test_rows:,}")

print("\nTraining distribution:")
print(
    f"  BENIGN (0): {train_benign:,}"
)
print(
    f"  ATTACK (1): {train_attack:,}"
)

print("\nTesting distribution:")
print(
    f"  BENIGN (0): {test_benign:,}"
)
print(
    f"  ATTACK (1): {test_attack:,}"
)

# ============================================================
# STEP 2
# Create balanced training data
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 - CREATING BALANCED TRAINING DATA")
print("=" * 70)

print("\nWe will keep ALL attack training samples.")

print(
    f"Attack training samples: {train_attack:,}"
)

print(
    "We will randomly select the same number "
    "of BENIGN samples."
)

# ------------------------------------------------------------
# We use a fixed random seed so the experiment is reproducible.
# ------------------------------------------------------------

RANDOM_STATE = 42

# ------------------------------------------------------------
# First pass:
# Collect attack rows and sample benign rows.
#
# Attack data is approximately 364,000 rows, which is
# manageable for our machine.
# ------------------------------------------------------------

attack_chunks = []

benign_chunks = []

for chunk_number, chunk in enumerate(
    pd.read_csv(IMBALANCED_TRAIN, chunksize=CHUNK_SIZE),
    start=1
):

    attack_part = chunk[
        chunk["Label"] == 1
    ].copy()

    benign_part = chunk[
        chunk["Label"] == 0
    ].copy()

    if len(attack_part) > 0:
        attack_chunks.append(attack_part)

    if len(benign_part) > 0:
        benign_chunks.append(benign_part)

    print(
        f"Reading training chunk {chunk_number}"
    )

# ------------------------------------------------------------
# Combine attack and benign training data
# ------------------------------------------------------------

print("\nCombining training data...")

attack_data = pd.concat(
    attack_chunks,
    ignore_index=True
)

benign_data = pd.concat(
    benign_chunks,
    ignore_index=True
)

# Free the lists
del attack_chunks
del benign_chunks

print(
    f"Attack samples available: "
    f"{len(attack_data):,}"
)

print(
    f"BENIGN samples available: "
    f"{len(benign_data):,}"
)

# ============================================================
# Select equal number of BENIGN samples
# ============================================================

balanced_benign = benign_data.sample(
    n=len(attack_data),
    random_state=RANDOM_STATE
)

# ------------------------------------------------------------
# Combine both classes
# ------------------------------------------------------------

balanced_train = pd.concat(
    [
        attack_data,
        balanced_benign
    ],
    ignore_index=True
)

# ------------------------------------------------------------
# Shuffle balanced dataset
# ------------------------------------------------------------

balanced_train = balanced_train.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

# ------------------------------------------------------------
# Save balanced training data
# ------------------------------------------------------------

balanced_train.to_csv(
    BALANCED_TRAIN,
    index=False
)

# ============================================================
# STEP 2 RESULTS
# ============================================================

balanced_counts = (
    balanced_train["Label"]
    .value_counts()
    .sort_index()
)

print("\n" + "-" * 70)
print("BALANCED TRAINING RESULTS")
print("-" * 70)

print(
    f"\nBalanced training rows: "
    f"{len(balanced_train):,}"
)

print(
    f"BENIGN (0): "
    f"{balanced_counts.get(0, 0):,}"
)

print(
    f"ATTACK (1): "
    f"{balanced_counts.get(1, 0):,}"
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATA PREPARATION COMPLETED")
print("=" * 70)

print("\nFiles created:")

print(
    f"\n1. Imbalanced training:"
    f"\n   {IMBALANCED_TRAIN}"
)

print(
    f"\n2. Balanced training:"
    f"\n   {BALANCED_TRAIN}"
)

print(
    f"\n3. Test set:"
    f"\n   {TEST_FILE}"
)

print("\n" + "-" * 70)

print("Experiment A:")
print("  Original imbalanced training data")

print("\nExperiment B:")
print("  Balanced training data")

print("\nBoth experiments will use:")
print("  The SAME test set")

print("\n" + "=" * 70)
print("NEXT STEP: TRAIN THE DNN")
print("=" * 70)