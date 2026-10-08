import pandas as pd
from pathlib import Path

# ============================================================
# CICIDS2017 - Full Dataset Inspection
# ============================================================

DATA_FILE = (
    Path(__file__).parent.parent
    / "data"
    / "MachineLearningCVE"
    / "Monday-WorkingHours.pcap_ISCX.csv"
)

CHUNK_SIZE = 50_000

print("=" * 70)
print("CICIDS2017 - FULL DATASET INSPECTION")
print("=" * 70)

print(f"\nDataset:")
print(DATA_FILE)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_FILE}"
    )

print("\n[OK] Dataset file found.")

# ============================================================
# Variables for the full inspection
# ============================================================

total_rows = 0
total_duplicates = 0
total_missing = None
total_infinite = None

label_counts = {}

numeric_columns = None
non_numeric_columns = None

# ============================================================
# Read dataset in chunks
# ============================================================

print("\nReading dataset in chunks...")
print(f"Chunk size: {CHUNK_SIZE:,} rows")

chunk_number = 0

for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE):

    chunk_number += 1

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    chunk.columns = chunk.columns.str.strip()

    # --------------------------------------------------------
    # Count rows
    # --------------------------------------------------------

    total_rows += len(chunk)

    # --------------------------------------------------------
    # Identify column types
    # --------------------------------------------------------

    if numeric_columns is None:
        numeric_columns = chunk.select_dtypes(
            include="number"
        ).columns.tolist()

        non_numeric_columns = chunk.select_dtypes(
            exclude="number"
        ).columns.tolist()

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing = chunk.isnull().sum()

    if total_missing is None:
        total_missing = missing
    else:
        total_missing = total_missing.add(
            missing,
            fill_value=0
        )

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    numeric_chunk = chunk.select_dtypes(include="number")

    infinite = pd.Series(
        False,
        index=numeric_chunk.columns
    )

    if not numeric_chunk.empty:
        infinite = numeric_chunk.isin(
            [float("inf"), float("-inf")]
        ).sum()

    if total_infinite is None:
        total_infinite = infinite
    else:
        total_infinite = total_infinite.add(
            infinite,
            fill_value=0
        )

    # --------------------------------------------------------
    # Duplicate rows inside each chunk
    # --------------------------------------------------------

    duplicates = chunk.duplicated().sum()

    total_duplicates += duplicates

    # --------------------------------------------------------
    # Label distribution
    # --------------------------------------------------------

    if "Label" in chunk.columns:

        counts = chunk["Label"].value_counts()

        for label, count in counts.items():

            if label not in label_counts:
                label_counts[label] = 0

            label_counts[label] += int(count)

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    print(
        f"Processed chunk {chunk_number} | "
        f"Rows processed: {total_rows:,}"
    )

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FULL DATASET RESULTS")
print("=" * 70)

print(f"\nTotal rows: {total_rows:,}")
print(f"Total columns: {len(numeric_columns) + len(non_numeric_columns)}")

print("\n" + "-" * 70)
print("COLUMN TYPES")
print("-" * 70)

print(f"Numeric columns: {len(numeric_columns)}")
print(f"Non-numeric columns: {len(non_numeric_columns)}")

print("\nNon-numeric columns:")

for column in non_numeric_columns:
    print(f"  - {column}")

# ============================================================
# Label distribution
# ============================================================

print("\n" + "-" * 70)
print("LABEL DISTRIBUTION")
print("-" * 70)

if label_counts:

    total_label_rows = sum(label_counts.values())

    for label, count in sorted(
        label_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        percentage = (
            count / total_label_rows
        ) * 100

        print(
            f"{label:<30} "
            f"{count:>10,} "
            f"({percentage:>6.2f}%)"
        )

else:
    print("[ERROR] Label column was not found.")

# ============================================================
# Missing values
# ============================================================

print("\n" + "-" * 70)
print("MISSING VALUES")
print("-" * 70)

missing_total = int(total_missing.sum())

print(f"Total missing values: {missing_total:,}")

if missing_total == 0:

    print("[OK] No missing values found.")

else:

    print("\nColumns containing missing values:")

    for column, count in total_missing.items():

        if count > 0:

            print(
                f"  {column}: {int(count):,}"
            )

# ============================================================
# Infinite values
# ============================================================

print("\n" + "-" * 70)
print("INFINITE VALUES")
print("-" * 70)

infinite_total = int(total_infinite.sum())

print(
    f"Total infinite values: "
    f"{infinite_total:,}"
)

if infinite_total == 0:

    print("[OK] No infinite values found.")

else:

    print("\nColumns containing infinite values:")

    for column, count in total_infinite.items():

        if count > 0:

            print(
                f"  {column}: {int(count):,}"
            )

# ============================================================
# Duplicates
# ============================================================

print("\n" + "-" * 70)
print("DUPLICATE ROWS")
print("-" * 70)

print(
    "Duplicate rows detected within chunks: "
    f"{total_duplicates:,}"
)

# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 70)
print("FULL INSPECTION SUMMARY")
print("=" * 70)

print(f"Rows              : {total_rows:,}")
print(f"Columns           : {len(numeric_columns) + len(non_numeric_columns)}")
print(f"Numeric features  : {len(numeric_columns)}")
print(f"Non-numeric cols  : {len(non_numeric_columns)}")
print(f"Missing values    : {missing_total:,}")
print(f"Infinite values   : {infinite_total:,}")
print(f"Duplicate rows    : {total_duplicates:,}")
print(f"Number of labels  : {len(label_counts)}")

print("\n" + "=" * 70)
print("FULL DATASET INSPECTION COMPLETED")
print("=" * 70)