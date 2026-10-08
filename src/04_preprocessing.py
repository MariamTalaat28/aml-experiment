import pandas as pd
from pathlib import Path

# ============================================================
# CICIDS2017 - Data Preprocessing
# ============================================================

DATA_DIR = (
    Path(__file__).parent.parent
    / "data"
    / "MachineLearningCVE"
)

OUTPUT_DIR = (
    Path(__file__).parent.parent
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "cicids2017_binary.csv"

CHUNK_SIZE = 50_000

print("=" * 70)
print("CICIDS2017 - DATA PREPROCESSING")
print("=" * 70)

print(f"\nInput directory:")
print(DATA_DIR)

print(f"\nOutput file:")
print(OUTPUT_FILE)

# ============================================================
# Find CSV files
# ============================================================

csv_files = sorted(DATA_DIR.glob("*.csv"))

print(f"\nCSV files found: {len(csv_files)}")

for file in csv_files:
    print(f"  - {file.name}")

# ============================================================
# Variables
# ============================================================

total_rows_read = 0
total_rows_saved = 0

first_write = True

# ============================================================
# Process each file
# ============================================================

for file in csv_files:

    print("\n" + "-" * 70)
    print(f"Processing: {file.name}")
    print("-" * 70)

    file_rows = 0
    file_saved = 0

    for chunk in pd.read_csv(
        file,
        chunksize=CHUNK_SIZE
    ):

        # ----------------------------------------------------
        # Clean column names
        # ----------------------------------------------------

        chunk.columns = chunk.columns.str.strip()

        # ----------------------------------------------------
        # Remove leading/trailing spaces from labels
        # ----------------------------------------------------

        chunk["Label"] = chunk["Label"].astype(str).str.strip()

        # ----------------------------------------------------
        # Convert labels to binary
        #
        # BENIGN = 0
        # Everything else = 1
        # ----------------------------------------------------

        chunk["Label"] = (
            chunk["Label"]
            .ne("BENIGN")
            .astype(int)
        )

        # ----------------------------------------------------
        # Replace infinite values with NaN
        # ----------------------------------------------------

        chunk = chunk.replace(
            [float("inf"), float("-inf")],
            pd.NA
        )

        # ----------------------------------------------------
        # Remove rows containing missing values
        # ----------------------------------------------------

        before_missing = len(chunk)

        chunk = chunk.dropna()

        removed_missing = (
            before_missing - len(chunk)
        )

        # ----------------------------------------------------
        # Remove duplicate rows
        # ----------------------------------------------------

        before_duplicates = len(chunk)

        chunk = chunk.drop_duplicates()

        removed_duplicates = (
            before_duplicates - len(chunk)
        )

        # ----------------------------------------------------
        # Save cleaned chunk
        # ----------------------------------------------------

        if len(chunk) > 0:

            chunk.to_csv(
                OUTPUT_FILE,
                mode="w" if first_write else "a",
                header=first_write,
                index=False
            )

            first_write = False

            file_saved += len(chunk)
            total_rows_saved += len(chunk)

        file_rows += CHUNK_SIZE
        total_rows_read += len(chunk)

    print(f"Rows saved from this file: {file_saved:,}")

# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED")
print("=" * 70)

print(f"\nOutput file:")
print(OUTPUT_FILE)

print(f"\nTotal rows saved: {total_rows_saved:,}")

print("\nLabels:")
print("  0 = BENIGN")
print("  1 = ATTACK")

print("\n" + "=" * 70)
print("NEXT STEP: VERIFY THE PROCESSED DATASET")
print("=" * 70)