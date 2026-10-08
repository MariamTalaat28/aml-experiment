import pandas as pd
from pathlib import Path

# ============================================================
# CICIDS2017 - Verify Processed Dataset
# ============================================================

DATA_FILE = (
    Path(__file__).parent.parent
    / "data"
    / "processed"
    / "cicids2017_binary.csv"
)

print("=" * 70)
print("CICIDS2017 - PROCESSED DATA VERIFICATION")
print("=" * 70)

print(f"\nDataset path:")
print(DATA_FILE)

# ============================================================
# 1. Check that the file exists
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"\nProcessed dataset not found:\n{DATA_FILE}"
    )

print("\n[OK] Processed dataset found.")

# ============================================================
# 2. Read the processed dataset in chunks
# ============================================================

CHUNK_SIZE = 50_000

total_rows = 0
total_missing = 0
total_infinite = 0

label_counts = {
    0: 0,
    1: 0
}

print("\nReading processed dataset in chunks...")
print(f"Chunk size: {CHUNK_SIZE:,} rows")

for chunk_number, chunk in enumerate(
    pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE),
    start=1
):

    total_rows += len(chunk)

    # --------------------------------------------------------
    # Count labels
    # --------------------------------------------------------

    counts = chunk["Label"].value_counts()

    label_counts[0] += counts.get(0, 0)
    label_counts[1] += counts.get(1, 0)

    # --------------------------------------------------------
    # Count missing values
    # --------------------------------------------------------

    total_missing += chunk.isnull().sum().sum()

    # --------------------------------------------------------
    # Count infinite values
    # --------------------------------------------------------

    numeric_data = chunk.select_dtypes(include="number")

    total_infinite += (
        numeric_data
        .isin([float("inf"), float("-inf")])
        .sum()
        .sum()
    )

    print(
        f"Processed chunk {chunk_number} | "
        f"Rows checked: {total_rows:,}"
    )

# ============================================================
# 3. Final results
# ============================================================

print("\n" + "=" * 70)
print("VERIFICATION RESULTS")
print("=" * 70)

print(f"\nTotal rows: {total_rows:,}")

print(f"Total columns: {len(chunk.columns)}")

# ============================================================
# 4. Label distribution
# ============================================================

print("\n" + "-" * 70)
print("LABEL DISTRIBUTION")
print("-" * 70)

benign_count = label_counts[0]
attack_count = label_counts[1]

benign_percentage = (
    benign_count / total_rows * 100
)

attack_percentage = (
    attack_count / total_rows * 100
)

print(
    f"BENIGN (0): {benign_count:,} "
    f"({benign_percentage:.2f}%)"
)

print(
    f"ATTACK (1): {attack_count:,} "
    f"({attack_percentage:.2f}%)"
)

# ============================================================
# 5. Missing values
# ============================================================

print("\n" + "-" * 70)
print("MISSING VALUES")
print("-" * 70)

print(f"Total missing values: {total_missing:,}")

# ============================================================
# 6. Infinite values
# ============================================================

print("\n" + "-" * 70)
print("INFINITE VALUES")
print("-" * 70)

print(f"Total infinite values: {total_infinite:,}")

# ============================================================
# 7. Final status
# ============================================================

print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

if total_missing == 0:
    print("[OK] No missing values found.")
else:
    print("[WARNING] Missing values still exist.")

if total_infinite == 0:
    print("[OK] No infinite values found.")
else:
    print("[WARNING] Infinite values still exist.")

if benign_count > 0 and attack_count > 0:
    print("[OK] Dataset contains both BENIGN and ATTACK traffic.")
else:
    print("[ERROR] Dataset does not contain both classes.")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETED")
print("=" * 70)