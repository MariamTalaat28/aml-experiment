import pandas as pd
from pathlib import Path

# ============================================================
# CICIDS2017 - Label Inspection Across All Files
# ============================================================

DATA_DIR = (
    Path(__file__).parent.parent
    / "data"
    / "MachineLearningCVE"
)

CHUNK_SIZE = 50_000

print("=" * 70)
print("CICIDS2017 - LABEL INSPECTION")
print("=" * 70)

# Find all CSV files
csv_files = sorted(DATA_DIR.glob("*.csv"))

print(f"\nCSV files found: {len(csv_files)}")

for file in csv_files:
    print(f"  - {file.name}")

# ============================================================
# Inspect each file
# ============================================================

overall_labels = {}

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION BY FILE")
print("=" * 70)

for file in csv_files:

    print("\n" + "-" * 70)
    print(file.name)
    print("-" * 70)

    file_labels = {}
    total_rows = 0

    for chunk in pd.read_csv(
        file,
        chunksize=CHUNK_SIZE,
        usecols=lambda column: column.strip() == "Label"
    ):

        # Clean column name
        chunk.columns = chunk.columns.str.strip()

        total_rows += len(chunk)

        counts = chunk["Label"].value_counts()

        for label, count in counts.items():

            if label not in file_labels:
                file_labels[label] = 0

            file_labels[label] += int(count)

    print(f"Total rows: {total_rows:,}")

    for label, count in sorted(
        file_labels.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        percentage = (count / total_rows) * 100

        print(
            f"{label:<35} "
            f"{count:>10,} "
            f"({percentage:>6.2f}%)"
        )

        # Add to overall distribution
        if label not in overall_labels:
            overall_labels[label] = 0

        overall_labels[label] += count


# ============================================================
# Overall label distribution
# ============================================================

print("\n" + "=" * 70)
print("OVERALL LABEL DISTRIBUTION")
print("=" * 70)

total_rows = sum(overall_labels.values())

print(f"\nTotal rows across all files: {total_rows:,}")
print(f"Number of unique labels: {len(overall_labels)}")

print()

for label, count in sorted(
    overall_labels.items(),
    key=lambda x: x[1],
    reverse=True
):

    percentage = (count / total_rows) * 100

    print(
        f"{label:<35} "
        f"{count:>10,} "
        f"({percentage:>6.2f}%)"
    )

print("\n" + "=" * 70)
print("LABEL INSPECTION COMPLETED")
print("=" * 70)