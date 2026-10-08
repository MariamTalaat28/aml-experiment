import pandas as pd
from pathlib import Path

# ============================================================
# CICIDS2017 - Data Inspection
# ============================================================

# Path to the Monday dataset
DATA_FILE = (
    Path(__file__).parent.parent
    / "data"
    / "MachineLearningCVE"
    / "Monday-WorkingHours.pcap_ISCX.csv"
)

print("=" * 70)
print("CICIDS2017 DATA INSPECTION")
print("=" * 70)

print(f"\nDataset path:")
print(DATA_FILE)

# ============================================================
# 1. Check that the file exists
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_FILE}"
    )

print("\n[OK] Dataset file found.")

# ============================================================
# 2. Load a small sample
# ============================================================

print("\nLoading first 5 rows...")

df = pd.read_csv(DATA_FILE, nrows=5)

# ============================================================
# 3. Basic information
# ============================================================

print("\n" + "=" * 70)
print("BASIC INFORMATION")
print("=" * 70)

print(f"Rows loaded: {len(df)}")
print(f"Number of columns: {len(df.columns)}")

# ============================================================
# 4. Column names
# ============================================================

print("\n" + "=" * 70)
print("COLUMN NAMES")
print("=" * 70)

for i, column in enumerate(df.columns):
    print(f"{i}: {repr(column)}")

# ============================================================
# 5. Data types
# ============================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)

# ============================================================
# 6. Sample data
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE DATA")
print("=" * 70)

print(df.head())

# ============================================================
# 7. Label distribution in sample
# ============================================================

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION IN SAMPLE")
print("=" * 70)

# Find the label column safely
label_column = None

for column in df.columns:
    if column.strip().lower() == "label":
        label_column = column
        break

if label_column is None:
    raise ValueError("Label column was not found.")

print(f"Label column: {repr(label_column)}")

print("\nLabels:")
print(df[label_column].value_counts())

# ============================================================
# 8. Missing values in sample
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES IN SAMPLE")
print("=" * 70)

missing_values = df.isnull().sum()

missing_values = missing_values[missing_values > 0]

if len(missing_values) == 0:
    print("[OK] No missing values found in the sample.")
else:
    print(missing_values)

# ============================================================
# 9. Duplicate rows in sample
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATES IN SAMPLE")
print("=" * 70)

duplicates = df.duplicated().sum()

print(f"Duplicate rows: {duplicates}")

# ============================================================
# 10. Numeric columns
# ============================================================

print("\n" + "=" * 70)
print("NUMERIC COLUMNS")
print("=" * 70)

numeric_columns = df.select_dtypes(include="number").columns

print(f"Number of numeric columns: {len(numeric_columns)}")

# ============================================================
# 11. Categorical columns
# ============================================================

print("\n" + "=" * 70)
print("NON-NUMERIC COLUMNS")
print("=" * 70)

non_numeric_columns = df.select_dtypes(exclude="number").columns

print(f"Number of non-numeric columns: {len(non_numeric_columns)}")

for column in non_numeric_columns:
    print(f"- {repr(column)}")

# ============================================================
# 12. Inspection summary
# ============================================================

print("\n" + "=" * 70)
print("INSPECTION SUMMARY")
print("=" * 70)

print(f"Dataset file : {DATA_FILE.name}")
print(f"Columns      : {len(df.columns)}")
print(f"Label column : {repr(label_column)}")
print(f"Numeric cols : {len(numeric_columns)}")
print(f"Other cols   : {len(non_numeric_columns)}")
print(f"Duplicates   : {duplicates}")

print("\n" + "=" * 70)
print("INSPECTION COMPLETED SUCCESSFULLY")
print("=" * 70)