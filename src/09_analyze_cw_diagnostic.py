import pandas as pd
import numpy as np

FILE = "/home/ubuntu/aml-experiment/results/cw_validated_1000_details.csv"

df = pd.read_csv(FILE)

print("=" * 80)
print("C&W DIAGNOSTIC ANALYSIS")
print("=" * 80)

# ------------------------------------------------------------
# 1. L2 / L-inf distribution
# ------------------------------------------------------------

successful = df[df["attack_success"] == 1].copy()

print("\nTotal samples:", len(df))
print("Successful attacks:", len(successful))

print("\nL2 statistics:")
print(successful["l2"].describe())

print("\nL-inf statistics:")
print(successful["linf"].describe())

# ------------------------------------------------------------
# 2. How many attacks have large perturbations?
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("L-INF THRESHOLDS")
print("=" * 80)

for threshold in [0.5, 1, 2, 5, 10, 20, 50]:

    count = (
        successful["linf"] > threshold
    ).sum()

    percentage = (
        count / len(successful) * 100
    )

    print(
        f"L-inf > {threshold:5}: "
        f"{count:4} samples "
        f"({percentage:.2f}%)"
    )

# ------------------------------------------------------------
# 3. Top extreme samples
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TOP 20 L-INF CASES")
print("=" * 80)

columns = [
    "sample_index",
    "true_label",
    "original_prediction",
    "adversarial_prediction",
    "attack_success",
    "l2",
    "linf",
    "max_perturbation_feature_name",
    "clean_probability",
    "adv_probability"
]

print(
    successful
    .sort_values("linf", ascending=False)
    [columns]
    .head(20)
    .to_string(index=False)
)

# ------------------------------------------------------------
# 4. RST Flag Count
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("RST FLAG COUNT ANALYSIS")
print("=" * 80)

rst_original = "original_RST Flag Count"
rst_adv = "adversarial_RST Flag Count"
rst_delta = "delta_RST Flag Count"

print("\nOriginal standardized values:")
print(
    df[rst_original].describe()
)

print("\nAdversarial standardized values:")
print(
    df[rst_adv].describe()
)

print("\nDelta values:")
print(
    df[rst_delta].describe()
)

print("\nLargest RST perturbations:")

rst_cols = [
    "sample_index",
    "true_label",
    "attack_success",
    rst_original,
    rst_adv,
    rst_delta,
    "l2",
    "linf"
]

print(
    df
    .sort_values(
        rst_delta,
        key=lambda x: x.abs(),
        ascending=False
    )[rst_cols]
    .head(10)
    .to_string(index=False)
)

# ------------------------------------------------------------
# 5. ECE Flag Count
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("ECE FLAG COUNT ANALYSIS")
print("=" * 80)

ece_original = "original_ECE Flag Count"
ece_adv = "adversarial_ECE Flag Count"
ece_delta = "delta_ECE Flag Count"

print("\nOriginal standardized values:")
print(
    df[ece_original].describe()
)

print("\nAdversarial standardized values:")
print(
    df[ece_adv].describe()
)

print("\nDelta values:")
print(
    df[ece_delta].describe()
)

print("\nLargest ECE perturbations:")

ece_cols = [
    "sample_index",
    "true_label",
    "attack_success",
    ece_original,
    ece_adv,
    ece_delta,
    "l2",
    "linf"
]

print(
    df
    .sort_values(
        ece_delta,
        key=lambda x: x.abs(),
        ascending=False
    )[ece_cols]
    .head(10)
    .to_string(index=False)
)

# ------------------------------------------------------------
# 6. How many extreme attacks?
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("EXTREME ATTACK SUMMARY")
print("=" * 80)

for threshold in [1, 2, 5, 10]:

    subset = successful[
        successful["linf"] > threshold
    ]

    print(
        f"\nL-inf > {threshold}"
    )

    print(
        "Count:",
        len(subset)
    )

    if len(subset) > 0:

        print(
            "Successful attack percentage:",
            f"{len(subset) / len(successful) * 100:.2f}%"
        )

        print(
            "Mean L2:",
            f"{subset['l2'].mean():.4f}"
        )

        print(
            "Mean L-inf:",
            f"{subset['linf'].mean():.4f}"
        )

print("\nFinished.")