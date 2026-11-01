# ============================================================
# EXPERIMENT: NORMALIZATION TECHNIQUES ON WINE QUALITY DATASET
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    MaxAbsScaler,
    Normalizer
)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. LOAD DATASET AND DISPLAY ITS CHARACTERISTICS
# ============================================================

file_path = r"C:\Users\jahanavi gorli\Downloads\winequality-red.csv"

output_folder = (
    r"C:\Users\jahanavi gorli\Downloads\wine_normalization_results"
)

os.makedirs(output_folder, exist_ok=True)

# Read the semicolon-separated CSV file
df = pd.read_csv(file_path, sep=";")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("=" * 60)
print("1. DATASET EXPLORATION")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFirst Five Records:")
print(df.head())


# Check the required target column
if "quality" not in df.columns:
    raise ValueError(
        "The quality column was not found. "
        "Check the dataset path and separator."
    )

if df.isnull().any().any():
    raise ValueError(
        "The dataset contains missing values. "
        "Handle them before continuing."
    )


# ============================================================
# 2. SEPARATE 11 INPUT FEATURES AND TARGET
# ============================================================

print("\n" + "=" * 60)
print("2. SEPARATE FEATURES AND TARGET")
print("=" * 60)

X = df.drop(columns=["quality"])
y = df["quality"]

print("\nInput Features:")
print(X.columns.tolist())

print("\nNumber of Input Features:", X.shape[1])

print("\nTarget Column: quality")

print("\nTarget Values:")
print(sorted(y.unique()))


# ============================================================
# 3. CALCULATE ORIGINAL DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("3. ORIGINAL FEATURE STATISTICS")
print("=" * 60)

feature_stats = pd.DataFrame({
    "Minimum": X.min(),
    "Maximum": X.max(),
    "Mean": X.mean(),
    "Standard Deviation": X.std(),
    "Median": X.median(),
    "IQR": X.quantile(0.75) - X.quantile(0.25)
})

print(feature_stats.round(4))

statistics_path = os.path.join(
    output_folder,
    "original_feature_statistics.csv"
)

feature_stats.to_csv(statistics_path)

print("\nStatistics saved to:", statistics_path)


# ============================================================
# 4. DEFINE FIVE SCALING TECHNIQUES
# ============================================================

scalers = {
    "MinMaxScaler": MinMaxScaler(),

    "StandardScaler": StandardScaler(),

    "RobustScaler": RobustScaler(),

    "MaxAbsScaler": MaxAbsScaler(),

    "Normalizer_L2": Normalizer(norm="l2")
}


# ============================================================
# 5. APPLY ALL FIVE SCALING TECHNIQUES
# ============================================================

print("\n" + "=" * 60)
print("5. APPLY SCALING TECHNIQUES")
print("=" * 60)

scaled_datasets = {}

for name, scaler in scalers.items():

    # Apply scaling to all 11 input features
    transformed_data = scaler.fit_transform(X)

    # Convert the transformed data into a DataFrame
    scaled_df = pd.DataFrame(
        transformed_data,
        columns=X.columns,
        index=X.index
    )

    # Keep the original target column unchanged
    scaled_df["quality"] = y

    scaled_datasets[name] = scaled_df

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print("\nFirst Five Transformed Rows:")
    print(scaled_df.head().round(4))

    # Save the complete transformed dataset as CSV
    output_file = os.path.join(
        output_folder,
        "wine_" + name.lower() + ".csv"
    )

    scaled_df.to_csv(output_file, index=False)

    print("\nSaved:", output_file)


# ============================================================
# 6. COMPARE NUMERICAL RANGES BEFORE AND AFTER SCALING
# ============================================================

print("\n" + "=" * 60)
print("6. NUMERICAL RANGE COMPARISON")
print("=" * 60)

range_results = []

# Original feature ranges
for column in X.columns:

    range_results.append({
        "Technique": "Original",
        "Feature": column,
        "Minimum": X[column].min(),
        "Maximum": X[column].max(),
        "Mean": X[column].mean(),
        "Standard Deviation": X[column].std()
    })

# Scaled feature ranges
for name, scaled_df in scaled_datasets.items():

    for column in X.columns:

        range_results.append({
            "Technique": name,
            "Feature": column,
            "Minimum": scaled_df[column].min(),
            "Maximum": scaled_df[column].max(),
            "Mean": scaled_df[column].mean(),
            "Standard Deviation": scaled_df[column].std()
        })

range_df = pd.DataFrame(range_results)

print(range_df.round(4).to_string(index=False))

range_file = os.path.join(
    output_folder,
    "numerical_range_comparison.csv"
)

range_df.to_csv(range_file, index=False)

print("\nRange comparison saved:", range_file)


# ============================================================
# 7. COMPARE DISTRIBUTIONS USING BOXPLOTS
# ============================================================

print("\n" + "=" * 60)
print("7. BOXPLOT COMPARISON")
print("=" * 60)

fig, axes = plt.subplots(3, 2, figsize=(18, 15))

axes = axes.flatten()

# Original data
sns.boxplot(
    data=X,
    orient="h",
    ax=axes[0]
)

axes[0].set_title("Original Features")

# Five scaling techniques
for index, (name, scaled_df) in enumerate(
    scaled_datasets.items(),
    start=1
):

    sns.boxplot(
        data=scaled_df[X.columns],
        orient="h",
        ax=axes[index]
    )

    axes[index].set_title(name)

plt.tight_layout()

boxplot_file = os.path.join(
    output_folder,
    "scaling_boxplots.png"
)

plt.savefig(boxplot_file, dpi=300)

plt.show()

print("Boxplots saved:", boxplot_file)


# ============================================================
# 8. COMPARE FEATURE DISTRIBUTIONS USING HISTOGRAMS
# ============================================================

print("\n" + "=" * 60)
print("8. HISTOGRAM COMPARISON")
print("=" * 60)

# Select three features for comparison
selected_features = [
    "total sulfur dioxide",
    "residual sugar",
    "alcohol"
]

fig, axes = plt.subplots(
    len(selected_features),
    6,
    figsize=(24, 12)
)

for row, column in enumerate(selected_features):

    # Original feature distribution
    sns.histplot(
        X[column],
        kde=True,
        ax=axes[row, 0]
    )

    axes[row, 0].set_title(
        "Original: " + column,
        fontsize=9
    )

    # Distributions after each scaling technique
    for col_index, (name, scaled_df) in enumerate(
        scaled_datasets.items(),
        start=1
    ):

        sns.histplot(
            scaled_df[column],
            kde=True,
            ax=axes[row, col_index]
        )

        axes[row, col_index].set_title(
            name,
            fontsize=9
        )

plt.tight_layout()

histogram_file = os.path.join(
    output_folder,
    "scaling_histograms.png"
)

plt.savefig(histogram_file, dpi=300)

plt.show()

print("Histograms saved:", histogram_file)


# ============================================================
# 9. CORRECT SCALING BEFORE MODEL TRAINING
# ============================================================

print("\n" + "=" * 60)
print("9. CORRECT MODEL TRAINING PRACTICE")
print("=" * 60)

# Split the original data before fitting any scaler
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)

# Demonstrate correct practice using a pipeline.
# The scaler is fitted only on training data.
# The model predicts the categorical quality score.

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000))
])

pipeline.fit(X_train, y_train)

test_accuracy = pipeline.score(X_test, y_test)

print("\nLogistic Regression Test Accuracy:")
print(round(test_accuracy, 4))

print(
    "\nThe pipeline fits the scaler on training data only "
    "and applies the same transformation to test data."
)


# ============================================================
# 10. FINAL OUTPUT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETED")
print("=" * 60)

print("\nScaling techniques applied:")

for name in scalers:
    print("-", name)

print("\nOutput folder:")
print(output_folder)

print("\nFiles saved:")
print("- Original feature statistics")
print("- Five transformed CSV datasets")
print("- Numerical range comparison")
print("- Scaling boxplots")
print("- Scaling histograms")