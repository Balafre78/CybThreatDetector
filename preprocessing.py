import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df_train = pd.read_csv("./cyberdataset_train.csv")
pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def check_column_types(df):
    for column in df.columns:
        expected_type = df[column].dtype

        print(f"\nColumn verification: '{column}' (expected type: {expected_type}) ---")

        invalid_rows = []

        for idx, value in df[column].items():
            if pd.api.types.is_numeric_dtype(expected_type):
                if not pd.api.types.is_number(value) and not pd.isna(value):
                    invalid_rows.append((idx, value))
            elif pd.api.types.is_string_dtype(expected_type):
                if not isinstance(value, str) and not pd.isna(value):
                    invalid_rows.append((idx, value))
        if invalid_rows:
            print(f"Invalid value found ({len(invalid_rows)}):")
            for row in invalid_rows[:5]:
                print(f"Line {row[0]} : {row[1]}")
        else:
            print("All values have the good type")

# Compute the correlation matrix
df_features = df_train.iloc[:, :-1]
corr_matrix = df_features.corr()
# Display the numeric correlation values
#print("=== Correlation Matrix ===")
#print(corr_matrix)


def heatmap_correlation(df):
    """
    Plots a correlation heatmap for all numeric columns in a DataFrame.
    - Automatically filters non-numeric columns
    - Drops rows with NaN values before computing correlations
    """
    numeric_df = df.select_dtypes(include=np.number)
    # Compute correlation matrix
    corr_matrix = numeric_df.corr()
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    # Adjust figure size dynamically based on number of columns
    n_cols = len(corr_matrix.columns)
    plt.figure(figsize=(min(0.6 * n_cols, 25), min(0.6 * n_cols, 25)))

    # Draw the heatmap
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap='coolwarm',  # red = positive, blue = negative
        center=0,
        annot=False,  # set to True to show numbers
        square=True,
        linewidths=0.3,
        cbar_kws={"shrink": .8}
    )
    # Add title and layout adjustments
    plt.title("Correlation Heatmap", fontsize=30, pad=12)
    plt.tight_layout()
    plt.show()

def clear_df(df, threshold=2829385):
    # Drop any rows containing NaN values
    cleaned_df = df.dropna()
    cleaned_df = cleaned_df.reset_index(drop=True)
    # Count number of zeros in each column
    zero_counts = (cleaned_df.iloc[:,:-1] == 0).sum()
    # Identify columns to drop
    cols_to_drop = zero_counts[zero_counts == threshold].index
    print(f"Columns dropped ({len(cols_to_drop)}):")
    print(list(cols_to_drop))
    # Drop those columns
    df_cleaned = cleaned_df.drop(columns=cols_to_drop)
    return df_cleaned

def count_labels(df):
    label_col = df.columns[-1]  # last column
    counts = df[label_col].value_counts()
    print("\n### Label counts ###")
    print(counts)

def get_highly_correlated_features(df, threshold=0.9):
    # df supposed to be cleared already
    df = df.iloc[:,:-1]
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    # Find features that have correlation greater than threshold
    to_drop = [
        column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)
    ]
    print(f"Columns with correlation > {threshold}:")
    print(to_drop)
    return to_drop


def drop_highly_correlated_features(df):
    to_keep = [
        'Subflow Fwd Packets',
        ' Total Backward Packets',
        ' Packet Length Mean',
        ' Fwd IAT Mean',
        ' Fwd Packets/s',
        ' SYN Flag Count',
        ' Fwd Header Length.1',
        ' Idle Mean'
    ]
    # Build the drop list (highly correlated columns not in to_keep)
    to_drop = [col for col in high_corr_cols if col not in to_keep]
    print(f"Columns dropped ({len(to_drop)}): {to_drop}")
    # Drop them
    df_cleaned = df.drop(columns=to_drop, errors='ignore')
    return df_cleaned

df_cleaned = clear_df(df_train)
print(df_cleaned.info())
print(df_cleaned.head())
count_labels(df_cleaned)
high_corr_cols = get_highly_correlated_features(df_cleaned)
print(high_corr_cols)
df_cleaned = drop_highly_correlated_features(df_cleaned)
heatmap_correlation(df_cleaned)














# =======================
#      PLOTTING UTILS
# =======================

def plot_feature_histograms(df, features, bins=50):
    """
    Plot histograms for selected numeric features.

    Why it's interesting:
      - Shows distribution shape (normal, skewed, multi-modal).
      - Makes it easy to spot outliers (extremely large values).
      - Helps decide whether scaling / log-transform might be needed.
    """
    label_col = df.columns[-1]

    n_features = len(features)
    n_cols = 2
    n_rows = int(np.ceil(n_features / n_cols))

    plt.figure(figsize=(8 * n_cols, 4 * n_rows))
    for i, feat in enumerate(features, 1):
        if feat not in df.columns:
            print(f"[plot_feature_histograms] Skipping missing feature '{feat}'.")
            continue
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(
            data=df,
            x=feat,
            hue=label_col,
            bins=bins,
            kde=True,
            element="step",
            stat="density"
        )
        plt.title(f"Histogram of {feat}")
        plt.xlabel(feat)
        plt.ylabel("Density")
        plt.yscale("linear")
    plt.tight_layout()
    plt.show()


def plot_fwd_bwd_scatter(df):
    """
    Scatter plot of forward vs backward packets.

    Why it's interesting:
      - Directly compares two features of the same "category" (traffic volume).
      - Points far from the diagonal line (fwd ≈ bwd) can indicate asymmetric flows,
        which often correlate with attacks (e.g. flood / data exfiltration).
      - Coloring by label shows whether some regions of this space are mostly attacks or normal.
    """
    candidates = [
        'Subflow Fwd Packets',
        ' Total Backward Packets'
    ]
    for c in candidates:
        if c not in df.columns:
            print(f"[plot_fwd_bwd_scatter] Column '{c}' not found in dataframe. Skipping scatter plot.")
            return

    label_col = df.columns[-1]

    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=df.sample(min(len(df), 5000), random_state=42),  # subsample for readability
        x='Subflow Fwd Packets',
        y=' Total Backward Packets',
        hue=label_col,
        alpha=0.5,
        s=20
    )
    max_val = max(df['Subflow Fwd Packets'].max(), df[' Total Backward Packets'].max())
    plt.plot([0, max_val], [0, max_val], color='gray', linestyle='--', linewidth=1, label='fwd = bwd')
    plt.title("Scatter plot: Subflow Fwd Packets vs Total Backward Packets")
    plt.xlabel("Subflow Fwd Packets")
    plt.ylabel("Total Backward Packets")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.show()


def plot_boxplot_by_label(df, feature):
    """
    Boxplot of one feature grouped by label.

    Why it's interesting:
      - Summarizes median, quartiles, and outliers for each class.
      - Helps see if a feature is discriminative (e.g. attacks have systematically
        higher 'Fwd Packets/s' or 'Idle Mean' than normal traffic).
    """
    if feature not in df.columns:
        print(f"[plot_boxplot_by_label] Column '{feature}' not found in dataframe. Skipping boxplot.")
        return
    label_col = df.columns[-1]

    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df,
        x=label_col,
        y=feature
    )
    plt.yscale("log")
    plt.title(f"Boxplot of {feature} by label")
    plt.xlabel("Label")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()


def plot_label_distribution(df):
    """
    Bar plot of label counts.

    Why it's interesting:
      - Shows class imbalance clearly.
      - Important for understanding if you need resampling, class weights, or special metrics
        (e.g. F1-score instead of accuracy).
    """
    label_col = df.columns[-1]
    plt.figure(figsize=(8, 4))
    sns.countplot(x=label_col, data=df)
    plt.title("Label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_pairwise_relationships(df, features, sample_size=3000):
    """
    Pairplot (scatter matrix) for selected features.

    Why it's interesting:
      - Shows pairwise relationships between multiple important features at once.
      - Diagonal plots show univariate distributions; off-diagonal show how classes
        separate in 2D feature space.
      - Helps detect clusters or combinations of features that separate attacks/normal.
    """
    features = [f for f in features if f in df.columns]
    if len(features) < 2:
        print("[plot_pairwise_relationships] Not enough features for pairplot.")
        return

    label_col = df.columns[-1]
    subset = df[features + [label_col]]
    if len(subset) > sample_size:
        subset = subset.sample(sample_size, random_state=42)

    sns.pairplot(
        subset,
        vars=features,
        hue=label_col,
        diag_kind="kde",
        plot_kws={"alpha": 0.5, "s": 15}
    )
    plt.suptitle("Pairwise relationships between key features", y=1.02)
    plt.show()


# ==========
# PLOT CALLS
# ==========

# Important features (make sure the names exactly match your columns)
important_features = [
    'Subflow Fwd Packets',
    ' Total Backward Packets',
    ' Packet Length Mean',
    ' Fwd IAT Mean',
    ' Fwd Packets/s',
    ' SYN Flag Count',
    ' Idle Mean'
]

# 1) Histograms of important features (colored by label)
#    -> Interesting to see distribution shape and outliers for each feature,
#       and how different labels occupy different ranges.
plot_feature_histograms(df_cleaned, important_features, bins=60)

# 2) Scatter plot: forward vs backward packets (fwd vs bwd)
#    -> Interesting to inspect asymmetric flows (points far from fwd=bwd line),
#       and whether those correspond to attacks or normal traffic.
plot_fwd_bwd_scatter(df_cleaned)

# 3) Boxplot by label for a traffic-rate feature
#    -> Interesting to compare medians/spread of 'Fwd Packets/s' between labels
#       and see if this feature helps separate classes.
plot_boxplot_by_label(df_cleaned, ' Fwd Packets/s')

# 4) Label distribution
#    -> Interesting to visualize class imbalance and decide on resampling/metrics.
plot_label_distribution(df_cleaned)

# 5) Pairwise relationships between a subset of features
#    -> Interesting to see 2D separation between classes across several key features at once.
plot_pairwise_relationships(
    df_cleaned,
    features=[
        'Subflow Fwd Packets',
        ' Total Backward Packets',
        ' Fwd Packets/s',
        ' Idle Mean'
    ],
    sample_size=3000
)