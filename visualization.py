import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_zero_value_features(df: pd.DataFrame, max_features: int = 20,
                                     display_decimals: int = 4,
                                     consider_almost_zero_pct: float = None):
    """
    Visualize the percentage of zero values in each numeric feature.
    - Highlights problematic features (above threshold)
    - Warns for columns that are 100% zero
    - Adds a vertical threshold line for clarity
    :param df: raw dataframe
    :param max_features: Max number of features to display
    :return:
    """
    # select feature columns (exclude last column = label)
    feature_cols = df.columns[:-1]
    numeric_cols = df[feature_cols].select_dtypes(include=np.number).columns

    if len(numeric_cols) == 0:
        print("No numeric features found for zero-value analysis.")
        return

    total_rows = len(df)
    # exact zero counts per column
    zero_counts = (df[numeric_cols] == 0).sum(axis=0).astype(int)

    # percentage (float) with full precision
    zero_pct = (zero_counts / total_rows) * 100.0
    zero_pct = zero_pct.sort_values(ascending=False)

    # limit features displayed
    if len(zero_pct) > max_features:
        zero_pct = zero_pct.head(max_features)
        zero_counts = zero_counts[zero_pct.index]

    # decide coloring:
    # - red if column is exactly all zeros (zero_count == total_rows)
    # - OR if consider_almost_zero_pct is set and pct >= threshold
    colors = []
    for col in zero_pct.index:
        count = zero_counts[col]
        pct = zero_pct.loc[col]
        if count == total_rows:
            colors.append("#C44E52")  # exact 100%
        elif (consider_almost_zero_pct is not None) and (pct >= consider_almost_zero_pct):
            colors.append("#E9A227")  # almost 100% (yellow/orange)
        else:
            colors.append("#55A868")  # normal (green)

    # plot
    plt.figure(figsize=(12, max(4, 0.35 * len(zero_pct))))
    ax = sns.barplot(x=zero_pct.values, y=zero_pct.index, hue= zero_pct.index, palette=colors)

    ax.set_title("Zero Values per Numeric Feature (%)", fontsize=18, pad=12)
    ax.set_xlabel("Percentage of Zero Values", fontsize=14)
    ax.set_ylabel("Numeric Feature", fontsize=14)

    # annotate with high precision
    fmt = f"{{:.{display_decimals}f}}%"
    for i, col in enumerate(zero_pct.index):
        pct = zero_pct.iloc[i]
        count = zero_counts[col]
        label = "<0.{}%".format("1" + "0"*(display_decimals-1)) if pct == 0 and display_decimals > 0 else fmt.format(pct)
        # also show raw counts optionally (uncomment if needed)
        # label = f"{fmt.format(pct)}  ({count}/{total_rows})"
        ax.text(pct + 0.1, i, label, va="center", ha="left", fontsize=9)

    # add vertical line at 100% for reference
    ax.axvline(x=100.0, color="red", linestyle="--", linewidth=1.0)
    plt.tight_layout()
    plt.show()

def plot_class_distribution(df: pd.DataFrame, log_scale: bool = True) -> None:
    """
    Label / class distribution (bar plot).
    :param df:
    :param log_scale:
    :return: plot
    """
    label_col = df.columns[-1]
    counts = df[label_col].value_counts().sort_values(ascending=False)
    percentages = counts / counts.sum() * 100
    plt.figure(figsize=(10, 6))
    # Use a single color instead of palette (avoids FutureWarning)
    ax = sns.barplot(
        x=counts.index.astype(str),
        y=counts.values,
        color="#4C72B0"
    )
    ax.set_title("Class Distribution", fontsize=22, pad=16)
    ax.set_xlabel("Class / Attack Type", fontsize=16)
    ax.set_ylabel("Number of Samples", fontsize=16)
    if log_scale:
        ax.set_yscale("log")

    for i, (count, pct) in enumerate(zip(counts.values, percentages.values)):
        ax.text(
            i,
            count,
            f"{pct:.4f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

def plot_benign_vs_attacks(df: pd.DataFrame) -> None:
    """
    Barplot: number of BENIGN vs all other attacks grouped together.
    Produces exactly two bars:
    - 'BENIGN'
    - 'Attacks' (sum of all non-BENIGN classes)
    :param df: raw dataframe
    :return: plot
    """
    label_col = df.columns[-1]

    # Count BENIGN and others
    benign_mask = df[label_col] == "BENIGN"
    benign_count = benign_mask.sum()
    other_count = len(df) - benign_count
    if benign_count == 0:
        print("No 'BENIGN' class found in the label column; cannot plot this comparison.")
        return
    counts = pd.Series( [benign_count, other_count], index=["BENIGN", "Attacks"])
    total = counts.sum()
    percentages = counts / total * 100
    plt.figure(figsize=(6, 6))
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=["#4C72B0", "#DD8452"], legend=False)
    ax.set_title("BENIGN vs All Attacks", fontsize=22, pad=16)
    ax.set_xlabel("Class Group", fontsize=16)
    ax.set_ylabel("Number of Samples", fontsize=16)

    # Annotate with percentages (two decimals as requested)
    for i, (count, pct) in enumerate(zip(counts.values, percentages.values)):
        label = f"{pct:.2f}%"
        ax.text(i, count, label, ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    plt.show()
