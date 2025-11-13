import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# from dataloader import dataframes

#verif = False
#test = input("Have the file cyberdata_train.csv been crearted? (y/n) ")
#if test == "y":
df_train = pd.read_csv("./cyberdataset_train.csv")
#else:
 #   df_train = pd.concat(dataframes, ignore_index=True)
  #  df_train.to_csv("Cyberdataset_train.csv", index=False)

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
    plt.title("Correlation Heatmap", fontsize=14, pad=12)
    plt.tight_layout()
    plt.show()

def clear_df(df, threshold=2722201):
    # Drop any rows containing NaN values
    cleaned_df = df.dropna()
    cleaned_df = cleaned_df.select_dtypes(include=np.number)
    # Count number of zeros in each column
    zero_counts = (cleaned_df == 0).sum()
    # Identify columns to drop
    cols_to_drop = zero_counts[zero_counts == threshold].index
    print(f"Columns dropped ({len(cols_to_drop)}):")
    print(list(cols_to_drop))
    # Drop those columns
    df_cleaned = cleaned_df.drop(columns=cols_to_drop)
    return df_cleaned

def get_highly_correlated_features(df, threshold=0.9):
    # df supposed to be cleared already
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
print((df_cleaned==0).sum())
high_corr_cols = get_highly_correlated_features(df_cleaned)