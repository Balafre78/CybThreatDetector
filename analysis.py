import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize

def _heatmap_correlation(df: pd.DataFrame) -> None:
    """
    Plots a correlation heatmap for all numeric columns in a DataFrame.
    - Automatically filters non-numeric columns
    - Drops rows with NaN values before computing correlations
    Args:
        df: DataFrame to plot
    """
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

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

def analyze_label_distribution(df: pd.DataFrame):
    """
    Displays the count of occurrences for each label (attack type) and plots a bar chart.
    df : Dataset containing a label column (last column).
    """
    label_col = df.columns[-1]
    label_counts = df[label_col].value_counts()
    print("\nLabel Distribution :")
    print(label_counts)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis")
    plt.xticks(rotation=75)
    plt.xlabel("Attack Type")
    plt.ylabel("Number of Samples")
    plt.title("Label Distribution (Attack Types)")
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels, title):
    """
    Plot the confusion matrix
    :param y_true: ToDo
    :param y_pred: ToDo
    :param labels: ToDo
    :param title: ToDo
    :return: ToDo
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(18, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, annot_kws={"size": 12})
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.title(title, fontsize=20)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, X, title, top_n=15):
    """
    Plot the feature importance
    :param model: ToDO
    :param X: ToDO
    :param title: Name of the model
    :param top_n: ToDO
    :return: ToDO
    """
    importances = model.feature_importances_
    index = np.argsort(importances)[::-1]
    df_plot = pd.DataFrame({"Feature": X.columns[index][:top_n], "Importance": importances[index][:top_n]})
    plt.figure(figsize=(12, 9))
    sns.barplot(
        data=df_plot,
        x="Importance",
        y="Feature",
        hue="Feature",
        palette="viridis",
        dodge=False,
        legend=False,
    )
    plt.title(f"{title} - Top {top_n} Features", fontsize=18)
    plt.xlabel("Importance", fontsize=14)
    plt.ylabel("Feature", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_multiclass_roc(model, X_test, y_test, title):
    classes = sorted(y_test.unique())
    y_test_bin = label_binarize(y_test, classes=classes)
    y_score = model.predict_proba(X_test)
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        plt.plot(fpr, tpr, label=f"{class_name}")
    plt.plot([0, 1], [0, 1], "k--")  # random classifier line
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def shap_analysis(model, X_train, X_test, max_samples=1000):
    """
    Description ToDO
    :param model: ToDO
    :param X_train: ToDO
    :param X_test: ToDO
    :param max_samples: ToDO
    :return: ToDO
    """
    print("\nRunning SHAP Analysis...")
    # SHAP explainer for tree-based models (fast, optimized)
    explainer = shap.TreeExplainer(model)
    if len(X_test) > max_samples:
        X_shap = X_test.sample(max_samples, random_state=39)
    else:
        X_shap = X_test
    print(f"Computing SHAP values on {len(X_shap)} samples...")
    # Compute SHAP values
    shap_values = explainer.shap_values(X_shap)
    # Summary plot (global importance)
    print("Displaying SHAP summary plot (global feature impact)...")
    shap.summary_plot(shap_values, X_shap, plot_type="dot")
    # Bar plot (mean SHAP)
    print("Displaying SHAP bar plot (mean absolute importance)...")
    shap.summary_plot(shap_values, X_shap, plot_type="bar")
    return shap_values
