import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier

from loading import inverse_map

def _log(message: str) -> None:
    print(f"\033[1;34m[\033[0;36mAnalysis\033[1;34m]\033[0m {message}\033[0m")


def _heatmap_correlation(df: pd.DataFrame) -> None:
    """
    Plots a correlation heatmap for all numeric columns in a DataFrame while automatically filtering non-numeric columns
    and dropping rows with NaN values before computing correlations
    :param df: A cleaned DataFrame to be plotted
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


def _analyze_label_distribution(df: pd.DataFrame) -> None:
    """
    Displays the number of occurrences for each label (attack type) and plots a bar chart.
    :param df : Dataset containing a label column (last column).
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


def plot_confusion_matrix(model: DecisionTreeClassifier | RandomForestClassifier | LogisticRegression | XGBClassifier, df_test: pd.DataFrame) -> None:
    """
    Plots the confusion matrix
    :param model: A given model between DecisionTreeClassifier, RandomForestClassifier, LogisticRegression and XGBClassifier
    :param df_test: A cleaned testing Dataframe
    """
    _log(f"Plotting Confusion Matrix for {type(model).__name__}")
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]
    if isinstance(model, XGBClassifier):
        y_pred_e = model.predict(X_test)
        y_pred = pd.Series(y_pred_e).map(lambda x: inverse_map[x])
    else:
        y_pred = model.predict(X_test)
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(18, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, annot_kws={"size": 12})
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.title(f"Confusion matrix - {type(model).__name__}", fontsize=20)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model: DecisionTreeClassifier | RandomForestClassifier | XGBClassifier, df_train: pd.DataFrame, top_n: int = 15) -> None:
    """
    Plots the feature importance.
    :param model: A given model between DecisionTreeClassifier, RandomForestClassifier and XGBClassifier
    :param df_train: A Dataframe used for its feature names
    :param top_n: Number of top features to display
    """
    _log(f"Plotting Feature Importance for {type(model).__name__}")
    X = df_train.iloc[:, :-1]
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
    plt.title(f"{type(model).__name__} - Top {top_n} Features", fontsize=18)
    plt.xlabel("Importance", fontsize=14)
    plt.ylabel("Feature", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_multiclass_roc(model: DecisionTreeClassifier | RandomForestClassifier | LogisticRegression | XGBClassifier, df_test: pd.DataFrame) -> None:
    """
    Plots the ROC Curve for every class in a multiclass classification setting.
    :param model: A given model between DecisionTreeClassifier, RandomForestClassifier, LogisticRegression and XGBClassifier
    :param df_test: A cleaned Dataframe used for testing
    """
    _log(f"Plotting ROC Curve for {type(model).__name__}")
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]
    classes = sorted(y_test.unique())
    y_test_bin = label_binarize(y_test, classes=classes)
    y_score = model.predict_proba(X_test)
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        plt.plot(fpr, tpr, label=f"{class_name}")
    plt.plot([0, 1], [0, 1], "k--")  # random classifier line
    plt.title(f"ROC Curve - {type(model).__name__}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def shap_analysis(model: DecisionTreeClassifier | RandomForestClassifier | XGBClassifier, df_test: pd.DataFrame, max_samples: int = 1000) -> None:
    """

    :param model: A given model between DecisionTreeClassifier, RandomForestClassifier and XGBClassifier
    :param df_test: A cleaned Dataframe used for testing
    :param max_samples: Maximum number of samples to use for SHAP analysis
    """
    _log(f"Running SHAP Analysis of {type(model).__name__}...")
    X_train = df_test.iloc[:, :-1]
    X_test = df_test.iloc[:, :-1]
    # SHAP explainer for tree-based models (fast, optimized)
    explainer = shap.TreeExplainer(model)
    if len(X_test) > max_samples:
        X_shap = X_test.sample(max_samples, random_state=39)
    else:
        X_shap = X_test
    _log(f"Computing SHAP values on {len(X_shap)} samples")
    # Compute SHAP values
    shap_values = explainer.shap_values(X_shap)
    # Summary plot (global importance)
    _log("Plotting SHAP summary (global feature impact)")
    shap.summary_plot(shap_values, X_shap, plot_type="dot")
    # Bar plot (mean SHAP)
    _log("Plotting SHAP bar plot (mean absolute importance)")
    shap.summary_plot(shap_values, X_shap, plot_type="bar")
    return shap_values
