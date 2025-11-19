from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from preprocessing import DEFAULT_TEST_DATASET_PATH


def _log(message: str) -> None:
    print(f"\033[1;34m[\033[0;36mTesting\033[1;34m]\033[0m {message}\033[0m")


def _load_test_dataset(
        csv_path: Path | str
) -> Tuple[pd.DataFrame, pd.Series]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"[Testing] Testing dataset file not found at {csv_path.resolve()}. Run preprocessing first.")
    _log(f"\033[1;33mLoading testing dataset from {csv_path.resolve()}")
    df = pd.read_csv(csv_path)
    return df.iloc[:, :-1], df.iloc[:, -1]


def evaluate_model(
        model: DecisionTreeClassifier | RandomForestClassifier,
        model_name: str,
        test_csv_path: Path | str = DEFAULT_TEST_DATASET_PATH,
) -> Dict[str, float]:
    X_test, y_test = _load_test_dataset(test_csv_path)

    metrics: Dict[str, float] = {}
    _log(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
    }
    print(f"- Accuracy: {metrics['accuracy']:.4f}")
    print(f"- F1 Macro: {metrics['f1_macro']:.4f}")
    print(f"- F1 Weighted: {metrics['f1_weighted']:.4f}")
    #print("Classification Report:\n", classification_report(y_test, y_pred))
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    #print("Confusion Matrix:")
    #print(cm)
    return metrics


"""def plot_feature_importance(model, X, title, top_n=15):
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
    plt.show()"""
