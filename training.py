from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from preprocessing import DEFAULT_CLEAN_DATASET_PATH

MODELS_DIR = Path("models")
TEST_SPLIT_FILE = "test_split.joblib"

def _log(message: str) -> None:
    print(f"\033[1;34m[\033[0;36mTraining\033[1;34m]\033[0m {message}\033[0m")

@dataclass(slots=True)
class ModelArtifacts:
    decision_tree: DecisionTreeClassifier
    random_forest: RandomForestClassifier
    X_test: pd.DataFrame
    y_test: pd.Series


def load_clean_dataset(csv_path: Path | str = DEFAULT_CLEAN_DATASET_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"[Training] Clean dataset file not found at {csv_path.resolve()}. Run preprocessing first.")
    _log(f"\033[1;33mLoading cleaned dataset from {csv_path.resolve()}")
    df = pd.read_csv(csv_path)
    return df.iloc[:, :-1], df.iloc[:, -1]


def train_models(
    csv_path: Path | str = DEFAULT_CLEAN_DATASET_PATH,
    models_dir: Path | str = MODELS_DIR,
    test_size: float = 0.2,
    random_state: int = 39,
) -> ModelArtifacts:
    X, y = load_clean_dataset(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, shuffle=True, stratify=y
    )

    _log("Training DecisionTreeClassifier...")
    dt_model = DecisionTreeClassifier(max_depth=None, random_state=random_state)
    start = time.time()
    dt_model.fit(X_train, y_train)
    end = time.time()
    _log(f"\033[1;32mDecisionTreeClassifier training completed in {end - start:.2f} seconds.")

    _log("Training RandomForestClassifier...")
    rf_model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=random_state)
    start = time.time()
    rf_model.fit(X_train, y_train)
    end = time.time()
    _log(f"\033[1;32mRandomForestClassifier training completed in {end - start:.2f} seconds.")

    artifacts = ModelArtifacts(decision_tree=dt_model, random_forest=rf_model, X_test=X_test, y_test=y_test)

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(dt_model, models_path / "decision_tree.joblib")
    joblib.dump(rf_model, models_path / "random_forest.joblib")
    joblib.dump({"X_test": X_test, "y_test": y_test}, models_path / TEST_SPLIT_FILE)
    _log(f"Saved models to {models_path.resolve()}")

    return artifacts


def load_model_artifacts_from_disk(models_dir: Path | str = MODELS_DIR) -> ModelArtifacts:
    models_path = Path(models_dir)
    dt_path = models_path / "decision_tree.joblib"
    rf_path = models_path / "random_forest.joblib"
    split_path = models_path / TEST_SPLIT_FILE

    missing = [str(path) for path in (dt_path, rf_path, split_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing trained model artifacts. Expected files: " + ", ".join(missing))

    dt_model: DecisionTreeClassifier = joblib.load(dt_path)
    rf_model: RandomForestClassifier = joblib.load(rf_path)
    split_payload = joblib.load(split_path)

    X_test = split_payload.get("X_test")
    y_test = split_payload.get("y_test")
    if X_test is None or y_test is None:
        raise ValueError("Corrupted test split payload; expected keys 'X_test' and 'y_test'.")

    return ModelArtifacts(decision_tree=dt_model, random_forest=rf_model, X_test=X_test, y_test=y_test)


def evaluate_models(artifacts: ModelArtifacts) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for model_name, model in (
        ("Decision Tree", artifacts.decision_tree),
        ("Random Forest", artifacts.random_forest),
    ):
        print(f"[evaluation] Evaluating {model_name}…")
        y_pred = model.predict(artifacts.X_test)
        metrics[model_name] = {
            "accuracy": accuracy_score(artifacts.y_test, y_pred),
            "f1_macro": f1_score(artifacts.y_test, y_pred, average="macro"),
            "f1_weighted": f1_score(artifacts.y_test, y_pred, average="weighted"),
        }
        print(f"Accuracy: {metrics[model_name]['accuracy']:.4f}")
        print(f"F1 Macro: {metrics[model_name]['f1_macro']:.4f}")
        print(f"F1 Weighted: {metrics[model_name]['f1_weighted']:.4f}")
        print("Classification Report:\n", classification_report(artifacts.y_test, y_pred))
        labels = sorted(artifacts.y_test.unique())
        cm = confusion_matrix(artifacts.y_test, y_pred, labels=labels)
        print("Confusion Matrix:")
        print(cm)
    return metrics


def plot_feature_importance(model, X, title, top_n=15):
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

