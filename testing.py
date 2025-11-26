from typing import Dict
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier


def _log(message: str) -> None:
    print(f"\033[1;34m[\033[0;36mTesting\033[1;34m]\033[0m {message}\033[0m")


def evaluate_model(model: DecisionTreeClassifier | RandomForestClassifier | LogisticRegression, df_test: pd.DataFrame) \
    -> Dict[str, float]:
    X_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]
    metrics: Dict[str, float] = {}
    _log(f"Evaluating {type(model).__name__}...")
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
    }
    print(f"- Accuracy: {metrics['accuracy']:.4f}")
    print(f"- F1 Macro: {metrics['f1_macro']:.4f}")
    print(f"- F1 Weighted: {metrics['f1_weighted']:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    labels = sorted(y_test.unique())
    return metrics

