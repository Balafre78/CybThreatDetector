from __future__ import annotations

import time
from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_DECISION_TREE_MODEL_PATH = Path("models/decision_tree.joblib")
DEFAULT_RANDOM_FOREST_MODEL_PATH = Path("models/random_forest.joblib")
DEFAULT_LOGISTIC_REGRESSION_MODEL_PATH = Path("models/logistic_regression.joblib")


def _log(message: str) -> None:
    print(f"\033[1;34m[\033[0;36mTraining\033[1;34m]\033[0m {message}\033[0m")


def train_model(
    df_train: pd.DataFrame,
    model_name: Literal["decision_tree", "random_forest", "logistic_regression"],
    model_export_path: Path | str,
    random_state: int = 39,
) -> DecisionTreeClassifier | RandomForestClassifier | LogisticRegression:
    X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]

    model = None
    model_export_path = Path(model_export_path)
    model_export_path.parent.mkdir(parents=True, exist_ok=True)

    match model_name:
        case "decision_tree":
            _log("Training DecisionTreeClassifier...")
            model = DecisionTreeClassifier(max_depth=None, random_state=random_state)
            start = time.time()
            model.fit(X_train, y_train)
            end = time.time()
            _log(f"\033[1;32mDecisionTreeClassifier training completed in {end - start:.2f} seconds.")

        case "random_forest":
            _log("Training RandomForestClassifier...")
            model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=random_state)
            start = time.time()
            model.fit(X_train, y_train)
            end = time.time()
            _log(f"\033[1;32mRandomForestClassifier training completed in {end - start:.2f} seconds.")

        case "logistic_regression":
            _log("Training LogisticRegression...")
            model = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                                                      n_jobs=-1, random_state=random_state))])
            start = time.time()
            model.fit(X_train, y_train)
            end = time.time()
            _log(f"\033[1;32mLogisticRegression training completed in {end - start:.2f} seconds.")
    joblib.dump(model, model_export_path)
    _log(f"Saved model to {model_export_path.resolve()}")
    return model
