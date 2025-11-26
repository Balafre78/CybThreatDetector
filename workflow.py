from pathlib import Path
from typing import Dict, Optional, Callable

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from disk import load_dataset, load_model
from loading import download_and_merge_dataset
from preprocessing import RAW_DATASET_PATH, preprocess_dataset, DEFAULT_TRAIN_DATASET_PATH, DEFAULT_TEST_DATASET_PATH
from testing import evaluate_model
from training import train_model, DEFAULT_DECISION_TREE_MODEL_PATH, DEFAULT_RANDOM_FOREST_MODEL_PATH, \
    DEFAULT_LOGISTIC_REGRESSION_MODEL_PATH
from analysis import plot_confusion_matrix, _heatmap_correlation, plot_feature_importance, plot_multiclass_roc, shap_analysis

class WorkflowManager:
    def __init__(
            self,
            raw_dataset_path: str | Path = RAW_DATASET_PATH,
            train_dataset_path: str | Path = DEFAULT_TRAIN_DATASET_PATH,
            test_dataset_path: str | Path = DEFAULT_TEST_DATASET_PATH,
            decision_tree_model_path: str | Path = DEFAULT_DECISION_TREE_MODEL_PATH,
            random_forest_model_path: str | Path = DEFAULT_RANDOM_FOREST_MODEL_PATH,
            logistic_regression_model_path: str | Path = DEFAULT_LOGISTIC_REGRESSION_MODEL_PATH,
        ) -> None:

        self.raw_dataset_path = raw_dataset_path
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.decision_tree_model_path = decision_tree_model_path
        self.random_forest_model_path = random_forest_model_path
        self.logistic_regression_model_path = logistic_regression_model_path

        self.raw_dataset: Optional[pd.DataFrame] = None
        self.train_dataset: Optional[pd.DataFrame] = None
        self.test_dataset: Optional[pd.DataFrame] = None
        self.decision_tree_model: Optional[DecisionTreeClassifier] = None
        self.random_forest_model: Optional[RandomForestClassifier] = None

        self.latest_metrics: Optional[Dict[str, Dict[str, float]]] = {}
        self._load()

    def _load(self):
        self.raw_dataset = load_dataset(self.raw_dataset_path)
        self.train_dataset = load_dataset(self.train_dataset_path)
        self.test_dataset = load_dataset(self.test_dataset_path)
        self.decision_tree_model = load_model(self.decision_tree_model_path)
        self.random_forest_model = load_model(self.random_forest_model_path)
        self.logistic_regression_model = load_model(self.logistic_regression_model_path)

    def load_data(self):
        self.raw_dataset = download_and_merge_dataset(output_csv=self.raw_dataset_path)

    def preprocess_data(self):
        self.train_dataset, self.test_dataset = preprocess_dataset(
            df_raw=self.raw_dataset,
            output_train_csv=self.train_dataset_path,
            output_test_csv=self.test_dataset_path
        )

    def train_all_models(self):
        self.train_decision_tree_model()
        self.train_random_forest_model()
        self.train_logistic_regression_model()

    def train_decision_tree_model(self):
        self.decision_tree_model = train_model(
            df_train=self.train_dataset,
            model_name="decision_tree",
            model_export_path=self.decision_tree_model_path
        )

    def train_random_forest_model(self):
        self.random_forest_model = train_model(
            df_train=self.train_dataset,
            model_name="random_forest",
            model_export_path=self.random_forest_model_path
        )

    def train_logistic_regression_model(self):
        self.logistic_regression_model = train_model(
            df_train=self.train_dataset,
            model_name="logistic_regression",
            model_export_path=self.logistic_regression_model_path
        )
    def test_all_models(self):
        self.test_decision_tree_model()
        self.test_random_forest_model()

    def test_decision_tree_model(self):
        evaluate_model(model=self.decision_tree_model, df_test=self.test_dataset)

    def test_random_forest_model(self):
        evaluate_model(model=self.random_forest_model, df_test=self.test_dataset)

    def test_logistic_regression_model(self):
        evaluate_model(model=self.logistic_regression_model, df_test=self.test_dataset)

    def analyze_confusion_matrix(self, model_name: str):
        model = {
            "decision_tree": self.decision_tree_model,
            "random_forest": self.random_forest_model,
            "logistic_regression": self.logistic_regression_model
        }.get(model_name)
        if model is None:
            print(f"Model '{model_name}' is not loaded.")
            return
        if self.test_dataset is None:
            print("Test dataset not loaded.")
            return
        # Prepare X_test and y_test (assume label is last column)
        X_test = self.test_dataset.iloc[:, :-1]
        y_test = self.test_dataset.iloc[:, -1]
        # Compute predictions
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            print("Error during prediction:", e)
            return
        labels = sorted(y_test.unique())
        plot_confusion_matrix(y_test, y_pred, labels, title=f"Confusion Matrix - {model_name}")

    def analyze_feature_importance(self, model_name: str):
        """
        Plot feature importance for decision tree and random forest.
        :param model_name: ToDO
        :return: ToDO
        """
        model = {
            "decision_tree": self.decision_tree_model,
            "random_forest": self.random_forest_model
        }.get(model_name)

        if model is None:
            print("Feature importance not supported for this model.")
            return

        if self.train_dataset is None:
            print("Train dataset not loaded.")
            return

        X = self.train_dataset.iloc[:, :-1]
        plot_feature_importance(model, X, f"{model_name} Feature Importance")

    def analyze_multiclass_roc(self, model_name: str):
        """
        Plot multiclass ROC for the selected model.
        Expects analysis.plot_multiclass_roc(model, X_test, y_test) signature;
        if your plot_multiclass_roc expects different args, adapt accordingly.
        """
        model = {
            "decision_tree": self.decision_tree_model,
            "random_forest": self.random_forest_model,
            "logistic_regression": self.logistic_regression_model
        }.get(model_name)
        if model is None:
            print(f"Model '{model_name}' unavailable.")
            return
        if self.test_dataset is None:
            print("Test dataset not loaded.")
            return
        X_test = self.test_dataset.iloc[:, :-1]
        y_test = self.test_dataset.iloc[:, -1]
        plot_multiclass_roc(model, X_test, y_test, title= f"ROC Curve - {model_name}")

    def analyze_shap(self, model_name: str):
        """
        Run SHAP analysis for a model
        :param model_name: ToDo
        :param max_samples: ToDo
        :return: ToDo
        """
        model = {
            "decision_tree": self.decision_tree_model,
            "random_forest": self.random_forest_model,
        }.get(model_name)
        if model is None:
            print("Model missing for SHAP.")
            return
        if self.train_dataset is None:
            print("Train dataset not loaded.")
            return
        if self.test_dataset is None:
            print("Test dataset not loaded.")
            return
        X_train = self.train_dataset.iloc[:, :-1]
        X_test = self.test_dataset.iloc[:, :-1]
        shap_analysis(model, X_train, X_test)

def _render_menu(actions: Dict[str, Dict[str, str | Callable]]) -> None:
    print(f'\u250C{45*'\u2500'}\u2510')
    print(f"\u2502 {'Cyber Threat Detector Workflow'.center(43)} \u2502")
    for key, value in actions.items():
        status = "\033[1;32mReady\033[0m" if value["can_execute"] else "\033[1;31mLocked\033[0m"
        print(f"\u2502{f' {key}. {value["name"]}'.ljust(35)}{f'[{status}] \u2502'.rjust(22)}")
    print(f'\u2514{45*'\u2500'}\u2518')


def run_cli() -> None:
    manager = WorkflowManager()
    print("Cyber Threat Detector interactive workflow.\nPress Ctrl+C to exit at any time.")
    while True:
        actions = {
            "1": {
                "name": "Load data",
                "func": manager.load_data,
                "can_execute": True
            },
            "2": {
                "name": "Preprocess data",
                "func": manager.preprocess_data,
                "can_execute": manager.raw_dataset is not None
            },
            "3": {
                "name": "Train decision tree model",
                "func": manager.train_decision_tree_model,
                "can_execute": manager.train_dataset is not None
            },
            "4": {
                "name": "Train random forest model",
                "func": manager.train_random_forest_model,
                "can_execute": manager.train_dataset is not None
            },
            "5": {
                "name": "Train logistic regression model",
                "func": manager.train_logistic_regression_model,
                "can_execute": manager.train_dataset is not None
            },
            "6": {
                "name": "Train all models",
                "func": manager.train_all_models,
                "can_execute": manager.train_dataset is not None
            },
            "7": {
                "name": "Test decision tree model",
                "func": manager.test_decision_tree_model,
                "can_execute": manager.test_dataset is not None and manager.decision_tree_model is not None
            },
            "8": {
                "name": "Test random forest model",
                "func": manager.test_decision_tree_model,
                "can_execute": manager.test_dataset is not None and manager.random_forest_model is not None
            },
            "9": {
                "name": "Test logistic regression model",
                "func": manager.test_logistic_regression_model,
                "can_execute": manager.test_dataset is not None and manager.logistic_regression_model is not None
            },
            "10": {
                "name": "Test all models",
                "func": manager.test_all_models,
                "can_execute": manager.test_dataset is not None and manager.decision_tree_model is not None and manager.random_forest_model is not None
                               and manager.logistic_regression_model is not None
            },
            "11": {
                "name": "Plot Confusion Matrix (DT)",
                "func": lambda: manager.analyze_confusion_matrix("decision_tree"),
                "can_execute": manager.test_dataset is not None and manager.decision_tree_model is not None
            },
            "12": {
                "name": "Plot Confusion Matrix (RF)",
                "func": lambda: manager.analyze_confusion_matrix("random_forest"),
                "can_execute": manager.test_dataset is not None and manager.random_forest_model is not None
            },
            "13": {
                "name": "Plot Confusion Matrix (LR)",
                "func": lambda: manager.analyze_confusion_matrix("logistic_regression"),
                "can_execute": manager.test_dataset is not None and manager.logistic_regression_model is not None
            },
            "14": {
                "name": "Feature Importance (DT)",
                "func": lambda: manager.analyze_feature_importance("decision_tree"),
                "can_execute": manager.train_dataset is not None and manager.decision_tree_model is not None
            },
            "15": {
                "name": "Feature Importance (RF)",
                "func": lambda: manager.analyze_feature_importance("random_forest"),
                "can_execute": manager.train_dataset is not None and manager.random_forest_model is not None
            },
            "17": {
                "name": "Multiclass ROC (DT)",
                "func": lambda: manager.analyze_multiclass_roc("decision_tree"),
                "can_execute": manager.test_dataset is not None and manager.decision_tree_model is not None
            },
            "18": {
                "name": "Multiclass ROC (RF)",
                "func": lambda: manager.analyze_multiclass_roc("random_forest"),
                "can_execute": manager.test_dataset is not None and manager.random_forest_model is not None
            },
            "19": {
                "name": "Multiclass ROC (LR)",
                "func": lambda:manager.analyze_multiclass_roc("logistic_regression"),
                "can_execute": manager.test_dataset is not None and manager.logistic_regression_model is not None
            },
            "20": {
                "name": "SHAP values (RF)",
                "func": lambda: manager.analyze_shap("random_forest"),
                "can_execute": manager.train_dataset is not None and manager.random_forest_model is not None
            },
            "21": {
                "name": "SHAP values (DT)",
                "func": lambda: manager.analyze_shap("decision_tree"),
                "can_execute": manager.train_dataset is not None and manager.decision_tree_model is not None
            },
            "q": {
                "name": "Quit",
                "func": lambda: exit(0),
                "can_execute": True
            },
        }
        _render_menu(actions)
        user_choice = input("Enter an option ❯ ").strip().lower()
        if user_choice not in actions.keys():
            print("Invalid selection.")
            continue
        if not actions[user_choice]["can_execute"]:
            print("This action is currently locked. Please complete the prerequisite steps first.")
            continue
        actions[user_choice]["func"]()