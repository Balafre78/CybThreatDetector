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
from analysis import plot_confusion_matrix, plot_feature_importance, plot_multiclass_roc, shap_analysis


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
        self.logistic_regression_model: Optional[LogisticRegression] = None

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

    def analyse_decision_tree(self):
        plot_confusion_matrix(model=self.decision_tree_model, df_test=self.test_dataset)
        plot_feature_importance(model=self.decision_tree_model, df_train=self.train_dataset)
        plot_multiclass_roc(model=self.decision_tree_model, df_test=self.test_dataset)
        shap_analysis(model=self.decision_tree_model, df_test=self.test_dataset)

    def analyse_random_forest(self):
        plot_confusion_matrix(model=self.random_forest_model, df_test=self.test_dataset)
        plot_feature_importance(model=self.random_forest_model, df_train=self.train_dataset)
        plot_multiclass_roc(model=self.random_forest_model, df_test=self.test_dataset)
        shap_analysis(model=self.random_forest_model, df_test=self.test_dataset)

    def analyse_logistic_regression(self):
        plot_confusion_matrix(model=self.logistic_regression_model, df_test=self.test_dataset)
        plot_multiclass_roc(model=self.logistic_regression_model, df_test=self.test_dataset)


def _render_menu(actions: Dict[str, Dict[str, str | Callable]]) -> None:
    print(f'\u250C{55*'\u2500'}\u2510')
    print(f"\u2502 {'Cyber Threat Detector Workflow'.center(53)} \u2502")
    for key, value in actions.items():
        status = "\033[1;32mReady\033[0m" if value["can_execute"] else "\033[1;31mLocked\033[0m"
        print(f"\u2502{f' {key}. {value["name"]}'.ljust(45)}{f'[{status}] \u2502'.rjust(22)}")
    print(f'\u2514{55*'\u2500'}\u2518')


def run_cli() -> None:
    manager = WorkflowManager()
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
                "name": "Analyse decision tree model",
                "func": manager.analyse_decision_tree,
                "can_execute": manager.test_dataset is not None and manager.decision_tree_model is not None
            },
            "12": {
                "name": "Analyse random forest model",
                "func": manager.analyse_random_forest,
                "can_execute": manager.test_dataset is not None and manager.random_forest_model is not None
            },
            "13": {
                "name": "Analyse logistic regression model",
                "func": manager.analyse_logistic_regression,
                "can_execute": manager.test_dataset is not None and manager.logistic_regression_model is not None
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