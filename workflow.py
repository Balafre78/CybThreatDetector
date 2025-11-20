from pathlib import Path
from typing import Dict, Optional, Callable

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from disk import load_dataset, load_model
from loading import download_and_merge_dataset
from preprocessing import RAW_DATASET_PATH, preprocess_dataset, DEFAULT_TRAIN_DATASET_PATH, DEFAULT_TEST_DATASET_PATH
from testing import evaluate_model
from training import train_model, DEFAULT_DECISION_TREE_MODEL_PATH, DEFAULT_RANDOM_FOREST_MODEL_PATH


class WorkflowManager:
    def __init__(
            self,
            raw_dataset_path: str | Path = RAW_DATASET_PATH,
            train_dataset_path: str | Path = DEFAULT_TRAIN_DATASET_PATH,
            test_dataset_path: str | Path = DEFAULT_TEST_DATASET_PATH,
            decision_tree_model_path: str | Path = DEFAULT_DECISION_TREE_MODEL_PATH,
            random_forest_model_path: str | Path = DEFAULT_RANDOM_FOREST_MODEL_PATH,
        ) -> None:

        self.raw_dataset_path = raw_dataset_path
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.decision_tree_model_path = decision_tree_model_path
        self.random_forest_model_path = random_forest_model_path

        self.raw_dataset: Optional[pd.DataFrame] = None
        self.train_dataset: Optional[pd.DataFrame] = None
        self.test_dataset: Optional[pd.DataFrame] = None
        self.decision_tree_model: Optional[DecisionTreeClassifier] = None
        self.random_forest_model: Optional[RandomForestClassifier] = None

        self.latest_metrics: Optional[Dict[str, Dict[str, float]]] = {}
        self._load()

    def _load(self):
        try:
            self.raw_dataset = load_dataset(self.raw_dataset_path)
        except FileNotFoundError:
            self.raw_dataset = None
        try:
            self.train_dataset = load_dataset(self.train_dataset_path)
        except FileNotFoundError:
            self.train_dataset = None
        try:
            self.test_dataset = load_dataset(self.test_dataset_path)
        except FileNotFoundError:
            self.test_dataset = None
        try:
            self.decision_tree_model = load_model(self.decision_tree_model_path)
        except FileNotFoundError:
            self.decision_tree_model = None
        try:
            self.random_forest_model = load_model(self.random_forest_model_path)
        except FileNotFoundError:
            self.random_forest_model = None

    def run_load_data(self):
        self.raw_dataset = download_and_merge_dataset(output_csv=self.raw_dataset_path)

    def run_preprocess_data(self):
        self.train_dataset, self.test_dataset = preprocess_dataset(
            df_raw=self.raw_dataset,
            output_train_csv=self.train_dataset_path,
            output_test_csv=self.test_dataset_path
        )

    def run_train_all_models(self):
        self.run_train_decision_tree_model()
        self.run_train_random_forest_model()

    def run_train_decision_tree_model(self):
        self.decision_tree_model = train_model(
            df_train=self.train_dataset,
            model_name="decision_tree",
            model_export_path=self.decision_tree_model_path
        )

    def run_train_random_forest_model(self):
        self.random_forest_model = train_model(
            df_train=self.train_dataset,
            model_name="random_forest",
            model_export_path=self.decision_tree_model_path
        )

    def run_test_all_models(self):
        self.run_test_decision_tree_model()
        self.run_test_random_forest_model()

    def run_test_decision_tree_model(self):
        evaluate_model(model=self.decision_tree_model, df_test=self.test_dataset)

    def run_test_random_forest_model(self):
        evaluate_model(model=self.random_forest_model, df_test=self.test_dataset)


def _format_step_line(index: int, txt: str, *, done: bool, unlocked: bool) -> str:
    if done:
        status = "\033[1;32mDone\033[0m"
    elif unlocked:
        status = "\033[1;33mReady\033[0m"
    else:
        status = "\033[1;31mLocked\033[0m"
    return f"\u2502{f' {index}. {txt}'.ljust(22)}{f'[{status}] \u2502'.rjust(22)}"


def _render_menu(actions: Dict[str, Dict[str, str | Callable]]) -> None:
    print(f'\u250C{32*'\u2500'}\u2510')
    print("\u2502 Cyber Threat Detector Workflow \u2502")
    """rows = [
        (WorkflowStep.LOAD_DATA, manager.raw_dataset is not None),
        (WorkflowStep.PREPROCESS_DATA, manager.train_dataset is not None and manager.test_dataset is not None),
        (WorkflowStep.TRAIN_MODEL, manager.models is not None),
        (WorkflowStep.TEST_MODEL, False),
    ]
    for idx, (step, done) in enumerate(rows, start=1):
        unlocked = manager.can_execute(step)
        print(_format_step_line(idx, step.label, done=done, unlocked=unlocked))"""
    for key, value in actions.items():
        print(f'\u2502 {key}. {value["name"]}'.ljust(30) + '\u2502')
    print(f'\u2514{32*'\u2500'}\u2518')

def run_cli() -> None:
    manager = WorkflowManager()
    actions = {
        "1": { "name": "Load data", "func": manager.run_load_data},
        "2": { "name": "Preprocess data", "func": manager.run_preprocess_data },
        "3": { "name": "Train decision tree model", "func": manager.run_train_decision_tree_model },
        "4": { "name": "Train random forest model", "func": manager.run_train_random_forest_model },
        "5": { "name": "Train all models", "func": manager.run_train_all_models},
        "6": { "name": "Test decision tree model", "func": manager.run_test_decision_tree_model},
        "7": { "name": "Test random forest model", "func": manager.run_test_decision_tree_model},
        "8": { "name": "Test all models", "func": manager.run_test_all_models },
        "q": { "name": "Quit", "func": lambda: exit(0) },
    }
    print("Cyber Threat Detector interactive workflow.\nPress Ctrl+C to exit at any time.")
    while True:
        _render_menu(actions)
        user_choice = input("Enter an option ❯ ").strip().lower()
        if user_choice not in actions.keys():
            print("Invalid selection.")
            continue
        actions[user_choice]["func"]()