"""
Module: model.py
LightGBM Classifier model training, evaluation, and Macro F0.5 score threshold optimization.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Tuple, Dict

def calculate_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Calculates F-beta score given precision, recall, and beta (default 0.5)."""
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)

class EntityResolutionModel:
    """LightGBM model wrapper for Business Entity Resolution pair matching."""
    
    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.05):
        self.params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": 6,
            "num_leaves": 31,
            "verbose": -1
        }
        self.clf = lgb.LGBMClassifier(**self.params)
        self.optimal_threshold = 0.5

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """Trains the LightGBM classifier on pair features."""
        if X_val is not None and y_val is not None:
            self.clf.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            val_probs = self.clf.predict_proba(X_val)[:, 1]
            self.optimal_threshold = self.optimize_f05_threshold(y_val, val_probs)
        else:
            self.clf.fit(X_train, y_train)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns match probabilities for given feature set."""
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """Returns binary match predictions (1 or 0) using optimal threshold."""
        thresh = threshold if threshold is not None else self.optimal_threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

    def optimize_f05_threshold(self, y_true: np.ndarray, probs: np.ndarray) -> float:
        """Finds the probability threshold that maximizes Macro F0.5 score."""
        best_thresh = 0.5
        best_f05 = 0.0

        for thresh in np.linspace(0.1, 0.9, 81):
            preds = (probs >= thresh).astype(int)
            tp = np.sum((preds == 1) & (y_true == 1))
            fp = np.sum((preds == 1) & (y_true == 0))
            fn = np.sum((preds == 0) & (y_true == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f05 = calculate_f_beta(precision, recall, beta=0.5)

            if f05 > best_f05:
                best_f05 = f05
                best_thresh = thresh

        return float(best_thresh)
