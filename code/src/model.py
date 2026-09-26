"""
Module: model.py
LightGBM Classifier model training, evaluation, and Macro F0.5 score threshold optimization.
Supports GPU acceleration.
"""
import os
import pickle
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
    
    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05, use_gpu: bool = False):
        self.params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": 8,
            "num_leaves": 63,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
            "verbose": -1,
            "n_jobs": -1
        }
        
        # Configure device (CPU by default with all cores; GPU if requested)
        if use_gpu:
            self.params["device"] = "gpu"
        else:
            self.params["device"] = "cpu"

        self.clf = lgb.LGBMClassifier(**self.params)
        self.optimal_threshold = 0.5

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None, val_groups: np.ndarray = None):
        """Trains the LightGBM classifier on pair features with automatic fallback to CPU if GPU/OpenCL is unavailable."""
        def _fit_model():
            if X_val is not None and y_val is not None:
                self.clf.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
            else:
                self.clf.fit(X_train, y_train)

        try:
            _fit_model()
        except lgb.basic.LightGBMError as e:
            if "OpenCL" in str(e) or "GPU" in str(e) or "gpu" in str(e):
                print("\n[Notice] No OpenCL/GPU device detected for LightGBM. Automatically falling back to multi-core CPU...", flush=True)
                self.params["device"] = "cpu"
                self.clf = lgb.LGBMClassifier(**self.params)
                _fit_model()
            else:
                raise e

        if X_val is not None and y_val is not None:
            val_probs = self.clf.predict_proba(X_val)[:, 1]
            self.optimal_threshold = self.optimize_f05_threshold(y_val, val_probs, groups=val_groups)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns match probabilities for given feature set."""
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """Returns binary match predictions (1 or 0) using optimal threshold."""
        thresh = threshold if threshold is not None else self.optimal_threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

    def optimize_f05_threshold(self, y_true: np.ndarray, probs: np.ndarray, groups: np.ndarray = None) -> float:
        """Finds the probability threshold that maximizes Macro F0.5 score (Official Competition Metric)."""
        best_thresh = 0.5
        best_f05 = 0.0

        if groups is None:
            # Fallback to Global (Micro) F0.5 if no groups provided
            for thresh in np.linspace(0.2, 0.85, 131):
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

        # Ultra-fast NumPy vectorized Macro F0.5 calculation
        _, group_idx = np.unique(groups, return_inverse=True)
        n_groups = len(np.unique(group_idx))

        for thresh in np.linspace(0.2, 0.85, 131):
            preds = (probs >= thresh).astype(int)
            
            # Boolean masks
            tp_mask = (preds == 1) & (y_true == 1)
            fp_mask = (preds == 1) & (y_true == 0)
            fn_mask = (preds == 0) & (y_true == 1)
            
            # Count TP, FP, FN per group
            tp_g = np.bincount(group_idx, weights=tp_mask, minlength=n_groups)
            fp_g = np.bincount(group_idx, weights=fp_mask, minlength=n_groups)
            fn_g = np.bincount(group_idx, weights=fn_mask, minlength=n_groups)
            
            p_den = tp_g + fp_g
            r_den = tp_g + fn_g
            
            # Precision and Recall per group (safe division)
            precision_g = np.divide(tp_g, p_den, out=np.zeros_like(tp_g, dtype=float), where=p_den!=0)
            recall_g = np.divide(tp_g, r_den, out=np.zeros_like(tp_g, dtype=float), where=r_den!=0)
            
            # F0.5 per group (safe division)
            f_den = 0.25 * precision_g + recall_g
            f05_g = np.divide(1.25 * precision_g * recall_g, f_den, out=np.zeros_like(precision_g), where=f_den!=0)
            
            # Official Competition Rule for Singletons:
            # If an entity has 0 true matches (tp_g == 0 and fn_g == 0),
            # and model correctly predicted 0 matches (fp_g == 0), it receives 1.0!
            singleton_perfect = (tp_g == 0) & (fp_g == 0) & (fn_g == 0)
            f05_g[singleton_perfect] = 1.0
            
            # Macro Average across all entities
            macro_f05 = np.mean(f05_g)
            
            if macro_f05 > best_f05:
                best_f05 = macro_f05
                best_thresh = thresh

        return float(best_thresh)

    def save(self, filepath: str):
        """Saves trained model, threshold, and validation metrics to disk."""
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        save_dict = {
            "clf": self.clf,
            "optimal_threshold": self.optimal_threshold,
            "val_metrics": getattr(self, "val_metrics", {})
        }
        with open(filepath, "wb") as f:
            pickle.dump(save_dict, f)
        print(f"Model successfully saved to {filepath}", flush=True)

    @classmethod
    def load(cls, filepath: str):
        """Loads trained model, threshold, and metrics from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        model = cls()
        model.clf = data["clf"]
        model.optimal_threshold = data["optimal_threshold"]
        model.val_metrics = data.get("val_metrics", {})
        print(f"Loaded trained model from {filepath} (optimal_threshold={model.optimal_threshold:.3f})", flush=True)
        return model
