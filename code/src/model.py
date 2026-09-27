"""
Module: model.py
LightGBM + CatBoost + XGBoost Multi-Model Stacking & Macro F0.5 Threshold Optimization.
Optimized for Competition Leaderboard Maximization.
"""
import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Tuple, Dict, Any, List, Set

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

def calculate_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Calculates F-beta score given precision, recall, and beta (default 0.5)."""
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)

class EntityResolutionModel:
    """Multi-Model Gradient Boosting Ensemble for Business Entity Resolution pair matching."""
    
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
        
        if use_gpu:
            self.params["device"] = "gpu"
        else:
            self.params["device"] = "cpu"

        self.clf = lgb.LGBMClassifier(**self.params)
        self.cb_clf = None
        self.xgb_clf = None
        self.optimal_threshold = 0.5
        self.val_metrics = {}

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None, val_groups: np.ndarray = None, val_entities: List[str] = None, val_pairs: List[Any] = None, gt_map: Dict[str, Set[str]] = None):
        """Trains LightGBM (and optional CatBoost & XGBoost) on pair features with early stopping."""
        # 1. Train LightGBM
        def _fit_lgb():
            if X_val is not None and y_val is not None:
                self.clf.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
            else:
                self.clf.fit(X_train, y_train)

        try:
            _fit_lgb()
        except lgb.basic.LightGBMError as e:
            if "OpenCL" in str(e) or "GPU" in str(e) or "gpu" in str(e):
                print("\n[Notice] Falling back LightGBM to CPU...", flush=True)
                self.params["device"] = "cpu"
                self.clf = lgb.LGBMClassifier(**self.params)
                _fit_lgb()
            else:
                raise e

        # 2. Train CatBoost (if available)
        if HAS_CATBOOST:
            try:
                print("Training CatBoost Classifier for Multi-Model Stacking...", flush=True)
                self.cb_clf = cb.CatBoostClassifier(
                    iterations=300,
                    learning_rate=0.06,
                    depth=6,
                    loss_function="Logloss",
                    eval_metric="Logloss",
                    random_seed=42,
                    verbose=0,
                    thread_count=-1
                )
                if X_val is not None and y_val is not None:
                    self.cb_clf.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30, verbose=False)
                else:
                    self.cb_clf.fit(X_train, y_train, verbose=False)
                print("CatBoost trained successfully.", flush=True)
            except Exception as e:
                print(f"[Notice] CatBoost training skipped ({e})", flush=True)
                self.cb_clf = None

        # 3. Train XGBoost (if available)
        if HAS_XGBOOST:
            try:
                print("Training XGBoost Classifier for Multi-Model Stacking...", flush=True)
                self.xgb_clf = xgb.XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.06,
                    max_depth=6,
                    eval_metric="logloss",
                    n_jobs=-1,
                    random_state=42
                )
                if X_val is not None and y_val is not None:
                    self.xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    self.xgb_clf.fit(X_train, y_train, verbose=False)
                print("XGBoost trained successfully.", flush=True)
            except Exception as e:
                print(f"[Notice] XGBoost training skipped ({e})", flush=True)
                self.xgb_clf = None

        # 4. Tune Macro F0.5 Threshold on Ensembled Validation Probabilities
        if X_val is not None and y_val is not None:
            val_probs = self.predict_proba(X_val)
            if val_entities is not None and val_pairs is not None and gt_map is not None:
                self.optimal_threshold = self.optimize_end_to_end_f05(val_entities, val_pairs, val_probs, gt_map)
            else:
                self.optimal_threshold = self.optimize_f05_threshold(y_val, val_probs, groups=val_groups)


    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns ensemble blend of match probabilities across available models."""
        preds = []
        weights = []

        # LightGBM
        if self.clf is not None:
            if hasattr(self.clf, "booster_"):
                p_lgb = self.clf.booster_.predict(X, num_threads=1)
            else:
                p_lgb = self.clf.predict_proba(X)[:, 1]
            preds.append(p_lgb)
            weights.append(0.50 if (self.cb_clf or self.xgb_clf) else 1.0)

        # CatBoost
        if self.cb_clf is not None:
            p_cb = self.cb_clf.predict_proba(X)[:, 1]
            preds.append(p_cb)
            weights.append(0.30)

        # XGBoost
        if self.xgb_clf is not None:
            p_xgb = self.xgb_clf.predict_proba(X)[:, 1]
            preds.append(p_xgb)
            weights.append(0.20)

        total_w = sum(weights)
        norm_w = [w / total_w for w in weights]
        blend = sum(p * w for p, w in zip(preds, norm_w))
        return blend

    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """Returns binary match predictions (1 or 0) using optimal threshold."""
        thresh = threshold if threshold is not None else self.optimal_threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

    def optimize_end_to_end_f05(self, val_entities: List[str], val_pairs: List[Any], probs: np.ndarray, gt_map: Dict[str, Set[str]]) -> float:
        """
        Directly optimizes probability threshold to maximize the OFFICIAL Competition Macro F0.5 Metric:
        - Computed per Source 1 entity across ALL validation entities (including singletons and missed candidates).
        - Correctly identified singletons (GT=0, Pred=0) receive 1.0.
        - False merges on singletons (GT=0, Pred>0) receive 0.0.
        - Missed blocking candidates are counted in Recall = TP / |GT|.
        """
        ent_to_idx = {e: i for i, e in enumerate(val_entities)}
        n_ents = len(val_entities)
        gt_counts = np.array([len(gt_map.get(e, set())) for e in val_entities], dtype=float)

        if len(val_pairs) == 0:
            return 0.5

        pair_ents = np.array([ent_to_idx[p[0]] for p in val_pairs])
        pair_is_gt = np.array([p[1] in gt_map.get(p[0], set()) for p in val_pairs])

        best_thresh = 0.5
        best_macro = 0.0

        for thresh in np.linspace(0.2, 0.85, 131):
            mask = (probs >= thresh)
            tp_pairs = mask & pair_is_gt
            fp_pairs = mask & (~pair_is_gt)

            tp_g = np.bincount(pair_ents, weights=tp_pairs, minlength=n_ents)
            fp_g = np.bincount(pair_ents, weights=fp_pairs, minlength=n_ents)

            p_den = tp_g + fp_g
            prec_g = np.divide(tp_g, p_den, out=np.zeros_like(tp_g, dtype=float), where=p_den != 0)
            rec_g = np.divide(tp_g, gt_counts, out=np.zeros_like(tp_g, dtype=float), where=gt_counts != 0)

            f_den = 0.25 * prec_g + rec_g
            f05_g = np.divide(1.25 * prec_g * rec_g, f_den, out=np.zeros_like(prec_g), where=f_den != 0)

            # Official Competition Singleton Rule
            singleton_mask = (gt_counts == 0)
            f05_g[singleton_mask & (fp_g == 0)] = 1.0
            f05_g[singleton_mask & (fp_g > 0)] = 0.0

            macro_val = float(np.mean(f05_g))
            if macro_val > best_macro:
                best_macro = macro_val
                best_thresh = thresh

        return float(best_thresh)

    def optimize_f05_threshold(self, y_true: np.ndarray, probs: np.ndarray, groups: np.ndarray = None) -> float:

        """Finds the probability threshold that maximizes Macro F0.5 score (Official Competition Metric)."""
        best_thresh = 0.5
        best_f05 = 0.0

        if groups is None:
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
            tp_g = np.bincount(group_idx, weights=(preds == 1) & (y_true == 1), minlength=n_groups)
            fp_g = np.bincount(group_idx, weights=(preds == 1) & (y_true == 0), minlength=n_groups)
            fn_g = np.bincount(group_idx, weights=(preds == 0) & (y_true == 1), minlength=n_groups)

            p_den = tp_g + fp_g
            r_den = tp_g + fn_g
            prec_g = np.divide(tp_g, p_den, out=np.zeros_like(tp_g, dtype=float), where=p_den != 0)
            rec_g = np.divide(tp_g, r_den, out=np.zeros_like(tp_g, dtype=float), where=r_den != 0)

            f_den = 0.25 * prec_g + rec_g
            f05_g = np.divide(1.25 * prec_g * rec_g, f_den, out=np.zeros_like(prec_g), where=f_den != 0)
            f05_g[(tp_g == 0) & (fp_g == 0) & (fn_g == 0)] = 1.0
            macro_f05 = np.mean(f05_g)

            if macro_f05 > best_f05:
                best_f05 = macro_f05
                best_thresh = thresh

        return float(best_thresh)

    def save(self, filepath: str):
        """Saves trained models, threshold, and validation metrics to disk."""
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        save_dict = {
            "clf": self.clf,
            "cb_clf": getattr(self, "cb_clf", None),
            "xgb_clf": getattr(self, "xgb_clf", None),
            "optimal_threshold": self.optimal_threshold,
            "val_metrics": getattr(self, "val_metrics", {})
        }
        with open(filepath, "wb") as f:
            pickle.dump(save_dict, f)
        print(f"Model successfully saved to {filepath}", flush=True)

    @classmethod
    def load(cls, filepath: str):
        """Loads trained models, threshold, and metrics from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        model = cls()
        model.clf = data.get("clf")
        model.cb_clf = data.get("cb_clf")
        model.xgb_clf = data.get("xgb_clf")
        model.optimal_threshold = data.get("optimal_threshold", 0.5)
        model.val_metrics = data.get("val_metrics", {})
        models_loaded = ["LightGBM"]
        if model.cb_clf is not None:
            models_loaded.append("CatBoost")
        if model.xgb_clf is not None:
            models_loaded.append("XGBoost")
        print(f"Loaded trained models {models_loaded} from {filepath} (optimal_threshold={model.optimal_threshold:.3f})", flush=True)
        return model
