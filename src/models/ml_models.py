# src/models/ml_models.py
"""
Classical ML models (XGBoost + helpers) for PhreshPhish classifier.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


# Default numerical features used during training
NUM_FEATURES = [
    "url_len", "domain_len", "path_len", "num_dots", "num_hyphens",
    "num_underscores", "num_digits", "num_params", "has_https",
    "has_ip", "num_subdomains", "text_len", "html_len_raw"
]

TEXT_COL = "input_text"


def build_preprocessor(
    max_tfidf_features: int = 80_000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> ColumnTransformer:
    """Create the same preprocessor used in training."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("tfidf", TfidfVectorizer(
                max_features=max_tfidf_features,
                ngram_range=ngram_range,
                min_df=3,
                max_df=0.95,
                sublinear_tf=True,
                dtype=np.float32,
            ), TEXT_COL),
        ],
        remainder="drop",
        n_jobs=-1,
    )


def train_xgboost(
    X: pd.DataFrame,
    y: np.ndarray,
    n_estimators: int = 400,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier.
    Expects X to already contain the numerical + text columns.
    """
    neg, pos = np.bincount(y)
    scale_pos_weight = neg / max(pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=random_state,
    )
    return model


def save_ml_bundle(
    preprocessor: ColumnTransformer,
    model: Any,
    threshold: float,
    save_dir: str | Path,
    label2id: Optional[Dict[str, int]] = None,
) -> Path:
    """Save preprocessor + model + metadata as a single joblib bundle."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "preprocessor": preprocessor,
        "model": model,
        "threshold": float(threshold),
        "label2id": label2id or {"benign": 0, "phish": 1},
        "id2label": {0: "benign", 1: "phish"},
        "feature_names": NUM_FEATURES,
        "model_type": "xgboost",
    }
    path = save_dir / "ml_bundle.joblib"
    joblib.dump(bundle, path)
    return path


def load_ml_bundle(bundle_path: str | Path) -> Dict[str, Any]:
    """Load a previously saved ML bundle."""
    return joblib.load(bundle_path)


def predict_ml(
    bundle: Dict[str, Any],
    X: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run prediction using a loaded ML bundle.
    Returns (labels, probabilities).
    """
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]
    threshold = bundle.get("threshold", 0.5)

    X_t = preprocessor.transform(X)
    probs = model.predict_proba(X_t)[:, 1]
    labels = (probs >= threshold).astype(int)
    return labels, probs