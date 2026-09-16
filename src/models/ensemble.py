# src/models/ensemble.py
"""
Simple ensemble of Classical ML (XGBoost) + Transformer
for lower false-positive rate.
"""

from typing import Dict, Any, List, Union, Optional
import numpy as np
import pandas as pd

from .ml_models import load_ml_bundle, predict_ml, NUM_FEATURES
from .dl_models import load_transformer, predict_transformer


class EnsembleClassifier:
    """
    Combines XGBoost + Transformer predictions.
    
    Strategy:
    - Average the probabilities (soft voting)
    - Or require both models to agree on "phish" (conservative / low-FP)
    """

    def __init__(
        self,
        ml_bundle_path: str,
        dl_model_dir: str,
        strategy: str = "average",          # "average" | "and" | "or"
        final_threshold: Optional[float] = None,
    ):
        self.ml_bundle = load_ml_bundle(ml_bundle_path)
        self.dl_bundle = load_transformer(dl_model_dir)
        self.strategy = strategy

        # If no final threshold given, use the average of both thresholds
        if final_threshold is None:
            self.final_threshold = (
                self.ml_bundle["threshold"] + self.dl_bundle["threshold"]
            ) / 2
        else:
            self.final_threshold = final_threshold

        self.id2label = {0: "benign", 1: "phish"}

    def predict(
        self,
        url: str,
        text: str,
        numerical_features: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Predict a single example.

        Parameters
        ----------
        url : str
        text : str
            Cleaned visible text (or already combined input_text)
        numerical_features : dict, optional
            Pre-computed URL + length features.
            If None, a minimal set is created (less accurate).
        """
        # ----- Transformer path -----
        input_text = f"{url} [SEP] {text}"
        dl_result = predict_transformer(self.dl_bundle, input_text)[0]
        dl_prob = dl_result["probability"]

        # ----- Classical ML path -----
        if numerical_features is None:
            # Minimal fallback (better to pass real features from preprocessor)
            numerical_features = {
                "url_len": len(url),
                "domain_len": 0,
                "path_len": 0,
                "num_dots": url.count("."),
                "num_hyphens": url.count("-"),
                "num_underscores": url.count("_"),
                "num_digits": sum(c.isdigit() for c in url),
                "num_params": url.count("="),
                "has_https": int(url.startswith("https")),
                "has_ip": 0,
                "num_subdomains": 0,
                "text_len": len(text),
                "html_len_raw": 0,
            }

        # Build a single-row DataFrame expected by the preprocessor
        row = {**numerical_features, "input_text": input_text}
        X = pd.DataFrame([row])

        # Ensure all expected columns exist
        for col in NUM_FEATURES:
            if col not in X.columns:
                X[col] = 0.0

        _, ml_probs = predict_ml(self.ml_bundle, X)
        ml_prob = float(ml_probs[0])

        # ----- Combine -----
        if self.strategy == "average":
            final_prob = (ml_prob + dl_prob) / 2
        elif self.strategy == "and":
            # Conservative: only high confidence if both agree
            final_prob = min(ml_prob, dl_prob)
        elif self.strategy == "or":
            final_prob = max(ml_prob, dl_prob)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        label_id = 1 if final_prob >= self.final_threshold else 0

        return {
            "label": self.id2label[label_id],
            "probability": round(final_prob, 4),
            "ml_probability": round(ml_prob, 4),
            "dl_probability": round(dl_prob, 4),
            "threshold": self.final_threshold,
            "strategy": self.strategy,
            "model": "ensemble",
        }

    def predict_batch(
        self,
        urls: List[str],
        texts: List[str],
        numerical_features_list: Optional[List[Dict[str, float]]] = None,
    ) -> List[Dict[str, Any]]:
        """Batch prediction (simple loop for clarity)."""
        results = []
        for i, (url, text) in enumerate(zip(urls, texts)):
            feats = None
            if numerical_features_list is not None:
                feats = numerical_features_list[i]
            results.append(self.predict(url, text, feats))
        return results