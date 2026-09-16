# app/inference.py
"""
Production inference logic for the PhreshPhish classifier.
Handles Classical ML, Transformer, and Ensemble predictions.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import sys

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.data.preprocessor import clean_html, extract_url_features
from src.models.ml_models import NUM_FEATURES, predict_ml
from src.models.dl_models import load_transformer, predict_transformer
from src.models.ensemble import EnsembleClassifier


class PhishingPredictor:
    """
    High-level predictor used by the Gradio / FastAPI app.
    """

    def __init__(
        self,
        prefer: str = "transformer",          # "transformer" | "ml" | "ensemble"
        ml_bundle_path: str = "models/exported/ml_bundle.joblib",
        dl_model_dir: str = "models/dl/best_model",
        ensemble_strategy: str = "average",
    ):
        self.prefer = prefer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ----- Classical ML -----
        self.ml_bundle = None
        ml_path = Path(ml_bundle_path)
        if ml_path.exists():
            self.ml_bundle = joblib.load(ml_path)
            print(f"✓ Loaded ML bundle from {ml_path}")
        else:
            print(f"⚠ ML bundle not found at {ml_path}")

        # ----- Transformer -----
        self.dl_bundle = None
        dl_path = Path(dl_model_dir)
        if dl_path.exists():
            self.dl_bundle = load_transformer(dl_path, device=self.device)
            print(f"✓ Loaded Transformer from {dl_path}")
        else:
            print(f"⚠ Transformer model not found at {dl_path}")

        # ----- Ensemble (optional) -----
        self.ensemble = None
        if prefer == "ensemble" and self.ml_bundle and self.dl_bundle:
            self.ensemble = EnsembleClassifier(
                ml_bundle_path=str(ml_path),
                dl_model_dir=str(dl_path),
                strategy=ensemble_strategy,
            )
            print("✓ Ensemble ready")

        if not any([self.ml_bundle, self.dl_bundle]):
            raise RuntimeError("No models could be loaded. Check paths.")

    def _prepare_features(self, url: str, html: str) -> Dict[str, Any]:
        """Clean HTML + extract URL features."""
        clean_text = clean_html(html or "")
        url_feats = extract_url_features(url)

        input_text = f"{url} [SEP] {clean_text}".strip()[:4500]

        features = {
            **url_feats,
            "text_len": len(clean_text),
            "html_len_raw": len(html or ""),
            "input_text": input_text,
            "text": clean_text,
        }
        return features

    def predict(self, url: str, html: str = "") -> Dict[str, Any]:
        """
        Main prediction entry point.
        Returns a dictionary with label, probability, model used, etc.
        """
        features = self._prepare_features(url, html)
        input_text = features["input_text"]

        # ---------- Ensemble ----------
        if self.prefer == "ensemble" and self.ensemble is not None:
            return self.ensemble.predict(
                url=url,
                text=features["text"],
                numerical_features={k: features[k] for k in NUM_FEATURES if k in features},
            )

        # ---------- Transformer (preferred) ----------
        if self.prefer == "transformer" and self.dl_bundle is not None:
            result = predict_transformer(self.dl_bundle, input_text)[0]
            return result

        # ---------- Classical ML fallback ----------
        if self.ml_bundle is not None:
            row = {k: features.get(k, 0) for k in NUM_FEATURES}
            row["input_text"] = input_text
            X = pd.DataFrame([row])

            labels, probs = predict_ml(self.ml_bundle, X)
            prob = float(probs[0])
            threshold = self.ml_bundle.get("threshold", 0.7)
            label = "phish" if prob >= threshold else "benign"

            return {
                "label": label,
                "probability": round(prob, 4),
                "threshold": threshold,
                "model": "xgboost",
            }

        raise RuntimeError("No model available for prediction.")