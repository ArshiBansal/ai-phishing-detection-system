# scripts/evaluate.py
"""
CLI evaluation script for the PhreshPhish classifier.
Evaluates Classical ML and/or Transformer models on a held-out set.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

from src.utils.helpers import set_seed, get_device
from src.models.ml_models import predict_ml, NUM_FEATURES
from src.models.dl_models import load_transformer, predict_transformer
from src.evaluation.metrics import evaluate_predictions, precision_at_recall


def evaluate_ml(bundle_path: Path, test_df: pd.DataFrame, threshold: float = None):
    print("\n" + "=" * 60)
    print("Evaluating Classical ML (XGBoost)")
    print("=" * 60)

    bundle = joblib.load(bundle_path)
    if threshold is None:
        threshold = bundle.get("threshold", 0.7)

    X = test_df[NUM_FEATURES + ["input_text"]]
    y = test_df["label_id"].values

    labels, probs = predict_ml(bundle, X)
    results = evaluate_predictions(
        y, probs, threshold=threshold, name="XGBoost"
    )
    return results


def evaluate_dl(model_dir: Path, test_df: pd.DataFrame, threshold: float = None):
    print("\n" + "=" * 60)
    print("Evaluating Transformer")
    print("=" * 60)

    bundle = load_transformer(model_dir)
    if threshold is None:
        threshold = bundle["threshold"]

    texts = test_df["input_text"].tolist()
    y = test_df["label_id"].values

    preds = predict_transformer(bundle, texts, batch_size=32)
    probs = np.array([p["probability"] for p in preds])

    results = evaluate_predictions(
        y, probs, threshold=threshold, name="Transformer"
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PhreshPhish models")
    parser.add_argument(
        "--data",
        type=str,
        default="processed/phreshphish_processed.parquet",
        help="Path to processed parquet file",
    )
    parser.add_argument(
        "--ml-bundle",
        type=str,
        default="models/exported/ml_bundle.joblib",
        help="Path to classical ML bundle",
    )
    parser.add_argument(
        "--dl-model",
        type=str,
        default="models/dl/best_model",
        help="Path to Transformer model directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["ml", "dl", "both"],
        default="both",
        help="Which model(s) to evaluate",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of data to use as test set",
    )
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()
    set_seed(args.seed)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    df = pd.read_parquet(data_path)
    print(f"Loaded data: {df.shape}")

    # Held-out test set (different seed from training)
    _, test_df = train_test_split(
        df,
        test_size=args.test_size,
        stratify=df["label_id"],
        random_state=args.seed,
    )
    print(f"Test size: {len(test_df)}")

    results = {}

    if args.model in ("ml", "both"):
        ml_path = Path(args.ml_bundle)
        if ml_path.exists():
            results["ml"] = evaluate_ml(ml_path, test_df)
        else:
            print(f"⚠ ML bundle not found: {ml_path}")

    if args.model in ("dl", "both"):
        dl_path = Path(args.dl_model)
        if dl_path.exists():
            results["dl"] = evaluate_dl(dl_path, test_df)
        else:
            print(f"⚠ Transformer model not found: {dl_path}")

    # Quick summary
    if results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for name, res in results.items():
            print(
                f"{res['name']:20s} | AP: {res['average_precision']:.4f} | "
                f"ROC-AUC: {res['roc_auc']:.4f}"
            )


if __name__ == "__main__":
    main()