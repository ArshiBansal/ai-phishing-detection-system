# scripts/train.py
"""
CLI entry point for training the PhreshPhish classifier.
Supports both Classical ML (XGBoost) and Transformer fine-tuning.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.helpers import set_seed, ensure_dir, load_yaml_config, get_device
from src.models.ml_models import (
    build_preprocessor,
    train_xgboost,
    save_ml_bundle,
    NUM_FEATURES,
)
from src.evaluation.metrics import find_threshold_for_recall, evaluate_predictions


def train_classical(config: dict, data_path: Path, output_dir: Path):
    """Train XGBoost baseline and save the bundle."""
    print("=" * 60)
    print("Training Classical ML (XGBoost)")
    print("=" * 60)

    df = pd.read_parquet(data_path)
    print(f"Loaded data: {df.shape}")

    X = df[NUM_FEATURES + ["input_text"]]
    y = df["label_id"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=config.get("seed", 42)
    )

    print("Building preprocessor...")
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    print("Training XGBoost...")
    model = train_xgboost(
        X_train, y_train,  # note: train_xgboost expects original X for now
        n_estimators=config.get("n_estimators", 400),
        max_depth=config.get("max_depth", 6),
        learning_rate=config.get("learning_rate", 0.05),
        random_state=config.get("seed", 42),
    )
    # We need to fit on the transformed data
    model.fit(X_train_t, y_train)

    # Find a good threshold for low FPR
    y_prob = model.predict_proba(X_val_t)[:, 1]
    threshold = find_threshold_for_recall(y_val, y_prob, target_recall=0.92)
    print(f"Chosen threshold (≈92% recall): {threshold:.4f}")

    evaluate_predictions(y_val, y_prob, threshold=threshold, name="XGBoost")

    # Save
    ensure_dir(output_dir)
    save_path = save_ml_bundle(
        preprocessor=preprocessor,
        model=model,
        threshold=threshold,
        save_dir=output_dir,
    )
    print(f"✓ Model saved to {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Train PhreshPhish classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="processed/phreshphish_processed.parquet",
        help="Path to processed parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/ml",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["ml", "dl"],
        default="ml",
        help="Which model to train (ml = XGBoost, dl = Transformer)",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        config = load_yaml_config(config_path)
        print(f"Loaded config from {config_path}")
    else:
        print(f"Config not found at {config_path} – using defaults")

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {data_path}\n"
            "Please run the preprocessing notebook first."
        )

    output_dir = Path(args.output)

    if args.model_type == "ml":
        train_classical(config, data_path, output_dir)
    else:
        print("Transformer training is handled in the notebook 04_dl_nlp_training.ipynb")
        print("You can also extend this script later to call the Trainer programmatically.")


if __name__ == "__main__":
    main()