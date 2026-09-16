# src/models/dl_models.py
"""
Transformer (Deep Learning) helpers for PhreshPhish classifier.
"""

from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)


def load_transformer(
    model_dir: str | Path,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load a fine-tuned Transformer model + tokenizer + metadata.

    Returns a dictionary containing:
        - model
        - tokenizer
        - threshold
        - max_length
        - id2label / label2id
        - device
    """
    model_dir = Path(model_dir)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Load meta if present
    meta_path = model_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        # fallback defaults
        meta = {
            "threshold": 0.70,
            "max_length": 256,
            "label2id": {"benign": 0, "phish": 1},
            "id2label": {0: "benign", 1: "phish"},
        }

    return {
        "model": model,
        "tokenizer": tokenizer,
        "threshold": float(meta.get("threshold", 0.70)),
        "max_length": int(meta.get("max_length", 256)),
        "label2id": meta.get("label2id", {"benign": 0, "phish": 1}),
        "id2label": meta.get("id2label", {0: "benign", 1: "phish"}),
        "device": device,
        "model_dir": str(model_dir),
    }


def predict_transformer(
    bundle: Dict[str, Any],
    texts: Union[str, List[str]],
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """
    Run inference with a loaded Transformer bundle.

    Parameters
    ----------
    bundle : dict
        Output of load_transformer()
    texts : str or list of str
        Input text(s). Should already be in the format "url [SEP] cleaned_html"
    batch_size : int
        Batch size for inference

    Returns
    -------
    list of dicts with keys: label, probability, threshold, model
    """
    if isinstance(texts, str):
        texts = [texts]

    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    threshold = bundle["threshold"]
    max_length = bundle["max_length"]
    id2label = bundle["id2label"]
    device = bundle["device"]

    results = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            encodings = tokenizer(
                batch,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)

            outputs = model(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()

            for prob in probs:
                label_id = 1 if prob >= threshold else 0
                results.append({
                    "label": id2label[label_id],
                    "probability": float(round(prob, 4)),
                    "threshold": threshold,
                    "model": "transformer",
                })

    return results


def create_pipeline(
    model_dir: str | Path,
    device: Optional[int] = None,
):
    """
    Convenience helper that returns a Hugging Face text-classification pipeline.
    """
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        "text-classification",
        model=str(model_dir),
        tokenizer=str(model_dir),
        device=device,
        top_k=None,
    )