# src/features/url_features.py
"""
Lexical URL feature extraction for PhreshPhish classifier.
"""

import re
from typing import Dict, Any
from urllib.parse import urlparse


def extract_url_features(url: str) -> Dict[str, Any]:
    """
    Extract lightweight, fast lexical features from a URL.

    Returns a dictionary of numeric features that can be used
    by classical ML models (Logistic Regression, XGBoost, etc.).
    """
    defaults = {
        "url_len": 0,
        "domain_len": 0,
        "path_len": 0,
        "num_dots": 0,
        "num_hyphens": 0,
        "num_underscores": 0,
        "num_digits": 0,
        "num_params": 0,
        "has_https": 0,
        "has_ip": 0,
        "num_subdomains": 0,
        "num_slashes": 0,
        "has_at_symbol": 0,
        "has_double_slash_redirect": 0,
    }

    try:
        full = str(url).lower().strip()
        if not full:
            return defaults

        parsed = urlparse(full)
        netloc = parsed.netloc
        path = parsed.path
        query = parsed.query

        features = {
            "url_len": len(full),
            "domain_len": len(netloc),
            "path_len": len(path),
            "num_dots": full.count("."),
            "num_hyphens": full.count("-"),
            "num_underscores": full.count("_"),
            "num_digits": sum(c.isdigit() for c in full),
            "num_params": full.count("="),
            "has_https": int(full.startswith("https")),
            "has_ip": int(bool(re.search(r"(?:\d{1,3}\.){3}\d{1,3}", netloc))),
            "num_subdomains": max(0, netloc.count(".") - 1) if netloc else 0,
            "num_slashes": full.count("/"),
            "has_at_symbol": int("@" in full),
            "has_double_slash_redirect": int("//" in full[full.find("://")+3:] if "://" in full else False),
        }
        return features

    except Exception:
        return defaults


def url_features_as_list(url: str) -> list:
    """
    Convenience helper that returns features as a fixed-order list
    (useful for some scikit-learn pipelines).
    """
    feats = extract_url_features(url)
    order = [
        "url_len", "domain_len", "path_len", "num_dots", "num_hyphens",
        "num_underscores", "num_digits", "num_params", "has_https",
        "has_ip", "num_subdomains", "num_slashes", "has_at_symbol",
        "has_double_slash_redirect",
    ]
    return [feats[k] for k in order]