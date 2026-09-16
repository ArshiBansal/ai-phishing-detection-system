# src/data/preprocessor.py
"""
HTML cleaning, text extraction and URL feature engineering
for the PhreshPhish classifier.
"""

import re
from typing import Dict, Any, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup


# Default limits (can be overridden)
MAX_HTML_CHARS = 30_000
MAX_TEXT_CHARS = 4_000


def clean_html(
    html: str,
    max_html_chars: int = MAX_HTML_CHARS,
    max_text_chars: int = MAX_TEXT_CHARS,
) -> str:
    """
    Aggressively clean and truncate HTML, returning visible text only.
    Designed to be memory-safe.
    """
    if not isinstance(html, str) or not html.strip():
        return ""

    # Hard truncate early
    html = html[:max_html_chars]

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        # Fallback to html.parser if lxml fails
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return ""

    # Remove noisy / non-visible tags
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "canvas", "template"]):
        tag.decompose()

    # Extract visible text
    text = soup.get_text(separator=" ", strip=True)

    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text[:max_text_chars]


def extract_url_features(url: str) -> Dict[str, Any]:
    """
    Extract lightweight lexical features from a URL.
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
    }

    try:
        full = str(url).lower().strip()
        if not full:
            return defaults

        parsed = urlparse(full)
        netloc = parsed.netloc
        path = parsed.path

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
            "has_ip": int(bool(re.search(r"\d+\.\d+\.\d+\.\d+", netloc))),
            "num_subdomains": max(0, netloc.count(".") - 1),
        }
        return features
    except Exception:
        return defaults


def process_example(
    example: Dict[str, Any],
    max_html_chars: int = MAX_HTML_CHARS,
    max_text_chars: int = MAX_TEXT_CHARS,
) -> Dict[str, Any]:
    """
    Full preprocessing pipeline for a single PhreshPhish example.
    Returns a clean dictionary ready for model input.
    """
    html_raw = example.get("html") or ""
    url = example.get("url") or ""

    clean_text = clean_html(
        html_raw,
        max_html_chars=max_html_chars,
        max_text_chars=max_text_chars,
    )
    url_feats = extract_url_features(url)

    # Combined text for Transformer / TF-IDF
    input_text = f"{url} [SEP] {clean_text}".strip()[:4500]

    return {
        "url": url,
        "label": example.get("label"),
        "target": example.get("target"),
        "lang": example.get("lang"),
        "text": clean_text,
        "text_len": len(clean_text),
        "html_len_raw": len(html_raw),
        "input_text": input_text,
        **url_feats,
    }