# tests/test_preprocessor.py
"""
Unit tests for the HTML / URL preprocessing utilities.
"""

import sys
from pathlib import Path

# Make src importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from src.data.preprocessor import clean_html, extract_url_features, process_example


# ----------------------------------------------------------------------
# clean_html
# ----------------------------------------------------------------------
def test_clean_html_basic():
    html = """
    <html>
      <head><title>Test Page</title><script>alert(1)</script></head>
      <body>
        <h1>Hello World</h1>
        <p>This is a paragraph.</p>
        <style>.x { color: red; }</style>
      </body>
    </html>
    """
    text = clean_html(html)
    assert "Hello World" in text
    assert "This is a paragraph" in text
    assert "alert(1)" not in text          # script removed
    assert "color: red" not in text        # style removed
    assert "<" not in text                 # no tags left


def test_clean_html_empty():
    assert clean_html("") == ""
    assert clean_html(None) == ""
    assert clean_html("   ") == ""


def test_clean_html_truncation():
    long_html = "<p>" + ("a" * 50_000) + "</p>"
    text = clean_html(long_html, max_html_chars=1000, max_text_chars=500)
    assert len(text) <= 500


# ----------------------------------------------------------------------
# extract_url_features
# ----------------------------------------------------------------------
def test_extract_url_features_https():
    url = "https://secure.example.com/login?user=test&id=123"
    feats = extract_url_features(url)

    assert feats["has_https"] == 1
    assert feats["url_len"] > 0
    assert feats["num_dots"] >= 2
    assert feats["num_params"] >= 1
    assert feats["has_ip"] == 0


def test_extract_url_features_ip():
    url = "http://192.168.1.1/admin"
    feats = extract_url_features(url)
    assert feats["has_ip"] == 1
    assert feats["has_https"] == 0


def test_extract_url_features_empty():
    feats = extract_url_features("")
    assert feats["url_len"] == 0
    assert feats["has_https"] == 0


# ----------------------------------------------------------------------
# process_example
# ----------------------------------------------------------------------
def test_process_example_full():
    example = {
        "url": "https://example.com/login",
        "html": "<html><body><h1>Login</h1><input type='password'></body></html>",
        "label": "phish",
        "target": "example",
        "lang": "en",
    }
    result = process_example(example)

    assert "input_text" in result
    assert "url" in result
    assert result["label"] == "phish"
    assert result["text_len"] > 0
    assert "Login" in result["text"]
    assert result["has_https"] == 1
    assert "[SEP]" in result["input_text"]


def test_process_example_missing_html():
    example = {
        "url": "https://example.com",
        "html": None,
        "label": "benign",
    }
    result = process_example(example)
    assert result["text"] == ""
    assert result["text_len"] == 0
    assert result["html_len_raw"] == 0


# ----------------------------------------------------------------------
# Run with: pytest tests/test_preprocessor.py -v
# ----------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])