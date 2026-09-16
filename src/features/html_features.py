# src/features/html_features.py
"""
HTML structural and content feature extraction for PhreshPhish.
Designed to be memory-safe (works on truncated HTML).
"""

import re
from typing import Dict, Any
from bs4 import BeautifulSoup


def extract_html_features(
    html: str,
    max_chars: int = 30_000,
) -> Dict[str, Any]:
    """
    Extract lightweight structural features from HTML.
    Truncates input early to avoid memory issues.
    """
    defaults = {
        "html_len": 0,
        "num_tags": 0,
        "num_links": 0,
        "num_forms": 0,
        "num_inputs": 0,
        "num_images": 0,
        "num_scripts": 0,
        "num_meta": 0,
        "has_password_input": 0,
        "has_email_input": 0,
        "has_login_form": 0,
        "title_len": 0,
        "num_external_links": 0,
        "num_hidden_inputs": 0,
    }

    if not isinstance(html, str) or not html.strip():
        return defaults

    # Hard truncate for safety
    html = html[:max_chars]
    html_len = len(html)

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            defaults["html_len"] = html_len
            return defaults

    # Basic counts
    all_tags = soup.find_all(True)
    num_tags = len(all_tags)

    links = soup.find_all("a", href=True)
    forms = soup.find_all("form")
    inputs = soup.find_all("input")
    images = soup.find_all("img")
    scripts = soup.find_all("script")
    metas = soup.find_all("meta")

    # Password / email / hidden inputs
    has_password = 0
    has_email = 0
    num_hidden = 0
    for inp in inputs:
        input_type = (inp.get("type") or "").lower()
        input_name = (inp.get("name") or "").lower()
        if input_type == "password" or "password" in input_name:
            has_password = 1
        if input_type == "email" or "email" in input_name or "user" in input_name:
            has_email = 1
        if input_type == "hidden":
            num_hidden += 1

    # Simple login-form heuristic
    has_login_form = 0
    for form in forms:
        form_html = str(form).lower()
        if any(kw in form_html for kw in ["login", "signin", "sign-in", "password", "credential"]):
            has_login_form = 1
            break

    # Title length
    title_tag = soup.find("title")
    title_len = len(title_tag.get_text(strip=True)) if title_tag else 0

    # External links (very rough)
    num_external = 0
    for a in links:
        href = a.get("href", "")
        if href.startswith("http") or href.startswith("//"):
            num_external += 1

    features = {
        "html_len": html_len,
        "num_tags": num_tags,
        "num_links": len(links),
        "num_forms": len(forms),
        "num_inputs": len(inputs),
        "num_images": len(images),
        "num_scripts": len(scripts),
        "num_meta": len(metas),
        "has_password_input": has_password,
        "has_email_input": has_email,
        "has_login_form": has_login_form,
        "title_len": title_len,
        "num_external_links": num_external,
        "num_hidden_inputs": num_hidden,
    }
    return features


def html_features_as_list(html: str, max_chars: int = 30_000) -> list:
    """
    Return features as a fixed-order list (useful for some pipelines).
    """
    feats = extract_html_features(html, max_chars=max_chars)
    order = [
        "html_len", "num_tags", "num_links", "num_forms", "num_inputs",
        "num_images", "num_scripts", "num_meta", "has_password_input",
        "has_email_input", "has_login_form", "title_len",
        "num_external_links", "num_hidden_inputs",
    ]
    return [feats[k] for k in order]