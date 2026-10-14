"""
feature_extraction.py — Extract numerical features from a raw URL string.

THIS IS THE SINGLE SOURCE OF TRUTH for feature engineering in the entire
project.  The *exact same* function is used:

    * during training  — applied to every URL in the PhiUSIIL dataset, and
    * during inference — applied to the URL the user types into Streamlit.

This guarantees that training and prediction use identical features.

Limitations
-----------
* Only URL-structural features are computed; the model has no access to the
  actual webpage content (HTML, images, JavaScript, redirects, reputation).
* Individual features are NOT proof that a URL is malicious.  For example,
  many legitimate URLs use IP addresses, long paths, or query parameters.
* HTTPS does not guarantee that a website is trustworthy.
"""

import math
import re
import ipaddress
from collections import Counter
from urllib.parse import urlparse, parse_qs
from typing import List, Dict, Any

# ── Canonical feature ordering (used everywhere) ─────────────────────────────
# IMPORTANT: Never change the order of this list — it must match the order
# in which the model was trained. Add new features at the END only.
FEATURE_NAMES: List[str] = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "num_dots",
    "num_dots_hostname",
    "num_subdomains",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_at_signs",
    "num_question_marks",
    "num_equals",
    "num_ampersands",
    "num_digits",
    "num_alpha",
    "num_special_chars",
    "digit_ratio",
    "alpha_ratio",
    "special_char_ratio",
    "num_query_params",
    "num_percent_encoded",
    "is_ip_address",
    "is_https",
    "has_non_standard_port",
    "has_double_slash_redirect",
    "path_depth",
    "url_entropy",
    "hostname_entropy",
    "tld_length",
]

# Class-label mapping (used in evaluation and the Streamlit app)
# 0 = Phishing,  1 = Legitimate
CLASS_NAMES: Dict[int, str] = {0: "Phishing", 1: "Legitimate"}


# ── Helper functions ──────────────────────────────────────────────────────────

def _is_ip_address(hostname: str) -> int:
    """Return 1 if *hostname* is a valid IPv4 or IPv6 address, else 0."""
    if not hostname:
        return 0
    try:
        ipaddress.ip_address(hostname.strip("[]"))
        return 1
    except (ValueError, AttributeError):
        return 0


def _shannon_entropy(text: str) -> float:
    """Shannon entropy of *text* (bits per character).  Higher = more random."""
    if not text:
        return 0.0
    freq = Counter(text)
    length = len(text)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in freq.values()
    )


def _get_tld(hostname: str) -> str:
    """Return the top-level domain (last dot-separated segment)."""
    if "." not in hostname:
        return ""
    return hostname.rsplit(".", 1)[-1]


def _safe_parse(url: str):
    """Normalise and parse a URL, handling missing schemes gracefully."""
    url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", url):
        url = "http://" + url
    return urlparse(url)


# ── Core public function ──────────────────────────────────────────────────────

def extract_url_features(url: str) -> Dict[str, Any]:
    """
    Compute 30 numerical features from a raw URL string.

    Parameters
    ----------
    url : str
        Any URL — with or without a scheme (``http://`` / ``https://``).

    Returns
    -------
    dict
        Keys are exactly ``FEATURE_NAMES`` in canonical order.
        On any error (malformed/empty URL), all values are 0.
    """
    try:
        raw = url.strip()

        if not raw:
            return {name: 0 for name in FEATURE_NAMES}

        parsed = _safe_parse(raw)
        hostname = (parsed.hostname or "").lower()
        path = parsed.path or ""
        query = parsed.query or ""
        length = max(len(raw), 1)  # avoid division by zero

        # Character category counts (on the full raw URL)
        n_digits = sum(c.isdigit() for c in raw)
        n_alpha = sum(c.isalpha() for c in raw)
        n_special = sum(not c.isalnum() for c in raw)

        # Non-standard port detection
        try:
            port = parsed.port
            has_non_std_port = int(port is not None and port not in (80, 443))
        except ValueError:
            has_non_std_port = 0

        features = {
            "url_length":                len(raw),
            "hostname_length":           len(hostname),
            "path_length":               len(path),
            "query_length":              len(query),
            "num_dots":                  raw.count("."),
            "num_dots_hostname":         hostname.count("."),
            "num_subdomains":            max(hostname.count(".") - 1, 0),
            "num_hyphens":               raw.count("-"),
            "num_underscores":           raw.count("_"),
            "num_slashes":               raw.count("/"),
            "num_at_signs":              raw.count("@"),
            "num_question_marks":        raw.count("?"),
            "num_equals":                raw.count("="),
            "num_ampersands":            raw.count("&"),
            "num_digits":                n_digits,
            "num_alpha":                 n_alpha,
            "num_special_chars":         n_special,
            "digit_ratio":               round(n_digits / length, 6),
            "alpha_ratio":               round(n_alpha / length, 6),
            "special_char_ratio":        round(n_special / length, 6),
            "num_query_params":          len(parse_qs(query)),
            "num_percent_encoded":       len(re.findall(r"%[0-9a-fA-F]{2}", raw)),
            "is_ip_address":             _is_ip_address(hostname),
            "is_https":                  int((parsed.scheme or "").lower() == "https"),
            "has_non_standard_port":     has_non_std_port,
            "has_double_slash_redirect": int("//" in path),
            "path_depth":                len([p for p in path.split("/") if p]),
            "url_entropy":               round(_shannon_entropy(raw), 6),
            "hostname_entropy":          round(_shannon_entropy(hostname), 6),
            "tld_length":                len(_get_tld(hostname)),
        }

    except Exception:
        # Malformed URL — return zero dict so the pipeline never crashes
        features = {name: 0 for name in FEATURE_NAMES}

    return features


def extract_features_batch(urls, show_progress: bool = True):
    """
    Extract features for every URL in *urls* (a list or pandas Series).

    Returns a ``pandas.DataFrame`` with columns == ``FEATURE_NAMES``.
    """
    import pandas as pd  # local import keeps module importable without pandas

    total = len(urls)
    rows = []

    for i, url in enumerate(urls):
        if show_progress and (i + 1) % 25_000 == 0:
            print(f"  Feature extraction: {i + 1:,} / {total:,}")
        rows.append(extract_url_features(str(url)))

    if show_progress:
        print(f"  Feature extraction: {total:,} / {total:,}  ✓")

    return pd.DataFrame(rows, columns=FEATURE_NAMES)
