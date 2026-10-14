"""
feature_extractor.py — Extract features from a raw URL string.

When a user types a URL into the Streamlit app we cannot look up the
pre-computed features in the dataset.  Instead we extract *URL-based*
features from the string itself.

IMPORTANT LIMITATION
--------------------
The original dataset also includes **webpage-content features** (e.g.
HasFavicon, NoOfImage, LineOfCode …).  Extracting those would require
actually fetching and parsing the live webpage, which is slow, may fail,
and raises security concerns.  Therefore this extractor only computes
URL-structural features and fills content-based features with safe
defaults (0).

This means predictions from the Streamlit app are *less accurate* than
the model's test-set performance.  See README for a full discussion.
"""

import re
import math
from urllib.parse import urlparse
from collections import Counter


def _count_chars(url: str, charset: str) -> int:
    return sum(1 for c in url if c in charset)


def _letter_count(url: str) -> int:
    return sum(1 for c in url if c.isalpha())


def _digit_count(url: str) -> int:
    return sum(1 for c in url if c.isdigit())


def _special_char_count(url: str) -> int:
    return sum(1 for c in url if not c.isalnum())


def _is_ip(domain: str) -> int:
    """Return 1 if the domain looks like an IP address."""
    parts = domain.split(".")
    if len(parts) == 4:
        try:
            return int(all(0 <= int(p) <= 255 for p in parts))
        except ValueError:
            pass
    return 0


def _subdomain_count(domain: str) -> int:
    parts = domain.split(".")
    return max(len(parts) - 2, 0)


def _char_continuation_rate(url: str) -> float:
    """Longest run of the same character / len(url)."""
    if not url:
        return 0.0
    max_run = 1
    current_run = 1
    for i in range(1, len(url)):
        if url[i] == url[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run / len(url)


def _url_char_prob(url: str) -> float:
    """Shannon entropy of the URL character distribution."""
    if not url:
        return 0.0
    freq = Counter(url)
    length = len(url)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def extract_features(url: str, feature_names: list[str]) -> dict:
    """
    Extract a feature dictionary from a raw URL string.

    Parameters
    ----------
    url : str
        The URL typed by the user.
    feature_names : list[str]
        The feature names expected by the model (from training).

    Returns
    -------
    dict
        Keys = feature_names, values = extracted or default values.
    """
    parsed = urlparse(url if "://" in url else f"http://{url}")
    domain = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    full_url = url

    # TLD extraction
    tld = ""
    if domain:
        parts = domain.split(".")
        if len(parts) >= 2:
            tld = parts[-1]

    # Known feature computations ──────────────────────────────────────────
    computed = {
        "URLLength": len(full_url),
        "DomainLength": len(domain),
        "IsDomainIP": _is_ip(domain),
        "NoOfSubDomain": _subdomain_count(domain),
        "HasObfuscation": int("%" in full_url or "@" in full_url),
        "NoOfObfuscatedChar": full_url.count("%") + full_url.count("@"),
        "ObfuscationRatio": (full_url.count("%") + full_url.count("@")) / max(len(full_url), 1),
        "NoOfLettersInURL": _letter_count(full_url),
        "LetterRatioInURL": _letter_count(full_url) / max(len(full_url), 1),
        "NoOfDegitsInURL": _digit_count(full_url),
        "DegitRatioInURL": _digit_count(full_url) / max(len(full_url), 1),
        "NoOfEqualsInURL": full_url.count("="),
        "NoOfQMarkInURL": full_url.count("?"),
        "NoOfAmpersandInURL": full_url.count("&"),
        "NoOfOtherSpecialCharsInURL": _special_char_count(full_url),
        "SpacialCharRatioInURL": _special_char_count(full_url) / max(len(full_url), 1),
        "IsHTTPS": int(parsed.scheme == "https"),
        "URLSimilarityIndex": 0,  # cannot compute without DB
        "CharContinuationRate": _char_continuation_rate(full_url),
        "URLCharProb": _url_char_prob(full_url),
        "TLDLegitimateProb": 0.5,  # neutral default
        "URLTitleMatchScore": 0,
        "HasFavicon": 0,
        "Robots": 0,
        "IsResponsive": 0,
        "NoOfURLRedirect": 0,
        "NoOfSelfRedirect": 0,
        "HasDescription": 0,
        "NoOfPopup": 0,
        "NoOfiFrame": 0,
        "HasExternalFormSubmit": 0,
        "HasSocialNet": 0,
        "HasSubmitButton": 0,
        "HasHiddenFields": 0,
        "HasPasswordField": 0,
        "BankForm": 0,
        "HasCopyrightInfo": 0,
        "HasImage": 0,
        "NoOfImage": 0,
        "NoOfCSS": 0,
        "NoOfJS": 0,
        "NoOfSelfRef": 0,
        "NoOfEmptyRef": 0,
        "NoOfExternalRef": 0,
        "LineOfCode": 0,
        "LargestLineLength": 0,
        "DomainTitleMatchScore": 0,
    }

    # Build the feature vector in the exact order the model expects
    feature_dict = {}
    for name in feature_names:
        feature_dict[name] = computed.get(name, 0)

    return feature_dict
