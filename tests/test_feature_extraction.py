"""
tests/test_feature_extraction.py — Unit tests for feature extraction.

Run:
    python -m pytest tests/ -v
"""

import os
import sys
import pytest
import pandas as pd

# ── Add project root so imports work ─────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.feature_extraction import (
    extract_url_features,
    extract_features_batch,
    FEATURE_NAMES,
)


# ── Helper ────────────────────────────────────────────────────────────────────

def _feat(url):
    """Shortcut to extract features for a single URL."""
    return extract_url_features(url)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFeatureNamesContract:
    """The returned dict must exactly match FEATURE_NAMES (keys and order)."""

    def test_keys_match_feature_names(self):
        feat = _feat("https://www.example.com/login")
        assert list(feat.keys()) == FEATURE_NAMES

    def test_count_matches(self):
        feat = _feat("http://test.org")
        assert len(feat) == len(FEATURE_NAMES)

    def test_all_values_are_numeric(self):
        feat = _feat("https://google.com/search?q=hello")
        for name, val in feat.items():
            assert isinstance(val, (int, float)), \
                f"Feature '{name}' has non-numeric value: {val!r}"


class TestStandardHttpsUrl:
    """Normal HTTPS URL with query parameter."""

    URL = "https://secure.bank.com/account/login?id=123"

    def test_is_https(self):
        assert _feat(self.URL)["is_https"] == 1

    def test_not_ip(self):
        assert _feat(self.URL)["is_ip_address"] == 0

    def test_query_params(self):
        assert _feat(self.URL)["num_query_params"] == 1

    def test_url_length(self):
        assert _feat(self.URL)["url_length"] == len(self.URL)

    def test_hostname_dots(self):
        # secure.bank.com → 2 dots
        assert _feat(self.URL)["num_dots_hostname"] == 2


class TestUrlWithoutScheme:
    """URL without http:// or https:// prefix."""

    URL = "www.mysite.org/dashboard"

    def test_hostname_extracted(self):
        assert _feat(self.URL)["hostname_length"] > 0

    def test_not_https(self):
        # No https:// prefix → is_https should be 0
        assert _feat(self.URL)["is_https"] == 0

    def test_dots_in_hostname(self):
        assert _feat(self.URL)["num_dots_hostname"] >= 1


class TestIpAddressHostname:
    """Hostname is a raw IPv4 address."""

    def test_is_ip_true(self):
        feat = _feat("http://192.168.1.100/admin")
        assert feat["is_ip_address"] == 1

    def test_valid_ipv6(self):
        feat = _feat("http://[2001:db8::1]/page")
        assert feat["is_ip_address"] == 1

    def test_regular_domain_not_ip(self):
        feat = _feat("https://example.com")
        assert feat["is_ip_address"] == 0


class TestPercentEncodingAndSpecialChars:
    """Percent-encoded sequences and @ symbol."""

    URL = "http://bad-site.com/%20/login%2f?token=abc@test#section"

    def test_percent_encoded_count(self):
        assert _feat(self.URL)["num_percent_encoded"] == 2

    def test_hyphen_count(self):
        assert _feat(self.URL)["num_hyphens"] >= 1

    def test_at_sign_count(self):
        assert _feat(self.URL)["num_at_signs"] >= 1


class TestEmptyAndMalformedUrls:
    """Edge cases: empty string, whitespace, garbled input."""

    def test_empty_string_returns_zeros(self):
        feat = _feat("")
        assert feat["url_length"] == 0
        assert all(v == 0 for v in feat.values())

    def test_whitespace_only_returns_zeros(self):
        feat = _feat("   ")
        assert feat["url_length"] == 0

    def test_malformed_does_not_raise(self):
        # These should not crash
        for bad in ["://???///%%", "!@#$%^", "http://"]:
            result = _feat(bad)
            assert isinstance(result, dict)
            assert len(result) == len(FEATURE_NAMES)

    def test_all_values_are_numeric_for_malformed(self):
        feat = _feat("://???")
        for name, val in feat.items():
            assert isinstance(val, (int, float)), f"'{name}' = {val!r}"


class TestBatchExtraction:
    """Batch function returns a consistent, null-free DataFrame."""

    URLS = [
        "https://google.com",
        "http://10.0.0.1/login.php?user=admin",
        "http://phish-update-account.biz",
        "ftp://files.example.org/data",
        "",  # empty — must not crash
    ]

    def test_returns_dataframe(self):
        df = extract_features_batch(self.URLS, show_progress=False)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self):
        df = extract_features_batch(self.URLS, show_progress=False)
        assert len(df) == len(self.URLS)

    def test_correct_columns(self):
        df = extract_features_batch(self.URLS, show_progress=False)
        assert list(df.columns) == FEATURE_NAMES

    def test_no_null_values(self):
        df = extract_features_batch(self.URLS, show_progress=False)
        assert not df.isnull().any().any(), "Batch extraction produced NaN values."

    def test_dtypes_numeric(self):
        df = extract_features_batch(self.URLS, show_progress=False)
        for col in FEATURE_NAMES:
            assert pd.api.types.is_numeric_dtype(df[col]), \
                f"Column '{col}' is not numeric."


class TestNonStandardPort:
    """Port detection."""

    def test_standard_port_80(self):
        assert _feat("http://example.com:80/page")["has_non_standard_port"] == 0

    def test_standard_port_443(self):
        assert _feat("https://example.com:443/page")["has_non_standard_port"] == 0

    def test_non_standard_port(self):
        assert _feat("http://example.com:8080/page")["has_non_standard_port"] == 1


class TestDoubleSlashRedirect:
    """Double-slash in path detection."""

    def test_double_slash_in_path(self):
        # After scheme the path is //redirect
        feat = _feat("http://evil.com//redirect.php")
        assert feat["has_double_slash_redirect"] == 1

    def test_normal_path_no_flag(self):
        feat = _feat("https://normal.com/about/us")
        assert feat["has_double_slash_redirect"] == 0
