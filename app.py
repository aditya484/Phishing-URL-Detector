"""
app.py — Streamlit Web Application for Phishing Website Detection.

Features:
- Safe URL input validation (no network calls).
- Uses the same extract_url_features() used during training.
- Shows prediction, phishing/legit probability bars.
- Lists suspicious structural indicators.
- Expandable feature table for inspection.
- Clear educational disclaimer throughout.

Launch:
    streamlit run app.py
"""

import os
import sys
import re
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Project root on sys.path ─────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.feature_extraction import extract_url_features, FEATURE_NAMES, CLASS_NAMES

MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "phishing_model.joblib")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phishing Website Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .result-phish { background:#fff0f0; border-left:5px solid #e74c3c;
                    padding:12px 16px; border-radius:6px; margin-bottom:10px; }
    .result-legit { background:#f0fff4; border-left:5px solid #2ecc71;
                    padding:12px 16px; border-radius:6px; margin-bottom:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Model loader (cached) ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load and cache the Joblib model package once per session."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        return None


# ── URL validator ─────────────────────────────────────────────────────────────
def validate_url(raw: str):
    """
    Basic structural validation — no network calls.

    Returns (is_valid: bool, error_message: str).
    """
    raw = raw.strip()
    if not raw:
        return False, "Please enter a URL."
    if len(raw) < 4:
        return False, "URL is too short to be valid."
    if len(raw) > 2048:
        return False, "URL is too long (max 2048 characters)."

    test = raw if re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", raw) else "http://" + raw
    try:
        parsed = urlparse(test)
        if not parsed.hostname:
            return False, "Cannot extract a hostname from this URL."
    except Exception:
        return False, "URL format is not recognisable."
    return True, ""


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ About")
    st.markdown(
        """
        **Dataset:**  
        [PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)  
        (UCI ML Repository — 235,795 URLs)

        **Algorithm:** Random Forest (200 trees)

        **Features:** 30 URL-structural features —  
        no webpage fetching, no third-party APIs.
        """
    )
    st.divider()
    st.subheader("⚠️ Disclaimer")
    st.warning(
        "This is an **educational ML demo**. Predictions are based on "
        "URL structural patterns only and are **not guarantees of safety**. "
        "Always rely on official browser security tools and antivirus software."
    )

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🌐 Phishing Website Detector")
st.markdown(
    "Paste any URL below. The model analyses its structural patterns "
    "to estimate whether it is likely phishing or legitimate."
)

package = load_model()

if package is None:
    st.warning("⚠️ **No model found.** Run training first:")
    st.code("python -m src.train", language="bash")
    st.stop()

pipeline      = package["pipeline"]
feature_names = package.get("feature_names", FEATURE_NAMES)

# ── Input ─────────────────────────────────────────────────────────────────────
user_url = st.text_input(
    "🔗 Website URL",
    placeholder="e.g.  https://www.example.com  or  paypal-secure-login.ru/verify",
    help="Enter a full URL or just a domain. No data is sent anywhere.",
)

col_btn, col_clear = st.columns([2, 1])
with col_btn:
    clicked = st.button("🔍 Analyse", type="primary", use_container_width=True)
with col_clear:
    if st.button("✖ Clear", use_container_width=True):
        st.rerun()

# ── Trusted High-Reputation Domains ──────────────────────────────────────────
# Real-world phishing detectors combine ML with high-reputation domain registries
# to prevent false positives on legitimate services like leetcode.com, google.com, etc.
TOP_TRUSTED_DOMAINS = {
    "google.com", "youtube.com", "facebook.com", "amazon.com", "wikipedia.org",
    "twitter.com", "x.com", "instagram.com", "linkedin.com", "apple.com",
    "microsoft.com", "github.com", "netflix.com", "yahoo.com", "reddit.com",
    "leetcode.com", "hackerrank.com", "codeforces.com", "geeksforgeeks.org",
    "stackoverflow.com", "stackexchange.com", "medium.com", "quora.com",
    "openai.com", "anthropic.com", "cloudflare.com", "gitlab.com", "bitbucket.org",
    "w3schools.com", "mozilla.org", "python.org", "kaggle.com", "arxiv.org",
    "coursera.org", "edx.org", "udemy.com", "khanacademy.org", "spotify.com",
    "zoom.us", "dropbox.com", "slack.com", "notion.so", "figma.com", "canva.com",
    "cnn.com", "bbc.com", "nytimes.com", "forbes.com", "reuters.com", "bloomberg.com"
}


def _get_registered_domain(hostname: str) -> str:
    """Extract root registered domain from hostname (e.g. 'sub.leetcode.com' -> 'leetcode.com')."""
    if not hostname:
        return ""
    parts = hostname.lower().strip().split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return hostname


# ── Prediction ────────────────────────────────────────────────────────────────
if clicked:
    valid, errmsg = validate_url(user_url or "")
    if not valid:
        st.error(f"❌ {errmsg}")
    else:
        with st.spinner("Extracting features…"):
            raw_input = user_url.strip()
            feats    = extract_url_features(raw_input)
            feat_df  = pd.DataFrame([feats])[feature_names]
            pred     = int(pipeline.predict(feat_df)[0])
            proba    = pipeline.predict_proba(feat_df)[0]

            # Map proba columns to classes via clf.classes_
            clf     = pipeline.named_steps["classifier"]
            classes = list(clf.classes_)
            p_phish = float(proba[classes.index(0)]) * 100   # class 0 prob
            p_legit = float(proba[classes.index(1)]) * 100   # class 1 prob

            # Parse domain for reputation check
            test_url = raw_input if re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", raw_input) else "http://" + raw_input
            parsed_domain = (urlparse(test_url).hostname or "").lower()
            root_domain = _get_registered_domain(parsed_domain)

            # Check if domain is a known trusted service with no obfuscations
            is_verified_trusted = False
            if root_domain in TOP_TRUSTED_DOMAINS and feats["is_ip_address"] == 0 and feats["num_at_signs"] == 0:
                is_verified_trusted = True
                pred = 1
                p_legit = max(p_legit, 99.2)
                p_phish = 100.0 - p_legit

        st.divider()

        # ── Result banner ────────────────────────────────────────────────────
        if is_verified_trusted:
            st.markdown(
                '<div class="result-legit">'
                "<h3>✅ Verified Legitimate Domain</h3>"
                f"<p>The root domain <b>{root_domain}</b> is a globally recognized, high-reputation service.<br>"
                f"<b>Legitimate confidence: {p_legit:.1f}%</b> (Verified Authority)</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        elif pred == 0:
            st.markdown(
                '<div class="result-phish">'
                "<h3>🚨 Suspected Phishing URL</h3>"
                f"<p>The URL matches patterns commonly found in phishing sites.<br>"
                f"<b>Phishing confidence: {p_phish:.1f}%</b></p>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-legit">'
                "<h3>✅ Likely Legitimate URL</h3>"
                f"<p>The URL's structure is consistent with legitimate sites.<br>"
                f"<b>Legitimate confidence: {p_legit:.1f}%</b></p>"
                "</div>",
                unsafe_allow_html=True,
            )

        # ── Probability gauges ───────────────────────────────────────────────
        st.subheader("📊 Confidence Scores")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("🔴 Phishing", f"{p_phish:.1f}%")
            st.progress(min(p_phish / 100.0, 1.0))
        with c2:
            st.metric("🟢 Legitimate", f"{p_legit:.1f}%")
            st.progress(min(p_legit / 100.0, 1.0))

        # ── Suspicious indicators ────────────────────────────────────────────
        st.subheader("🔬 Structural Indicators")
        flags = []
        if feats["is_ip_address"]:
            flags.append("⚠️ Hostname is a raw IP address (e.g. `http://192.168.1.1/`).")
        if not feats["is_https"]:
            flags.append("⚠️ URL does not use HTTPS.")
        if feats["num_percent_encoded"] > 0:
            flags.append(
                f"⚠️ {feats['num_percent_encoded']} percent-encoded character(s) "
                "found (possible obfuscation)."
            )
        if feats["has_double_slash_redirect"]:
            flags.append("⚠️ Double slash `//` in path — possible open-redirect technique.")
        if feats["num_subdomains"] >= 3:
            flags.append(
                f"⚠️ {feats['num_subdomains']} subdomain levels — "
                "legitimate brands rarely use this many."
            )
        if feats["url_entropy"] > 4.5:
            flags.append(
                f"⚠️ High URL entropy ({feats['url_entropy']:.2f} bits) — "
                "suggests a randomly generated or obfuscated domain."
            )
        if feats["num_at_signs"] > 0:
            flags.append("⚠️ `@` symbol in URL — browsers ignore everything before `@`.")

        if flags:
            for f in flags:
                st.write(f)
        else:
            st.success("✔ No obvious structural anomalies detected.")

        # ── Feature detail expander ──────────────────────────────────────────
        with st.expander("📋 All 30 Extracted Feature Values"):
            display = pd.DataFrame(
                {"Feature": list(feats.keys()), "Value": list(feats.values())}
            )
            st.dataframe(display, use_container_width=True, hide_index=True)

        st.divider()
        st.caption(
            "🔒 **Remember:** A 'Legitimate' result does not guarantee the site is safe. "
            "The model cannot inspect live page content, JavaScript, or server behaviour. "
            "Never enter passwords or payment details on a site you do not fully trust."
        )
