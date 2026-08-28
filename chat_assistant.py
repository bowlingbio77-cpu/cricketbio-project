"""
chat_assistant.py

Drop-in chat widget for the Cricket Bowling Biomechanics dashboard.
Talks to a locally running Ollama model and grounds its answers in the
current bowler's analysis results (features, predictions, SHAP values,
coaching recommendations) pulled from Streamlit session_state.

Usage in app.py:

    from chat_assistant import render_chat_widget
    ...
    # after you've computed features / predictions / shap_values / recommendations
    # and stored them in st.session_state (see the CONTEXT KEYS section below)
    render_chat_widget()

Requirements:
    pip install requests
    Ollama running locally: ``ollama serve`` (default http://localhost:11434)
    A model pulled, e.g.: ``ollama pull llama3.1``

Configuration (environment variables):
    OLLAMA_BASE_URL  - Base URL of the Ollama server (default: http://localhost:11434)
    OLLAMA_MODEL     - Model name to use (default: llama3.2:1b)
    OLLAMA_TIMEOUT   - Request timeout in seconds (default: 30)
"""

import os
import hashlib
from typing import Optional
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config -- all overridable via environment variables
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "30"))

_MAX_RESPONSE_CHARS = 4000  # sanity cap on LLM output displayed in UI

SYSTEM_PROMPT = (
    "You are a cricket fast-bowling biomechanics assistant embedded in a "
    "coaching dashboard. You are given the current bowler's computed "
    "biomechanical features, model predictions (performance score, injury "
    "risk), SHAP feature-importance values, and rule-based coaching "
    "recommendations. Answer the coach's or player's questions using ONLY "
    "this data plus general biomechanics knowledge \u2014 do not invent numbers "
    "that aren't provided. If no analysis has been run yet, say so and ask "
    "them to run one first. Keep answers concise and practical. Always "
    "remind users that elbow-flexion / ICC-legality readings here are a "
    "screening signal, not an official ruling, and that this tool supports "
    "but doesn't replace a qualified coach, biomechanist, or physician."
)

# ---------------------------------------------------------------------------
# Model availability check
# ---------------------------------------------------------------------------

_MODEL_CHECKED_KEY = "_ollama_model_checked"
_MODEL_AVAILABLE_KEY = "_ollama_model_available"


def _check_model_available() -> bool:
    """Check if the configured Ollama model is available.  Caches the result
    for the lifetime of the Streamlit session so we don't re-probe on every
    rerun."""
    if _MODEL_CHECKED_KEY in st.session_state:
        return st.session_state[_MODEL_AVAILABLE_KEY]

    try:
        tags_url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        resp = requests.get(tags_url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = [m.get("name", "") for m in data.get("models", [])]
        available = any(OLLAMA_MODEL in name for name in models)
    except (requests.ConnectionError, requests.Timeout, requests.HTTPError, Exception):
        available = False

    st.session_state[_MODEL_CHECKED_KEY] = True
    st.session_state[_MODEL_AVAILABLE_KEY] = available
    return available


# ---------------------------------------------------------------------------
# CONTEXT KEYS -- adjust these to match what your app.py actually stores.
#
# app.py stores these in st.session_state after each analysis run:
#   features            -> dict(result.feature_vector)          (name -> value)
#   performance_score   -> result.performance_score
#   injury_risk         -> result.injury_risk   (dict: risk_level/probabilities)
#   shap_values         -> merged performance + injury SHAP contributions
#   recommendations     -> result.coaching_notes                (list[str])
# ---------------------------------------------------------------------------

def _format_injury_risk(injury_risk):
    """Handles both the dict returned by pipeline.AnalysisResult
    (risk_level + probabilities) and a plain 0/1/2 label."""
    if isinstance(injury_risk, dict):
        level = str(injury_risk.get("risk_level", "unknown")).lower()
        probs = injury_risk.get("probabilities")
        parts = [f"Injury risk: {level}"]
        if isinstance(probs, (list, tuple)) and len(probs) >= 3:
            try:
                parts.append(
                    f"(P(low)={probs[0]:.2f}, P(moderate)={probs[1]:.2f}, "
                    f"P(high)={probs[2]:.2f})"
                )
            except (TypeError, ValueError):
                pass
        return " ".join(parts)
    risk_labels = {0: "low", 1: "moderate", 2: "high"}
    label = risk_labels.get(injury_risk, injury_risk)
    return f"Injury risk: {label}"


def _build_context_block() -> str:
    features = st.session_state.get("features")               # dict: name -> value
    performance_score = st.session_state.get("performance_score")
    injury_risk = st.session_state.get("injury_risk")
    shap_values = st.session_state.get("shap_values")          # dict: name -> shap value
    recommendations = st.session_state.get("recommendations")  # list[str] or str

    if not features:
        return (
            "No analysis has been run yet in this session \u2014 no feature "
            "values, predictions, or recommendations are available."
        )

    lines = ["Current bowler analysis:"]

    lines.append("\nBiomechanical features:")
    for name, value in features.items():
        lines.append(f"- {name}: {value}")

    if performance_score is not None:
        lines.append(f"\nPerformance score: {performance_score}")
    if injury_risk is not None:
        lines.append(_format_injury_risk(injury_risk))

    if shap_values:
        lines.append("\nSHAP feature importances (impact on prediction):")
        # sort by absolute impact, most influential first
        ranked = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)
        for name, val in ranked:
            lines.append(f"- {name}: {val:+.3f}")

    if recommendations:
        lines.append("\nCoaching recommendations already generated:")
        if isinstance(recommendations, (list, tuple)):
            for rec in recommendations:
                lines.append(f"- {rec}")
        else:
            lines.append(f"- {recommendations}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response cache  (keyed on content hash of the messages list)
# ---------------------------------------------------------------------------

def _messages_hash(messages: list) -> str:
    """Deterministic hash of the messages list for caching."""
    raw = repr(messages)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _get_cached_response(messages: list) -> Optional[str]:
    cache = st.session_state.get("_chat_cache", {})
    return cache.get(_messages_hash(messages))


def _set_cached_response(messages: list, response: str):
    cache = st.session_state.setdefault("_chat_cache", {})
    # keep cache bounded (last 50 unique exchanges)
    if len(cache) > 50:
        oldest_key = next(iter(cache))
        del cache[oldest_key]
    cache[_messages_hash(messages)] = response


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def _call_ollama(messages: list) -> str:
    """Send messages to Ollama with timeout, model validation, and caching."""
    # --- check cache first ---
    cached = _get_cached_response(messages)
    if cached is not None:
        return cached

    # --- check model availability ---
    if not _check_model_available():
        return (
            f"\u26a0\ufe0f Model **{OLLAMA_MODEL}** not found at {OLLAMA_BASE_URL}. "
            f"Make sure Ollama is running (`ollama serve`) and the model is pulled "
            f"(`ollama pull {OLLAMA_MODEL}`). The chat assistant is unavailable until "
            f"the model is ready."
        )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "").strip()
        if not content:
            return "\u26a0\ufe0f The model returned an empty response. Try rephrasing your question."
        # Truncate extremely long responses for UI sanity
        if len(content) > _MAX_RESPONSE_CHARS:
            content = content[:_MAX_RESPONSE_CHARS] + "\n\n[response truncated]"
        _set_cached_response(messages, content)
        return content
    except requests.ConnectionError:
        return (
            f"\u26a0\ufe0f Couldn't reach Ollama at **{OLLAMA_BASE_URL}**. "
            f"Make sure it's running (`ollama serve`) and that you've pulled "
            f"the model (`ollama pull {OLLAMA_MODEL}`)."
        )
    except requests.Timeout:
        return (
            f"\u26a0\ufe0f Ollama request timed out after {OLLAMA_TIMEOUT}s. "
            f"The model may be loading or too slow for this query. "
            f"Try a shorter question or increase OLLAMA_TIMEOUT."
        )
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        return (
            f"\u26a0\ufe0f Ollama returned HTTP {status}. "
            f"The model **{OLLAMA_MODEL}** may not be available. "
            f"Check `ollama list` and try again."
        )
    except Exception as exc:
        return f"\u26a0\ufe0f Unexpected error calling Ollama: {exc}"


# ---------------------------------------------------------------------------
# Streamlit widget
# ---------------------------------------------------------------------------

def render_chat_widget():
    """Renders a chat panel in the sidebar, grounded in current analysis."""

    st.sidebar.markdown("---")
    st.sidebar.subheader("\U0001f4ac Ask about this bowler")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of {"role": ..., "content": ...}

    # replay history
    for msg in st.session_state.chat_history:
        with st.sidebar.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.sidebar.chat_input("e.g. Why is the injury risk high?")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.sidebar.chat_message("user"):
            st.markdown(user_input)

        context_block = _build_context_block()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + context_block},
        ]
        # include recent turns so it's a real conversation, not one-shot
        messages += st.session_state.chat_history[-10:]

        with st.sidebar.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = _call_ollama(messages)
            st.markdown(reply)

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
