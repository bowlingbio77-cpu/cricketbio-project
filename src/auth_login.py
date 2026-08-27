"""
src/auth_login.py

Animated glassmorphism login screen for PaceAI, gating the rest of app.py
behind a session-based login. Built entirely with native Streamlit widgets
(so keyboard nav / focus / screen readers work out of the box) reskinned
via injected CSS, plus a pure-CSS decorative background (orbs, grid,
particles) that needs no JavaScript and respects prefers-reduced-motion.

--------------------------------------------------------------------------
HONEST LIMITATIONS vs. a hand-built React login screen
--------------------------------------------------------------------------
Streamlit re-renders from Python on every interaction (it's not a JS SPA),
so a few things from a typical spec are approximated rather than exact:

  - Floating labels: true "label slides from inside the input to above it"
    needs the label positioned *after* the input in the DOM so a CSS
    sibling selector can react to :focus. Streamlit renders the label
    *before* the input, so this isn't reliably stylable. Instead: a small
    static label above the field, with the input itself glowing/lifting
    on focus. Visually clean, just not the exact floating-label motion.
  - "Client-side" validation: there's no separate JS validation layer --
    submitting re-runs the Python script, which validates and reruns with
    an error shown beneath the field. It feels instant locally but is
    technically a server (rerun) round-trip, not pure client-side JS.
  - Password show/hide: Streamlit's password input already ships a built
    -in eye icon (no custom JS needed) -- verify your Streamlit version
    is recent enough (`pip install -U streamlit` if you don't see it).

--------------------------------------------------------------------------
WIRING INTO app.py
--------------------------------------------------------------------------
1. Save this file as `src/auth_login.py` (next to your other src/ modules).

2. Near the top of app.py, after your existing imports, add:

       from src.auth_login import render_login_page, is_authenticated

3. Right after `st.set_page_config(...)` (keep that call first, Streamlit
   requires it), and BEFORE your existing custom CSS / dashboard body,
   add the auth gate:

       if not is_authenticated():
           render_login_page()
           st.stop()

   Everything below that line only ever runs for a logged-in session.

4. Set real credentials via environment variables (don't hardcode a
   password in source). In your `.env`:

       PACEAI_LOGIN_EMAIL=you@example.com
       PACEAI_LOGIN_PASSWORD_HASH=<sha256 hex digest of your password>

   Generate the hash once with:

       python -c "import hashlib; print(hashlib.sha256(b'yourpassword').hexdigest())"

   If those env vars aren't set, it falls back to a demo login
   (admin@paceai.local / admin123) and shows a visible warning banner --
   replace `check_credentials()` with a real user lookup / API call
   whenever you're ready; it's the one function you need to swap.

5. Add a "Log out" control wherever makes sense in your sidebar:

       if st.sidebar.button("Log out"):
           st.session_state["authenticated"] = False
           st.rerun()
"""

from __future__ import annotations

import hashlib
import os
import time

import streamlit as st

# ---------------------------------------------------------------------------
# Pluggable auth check -- replace this with a real API / DB lookup later.
# Everything else in this file only calls this one function.
# ---------------------------------------------------------------------------

_DEMO_EMAIL = "admin@paceai.local"
_DEMO_PASSWORD_HASH = hashlib.sha256(b"admin123").hexdigest()


def _using_demo_credentials() -> bool:
    return not (os.environ.get("PACEAI_LOGIN_EMAIL") and os.environ.get("PACEAI_LOGIN_PASSWORD_HASH"))


def check_credentials(email: str, password: str) -> bool:
    """Swap this out for a real backend call when you're ready.

    Must keep the same signature: (email: str, password: str) -> bool.
    """
    expected_email = os.environ.get("PACEAI_LOGIN_EMAIL", _DEMO_EMAIL)
    expected_hash = os.environ.get("PACEAI_LOGIN_PASSWORD_HASH", _DEMO_PASSWORD_HASH)

    if email.strip().lower() != expected_email.strip().lower():
        return False
    return hashlib.sha256(password.encode("utf-8")).hexdigest() == expected_hash


def is_authenticated() -> bool:
    return bool(st.session_state.get("authenticated", False))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _valid_email(value: str) -> bool:
    if not value or "@" not in value:
        return False
    local, _, domain = value.rpartition("@")
    return bool(local) and "." in domain and not domain.startswith(".")


def _valid_password(value: str) -> bool:
    return len(value) >= 6


# ---------------------------------------------------------------------------
# CSS -- decorative background + glassmorphism card, matching PaceAI's
# existing palette (#0d1117 base, Inter font) with the blue/cyan/purple
# accent range the brief asked for.
# ---------------------------------------------------------------------------

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

/* Hide Streamlit chrome while on the login screen */
section[data-testid="stSidebar"],
header[data-testid="stHeader"],
footer { display: none !important; }

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: #05070c;
    overflow: hidden;
}

/* ---------------- Decorative animated background ---------------- */
.pace-login-bg {
    position: fixed;
    inset: 0;
    z-index: 0;
    overflow: hidden;
    pointer-events: none;
}
.pace-login-bg .grid {
    position: absolute;
    inset: -2px;
    background-image:
        linear-gradient(rgba(41, 182, 246, 0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(41, 182, 246, 0.05) 1px, transparent 1px);
    background-size: 42px 42px;
    mask-image: radial-gradient(ellipse at center, black 0%, transparent 75%);
}
.pace-login-bg .orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(70px);
    opacity: 0.55;
    animation: orbFloat 18s ease-in-out infinite;
}
.orb-cyan   { width: 420px; height: 420px; background: #29b6f6; top: -10%; left: -8%; animation-delay: 0s; }
.orb-purple { width: 380px; height: 380px; background: #7c5cff; bottom: -12%; right: -6%; animation-delay: -6s; }
.orb-blue   { width: 300px; height: 300px; background: #1e6fd9; top: 40%; left: 55%; animation-delay: -12s; }

.pace-login-bg .spark {
    position: absolute;
    width: 3px;
    height: 3px;
    border-radius: 50%;
    background: #7dd3fc;
    box-shadow: 0 0 8px #7dd3fc;
    animation: sparkGlow 4.5s ease-in-out infinite;
}

@keyframes orbFloat {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50% { transform: translate(30px, -25px) scale(1.08); }
}
@keyframes sparkGlow {
    0%, 100% { opacity: 0.15; transform: scale(1); }
    50% { opacity: 0.9; transform: scale(1.6); }
}

/* ---------------- Layout ---------------- */
.pace-login-wrap {
    position: relative;
    z-index: 1;
}

.pace-hero {
    padding: 48px 32px;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.pace-hero .logo-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 28px;
}
.pace-hero .logo-mark {
    font-size: 1.6rem;
}
.pace-hero .logo-text {
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: 0.5px;
    color: #ffffff;
}
.pace-hero h1 {
    font-size: 2.3rem;
    font-weight: 800;
    line-height: 1.2;
    background: linear-gradient(120deg, #ffffff 30%, #7dd3fc 75%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 14px 0;
}
.pace-hero p {
    color: #8b949e;
    font-size: 0.98rem;
    line-height: 1.55;
    max-width: 380px;
    margin: 0;
}

/* ---------------- Glassmorphism card ---------------- */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.pace-card-anchor) {
    background: rgba(20, 26, 38, 0.55);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(125, 211, 252, 0.18);
    border-radius: 20px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.45), 0 0 40px rgba(41, 182, 246, 0.06);
    padding: 8px;
    animation: cardEnter 0.7s cubic-bezier(0.16, 1, 0.3, 1) both;
}

@keyframes cardEnter {
    from { opacity: 0; transform: scale(0.96) translateY(8px); filter: blur(6px); }
    to   { opacity: 1; transform: scale(1) translateY(0); filter: blur(0); }
}

.pace-card-header {
    text-align: center;
    padding: 22px 12px 6px 12px;
}
.pace-card-header .mark {
    font-size: 1.8rem;
    margin-bottom: 6px;
}
.pace-card-header h2 {
    color: #ffffff;
    font-size: 1.5rem;
    font-weight: 800;
    margin: 4px 0 4px 0;
}
.pace-card-header p {
    color: #8b949e;
    font-size: 0.88rem;
    margin: 0;
}

/* Inputs */
div[data-testid="stTextInput"] label p {
    color: #a9b4c0 !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px;
}
div[data-testid="stTextInput"] input {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.10) !important;
    border-radius: 10px !important;
    color: #e6edf3 !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.15s ease;
}
div[data-testid="stTextInput"] input:hover {
    border-color: rgba(125, 211, 252, 0.35) !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #29b6f6 !important;
    box-shadow: 0 0 0 3px rgba(41, 182, 246, 0.15), 0 0 18px rgba(41, 182, 246, 0.18) !important;
    transform: translateY(-1px);
}

.pace-field-error {
    color: #ef5350;
    font-size: 0.78rem;
    margin: -10px 0 8px 2px;
    animation: errorIn 0.2s ease-out both;
}
@keyframes errorIn {
    from { opacity: 0; transform: translateY(-3px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Checkbox row */
div[data-testid="stCheckbox"] label p {
    color: #a9b4c0 !important;
    font-size: 0.85rem !important;
}

.pace-forgot {
    text-align: right;
    padding-top: 6px;
}
.pace-forgot a {
    color: #7dd3fc;
    font-size: 0.85rem;
    text-decoration: none;
    transition: color 0.15s ease;
}
.pace-forgot a:hover {
    color: #ffffff;
    text-decoration: underline;
}

/* Primary login button */
div[data-testid="stButton"]:has(button[kind="primary"]) button {
    background: linear-gradient(90deg, #29b6f6, #7c5cff) !important;
    background-size: 160% 100% !important;
    background-position: 0% 0% !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px;
    padding: 0.65rem 0 !important;
    box-shadow: 0 6px 20px rgba(41, 182, 246, 0.25);
    transition: transform 0.15s ease, box-shadow 0.2s ease, background-position 0.4s ease;
}
div[data-testid="stButton"]:has(button[kind="primary"]) button:hover {
    transform: translateY(-2px);
    background-position: 100% 0% !important;
    box-shadow: 0 10px 26px rgba(41, 182, 246, 0.35);
}
div[data-testid="stButton"]:has(button[kind="primary"]) button:active {
    transform: translateY(0) scale(0.98);
}

/* Secondary (social) buttons */
div[data-testid="stButton"]:has(button[kind="secondary"]) button {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 10px !important;
    color: #e6edf3 !important;
    transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease;
}
div[data-testid="stButton"]:has(button[kind="secondary"]) button:hover {
    transform: translateY(-1px);
    border-color: rgba(125, 211, 252, 0.4) !important;
    background: rgba(255, 255, 255, 0.06) !important;
}

.pace-divider {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #6b7280;
    font-size: 0.75rem;
    margin: 6px 0 12px 0;
}
.pace-divider::before, .pace-divider::after {
    content: "";
    flex: 1;
    height: 1px;
    background: rgba(255, 255, 255, 0.08);
}

.pace-signup {
    text-align: center;
    color: #8b949e;
    font-size: 0.85rem;
    padding: 4px 0 18px 0;
}
.pace-signup a {
    color: #7dd3fc;
    text-decoration: none;
    font-weight: 600;
}
.pace-signup a:hover { text-decoration: underline; }

.pace-demo-warning {
    background: rgba(251, 192, 45, 0.08);
    border: 1px solid rgba(251, 192, 45, 0.3);
    color: #fbc02d;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.78rem;
    margin-bottom: 14px;
    text-align: center;
}

/* Success state */
.pace-success {
    text-align: center;
    padding: 30px 10px;
}
.pace-success .check {
    width: 56px;
    height: 56px;
    margin: 0 auto 14px auto;
    border-radius: 50%;
    background: rgba(0, 230, 118, 0.12);
    border: 1px solid rgba(0, 230, 118, 0.4);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.6rem;
    color: #00e676;
    animation: successPop 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) both;
}
@keyframes successPop {
    from { transform: scale(0.5); opacity: 0; }
    to { transform: scale(1); opacity: 1; }
}
.pace-success p {
    color: #e6edf3;
    font-weight: 600;
    margin: 0;
}

/* Mobile: hide the hero column, keep the card centered */
@media (max-width: 820px) {
    div[data-testid="column"]:has(.pace-hero) { display: none !important; }
}

/* Respect reduced motion */
@media (prefers-reduced-motion: reduce) {
    .pace-login-bg .orb, .pace-login-bg .spark { animation: none !important; }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.pace-card-anchor) { animation: none !important; }
    .pace-success .check { animation: none !important; }
}
</style>
"""


def _render_background() -> None:
    sparks = "".join(
        f'<div class="spark" style="top:{y}%; left:{x}%; animation-delay:{d}s;"></div>'
        for x, y, d in [
            (12, 20, 0.0), (78, 15, 1.1), (35, 65, 2.3), (88, 70, 0.6),
            (55, 85, 1.8), (20, 45, 3.0), (65, 30, 2.6), (8, 80, 1.4),
        ]
    )
    st.markdown(
        f"""
        <div class="pace-login-bg">
            <div class="grid"></div>
            <div class="orb orb-cyan"></div>
            <div class="orb orb-purple"></div>
            <div class="orb orb-blue"></div>
            {sparks}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    st.markdown(
        """
        <div class="pace-hero">
            <div class="logo-row">
                <span class="logo-mark">⚡</span>
                <span class="logo-text">PaceAI</span>
            </div>
            <h1>Bowling biomechanics,<br/>quantified.</h1>
            <p>
                Kinematic chain profiling, ICC arm-legality checks, and
                injury-risk analytics for coaches, biomechanists, and
                sports physiotherapists.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_login_page() -> None:
    """Renders the full login screen. Call this instead of the rest of
    your dashboard while `is_authenticated()` is False, then `st.stop()`.
    """
    st.markdown(_CSS, unsafe_allow_html=True)
    _render_background()

    # Success screen (shown briefly after a correct login, before rerun
    # hands control back to app.py's authenticated branch)
    if st.session_state.get("_login_success_pending"):
        st.session_state["_login_success_pending"] = False
        st.session_state["authenticated"] = True
        placeholder = st.empty()
        with placeholder.container():
            st.markdown(
                """
                <div class="pace-success">
                    <div class="check">✓</div>
                    <p>Login successful</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        time.sleep(0.7)
        st.rerun()
        return

    left, right = st.columns([1, 1], gap="large")

    with left:
        _render_hero()

    with right:
        st.markdown("<div style='height:6vh'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<span class="pace-card-anchor"></span>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="pace-card-header">
                    <div class="mark">⚡</div>
                    <h2>Welcome back</h2>
                    <p>Sign in to continue to your account</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            padded = st.container()
            with padded:
                if _using_demo_credentials():
                    st.markdown(
                        '<div class="pace-demo-warning">'
                        "Demo credentials active — admin@paceai.local / admin123. "
                        "Set PACEAI_LOGIN_EMAIL / PACEAI_LOGIN_PASSWORD_HASH to replace them."
                        "</div>",
                        unsafe_allow_html=True,
                    )

                email = st.text_input(
                    "Email", key="login_email", placeholder="you@example.com"
                )
                if st.session_state.get("_login_touched") and not _valid_email(email):
                    st.markdown(
                        '<div class="pace-field-error">Please enter a valid email address.</div>',
                        unsafe_allow_html=True,
                    )

                password = st.text_input(
                    "Password", key="login_password", type="password",
                    placeholder="Enter your password",
                )
                if st.session_state.get("_login_touched") and not _valid_password(password):
                    st.markdown(
                        '<div class="pace-field-error">Password must be at least 6 characters.</div>',
                        unsafe_allow_html=True,
                    )

                c1, c2 = st.columns([1, 1])
                with c1:
                    st.checkbox("Remember me", key="login_remember")
                with c2:
                    st.markdown(
                        '<div class="pace-forgot"><span style="color:#6b7280; font-size:0.85rem; cursor:not-allowed; opacity:0.6;" title="Coming soon">Forgot password?</span></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                login_clicked = st.button(
                    "Login →", key="login_submit", type="primary", use_container_width=True
                )

                if st.session_state.get("_login_auth_error"):
                    st.markdown(
                        '<div class="pace-field-error" style="text-align:center;">'
                        "Incorrect email or password." "</div>",
                        unsafe_allow_html=True,
                    )
                    st.session_state["_login_auth_error"] = False

                if login_clicked:
                    st.session_state["_login_touched"] = True
                    if _valid_email(email) and _valid_password(password):
                        with st.spinner("Signing in..."):
                            time.sleep(1.5)  # simulated auth latency
                            ok = check_credentials(email, password)
                        if ok:
                            st.session_state["_login_success_pending"] = True
                            st.rerun()
                        else:
                            st.session_state["_login_auth_error"] = True
                            st.rerun()

                st.markdown('<div class="pace-divider">OR</div>', unsafe_allow_html=True)

                if st.button("🔍  Continue with Google", key="login_google", use_container_width=True, disabled=True):
                    pass
                if st.button("🐙  Continue with GitHub", key="login_github", use_container_width=True, disabled=True):
                    pass

                st.markdown(
                    '<div class="pace-signup"><span style="color:#6b7280; opacity:0.6;">Don\'t have an account? <span style="cursor:not-allowed;" title="Coming soon">Sign up</span></span></div>',
                    unsafe_allow_html=True,
                )
