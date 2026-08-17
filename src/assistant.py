"""
Stage 10: AI assistant layer over a delivery analysis.

Bridges the biomechanics pipeline to the self-hosted Odysseus AI workspace
(clone of https://github.com/odysseus-dev/odysseus). Builds a structured
markdown report from an :class:`AnalysisResult`, then submits it as context to
an Odysseus chat session using a scoped API token so the LLM can interpret and
annotate the machine-generated analysis (summary, technical priorities, injury
risk narrative).

Degrades gracefully: when Odysseus is not configured or not running, callers
still get the local report. All access goes through the stdlib (urllib) so no
extra dependency is required.

Odysseus contract used here (from its routes/):
  * auth      : ``Authorization: Bearer <ody_...>`` scoped API token
  * health    : GET  /api/health                     (auth-exempt)
  * session   : POST /api/session  (form fields)     -> {"id": ...}
  * chat      : POST /api/chat     (JSON ChatRequest) -> {"response": ...}
"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request

from .config import FEATURE_NAMES, ICC_ELBOW_EXTENSION_LIMIT_DEG

# --------------------------------------------------------------------------- #
# Configuration (all overridable via environment variables)
# --------------------------------------------------------------------------- #
DEFAULT_BASE_URL = "http://localhost:7000"
DEFAULT_TIMEOUT = 120.0

ENV_BASE_URL = "ODYSSEUS_BASE_URL"
ENV_API_TOKEN = "ODYSSEUS_API_TOKEN"
ENV_ENDPOINT_ID = "ODYSSEUS_ENDPOINT_ID"
ENV_ENDPOINT_URL = "ODYSSEUS_ENDPOINT_URL"
ENV_MODEL = "ODYSSEUS_MODEL"
ENV_TIMEOUT = "ODYSSEUS_TIMEOUT"

DEFAULT_SESSION_NAME = "cricket-biomech-assistant"

DEFAULT_QUESTION = (
    "Analyze this delivery as an expert cricket biomechanics coach. Summarize the "
    "delivery in 2-3 sentences, then highlight the strongest and weakest technical "
    "aspects, flag any injury-risk concerns, and recommend 3-4 concrete, prioritized "
    "technical/training adjustments."
)

COHORT_QUESTION = (
    "Compare this cohort of deliveries as an expert cricket biomechanics coach. "
    "Identify cross-delivery patterns in performance, injury risk, technique "
    "consistency, and ball-tracking quality. Rank the deliveries most in need of "
    "attention (and why), and list any repeat technical or risk flags."
)

# Human-friendly units for the kinematics table (mirrors config.FEATURE_NAMES).
FEATURE_UNITS = {
    "shoulder_rotation_deg": "deg",
    "elbow_flexion_deg": "deg (flexion from straight)",
    "wrist_angle_deg": "deg",
    "hip_rotation_deg": "deg",
    "knee_flexion_deg": "deg",
    "trunk_lean_deg": "deg",
    "stride_length_norm": "x height",
    "release_angle_deg": "deg",
    "angular_velocity_deg_s": "deg/s",
    "ground_contact_time_s": "s",
}

MAX_MESSAGE_CHARS = 48000


class AssistantError(Exception):
    """Base error for the assistant layer."""


class AssistantNotConfigured(AssistantError):
    """Odysseus is reachable but required credentials/settings are missing."""


class AssistantNotRunning(AssistantError):
    """Odysseus could not be reached (server down or unreachable)."""


class AssistantSetupError(AssistantError):
    """A chat session could not be created or the server rejected the request."""


# --------------------------------------------------------------------------- #
# Report builder (local, deterministic -- no LLM required)
# --------------------------------------------------------------------------- #
def _fmt(value) -> str:
    """Format a numeric value for the report; None/na -> 'n/a'."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".") if abs(value) < 1000 else f"{value:.2e}"
    return str(value)


def _elbow_legality(feature_vector: dict) -> str:
    elbow = feature_vector.get("elbow_flexion_deg")
    if elbow is None:
        return "n/a"
    status = "LEGAL" if elbow <= ICC_ELBOW_EXTENSION_LIMIT_DEG else "EXCEEDS ICC LIMIT"
    return f"{status} (ICC limit: <= {ICC_ELBOW_EXTENSION_LIMIT_DEG} deg extension at release)"


def _top_contributors(shap_values: dict, n: int = 5) -> str:
    if not shap_values:
        return "n/a"
    ordered = sorted(shap_values.items(), key=lambda kv: kv[1], reverse=True)
    lines = []
    for name, val in ordered[:n]:
        lines.append(f"- {name}: {val:+.3f}")
    return "\n".join(lines)


def _fmt_probabilities(probs) -> str:
    """Render model class probabilities (dict {class: p} or list aligned to the
    bundle's label map, e.g. [no, minor, moderate, severe])."""
    if not probs:
        return "n/a"
    if isinstance(probs, dict):
        return ", ".join(f"{k}: {v:.3f}" for k, v in probs.items())
    return ", ".join(f"class {i}: {v:.3f}" for i, v in enumerate(probs))


def assistant_report(result, include_timings: bool = True) -> str:
    """Render an :class:`AnalysisResult` as a structured markdown report."""
    parts = ["# Delivery Biomechanics Analysis Report", ""]

    meta = []
    if getattr(result, "video_path", None):
        meta.append(f"- **Video**: {result.video_path}")
    meta.append(f"- **Bowling arm**: {result.bowling_arm or 'n/a'}")
    meta.append(f"- **Camera view**: {result.camera_view or 'unknown'}")
    parts.append("\n".join(meta))
    parts.append("")

    parts.append("## Kinematics (release phase)")
    rows = ["| Feature | Value | Unit |", "|---|---|---|"]
    for name in FEATURE_NAMES:
        rows.append(f"| {name} | {_fmt(result.feature_vector.get(name))} | {FEATURE_UNITS.get(name, '')} |")
    parts.append("\n".join(rows))
    parts.append("")

    parts.append(f"**ICC elbow legality**: {_elbow_legality(result.feature_vector)}")
    parts.append("")

    if result.performance_score is not None:
        parts.append("## Performance model")
        parts.append(f"- **Score**: {result.performance_score:.1f} / 100")
        parts.append("- **Top contributors (SHAP)**:")
        parts.append(_top_contributors(result.shap_contributions_performance))
        parts.append("")
    else:
        parts.append("## Performance model")
        parts.append("- **Score**: n/a (ML bundles not supplied)")
        parts.append("")

    if result.injury_risk:
        risk = result.injury_risk
        parts.append("## Injury risk model")
        parts.append(f"- **Level**: {risk.get('risk_level', 'n/a')}")
        parts.append(f"- **Risk index**: {_fmt(risk.get('risk_index'))}")
        probs = risk.get("probabilities")
        if probs:
            parts.append(f"- **Probabilities**: {_fmt_probabilities(probs)}")
        parts.append("- **Top contributors (SHAP)**:")
        parts.append(_top_contributors(result.shap_contributions_injury))
        parts.append("")
    else:
        parts.append("## Injury risk model")
        parts.append("- **Level**: n/a (ML bundles not supplied)")
        parts.append("")

    ball_stats = result.ball_stats or {}
    if ball_stats:
        parts.append("## Ball tracking")
        parts.append(f"- **Outcome**: {ball_stats.get('outcome', 'n/a')}")
        if ball_stats.get("release_idx") is not None:
            parts.append(f"- **Release frame**: {ball_stats['release_idx']}")
        if ball_stats.get("impact_idx") is not None:
            parts.append(f"- **Impact frame**: {ball_stats['impact_idx']}")
        parts.append(f"- **Frames**: {ball_stats.get('n_frames', 'n/a')} "
                     f"(detected {ball_stats.get('n_detected', 'n/a')}, "
                     f"interpolated {ball_stats.get('n_interpolated', 'n/a')})")
        parts.append(f"- **YOLO coverage**: {_fmt(ball_stats.get('coverage_pct'))}%")
        parts.append(f"- **Avg speed**: {_fmt(ball_stats.get('avg_speed_px_s'))} px/s")
        parts.append(f"- **Max speed**: {_fmt(ball_stats.get('max_speed_px_s'))} px/s")
        parts.append("")

    if result.coaching_notes:
        parts.append("## Coaching notes (rule-based)")
        for note in result.coaching_notes:
            parts.append(f"- {note}")
        parts.append("")

    if result.warnings:
        parts.append("## Warnings")
        for warning in result.warnings:
            parts.append(f"- {warning}")
        parts.append("")

    if include_timings and result.stage_times:
        parts.append("## Stage timings (s)")
        for stage, secs in result.stage_times.items():
            parts.append(f"- {stage}: {secs:.2f}")
        parts.append("")

    return "\n".join(parts).strip()


def _build_chat_message(report: str, question: str = None) -> str:
    """Compose the single chat message that carries report context + question."""
    question = (question or DEFAULT_QUESTION).strip()
    message = (
        "You are an expert cricket biomechanics analyst reviewing a machine-generated "
        "analysis of one bowling delivery. Interpret the numbers, connect them to "
        "cricket biomechanics, and answer only using the report below unless asked for "
        "outside knowledge.\n\n"
        "--- DELIVERY ANALYSIS REPORT ---\n"
        f"{report}\n"
        "--- END REPORT ---\n\n"
        f"QUESTION: {question}"
    )
    if len(message) > MAX_MESSAGE_CHARS:
        excess = len(message) - MAX_MESSAGE_CHARS + 200
        message = message[: MAX_MESSAGE_CHARS - 200] + f"\n\n[report truncated by {excess} chars]"
    return message


def cohort_report(labeled_results, include_timings: bool = False) -> str:
    """Render several labeled deliveries as one compact cohort report.

    `labeled_results` is an iterable of ``(label, AnalysisResult)`` pairs. Each
    delivery gets a compact block (scores, legality, key features, ball outcome)
    plus a deterministic machine summary (mean performance, injury-level counts,
    ICC legality count, ball-outcome counts, recurring warnings). Intended for
    comparative / trend questions to the assistant.
    """
    key_features = ("elbow_flexion_deg", "trunk_lean_deg", "knee_flexion_deg",
                    "stride_length_norm", "angular_velocity_deg_s",
                    "ground_contact_time_s", "release_angle_deg")
    results = list(labeled_results)
    parts = [f"# Delivery Cohort Analysis Report ({len(results)} deliveries)", ""]

    performance_scores = []
    risk_counts = {}
    legal_count = 0
    elbow_seen = 0
    outcome_counts = {}
    warning_counts = {}
    for label, result in results:
        name = label or os.path.basename(result.video_path or "clip")
        parts.append(f"## {name}")
        if result.performance_score is not None:
            performance_scores.append(result.performance_score)
            parts.append(f"- **Performance**: {result.performance_score:.1f} / 100")
        if result.injury_risk:
            level = result.injury_risk.get("risk_level", "n/a")
            risk_counts[level] = risk_counts.get(level, 0) + 1
            parts.append(f"- **Injury risk**: {level} "
                         f"(index {_fmt(result.injury_risk.get('risk_index'))})")
        elbow = result.feature_vector.get("elbow_flexion_deg")
        if elbow is not None:
            elbow_seen += 1
            if elbow <= ICC_ELBOW_EXTENSION_LIMIT_DEG:
                legal_count += 1
            parts.append(f"- **ICC legality**: "
                         f"{'LEGAL' if elbow <= ICC_ELBOW_EXTENSION_LIMIT_DEG else 'EXCEEDS LIMIT'} "
                         f"(elbow {_fmt(elbow)} deg)")
        for feat in key_features:
            parts.append(f"- {feat}: {_fmt(result.feature_vector.get(feat))}")
        ball_stats = result.ball_stats or {}
        outcome = ball_stats.get("outcome")
        if outcome:
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            parts.append(f"- **Ball outcome**: {outcome} "
                         f"(coverage {_fmt(ball_stats.get('coverage_pct'))}%)")
        for warning in result.warnings:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
        parts.append("")

    parts.append("## Cohort summary (machine-generated)")
    if performance_scores:
        mean_p = sum(performance_scores) / len(performance_scores)
        parts.append(f"- **Mean performance**: {mean_p:.1f} / 100 "
                     f"(range {min(performance_scores):.1f}-{max(performance_scores):.1f}, "
                     f"n={len(performance_scores)})")
    if risk_counts:
        parts.append("- **Injury risk distribution**: "
                     + ", ".join(f"{k}: {v}" for k, v in sorted(risk_counts.items())))
    if elbow_seen:
        parts.append(f"- **ICC legal deliveries**: {legal_count} / {elbow_seen}")
    if outcome_counts:
        parts.append("- **Ball-tracking outcomes**: "
                     + ", ".join(f"{k}: {v}" for k, v in sorted(outcome_counts.items())))
    top_warnings = sorted(warning_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
    if top_warnings:
        parts.append("- **Most common warnings**:")
        for warning, count in top_warnings:
            parts.append(f"  - ({count}x) {warning}")
    parts.append("")

    if include_timings:
        parts.append("## Per-delivery stage timings (s)")
        for label, result in results:
            name = label or os.path.basename(result.video_path or "clip")
            timings = ", ".join(f"{k}={v:.2f}" for k, v in result.stage_times.items())
            parts.append(f"- {name}: {timings}")
        parts.append("")

    return "\n".join(parts).strip()


# --------------------------------------------------------------------------- #
# Odysseus client (stdlib HTTP only)
# --------------------------------------------------------------------------- #
class OdysseusAssistant:
    """Minimal programmatic client for the Odysseus chat workspace.

    Creates one chat session per process and submits the analysis report as
    context. Requires a scoped API token with ``chat`` scope plus a configured
    model endpoint (``ODYSSEUS_ENDPOINT_ID`` is preferred for API-token callers
    because non-admin tokens cannot supply a raw endpoint URL).
    """

    def __init__(self, base_url: str = None, api_token: str = None,
                 endpoint_id: str = None, endpoint_url: str = None,
                 model: str = None, timeout: float = None,
                 session_name: str = DEFAULT_SESSION_NAME):
        self.base_url = (base_url or os.environ.get(ENV_BASE_URL) or DEFAULT_BASE_URL).rstrip("/")
        self.api_token = api_token or os.environ.get(ENV_API_TOKEN)
        self.endpoint_id = endpoint_id or os.environ.get(ENV_ENDPOINT_ID)
        self.endpoint_url = endpoint_url or os.environ.get(ENV_ENDPOINT_URL)
        self.model = model or os.environ.get(ENV_MODEL)
        self.timeout = timeout if timeout is not None else float(os.environ.get(ENV_TIMEOUT, DEFAULT_TIMEOUT))
        self.session_name = session_name
        self._session_id = None

    # -- readiness --------------------------------------------------------- #
    def configured(self) -> bool:
        """True when we have enough to talk to Odysseus (token + an endpoint)."""
        return bool(self.api_token) and bool(self.endpoint_id or self.endpoint_url)

    def missing_settings(self) -> list:
        missing = []
        if not self.api_token:
            missing.append(ENV_API_TOKEN)
        if not (self.endpoint_id or self.endpoint_url):
            missing.append(f"{ENV_ENDPOINT_ID} or {ENV_ENDPOINT_URL}")
        return missing

    def ping(self, timeout: float = 3.0) -> bool:
        """Cheap reachability probe (auth-exempt /api/health)."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/health", timeout=timeout) as resp:
                return resp.status == 200
        except OSError:
            return False

    # -- low-level requests ------------------------------------------------ #
    def _headers(self) -> dict:
        headers = {"Accept": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    def _request(self, url: str, data: bytes, content_type: str) -> dict:
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={**self._headers(), "Content-Type": content_type},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise AssistantError(f"Odysseus HTTP {exc.code} at {url}: {detail}") from exc
        except OSError as exc:
            raise AssistantNotRunning(
                f"Could not reach Odysseus at {self.base_url}. Start it with "
                f"`docker compose up` (from the odysseus repo)."
            ) from exc

    # -- session lifecycle -------------------------------------------------- #
    def ensure_session(self) -> str:
        """Create (once) and return a chat session id for this assistant."""
        if self._session_id:
            return self._session_id
        if not self.api_token:
            raise AssistantNotConfigured(
                f"Missing {ENV_API_TOKEN}. Create a scoped API token with `chat` "
                "scope in Odysseus (Settings -> API tokens)."
            )
        if not (self.endpoint_id or self.endpoint_url):
            raise AssistantNotConfigured(
                f"Set {ENV_ENDPOINT_ID} (preferred for API tokens) or "
                f"{ENV_ENDPOINT_URL} to the LLM model endpoint Odysseus should use."
            )
        form = {"name": self.session_name}
        if self.endpoint_id:
            form["endpoint_id"] = self.endpoint_id
        else:
            form["endpoint_url"] = self.endpoint_url
        if self.model:
            form["model"] = self.model
        payload = self._request(
            f"{self.base_url}/api/session",
            urllib.parse.urlencode(form).encode("utf-8"),
            "application/x-www-form-urlencoded",
        )
        sid = payload.get("id") or payload.get("session_id")
        if not sid:
            raise AssistantSetupError(f"Unexpected session response from Odysseus: {payload}")
        self._session_id = sid
        return sid

    # -- chat --------------------------------------------------------------- #
    def chat(self, message: str, session_id: str = None,
             use_web: bool = False, use_research: bool = False) -> str:
        """Send one message to a session and return the assistant reply."""
        sid = session_id or self.ensure_session()
        payload = {
            "message": message,
            "session": sid,
            "use_web": use_web,
            "use_research": use_research,
        }
        data = self._request(
            f"{self.base_url}/api/chat",
            json.dumps(payload).encode("utf-8"),
            "application/json",
        )
        response = data.get("response")
        if not isinstance(response, str) or not response.strip():
            raise AssistantError(f"Unexpected chat response from Odysseus: {data}")
        return response

    # -- analysis entry point ---------------------------------------------- #
    def analyze(self, result, question: str = None, session_id: str = None) -> str:
        """Render `result` to a report and ask Odysseus to interpret it."""
        message = _build_chat_message(assistant_report(result), question)
        return self.chat(message, session_id=session_id)
