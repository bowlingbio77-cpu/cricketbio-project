"""Tests for the AI-assistant layer (report builder + Odysseus client).

The client tests spin up a tiny stdlib HTTP server to prove the exact request
shape Odysseus expects (Bearer auth, form-encoded session creation, JSON chat)
and the graceful fallback paths (server down / missing token / HTTP errors).
No Odysseus instance is required.
"""
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src import assistant
from src.assistant import (
    AssistantError,
    AssistantNotConfigured,
    AssistantNotRunning,
    OdysseusAssistant,
)
from src.config import FEATURE_NAMES
from src.pipeline import AnalysisResult


def make_result(**overrides):
    kwargs = dict(
        feature_vector={
            "shoulder_rotation_deg": 41.0,
            "elbow_flexion_deg": 6.2,
            "wrist_angle_deg": 150.5,
            "hip_rotation_deg": 12.3,
            "knee_flexion_deg": 18.7,
            "trunk_lean_deg": 27.9,
            "stride_length_norm": 0.95,
            "release_angle_deg": -11.4,
            "angular_velocity_deg_s": 842.3,
            "ground_contact_time_s": 0.16,
        },
        performance_score=73.4,
        injury_risk={"risk_level": "moderate", "risk_index": 0.52,
                     "probabilities": {"none": 0.1, "minor": 0.3, "moderate": 0.4, "severe": 0.2}},
        shap_contributions_performance={"stride_length_norm": 4.2, "elbow_flexion_deg": -1.1},
        shap_contributions_injury={"knee_flexion_deg": 0.7},
        coaching_notes=["note one", "note two"],
        stage_times={"preprocess": 1.2, "ball_tracking": 3.4, "total": 9.8},
        warnings=["warning one"],
        camera_view="behind",
        bowling_arm="right",
        video_path="clip_1.avi",
        ball_stats={"outcome": "ok", "release_idx": 28, "impact_idx": 41,
                    "n_frames": 14, "n_detected": 11, "n_interpolated": 3,
                    "coverage_pct": 78.6, "avg_speed_px_s": 120.0, "max_speed_px_s": 410.0},
    )
    kwargs.update(overrides)
    return AnalysisResult(**kwargs)


# --------------------------------------------------------------------------- #
# Report builder
# --------------------------------------------------------------------------- #
def test_report_contains_all_sections():
    report = assistant.assistant_report(make_result())
    assert "# Delivery Biomechanics Analysis Report" in report
    assert "## Kinematics" in report
    assert "## Performance model" in report
    assert "## Injury risk model" in report
    assert "## Ball tracking" in report
    assert "## Coaching notes" in report
    assert "## Warnings" in report
    for name in FEATURE_NAMES:
        assert name in report
    assert "73.4" in report
    assert "moderate" in report
    assert "release frame" in report.lower()
    assert "78.6%" in report
    assert "LEGAL" in report


def test_report_handles_missing_ml():
    report = assistant.assistant_report(make_result(performance_score=None, injury_risk=None,
                                                    shap_contributions_performance=None,
                                                    shap_contributions_injury=None))
    assert "n/a (ML bundles not supplied)" in report
    assert "LEGAL" in report


def test_report_handles_probabilities_as_list():
    injury = {"risk_level": "moderate", "risk_index": 2,
              "probabilities": [0.1, 0.3, 0.4, 0.2]}
    report = assistant.assistant_report(make_result(injury_risk=injury))
    assert "class 0: 0.100" in report and "class 2: 0.400" in report


def test_report_elbow_legality_exceeded():
    report = assistant.assistant_report(make_result(feature_vector={
        **make_result().feature_vector, "elbow_flexion_deg": 21.0}))
    assert "EXCEEDS ICC LIMIT" in report


def test_report_top_contributors_ordered():
    top = assistant._top_contributors({"a": 0.5, "b": 2.0, "c": -1.0})
    lines = top.splitlines()
    assert "b: +2.000" in lines[0]
    assert "a: +0.500" in lines[1]
    assert "c: -1.000" in lines[2]


def test_report_na_when_ball_stats_empty():
    report = assistant.assistant_report(make_result(ball_stats={}))
    assert "## Ball tracking" not in report


def test_build_message_includes_report_and_question():
    report = assistant.assistant_report(make_result())
    message = assistant._build_chat_message(report, question="Is the action legal?")
    assert "DELIVERY ANALYSIS REPORT" in message
    assert "QUESTION: Is the action legal?" in message


def test_build_message_default_question_when_none():
    report = assistant.assistant_report(make_result())
    message = assistant._build_chat_message(report)
    assert "QUESTION: " in message
    assert assistant.DEFAULT_QUESTION in message


def test_build_message_truncates_oversized_reports():
    big = "x" * (assistant.MAX_MESSAGE_CHARS * 2)
    message = assistant._build_chat_message(big)
    assert len(message) <= assistant.MAX_MESSAGE_CHARS
    assert "[report truncated" in message


def test_cohort_report_summary():
    r1 = make_result(performance_score=80.0, video_path="a.avi",
                     injury_risk={"risk_level": "low", "risk_index": 0.2})
    r2 = make_result(performance_score=60.0, video_path="b.avi",
                     injury_risk={"risk_level": "high", "risk_index": 0.8},
                     feature_vector={**make_result().feature_vector, "elbow_flexion_deg": 20.0})
    report = assistant.cohort_report([("delivery-A", r1), ("delivery-B", r2)])
    assert "# Delivery Cohort Analysis Report (2 deliveries)" in report
    assert "## delivery-A" in report and "## delivery-B" in report
    assert "Mean performance" in report and "70.0" in report
    assert "low: 1" in report and "high: 1" in report
    assert "**ICC legal deliveries**: 1 / 2" in report


# --------------------------------------------------------------------------- #
# Client against a mock Odysseus server
# --------------------------------------------------------------------------- #
class _MockHandler(BaseHTTPRequestHandler):
    sessions_created = 0
    chat_bodies = []
    auth_headers = []
    fail_chat = False

    def log_message(self, *args):
        pass

    def _reply(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/health":
            self._reply(200, {"status": "ok"})
        else:
            self._reply(404, {"detail": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        type(self).auth_headers.append(self.headers.get("Authorization"))
        if self.path == "/api/session":
            type(self).sessions_created += 1
            self._reply(200, {"id": "sess-1", "name": "x", "model": "m"})
        elif self.path == "/api/chat":
            payload = json.loads(body)
            type(self).chat_bodies.append(payload)
            if type(self).fail_chat:
                self._reply(400, {"detail": "No model selected for this chat"})
            else:
                self._reply(200, {"response": "The delivery looks solid."})
        else:
            self._reply(404, {"detail": "not found"})


@pytest.fixture()
def mock_server():
    _MockHandler.sessions_created = 0
    _MockHandler.chat_bodies = []
    _MockHandler.auth_headers = []
    _MockHandler.fail_chat = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield base, _MockHandler
    server.shutdown()
    thread.join(timeout=5)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_ping_true_when_server_up(mock_server):
    base, _ = mock_server
    assert OdysseusAssistant(base_url=base, api_token="tok").ping() is True


def test_ping_false_when_server_down():
    client = OdysseusAssistant(base_url=f"http://127.0.0.1:{_free_port()}", api_token="tok")
    assert client.ping() is False


def test_missing_token_raises_not_configured(mock_server):
    base, _ = mock_server
    client = OdysseusAssistant(base_url=base, endpoint_url="http://llm:11434")
    with pytest.raises(AssistantNotConfigured):
        client.ensure_session()


def test_missing_endpoint_raises_not_configured(mock_server):
    base, _ = mock_server
    client = OdysseusAssistant(base_url=base, api_token="tok")
    with pytest.raises(AssistantNotConfigured):
        client.ensure_session()


def test_ensure_session_uses_bearer_and_creates_once(mock_server):
    base, handler = mock_server
    client = OdysseusAssistant(base_url=base, api_token="ody_123", endpoint_id="ep-9")
    sid = client.ensure_session()
    assert sid == "sess-1"
    assert client.ensure_session() == "sess-1"
    assert handler.sessions_created == 1
    assert handler.auth_headers[0] == "Bearer ody_123"


def test_chat_sends_report_context(mock_server):
    base, handler = mock_server
    client = OdysseusAssistant(base_url=base, api_token="ody_123", endpoint_id="ep-9")
    reply = client.analyze(make_result(), question="Is the action legal?")
    assert reply == "The delivery looks solid."
    body = handler.chat_bodies[0]
    assert body["session"] == "sess-1"
    assert body["use_web"] is False and body["use_research"] is False
    assert "DELIVERY ANALYSIS REPORT" in body["message"]
    assert "QUESTION: Is the action legal?" in body["message"]


def test_chat_http_error_raises_assistant_error(mock_server):
    base, handler = mock_server
    handler.fail_chat = True
    client = OdysseusAssistant(base_url=base, api_token="ody_123", endpoint_id="ep-9")
    with pytest.raises(AssistantError, match="Odysseus HTTP 400"):
        client.chat("hello")


def test_chat_server_down_raises_not_running():
    client = OdysseusAssistant(base_url=f"http://127.0.0.1:{_free_port()}", api_token="tok",
                               endpoint_id="ep-9", timeout=2.0)
    with pytest.raises(AssistantNotRunning):
        client.chat("hello")
