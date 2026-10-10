"""
Phase 13 -- separated confidence channels.

The failure mode this module exists to prevent: a 0.95 detector score being
displayed next to a biomechanical number, so the reader concludes the
*measurement* is 95% reliable.  It is not.  Detector confidence says nothing
about pose quality, delivery framing, or whether the right person was analysed.

Each channel is an independent, named quantity with an explicit
``available``/``value`` pair.  ``value`` is ``None`` when the channel could not
be measured -- it is never back-filled from a neighbouring channel, and never
defaulted to a number that would look like a measurement.

Derivation rules (deliberately conservative)
---------------------------------------------
``detector_confidence``    mean detection score of the LOCKED bowler track
``tracking_confidence``    share of the delivery window in which the locked
                           track was present, penalised by observed identity
                           switches
``role_confidence``        role-model probability for BOWLER, if the role model
                           ran; otherwise ``None`` (NOT the heuristic score --
                           those are different things and conflating them is
                           exactly the bug)
``action_confidence``      action-model confidence for the predicted phase
``pose_confidence``        mean landmark visibility actually observed
``delivery_confidence``    evidence that a single complete delivery is in frame
``measurement_confidence`` the minimum of the channels that a biomechanical
                           number genuinely depends on (pose, delivery, and the
                           identity lock).  Reported separately, never as
                           "accuracy".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

CHANNELS = (
    "detector_confidence",
    "tracking_confidence",
    "role_confidence",
    "action_confidence",
    "pose_confidence",
    "delivery_confidence",
    "measurement_confidence",
)

#: Channels a biomechanical number depends on. ``measurement_confidence`` is
#: the min over the *available* subset; if a required channel is unavailable
#: the whole measurement confidence is unavailable, not silently high.
MEASUREMENT_DEPENDENCIES = ("tracking_confidence", "pose_confidence",
                            "delivery_confidence")


@dataclass
class Channel:
    name: str
    value: Optional[float]
    available: bool
    source: str = ""
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def display(self) -> str:
        if not self.available or self.value is None:
            return "NOT MEASURED" + (f" ({self.reason})" if self.reason else "")
        return f"{self.value:.3f}" + (f" [{self.source}]" if self.source else "")


@dataclass
class ConfidenceBundle:
    channels: Dict[str, Channel] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    # -- construction ------------------------------------------------------- #

    @classmethod
    def empty(cls) -> "ConfidenceBundle":
        b = cls()
        for c in CHANNELS:
            b.channels[c] = Channel(c, None, False, reason="not evaluated")
        return b

    def set(self, name: str, value: Optional[float], source: str,
            reason: str = "") -> "ConfidenceBundle":
        if name not in self.channels:
            raise KeyError(f"unknown confidence channel {name!r}; "
                           f"expected one of {CHANNELS}")
        if value is None:
            self.channels[name] = Channel(name, None, False, source, reason or "unavailable")
        else:
            v = float(value)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {v}")
            self.channels[name] = Channel(name, v, True, source, reason)
        return self

    def get(self, name: str) -> Channel:
        if name not in self.channels:
            raise KeyError(f"unknown confidence channel {name!r}")
        return self.channels[name]

    def value(self, name: str) -> Optional[float]:
        c = self.channels.get(name)
        return c.value if c and c.available else None

    def available(self, name: str) -> bool:
        c = self.channels.get(name)
        return bool(c and c.available and c.value is not None)

    # -- derived ------------------------------------------------------------ #

    def finalize(self) -> "ConfidenceBundle":
        """Compute ``measurement_confidence`` from its declared dependencies.

        Conservative by design: if ANY dependency is unavailable the derived
        channel is unavailable too.  A missing channel must not raise the
        reported number.
        """
        missing = [d for d in MEASUREMENT_DEPENDENCIES if not self.available(d)]
        if missing:
            self.channels["measurement_confidence"] = Channel(
                "measurement_confidence", None, False,
                source="min(" + ",".join(MEASUREMENT_DEPENDENCIES) + ")",
                reason="unavailable dependency: " + ", ".join(missing))
            self.warnings.append(
                "measurement_confidence NOT MEASURED because "
                + ", ".join(missing) + " could not be measured. A biomechanical "
                "number without a measurement confidence must not be presented "
                "as reliable.")
            return self
        vals = [float(self.value(d)) for d in MEASUREMENT_DEPENDENCIES]
        self.channels["measurement_confidence"] = Channel(
            "measurement_confidence", min(vals), True,
            source="min(" + ",".join(MEASUREMENT_DEPENDENCIES) + ")")
        return self

    # -- serialisation ------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channels": {k: v.to_dict() for k, v in self.channels.items()},
            "measurement_dependencies": list(MEASUREMENT_DEPENDENCIES),
            "warnings": list(self.warnings),
            "disclaimer": (
                "These channels are separate on purpose. A high detector "
                "confidence does NOT imply a high-confidence biomechanical "
                "measurement, and none of these is an accuracy figure."),
        }

    def to_rows(self) -> List[Dict[str, Any]]:
        return [
            {"channel": k, "value": v.value if v.available else None,
             "available": v.available, "source": v.source, "reason": v.reason}
            for k, v in self.channels.items()
        ]

    def summary_line(self) -> str:
        parts = [f"{k}={self.channels[k].display()}" for k in CHANNELS]
        return " | ".join(parts)


# --------------------------------------------------------------------------- #
# Builders from pipeline facts
# --------------------------------------------------------------------------- #

def from_pipeline(*, detection_confidences: Optional[List[float]] = None,
                  track_coverage: Optional[float] = None,
                  identity_switches: Optional[int] = None,
                  role_confidence: Optional[float] = None,
                  action_confidence: Optional[float] = None,
                  pose_confidence: Optional[float] = None,
                  delivery_reliable: Optional[bool] = None,
                  bowler_confirmed: Optional[bool] = None) -> ConfidenceBundle:
    """Build a bundle from whatever the pipeline actually measured.

    Anything not passed in stays ``NOT MEASURED``.  Note that the heuristic
    bowler score (src/tracking.py ``_confidence``) is deliberately NOT used as
    ``role_confidence``: it is a different quantity, and substituting one for
    the other is the confusion this module exists to prevent.
    """
    b = ConfidenceBundle.empty()

    if detection_confidences:
        b.set("detector_confidence", sum(detection_confidences) / len(detection_confidences),
              "mean YOLO score on the locked track")
    else:
        b.set("detector_confidence", None, "yolo",
              "no per-frame detector scores available for the locked track")

    if track_coverage is not None:
        cov = max(0.0, min(1.0, float(track_coverage)))
        if identity_switches:
            cov *= max(0.0, 1.0 - 0.2 * int(identity_switches))
            b.warnings.append(
                f"tracking_confidence penalised for {identity_switches} observed "
                "bowler identity switch(es) across half-clip windows.")
        b.set("tracking_confidence", cov, "frame coverage x identity-switch penalty")
    else:
        b.set("tracking_confidence", None, "bytetrack", "track coverage not measured")

    if role_confidence is not None:
        b.set("role_confidence", role_confidence, "cricket role model")
    else:
        b.set("role_confidence", None, "cricket role model",
              "no validated role model; the heuristic bowler score is a "
              "DIFFERENT quantity and is not substituted here")

    if action_confidence is not None:
        b.set("action_confidence", action_confidence, "bowling action model")
    else:
        b.set("action_confidence", None, "bowling action model", "action model not run")

    if pose_confidence is not None:
        b.set("pose_confidence", pose_confidence, "mean MediaPipe landmark visibility")
    else:
        b.set("pose_confidence", None, "mediapipe", "pose not estimated")

    if delivery_reliable is True and bowler_confirmed is not False:
        b.set("delivery_confidence", 1.0, "analyze_delivery_phases reliable=True")
    elif delivery_reliable is False:
        b.set("delivery_confidence", 0.0, "analyze_delivery_phases reliable=False",
              "the clip did not contain a single clean delivery")
    else:
        b.set("delivery_confidence", None, "feature_engineering",
              "delivery reliability not evaluated")

    if bowler_confirmed is False:
        b.warnings.append(
            "BOWLER NOT CONFIRMED: role/action evidence is advisory only. "
            "Normal bowler biomechanics must not be produced.")
    return b.finalize()


NOT_CONFIRMED_MESSAGE = "BOWLER NOT CONFIRMED"


def not_confirmed_result(reason: str) -> Dict[str, Any]:
    """The explicit refusal object for a failed bowler confirmation."""
    b = ConfidenceBundle.empty()
    b.set("delivery_confidence", None, "", f"bowler not confirmed: {reason}")
    b.finalize()
    return {
        "status": NOT_CONFIRMED_MESSAGE,
        "reason": reason,
        "biomechanics_available": False,
        "coaching_available": False,
        "confidence": b.to_dict(),
        "message": (
            f"{NOT_CONFIRMED_MESSAGE} ({reason}). The bowler could not be "
            "identified with sufficient evidence, so no bowler biomechanics "
            "result is produced. measurement_confidence is NOT MEASURED and no "
            "biomechanical number is reported at all. Re-run and confirm the "
            "bowler manually, or choose a clip with a clearer run-up."),
    }
