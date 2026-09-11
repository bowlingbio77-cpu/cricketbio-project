"""
Stage 9: Coaching Recommendation

Turns (feature_vector, performance_score, injury_risk, SHAP contributions)
into plain-English coaching notes. Rule thresholds are illustrative, literature-informed
starting points based on published fast-bowling biomechanics literature (ICC elbow
reference value, typical elite ranges for trunk lean / knee flexion / stride length) --
a real deployment should calibrate these against a labeled dataset from the target
population (age group, format, injury history) via the ML module's targets
rather than hard-coding numbers indefinitely.

Terminology note: risk output is a "biomechanical risk indicator" derived from
literature-informed biomechanical thresholds -- it is NOT a clinical diagnosis,
and the ICC elbow check is a screening indicator (requires lab-grade 3D motion
capture per ICC protocol for an official legality ruling).
"""
from typing import List, Dict
from . import config

# (feature, low, high, note_if_low, note_if_high)
_RULES = [
    ("elbow_flexion_deg", 0, config.ICC_ELBOW_EXTENSION_LIMIT_DEG,
     None,
     "Elbow extension at release exceeds the {limit}° ICC screening reference -- "
     "screening indicator only (not an official ICC measurement). Work on a "
     "straighter arm path through release."),
    ("trunk_lean_deg", 15, 40,
     "Very upright trunk at release -- more lateral flexion toward the target can add "
     "pace and downward trajectory on the ball.",
     "Excessive lateral trunk flexion -- associated with elevated lower-back (lumbar "
     "stress fracture) risk in fast bowlers. Strengthen obliques and reduce lean."),
    ("knee_flexion_deg", 5, 30,
     None,
     "High front-knee flexion at bracing (a 'bent knee' technique) increases vertical "
     "ground reaction force and knee-joint loading. A more braced, extended front leg "
     "reduces impact stress."),
    ("stride_length_norm", 0.6, 1.3,
     "Short stride length relative to height -- a longer, more aggressive final stride "
     "typically improves momentum transfer into the delivery.",
     "Overstriding can reduce front-leg bracing efficiency and increase impact loading -- "
     "check run-up rhythm isn't rushed into the crease."),
    ("shoulder_rotation_deg", 20, 60,
     "Limited shoulder-hip separation -- more counter-rotation ('X-factor') between hips "
     "and shoulders generates additional racket-arm/whip speed.",
     None),
    ("angular_velocity_deg_s", 300, 1200,
     "Lower peak rotational speed through the delivery -- rotational power work "
     "(medicine-ball throws, cable rotations) may help increase release speed.",
     None),
    ("ground_contact_time_s", 0.08, 0.25,
     None,
     "Long front-foot ground contact time can indicate reduced bracing stiffness -- "
     "plyometric and eccentric strength work for the front leg may help."),
]


def generate_recommendations(feature_vector: Dict[str, float],
                              performance_score: float = None,
                              injury_risk: dict = None,
                              shap_contributions: Dict[str, float] = None) -> List[str]:
    notes = []

    for feat, low, high, note_low, note_high in _RULES:
        val = feature_vector.get(feat)
        if val is None:
            continue
        if val < low and note_low:
            notes.append(note_low.format(limit=config.ICC_ELBOW_EXTENSION_LIMIT_DEG))
        elif val > high and note_high:
            notes.append(note_high.format(limit=config.ICC_ELBOW_EXTENSION_LIMIT_DEG))

    if injury_risk and injury_risk.get("risk_level") in ("moderate", "high"):
        # Surface the top SHAP-driven contributor as the priority intervention.
        if shap_contributions:
            top_feat = max(shap_contributions, key=lambda k: shap_contributions[k])
            notes.insert(0,
                f"The biomechanical risk indicator is {injury_risk['risk_level'].upper()}, most driven by "
                f"'{top_feat.replace('_', ' ')}' -- prioritize reviewing this before increasing bowling workload.")
        else:
            notes.insert(0, f"The biomechanical risk indicator is {injury_risk['risk_level'].upper()} -- "
                             f"consider a workload review with the medical/S&C staff.")

    if performance_score is not None:
        if performance_score >= 80:
            notes.append(f"Demonstration performance score: {performance_score:.0f}/100 -- strong, "
                          f"technically sound action vs the literature-informed benchmark. "
                          f"Focus on consistency and repeatability.")
        elif performance_score >= 60:
            notes.append(f"Demonstration performance score: {performance_score:.0f}/100 -- solid "
                          f"foundation with room to refine the flagged areas above.")
        else:
            notes.append(f"Demonstration performance score: {performance_score:.0f}/100 -- several "
                          f"technical elements below the benchmark range; recommend focused net sessions "
                          f"on the items above under coach supervision.")

    if not notes:
        notes.append("No significant technical flags detected -- action is within typical ranges "
                      "across the measured features.")

    return notes
