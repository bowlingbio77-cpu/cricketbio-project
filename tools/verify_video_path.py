"""End-to-end check of the REAL video path into the new result page.

Runs the actual CV pipeline on a ground-truth clip, then reproduces app.py's
merge + render_result_page wiring. This is the one path AppTest cannot reach,
and the one where the discarded-AnalysisResult bug lived.

Run:  python tools/verify_video_path.py [clip.avi]
"""
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, ml_models, pipeline, result_view

# Must stay identical to app._VIDEO_ONLY_FIELDS.
_VIDEO_ONLY_FIELDS = (
    "feature_provenance", "landmark_source_summary", "stage_backends",
    "bowler_bboxes", "bowler_track_id", "bowler_confidence",
    "bowler_confirmed", "bowler_confirm_reason", "bowler_candidates",
    "identity_switch_count", "batting_stances", "striker_track_id",
    "non_striker_track_id", "original_frame_dims", "player_roles",
    "ball_stats", "video_path", "pose_video_path", "reels_video_path",
    "analysis_replay_path", "bowling_arm", "camera_view", "warnings",
    "subject_verified", "delivery_reliable", "scoring_blocked_reason",
)


def merge_video_result(result, video_result):
    if video_result is None:
        return result
    overrides = {}
    for name in _VIDEO_ONLY_FIELDS:
        value = getattr(video_result, name, None)
        if value is None or value == {} or value == []:
            continue
        overrides[name] = value
    if not overrides:
        return result
    return dataclasses.replace(result, **overrides)


def main():
    clip = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        "data", "gt_clips", "fast_left_clean", "clip.avi")
    clip = os.path.abspath(clip)
    print(f"clip: {clip}")

    perf = injury = None
    perf_path = os.path.join(config.MODEL_DIR, "performance_random_forest.joblib")
    injury_path = os.path.join(config.MODEL_DIR, "injury_random_forest.joblib")
    for label, path in (("perf", perf_path), ("injury", injury_path)):
        try:
            bundle = ml_models.load_bundle(path)
        except Exception as e:
            print(f"{label} bundle unavailable ({path}): {e}")
            continue
        if label == "perf":
            perf = bundle
        else:
            injury = bundle
    if perf is None or injury is None:
        print("run `python train_demo_model.py` first")

    print("running CV pipeline (this takes a while)...")
    # The bundled gt_clips are synthetic renders, so the cricket precheck
    # rejects them ("no pose detected"). We bypass it only to validate the
    # WIRING -- that the fields analyze_video returns are the ones we merge.
    # The biomechanics from a synthetic clip are meaningless; ignore them.
    try:
        video_result = pipeline.analyze_video(
            clip, bowling_arm="right", performance_bundle=perf,
            injury_bundle=injury, target_fps=20, precheck=False)
    except pipeline.CricketPrecheckError as e:
        print(f"PRECHECK REJECTED: {e}")
        print("No real footage in the repo; cannot verify the video path.")
        return 2
    except Exception as e:
        print(f"PIPELINE FAILED: {type(e).__name__}: {e}")
        return 3
    print("pipeline returned:", type(video_result).__name__)

    # The exact app.py sequence: ML-only result, then overlay video fields.
    ml_only = pipeline.analyze_feature_vector(
        dict(video_result.feature_vector or {}), perf, injury,
        subject_verified=video_result.subject_verified,
        reliable=video_result.delivery_reliable)
    merged = merge_video_result(ml_only, video_result)

    print()
    print("--- what the OLD code showed (ml_only) ---")
    print("  provenance entries :", len(ml_only.feature_provenance or {}))
    print("  landmark summary   :", bool(ml_only.landmark_source_summary))
    print("  bowler track id    :", ml_only.bowler_track_id)
    print("  replay path        :", bool(ml_only.analysis_replay_path))
    print("  bowler bboxes      :", len(ml_only.bowler_bboxes or []))

    print("--- what the NEW code shows (merged) ---")
    print("  provenance entries :", len(merged.feature_provenance or {}))
    print("  landmark summary   :", bool(merged.landmark_source_summary))
    print("  bowler track id    :", merged.bowler_track_id)
    print("  replay path        :", bool(merged.analysis_replay_path))
    print("  replay exists      :", bool(merged.analysis_replay_path
                                         and os.path.exists(merged.analysis_replay_path)))
    print("  bowler bboxes      :", len(merged.bowler_bboxes or []))
    print("  subject_verified   :", merged.subject_verified)
    print("  delivery_reliable  :", merged.delivery_reliable)
    print("  blocked_reason     :", merged.scoring_blocked_reason)
    print("  performance_score  :", merged.performance_score)
    print("  injury_risk        :", (merged.injury_risk or {}).get("risk_level"))

    status = result_view.assess_delivery(merged, perf_bundle=perf, is_video=True)
    findings = result_view.build_findings(merged, status)
    print()
    print("--- interpretation ---")
    print("  state       :", status.state, status.states)
    print("  completeness:", status.completeness)
    print("  measured    :", status.measured, "degraded:", status.degraded,
          "missing:", status.missing, "low_conf:", status.low_confidence)
    print("  findings    :", len(findings))
    for f in findings:
        print(f"    [{f.severity}] {f.title}")

    # The regression that started all this: without the merge these are all empty.
    print()
    ok = True
    # The core contract: every field app.py merges must actually exist on the
    # object analyze_video returns. A typo here silently drops data again.
    missing_fields = [f for f in _VIDEO_ONLY_FIELDS
                      if not hasattr(video_result, f)]
    if missing_fields:
        print("FAIL: fields we merge that analyze_video does not provide:")
        for f in missing_fields:
            print("   ", f)
        ok = False
    else:
        print(f"all {len(_VIDEO_ONLY_FIELDS)} merged fields exist on the result")

    if not merged.feature_provenance:
        print("FAIL: provenance lost"); ok = False
    if not merged.landmark_source_summary:
        print("FAIL: landmark source lost"); ok = False
    if merged.analysis_replay_path and not os.path.exists(merged.analysis_replay_path):
        print("FAIL: replay path points at a missing file"); ok = False
    if len(merged.bowler_bboxes or []) == 0:
        print("WARN: no bowler bboxes on this clip")

    print()
    print("VIDEO WIRING OK" if ok else "VIDEO WIRING FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
