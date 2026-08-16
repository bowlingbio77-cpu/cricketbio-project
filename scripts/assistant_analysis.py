"""AI-assistant analysis CLI: video -> biomechanics analysis -> LLM interpretation.

Runs the full delivery pipeline (`pipeline.analyze_video`) on one or more
clips, renders the machine-generated `AnalysisResult` as a structured report,
then submits it as context to the self-hosted Odysseus AI workspace so an LLM
coach can interpret it. Degrades gracefully when Odysseus is not configured or
not running (`--no-llm` prints only the local report).

Modes:
  single : analyze one clip, optionally ask a follow-up question (`--ask`).
  batch  : analyze every clip in `--dir`, then ask Odysseus a comparative
           cohort question across all deliveries (trend/consistency analysis).

Requirements for the LLM leg:
  * Odysseus running (docker compose up from the odysseus repo)
  * scoped API token with `chat` scope  -> ODYSSEUS_API_TOKEN
  * a configured model endpoint         -> ODYSSEUS_ENDPOINT_ID (preferred)
                                          or ODYSSEUS_ENDPOINT_URL (admin only)
  * optional model override             -> ODYSSEUS_MODEL
  * optional base URL override          -> ODYSSEUS_BASE_URL (default :7000)

Usage:
  python scripts/assistant_analysis.py --video corrected_all_data/bowling/x.avi
  python scripts/assistant_analysis.py --video x.avi --ask "What is the biggest red flag?"
  python scripts/assistant_analysis.py --video x.avi --no-llm --out report.md
  python scripts/assistant_analysis.py --batch --dir corrected_all_data/bowling --limit 8
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import assistant, config, ml_models, pipeline            # noqa: E402


def load_bundles(model_name: str):
    perf_path = os.path.join(config.MODEL_DIR, f"performance_{model_name}.joblib")
    injury_path = os.path.join(config.MODEL_DIR, f"injury_{model_name}.joblib")
    perf = ml_models.load_bundle(perf_path) if os.path.exists(perf_path) else None
    injury = ml_models.load_bundle(injury_path) if os.path.exists(injury_path) else None
    return perf, injury


def analyze_one(video_path: str, bowling_arm: str, camera_view: str,
                model_name: str) -> pipeline.AnalysisResult:
    perf_bundle, injury_bundle = load_bundles(model_name)
    return pipeline.analyze_video(
        video_path,
        bowling_arm=bowling_arm,
        performance_bundle=perf_bundle,
        injury_bundle=injury_bundle,
        camera_view=camera_view,
    )


def list_clips(directory: str, limit: int = None) -> list:
    exts = (".avi", ".mp4", ".mov", ".mkv")
    clips = sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(exts)
    )
    if limit:
        clips = clips[:limit]
    return clips


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", help="Path to a delivery clip (single mode).")
    ap.add_argument("--batch", action="store_true", help="Analyze all clips in --dir.")
    ap.add_argument("--dir", default="corrected_all_data/bowling",
                    help="Directory of clips for --batch.")
    ap.add_argument("--limit", type=int, default=None, help="Max clips in batch mode.")
    ap.add_argument("--arm", default="right", choices=["right", "left"])
    ap.add_argument("--camera-view", default="behind", choices=["behind", "side"])
    ap.add_argument("--model-name", default="random_forest",
                    help="Trained model bundle family (random_forest/xgboost/catboost/...).")
    ap.add_argument("--ask", help="Specific question for the assistant.")
    ap.add_argument("--no-llm", action="store_true",
                    help="Run the pipeline and print the local report only (no Odysseus).")
    ap.add_argument("--out", help="Optional path to write the report(s) to.")
    args = ap.parse_args()

    if not args.video and not args.batch:
        ap.error("Provide --video or --batch.")
    if args.batch and args.video:
        ap.error("Use --batch or --video, not both.")

    results = []
    if args.batch:
        clips = list_clips(args.dir, args.limit)
        if not clips:
            print(f"No video clips found in {args.dir}", file=sys.stderr)
            sys.exit(2)
        print(f"Analyzing {len(clips)} clips from {args.dir} ...", file=sys.stderr)
        for clip in clips:
            print(f"  -> {os.path.basename(clip)}", file=sys.stderr)
            try:
                results.append(analyze_one(clip, args.arm, args.camera_view, args.model_name))
            except RuntimeError as exc:
                print(f"  !! skipped ({exc})", file=sys.stderr)
        if not results:
            print("No clips produced an analysis.", file=sys.stderr)
            sys.exit(2)
    else:
        results.append(analyze_one(args.video, args.arm, args.camera_view, args.model_name))

    if len(results) == 1:
        report = assistant.assistant_report(results[0])
    else:
        report = assistant.cohort_report(
            [(os.path.basename(r.video_path or f"clip-{i}"), r) for i, r in enumerate(results)]
        )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
        print(f"Report written to {args.out}", file=sys.stderr)

    print(report)
    print()

    if args.no_llm:
        return

    client = assistant.OdysseusAssistant()
    if not client.configured():
        print("Odysseus not configured:", file=sys.stderr)
        for setting in client.missing_settings():
            print(f"  - set {setting}", file=sys.stderr)
        print("Run with --no-llm for the local report only.", file=sys.stderr)
        sys.exit(3)

    print("Querying Odysseus assistant ...", file=sys.stderr)
    try:
        if len(results) == 1:
            reply = client.analyze(results[0], question=args.ask)
        else:
            reply = _ask_cohort(client, results, args.ask)
    except assistant.AssistantError as exc:
        print(f"Assistant error: {exc}", file=sys.stderr)
        sys.exit(4)

    print("--- ASSISTANT ---")
    print(reply)


def _ask_cohort(client: assistant.OdysseusAssistant, results: list, custom_question: str = None) -> str:
    message = assistant._build_chat_message(
        assistant.cohort_report([(os.path.basename(r.video_path or f"clip-{i}"), r)
                                 for i, r in enumerate(results)]),
        question=custom_question or assistant.COHORT_QUESTION,
    )
    return client.chat(message)


if __name__ == "__main__":
    main()
