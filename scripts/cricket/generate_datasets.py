"""Phase 7/8/9 driver: generate every training dataset and write the spec.

    python scripts/cricket/generate_datasets.py --dry-run
    python scripts/cricket/generate_datasets.py --only det role
    python scripts/cricket/generate_datasets.py --extract-pose

With NO human annotations this produces EMPTY datasets and says so. It never
falls back to the existing model-derived auto-labels.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.cricket_understanding import generators, registry, schema  # noqa: E402
from src.cricket_understanding import splits as splits_mod           # noqa: E402

ALL = ["det", "role", "track_seq", "action", "delivery"]


def _video_stats(records):
    ok = [r for r in records if r.probe_status == "OK"]
    frames = [r.frame_count for r in ok if r.frame_count]
    return {
        "n_registered": len(records),
        "n_probed_ok": len(ok),
        "n_total_frames": sum(frames),
        "median_frames_per_video": (sorted(frames)[len(frames) // 2] if frames else None),
        "min_frames": min(frames) if frames else None,
        "max_frames": max(frames) if frames else None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=ALL, choices=ALL)
    ap.add_argument("--dry-run", action="store_true",
                    help="full bookkeeping + manifests, no pixels written")
    ap.add_argument("--imsize", type=int, default=None,
                    help="letterbox-free long-side resize for the det dataset")
    ap.add_argument("--registry", default=registry.VIDEOS_JSONL)
    ap.add_argument("--clean", action="store_true",
                    help="remove each generated dataset dir before writing")
    ap.add_argument("--extract-pose", action="store_true",
                    help="run Phase 8 pose extraction first (needs mediapipe)")
    ap.add_argument("--pose-limit", type=int, default=0, help="0 = all deliveries")
    args = ap.parse_args(argv)

    schema.ensure_dirs()
    records = registry.load_records(args.registry)
    if not records:
        print("No registered videos. Run scripts/cricket/register_videos.py first.",
              file=sys.stderr)
        return 2
    split_map = splits_mod.load_splits() or {}
    contexts = generators.load_contexts(records, split_map, require_labels=True)

    print("=" * 74)
    print("Cricket Understanding v1 -- dataset generation")
    print("=" * 74)
    print(json.dumps(_video_stats(records), indent=2))
    print(f"\nLABEL SOURCE : {generators.LABEL_ROOT}")
    print("LABEL NATURE : human-annotated only (Phase 3 GUI)")
    print(f"REJECTED     : {list(generators.FORBIDDEN_LABEL_ROOTS)}")
    print(f"\nvideos with human labels: {len(contexts)} / {len(records)}")
    if not contexts:
        print("\n*** NO HUMAN ANNOTATIONS EXIST. ***")
        print("Every generated dataset will be EMPTY. This is the correct and")
        print("honest outcome: PaceAI currently has no cricket ground truth.")
        print("Run:  python tools/annotate_cricket.py --annotator_id <ID> --list")
        print("Do NOT substitute the existing auto-labelled trees; they are model")
        print("output, and training on them would only re-learn the current model.")

    if args.extract_pose and contexts:
        n = extract_pose(args, contexts)
        print(f"\nPhase 8 pose extraction: {n} delivery sequence(s) written")

    if args.clean:
        import shutil
        for name in args.only:
            d = os.path.join(schema.p("extracted"), generators.GENERATORS[name].__name__.replace("build_", "").replace("_dataset", ""))
            if os.path.isdir(d):
                shutil.rmtree(d)
                print(f"cleaned {d}")

    summaries = {}
    for name in args.only:
        fn = generators.GENERATORS[name]
        print(f"\n--- {name} ---")
        s = fn(contexts, None, dry_run=args.dry_run) if name == "det" else fn(contexts, None, dry_run=args.dry_run)
        summaries[name] = s
        print(json.dumps({k: v for k, v in s.items()
                          if k not in ("forbidden_label_roots",)}, indent=2)[:2200])

    out = os.path.join("evaluation", "generated_dataset_spec.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    payload = {
        "generator_version": generators.GENERATOR_VERSION,
        "schema_version": schema.SCHEMA_VERSION,
        "dry_run": args.dry_run,
        "video_stats": _video_stats(records),
        "splits": {k: len(v) for k, v in split_map.items()},
        "n_contexts_with_labels": len(contexts),
        "label_source": generators.LABEL_ROOT,
        "label_source_is_human": True,
        "rejected_label_roots": list(generators.FORBIDDEN_LABEL_ROOTS),
        "class_map": schema.YOLO_CLASS_MAP,
        "role_feature_names": generators.ROLE_FEATURE_NAMES,
        "track_seq_feature_names": generators.TRACK_SEQ_FEATURE_NAMES,
        "action_feature_names": generators.ACTION_FEATURE_NAMES,
        "summaries": summaries,
    }
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(schema.dumps(payload))
    print(f"\nSpec: {out}")
    return 0


def extract_pose(args, contexts) -> int:
    """Phase 8 -- pose sequences for every human-annotated bowler delivery."""
    import cv2
    from src.cricket_understanding import pose_seq

    n = 0
    for ctx in contexts:
        if not ctx.deliveries:
            continue
        bowler = {d.bowler_track_id for d in ctx.deliveries}
        boxes = {t: {p.frame_id: p.bbox for p in ctx.people if p.track_id == t}
                 for t in bowler}
        cap = cv2.VideoCapture(ctx.path)
        cache = {}

        def shape_fn(f, cap=cap, cache=cache):
            if f in cache:
                return cache[f]
            if f not in cache:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, fr = cap.read()
                cache.clear()
                cache[f] = fr if ok else None
            return cache.get(f)

        entries = []
        for d in ctx.deliveries:
            if args.pose_limit and n >= args.pose_limit:
                break
            payload = pose_seq.extract_for_delivery(
                ctx.path, ctx.video_id, d, boxes.get(d.bowler_track_id, {}), shape_fn)
            if payload.get("error"):
                print(f"  ! {ctx.video_id}/{d.delivery_id}: {payload['error'][:80]}")
                continue
            pose_seq.save(payload)
            entries.append({
                "delivery_id": d.delivery_id, "bowler_track_id": d.bowler_track_id,
                "n_frames": payload["n_frames"], "n_observed": payload["n_observed"],
                "n_missing": payload["n_missing"],
                "mean_pose_confidence": round(payload["mean_pose_confidence"], 4),
            })
            print(f"  {ctx.video_id}/{d.delivery_id}: "
                  f"{payload['n_observed']}/{payload['n_frames']} observed, "
                  f"{payload['n_missing']} missing, "
                  f"conf={payload['mean_pose_confidence']:.3f}")
            n += 1
        cap.release()
        if entries:
            pose_seq.save_index(ctx.video_id, entries)
    return n


if __name__ == "__main__":
    raise SystemExit(main())
