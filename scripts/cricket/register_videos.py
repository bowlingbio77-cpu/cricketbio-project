"""Register cricket videos BY REFERENCE into the Phase 1/6 dataset root.

    python scripts/cricket/register_videos.py --source corrected_all_data/bowling
    python scripts/cricket/register_videos.py --source corrected_all_data/bowling --limit 60
    python scripts/cricket/register_videos.py --source corrected_all_data/bowling --view behind_bowler

Nothing is copied. Each clip gets a content-hash ``video_id`` (see
``schema.video_id_for_path``) because the corpus contains 410 filename
collisions from mixed 7-/8-digit zero padding.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.cricket_understanding import registry  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="directory or single video file")
    ap.add_argument("--out", default=registry.VIDEOS_JSONL)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--view", default="unknown", choices=list(
        __import__("src.cricket_understanding.schema", fromlist=["x"]).CAMERA_VIEWS),
        help="only meaningful if you actually know the view; leave 'unknown'")
    ap.add_argument("--bowler-id", default="", help="only if the same bowler is known")
    ap.add_argument("--bowling-side", default="unknown", choices=["right", "left", "unknown"])
    ap.add_argument("--merge", action="store_true",
                    help="merge with the existing registry instead of replacing it")
    ap.add_argument("--filename-prefix", default="",
                    help="only register files whose name starts with this, e.g. "
                         "fast_left. Repeatable is not supported; use one call per view.")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip OpenCV probing (fast, but fps/frame_count stay null)")
    args = ap.parse_args(argv)

    if not os.path.exists(args.source):
        print(f"source not found: {args.source}", file=sys.stderr)
        return 2

    paths = list(registry.iter_videos(args.source))
    if args.filename_prefix:
        paths = [p for p in paths
                 if os.path.basename(p).startswith(args.filename_prefix)]
    paths = paths[args.offset:]
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        print("no video files found", file=sys.stderr)
        return 2

    existing = registry.load_records(args.out) if args.merge else []
    merged = {r.video_id: r for r in existing}

    n_new = n_dup = n_fail = 0
    for i, p in enumerate(paths, 1):
        if args.no_probe:
            info = {"probe_status": "SKIPPED", "probe_error": "--no-probe"}
            rec = registry.VideoRecord(
                video_id=__import__("src.cricket_understanding.schema",
                                    fromlist=["x"]).video_id_for_path(p),
                path=os.path.abspath(p), filename=os.path.basename(p),
                bytes=os.path.getsize(p), probe_status="SKIPPED",
                probe_error="--no-probe")
        else:
            rec = registry.make_record(p)
            rec.camera_view = args.view
            rec.bowling_side_if_known = args.bowling_side
            rec.bowler_id_if_known = args.bowler_id
        if rec.probe_status != "OK":
            n_fail += 1
        if rec.video_id in merged:
            n_dup += 1
        merged[rec.video_id] = rec
        n_new += 1
        if i % 25 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)} probed ({n_fail} un-probeable, {n_dup} already known)")

    out = sorted(merged.values(), key=lambda r: r.video_id)
    registry.save_records(out, args.out)
    print(f"\nregistered {len(out)} video(s) -> {args.out}")
    print(f"  new/updated : {n_new}")
    print(f"  duplicates  : {n_dup}")
    print(f"  un-probeable: {n_fail}  (recorded with probe_status=FAILED, NOT dropped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
