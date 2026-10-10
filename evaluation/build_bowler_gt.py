"""Build bowler-identity ground truth from raw Sportradar annotations.

PHASE 24 infrastructure. Replaces the substring yardstick previously used at
``evaluation/single_video_benchmark/run_validation.py:155-156``
(``if "bowler" in raw_lower``), which cannot distinguish ``bowler`` from
``non_striker``-style labels and silently caps any reported accuracy.

Source schema (``dataset/data/<video_id>_2overs.json``)::

    person: {frame_index(str) -> [ {id, role, box:[x1,y1,x2,y2,?]}, ... ]}
    event:  {frame_index(str) -> [ {event, bowling_type, bowling_from,
                                   box, bowler_hand}, ... ]}

Identity policy
---------------
The annotations fragment identity: the benchmark video
``20221001_4139132_2overs`` carries 15 distinct bowler IDs across two overs, and a
whole-video dominant ID accounts for only ~35% of bowler-role frames on average --
because a video contains two different bowlers, one per over.

``PER_DELIVERY`` policy (default): the annotation stream marks a contiguous
``event == "is bowling"`` window per delivery. Within each such window the bowler
is the ``role == "bowler"`` person-id covering the most frames. Identity is
therefore resolved *per delivery* rather than per video, which matches the fact
that each over has a different bowler. Frames carrying a different bowler id
inside a window are counted as ``fragmented_frames_excluded`` and excluded from
the evaluation denominator rather than silently scored as misses.

``DOMINANT_ID`` (whole video) remains available via ``--policy`` for comparison.

The policy, its coverage, and its ceiling are recorded in the manifest so no
consumer can read these files without seeing the limitation.

Output
------
``evaluation/bowler_identification/ground_truth/<video_id>.json``
``evaluation/bowler_identification/ground_truth/manifest.json``
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW_DIR = REPO / "evaluation" / "single_video_benchmark" / "dataset" / "data"
VIDEO_DIR = REPO / "evaluation" / "single_video_benchmark" / "dataset" / "videos"
OUT_DIR = REPO / "evaluation" / "bowler_identification" / "ground_truth"

BOWLER_ROLE = "bowler"
MIN_BOX_AREA = 1.0
MIN_BOX_SIDE = 2.0
BOWLING_EVENT = "is bowling"
DELIVERY_GAP = 2


def _group_deliveries(frames: list[int], gap: int) -> list[list[int]]:
    """Collapse a sorted frame list into contiguous runs separated by > ``gap``."""
    groups: list[list[int]] = []
    for frame in frames:
        if groups and frame - groups[-1][-1] <= gap:
            groups[-1].append(frame)
        else:
            groups.append([frame])
    return groups


def _clean_box(raw) -> list[float] | None:
    """Normalise a 5-value annotation box to a well-formed [x1, y1, x2, y2]."""
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if (x2 - x1) < MIN_BOX_SIDE or (y2 - y1) < MIN_BOX_SIDE:
        return None
    if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
        return None
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def _as_int_key(key) -> int:
    try:
        return int(key)
    except (TypeError, ValueError):
        return -1


def build_for_video(path: Path, policy: str = "PER_DELIVERY") -> dict | None:
    """Extract bowler ground truth for one annotation file under ``policy``."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"error": f"unreadable: {type(exc).__name__}: {exc}"}

    person = data.get("person") or {}
    event = data.get("event") or {}
    if not isinstance(person, dict) or not person:
        return {"error": "no person annotations"}

    # Frame -> id -> box, restricted to role == "bowler".
    per_frame: dict[int, dict[str, list[float]]] = defaultdict(dict)
    id_frames: Counter = Counter()

    for raw_frame, entries in person.items():
        frame = _as_int_key(raw_frame)
        if frame < 0 or not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("role", "")).strip().lower() != BOWLER_ROLE:
                continue
            box = _clean_box(entry.get("box"))
            if box is None:
                continue
            ann_id = str(entry.get("id", "")).strip()
            if not ann_id:
                continue
            # First annotation wins for a given (frame, id) pair.
            per_frame[frame].setdefault(ann_id, box)
            id_frames[ann_id] += 1

    if not id_frames:
        return {"error": "no bowler-role annotations"}

    # Independent semantic signal: frames explicitly marked as a bowling action.
    bowling_frames = sorted(
        {
            _as_int_key(f)
            for f, evs in event.items()
            if isinstance(evs, list)
            and any(
                isinstance(e, dict)
                and str(e.get("event", "")).strip().lower() == BOWLING_EVENT
                for e in evs
            )
            and _as_int_key(f) >= 0
        }
    )
    deliveries = _group_deliveries(bowling_frames, DELIVERY_GAP)

    total_bowler_frames = sum(id_frames.values())
    boxes: dict[str, list[float]] = {}
    fragmented = 0
    per_delivery_ids: list[dict] = []

    if policy == "PER_DELIVERY" and deliveries:
        # Identity is resolved inside each contiguous bowling window, so the two
        # bowlers in a two-over video are not collapsed into one dominant id.
        for window in deliveries:
            counts: Counter = Counter()
            for frame in window:
                for ann_id in per_frame.get(frame, {}):
                    counts[ann_id] += 1
            if not counts:
                per_delivery_ids.append({"window": [window[0], window[-1]], "bowler_id": None})
                continue
            win_id, win_count = counts.most_common(1)[0]
            per_delivery_ids.append(
                {
                    "window": [window[0], window[-1]],
                    "bowler_id": win_id,
                    "window_frames": len(window),
                    "identified_frames": win_count,
                    "candidate_ids": dict(counts.most_common(4)),
                }
            )
            for frame in window:
                id_map = per_frame.get(frame, {})
                if win_id in id_map:
                    boxes[str(frame)] = id_map[win_id]
                elif id_map:
                    fragmented += 1
    else:
        # Whole-video fallback.
        dominant_id, _ = id_frames.most_common(1)[0]
        for frame, id_map in per_frame.items():
            if dominant_id in id_map:
                boxes[str(frame)] = id_map[dominant_id]
            else:
                fragmented += 1
        per_delivery_ids = [{"bowler_id": dominant_id, "policy": "whole_video"}]

    dominant_ids = sorted({d["bowler_id"] for d in per_delivery_ids if d.get("bowler_id")})
    identified_frames = sum(d.get("identified_frames", 0) for d in per_delivery_ids)

    return {
        "video_id": path.stem,
        "identity_policy": policy,
        "distinct_bowler_ids": len(id_frames),
        "resolved_bowler_ids": len(dominant_ids),
        "total_bowler_role_frames": total_bowler_frames,
        "action_window_frames": len(boxes),
        "scored_frames": len(boxes),
        "scored_fraction_of_bowler_frames": (
            round(len(boxes) / total_bowler_frames, 6) if total_bowler_frames else 0.0
        ),
        "fragmented_frames_excluded": fragmented,
        "bowling_event_frames": len(bowling_frames),
        "detected_deliveries": len(deliveries),
        "per_delivery": per_delivery_ids,
        "bowler_bboxes": boxes,
        "bowling_action_frames": bowling_frames,
        "delivery_groups": deliveries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--policy",
        choices=("PER_DELIVERY", "DOMINANT_ID"),
        default="PER_DELIVERY",
        help="How bowler identity is resolved from fragmented annotation ids.",
    )
    args = parser.parse_args()

    raw_files = sorted(RAW_DIR.glob("*_2overs.json"))
    if not raw_files:
        print(f"ERROR: no annotations under {RAW_DIR}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    entries, errors = [], []

    for raw in raw_files:
        built = build_for_video(raw, policy=args.policy)
        video_id = raw.stem
        if built is None or "error" in built:
            errors.append({"video_id": video_id, "error": (built or {}).get("error", "unknown")})
            print(f"  SKIP  {video_id}: {(built or {}).get('error', 'unknown')}")
            continue

        video_file = VIDEO_DIR / f"{video_id}.mp4"
        built["has_video"] = video_file.exists()
        (args.out / f"{video_id}.json").write_text(
            json.dumps(built, separators=(",", ":")), encoding="utf-8"
        )
        entries.append(built)

        print(
            f"  OK    {video_id}\n"
            f"          ids={built['resolved_bowler_ids']} of "
            f"{built['distinct_bowler_ids']} distinct\n"
            f"          action_window_frames={built['action_window_frames']}  "
            f"fragmented_excluded={built['fragmented_frames_excluded']}  "
            f"deliveries={built['detected_deliveries']}  "
            f"video={'present' if video_file.exists() else 'ABSENT'}"
        )

    manifest = {
        "description": "Bowler-identity ground truth from raw Sportradar annotations.",
        "source": str(RAW_DIR.relative_to(REPO)).replace("\\", "/"),
        "identity_policy": args.policy,
        "policy_note": (
            "Annotations fragment bowler identity across multiple person-ids. Under PER_DELIVERY, "
            "identity is resolved inside each contiguous event=='is bowling' window, because a "
            "video contains two different bowlers (one per over). Frames annotated as bowler under "
            "a different id inside a window are counted as fragmented_frames_excluded and are NOT "
            "scored as misses, so accuracy uses a conservative denominator. "
            "action_window_frames is the evaluation set: it deliberately covers only frames inside a "
            "bowling action, which is the semantically correct unit for 'did the system pick the "
            "bowler during his delivery'. It is a subset of all bowler-role frames "
            "(scored_fraction_of_bowler_frames), which also include the bowler standing at the mark "
            "or walking back -- those frames are not evaluated."
        ),
        "not_biomechanics_ground_truth": (
            "This is identity/bbox ground truth only. It contains no per-delivery biomechanical "
            "labels, so no biomechanics accuracy may be derived from it."
        ),
        "videos": len(entries),
        "videos_with_footage": sum(1 for e in entries if e["has_video"]),
        "errors": errors,
        "per_video": [
            {
                "video_id": e["video_id"],
                "resolved_bowler_ids": e["resolved_bowler_ids"],
                "distinct_bowler_ids": e["distinct_bowler_ids"],
                "scored_fraction_of_bowler_frames": e["scored_fraction_of_bowler_frames"],
                "action_window_frames": e["action_window_frames"],
                "fragmented_frames_excluded": e["fragmented_frames_excluded"],
                "detected_deliveries": e["detected_deliveries"],
                "has_video": e["has_video"],
            }
            for e in entries
        ],
    }
    manifest["total_scored_frames"] = sum(e["action_window_frames"] for e in entries)
    manifest["total_fragmented_excluded"] = sum(
        e["fragmented_frames_excluded"] for e in entries
    )
    manifest["mean_scored_fraction_of_bowler_frames"] = (
        round(
            sum(e["scored_fraction_of_bowler_frames"] for e in entries) / len(entries), 6
        )
        if entries
        else 0.0
    )
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(
        f"\npolicy={args.policy}  videos={len(entries)}  "
        f"with_footage={manifest['videos_with_footage']}  "
        f"action_window_frames={manifest['total_scored_frames']}  "
        f"fragmented_excluded={manifest['total_fragmented_excluded']}  "
        f"deliveries={sum(e['detected_deliveries'] for e in entries)}"
    )
    if errors:
        print(f"{len(errors)} videos skipped (see manifest.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
