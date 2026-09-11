"""
PaceAI P1.1 -- Annotation status audit report.

Builds `evaluation/ground_truth/annotation_status.md` from:
  * evaluation/ground_truth/video_inventory.csv   (human review decisions)
  * evaluation/ground_truth/ground_truth.csv      (human annotations)

The report is deliberately honest: every counter is derived from those two
files, and it never claims annotation is complete. If ground_truth.csv has no
rows, the report says so and states that human annotation is required.

Usage:
    python scripts/build_annotation_status.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.ground_truth_validation import (
    GT_PATH, INVENTORY_PATH, MEASUREMENT_FIELDS,
)

OUT_PATH = os.path.join(os.path.dirname(GT_PATH), "annotation_status.md")


def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def nonblank(value):
    value = (value or "").strip()
    return "" if value in ("", "UNKNOWN", "unknown") else value


def missing_measurements(rows_for_clip):
    present = set()
    for r in rows_for_clip:
        for field in MEASUREMENT_FIELDS:
            if nonblank(r.get(field)):
                present.add(field)
    return [f for f in MEASUREMENT_FIELDS if f not in present]


def build_status_markdown(inventory: list, gt_rows: list) -> tuple:
    """Return (markdown_text, counts). `counts` is:
    {total, reviewed, usable, annotated, missing}."""
    rows_by_clip = {}
    for r in gt_rows:
        rows_by_clip.setdefault(r.get("video_id") or "", []).append(r)

    total = len(inventory)
    reviewed = 0
    usable = 0
    annotated = 0
    clip_rows_out = []

    for row in inventory:
        video_id = row.get("video_id", "")
        view_type = nonblank(row.get("view_type")) or "UNKNOWN"
        usable_flag = nonblank(row.get("usable_for_annotation")) or "UNKNOWN"
        notes = (row.get("notes") or "").replace("|", "/")
        clip_rows = rows_by_clip.get(video_id, [])

        if view_type != "UNKNOWN":
            reviewed += 1
            if usable_flag == "yes":
                usable += 1

        if not clip_rows:
            status = "not_annotated"
            annotators = "-"
            missing = "all measurements"
        else:
            annotated += 1
            status = "annotated"
            annotators = ", ".join(sorted({r.get("annotator_id") or "-"
                                           for r in clip_rows}))
            miss = missing_measurements(clip_rows)
            if miss:
                status = "partially_annotated"
            missing = ", ".join(miss) if miss else "none"

        clip_rows_out.append(
            f"| {video_id} | {view_type} | {usable_flag} | {status} | "
            f"{annotators} | {missing} | {notes} |")

    counts = {
        "total": total,
        "reviewed": reviewed,
        "usable": usable,
        "annotated": annotated,
        "missing": max(0, total - annotated),
    }

    lines = []
    lines.append("# PaceAI P1 — Annotation Status")
    lines.append("")
    lines.append("> Human annotation is the only source of ground truth. This "
                 "report is generated from the two canonical files; it does not "
                 "claim completion.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|---|---|")
    lines.append(f"| Total real clips | {total} |")
    lines.append(f"| Reviewed (human `view_type`/`usable` decision) | {reviewed} |")
    lines.append(f"| Reviewed & usable for annotation | {usable} |")
    lines.append(f"| Annotated (rows in ground_truth.csv) | {annotated} |")
    lines.append(f"| Missing (no annotation yet) | {counts['missing']} |")
    lines.append("")

    if annotated == 0:
        lines.append("**Human annotation required.** No rows exist in "
                     "`evaluation/ground_truth/ground_truth.csv`; no accuracy "
                     "statistics can be computed until independent annotations "
                     "are entered.")
        lines.append("")

    lines.append("## Per-clip status")
    lines.append("")
    lines.append("| video_id | view_type | usable_for_annotation | "
                 "annotation_status | annotator_id(s) | missing_measurements | "
                 "notes |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.extend(clip_rows_out)

    lines.append("")
    lines.append("Note: clips with `view_type=unusable` or "
                 "`usable_for_annotation=no` are not annotated.")
    return "\n".join(lines) + "\n", counts


def main() -> None:
    inventory = read_csv_rows(INVENTORY_PATH)
    gt_rows = read_csv_rows(GT_PATH)
    text, counts = build_status_markdown(inventory, gt_rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {OUT_PATH}")
    print(f"total={counts['total']} reviewed={counts['reviewed']} "
          f"usable={counts['usable']} annotated={counts['annotated']}")


if __name__ == "__main__":
    main()