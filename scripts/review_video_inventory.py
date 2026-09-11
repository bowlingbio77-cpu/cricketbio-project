"""
PaceAI P1.1 -- Human review of the real-video inventory.

The columns video_inventory.csv can only be filled by a *human* looking at the
clips: `view_type`, `usable_for_annotation`, `notes`, and `review_date`.
This tool drives that review interactively. It never classifies a clip on its
own and never uses PaceAI predictions.

Usage:
    python scripts/review_video_inventory.py --show
    python scripts/review_video_inventory.py                # review unreviewed clips
    python scripts/review_video_inventory.py --all          # re-review every clip

Prompts:
    view_type      (side_view / near_side_view / other / unusable)
    usable_for_annotation   (yes / no)
    notes          free text (why the decision was made)
"""
import csv
import datetime
import os
import sys

INVENTORY_PATH = os.path.join("evaluation", "ground_truth", "video_inventory.csv")

VIEW_TYPES = ["side_view", "near_side_view", "other", "unusable"]
USABLE_OPTIONS = ["yes", "no"]


def load_rows(path: str) -> list:
    if not os.path.exists(path):
        sys.exit(f"inventory not found: {path}")
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: str, rows: list, columns: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def prompt(label: str, choices) -> str:
    while True:
        value = input(f"{label} [{('/'.join(choices))}]: ").strip().lower()
        if value in choices:
            return value
        print(f"  valid choices: {', '.join(choices)}")


def main() -> None:
    if "--show" in sys.argv:
        for row in load_rows(INVENTORY_PATH):
            print("{video_id:20} view={view_type:14} usable={usable_for_annotation:8} "
                  "reviewed={review_date}  {notes}".format(**row))
        return

    rows = load_rows(INVENTORY_PATH)
    columns = list(rows[0].keys())
    force_all = "--all" in sys.argv
    today = datetime.date.today().isoformat()

    for row in rows:
        already = row.get("view_type", "") not in ("", "UNKNOWN") and \
            row.get("usable_for_annotation", "") not in ("", "UNKNOWN")
        if already and not force_all:
            print(f"already reviewed: {row['video_id']} "
                  f"[view={row.get('view_type')} usable={row.get('usable_for_annotation')}]")
            continue

        print(f"\nclip: {row['video_id']}  (file: {row.get('path')})")
        print(f"  fps={row.get('fps')}  frames={row.get('frame_count')}  "
              f"size={row.get('width')}x{row.get('height')}  "
              f"duration={row.get('duration_s')}s")
        print(f"  current notes: {row.get('notes', '')}")

        view_type = prompt("view_type", VIEW_TYPES)
        usable = prompt("usable_for_annotation", USABLE_OPTIONS)
        notes = input("notes (reason for the decision): ").strip()
        if notes:
            notes = f"review={today}: {notes}"
        else:
            notes = f"review={today}"

        row["view_type"] = view_type
        row["usable_for_annotation"] = usable
        row["notes"] = notes
        row["review_date"] = today

    write_rows(INVENTORY_PATH, rows, columns)
    print(f"\nWrote {len(rows)} rows -> {INVENTORY_PATH}")


if __name__ == "__main__":
    main()