# PaceAI P1 — Annotation Status

> Human annotation is the only source of ground truth. This report is generated from the two canonical files; it does not claim completion.

## Summary

| Metric | Count |
|---|---|
| Total real clips | 8 |
| Reviewed (human `view_type`/`usable` decision) | 0 |
| Reviewed & usable for annotation | 0 |
| Annotated (rows in ground_truth.csv) | 0 |
| Missing (no annotation yet) | 8 |

**Human annotation required.** No rows exist in `evaluation/ground_truth/ground_truth.csv`; no accuracy statistics can be computed until independent annotations are entered.

## Per-clip status

| video_id | view_type | usable_for_annotation | annotation_status | annotator_id(s) | missing_measurements | notes |
|---|---|---|---|---|---|---|
| fast_left_00000001 | UNKNOWN | UNKNOWN | not_annotated | - | all measurements | bowling_type=fast; PARI-F1 plausibility sweep clip (see evaluation/validation_report.md). No per-delivery ground truth exists for this clip. |
| fast_left_0000001 | UNKNOWN | UNKNOWN | not_annotated | - | all measurements | bowling_type=fast; PARI-F1 plausibility sweep clip (see evaluation/validation_report.md). No per-delivery ground truth exists for this clip. |
| fast_right_00000001 | UNKNOWN | UNKNOWN | not_annotated | - | all measurements | bowling_type=fast; PARI-F1 plausibility sweep clip (see evaluation/validation_report.md). No per-delivery ground truth exists for this clip. |
| fast_right_00000002 | UNKNOWN | UNKNOWN | not_annotated | - | all measurements | bowling_type=fast; PARI-F1 plausibility sweep clip (see evaluation/validation_report.md). No per-delivery ground truth exists for this clip. |
| leg_right_00000001 | UNKNOWN | UNKNOWN | not_annotated | - | all measurements | bowling_type=leg; PARI-F1 plausibility sweep clip (see evaluation/validation_report.md). No per-delivery ground truth exists for this clip. |
| leg_right_00000029 | UNKNOWN | UNKNOWN | not_annotated | - | all measurements | bowling_type=leg; PARI-F1 plausibility sweep clip (see evaluation/validation_report.md). No per-delivery ground truth exists for this clip. |
| off_left_00000001 | UNKNOWN | UNKNOWN | not_annotated | - | all measurements | bowling_type=off; PARI-F1 plausibility sweep clip (see evaluation/validation_report.md). No per-delivery ground truth exists for this clip. |
| off_right_00000042 | UNKNOWN | UNKNOWN | not_annotated | - | all measurements | bowling_type=off; PARI-F1 plausibility sweep clip (see evaluation/validation_report.md). No per-delivery ground truth exists for this clip. |

Note: clips with `view_type=unusable` or `usable_for_annotation=no` are not annotated.
