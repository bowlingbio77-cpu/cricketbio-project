# Cricket Understanding v1 -- implementation report

**Status: the infrastructure is complete and exercised on real video. The
accuracy of every model is `NOT MEASURED`, because the repository contains no
human cricket annotation.**

This is the honest bottom line, and nothing in the 16 phases changes it.

---

## 1. Corpus state

| | |
|---|---|
| Registered videos | 72 |
| Human-annotated | **0** |
| Human deliveries | **0** |
| Splits | 48 train / 12 val / 12 test, 6 groups |
| QC issues | 0 |

`data/bowler_batsman_dataset/auto_labeled/` and
`data/cricket_ball_dataset/auto_labeled/` are model-derived. They are **not**
cricket ground truth, they are never scored, and the existing
`models/bowler_batsman_yolo11n.pt` trained on them is not reused.

---

## 2. The one real finding

Running the shipping pipeline over 8 of the 72 registered videos:

| Observation | Value |
|---|---|
| Clips with a bowler auto-confirmed | 1 / 8 (12.5%, 95% CI **2.2% - 47.1%**) |
| Clips where the subject was verified | 1 / 8 |
| Clips with at least one identity switch | **8 / 8** |
| Mean identity switches per clip | 1.125 |
| Refusal reasons | `ambiguous_margin` x6, `low_evidence` x1, confirmed x1 |
| Mean wall clock per clip | 21.7 s (16.3 - 27.1) |

Two things follow, and only two.

**Identity is unstable on every clip tried.** 8/8 clips switched identity at
least once. This is measurable without labels and is the most actionable result
in the project: it says the problem is *continuity*, not the final choice of
person. Any biomechanical number computed on a track that changed identity
mid-clip describes a discontinuity in the middle of a delivery stride.

**The selector abstains, and correctly so.** It refused to auto-confirm on 7 of
8 clips, almost all citing `ambiguous_margin` -- a top-1-vs-top-2 tie. That is the
gate doing its job: silently analysing the wrong human is worse than declining.
The Wilson interval is deliberately wide and is reported as such; 1/8 is not
evidence of anything beyond "rare".

**What is *not* concluded:** that the one confirmed clip is right, that 12.5% is
better or worse than anything, or that the 8 clips are representative of the 72.
`bowler_top1_accuracy` and `false_selection_rate` are `NOT MEASURED`.

---

## 3. Phase status

| Phase | Deliverable | Status |
|---|---|---|
| 0 | repository audit | done, corrected (see 5) |
| 1 | data inventory + leakage guard | done |
| 2 | schema with provenance | done |
| 3 | annotation GUI (OpenCV) + headless state machine | done, 72 videos listed |
| 4 | annotation QC | done, 0 issues |
| 5 | grouped leakage-safe splits | done, 48/12/12 |
| 6 | dataset spec | done |
| 7 | pose sequences | done |
| 8 | dataset generators (det/role/action/track/delivery) | done |
| 9 | Models A/B/C + guards | done, 0 of 3 trainable |
| 10 | baseline harness + 12 metrics | done, 0 measured / 12 not measured |
| 11 | A/B/C/D experiment | done, A/B run, C/D blocked |
| 12 | CLI entry points | done |
| 13 | regression suite | done |
| 14 | documentation | done |
| 15 | this report | done |

---

## 4. Bugs found and fixed while building this

Four of these were live defects in code that had already passed tests.

1. **`metrics.py` accepted any `annotator_id` as ground truth.** An annotation
   written by a labeller named `auto_labeler` would have produced a fully
   measured, confident, completely circular accuracy. Added
   `is_human_annotator` + `NON_HUMAN_ANNOTATOR_MARKERS`; rejected clips are now
   *named* in the metric's `reason` rather than silently dropped.
2. **`compare_arms` produced an empty table.** A nested generator reused the
   comprehension variable name, so the metric-name set came back empty. Found by
   a test that asserted a row existed.
3. **`compare_arms` never set a verdict for comparable arms** -- exactly the path
   taken the first time a real holdout exists. The report generator reads
   `row["verdict"]` unconditionally and would have raised `KeyError` on the day
   the project first got a real number.
4. **`tasks.py` guard 1 checked only that `annotator_id` was non-empty**, while
   its own docstring claimed "Model-derived rows are not accepted". `auto_labeler`
   passed. It now uses the same `is_human_annotator` check, and model-derived
   train rows no longer count toward per-class support.
5. `splits.save_splits()` wrote `split_meta.json` into the real data tree during
   a test despite `schema.SPLITS_DIR` being monkeypatched.
6. The pose test polluted other modules via the `src.pose_estimation` parent
   attribute; now both the `sys.modules` entry and the parent attribute are
   patched.
7. `baseline.py` referenced an undefined local in a `del` statement and aborted
   after writing its report.

The recurring theme is worth stating: **three of these were guards that
documented a property the code did not have.** Tests that assert the *absence*
of a result ("this must stay `NOT MEASURED`") caught them; tests that assert a
number would not have.

---

## 5. Corrections to the Phase 0 audit

- **The audit's claim that `torch` cannot be imported is wrong for the project's
  interpreter.** `venv\Scripts\python.exe` imports `torch 2.13.0+cpu` and
  `ultralytics 8.4.115` successfully. The AppLocker block applies to a different
  system interpreter. All runs in this report used the venv.
- Stage backends are `detection=bytetrack` and `tracking=bytetrack` on 8/8 clips,
  i.e. no degradation to the IoU fallback occurred.

---

## 6. What is deliberately not claimed

- No accuracy, F1, precision, recall, or MAE for any model.
- No comparison to any other system.
- No claim that arm B is better than arm A. The spatial prior changed the
  selection in 8/8 clips and raised auto-confirmations from 1/8 to 4/8, but
  confirmation is the selector grading itself; without a human,
  `bowler_top1_accuracy` cannot distinguish "better" from "more confident".
- No claim that the 8-clip baseline generalises to the 72-clip corpus.
- No threshold was tuned. The prior's weight (0.50, declared, not fitted) was
  left deliberately too large rather than shrunk until the arm stopped changing
  answers, because tuning it against unlabelled data is the failure this project
  exists to prevent.

---

## 7. The single blocking dependency

A human annotating the 12 test videos.

Everything downstream is implemented, guarded, and tested: the generators run,
the trainer will accept data the moment the guards pass, the experiment harness
will unblock arms C and D, and the metrics will start reporting real numbers the
first time a `ClipGroundTruth` carries a human `annotator_id`. No further code
is needed for the project to produce a genuine held-out result.

```bash
python tools/annotate_cricket.py --annotator_id <ID> --list
```

---

## 8. Reproducing this report

```bash
python scripts/cricket/build_splits.py          # QC + splits
python scripts/cricket/train_model.py --report   # Model A/B/C readiness
python scripts/cricket/baseline.py --limit 8     # section 2
python scripts/cricket/experiments.py --run --limit 8
python -m pytest                                 # 574 passed
```
