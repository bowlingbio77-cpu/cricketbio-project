# Cricket Understanding -- training and evaluation

How to annotate, train, and evaluate the cricket-understanding models, and what
the current numbers actually mean.

**Read this first:** no metric in this project has been measured against human
ground truth, because there is no human cricket annotation in the repository
yet. Every accuracy-shaped number below is `NOT MEASURED`. That is the correct
state, not a bug and not a placeholder to be filled in later from model output.

---

## 1. What is real today

| Quantity | Value |
|---|---|
| Registered videos | 72 |
| Human-annotated videos | **0** |
| Human-labelled deliveries | **0** |
| Human-attributable training rows (Models A/B/C) | **0** |
| Tasks trainable | **0 of 3** |
| Accuracy metrics measured | **0 of 12** |

The pipelines that produce those facts *are* working and are exercised on real
video. The only missing ingredient is a human decision.

---

## 2. The annotation loop

```bash
# 1. see what needs annotating
python tools/annotate_cricket.py --annotator_id <YOUR_ID> --list

# 2. draw bowler / striker / non-striker / keeper / fielder boxes, then the
#    delivery window and release frame, then the per-frame bowling phases
python tools/annotate_cricket.py --annotator_id <YOUR_ID> --video v_02a8e776ad75

# 3. check the annotations for leakage, ordering errors, and gaps
python scripts/cricket/build_splits.py
```

`--annotator_id` is not optional and is not free text you can leave blank: the
annotation dataclasses raise on an empty id, because ground truth must be
attributable to a person. An id that names an automatic labeller
(`auto_...`, `model_...`, `yolo...`) is rejected downstream by
`metrics.is_human_annotator`, so a model can never end up scoring itself.

QC runs automatically as part of `build_splits.py` and reports rather than fixes.
It will tell you a release frame precedes a gather frame; it will not silently
reorder it for you.

---

## 3. The three learning tasks

Declared once in `src/cricket_understanding/tasks.py`; the tables in
`python scripts/cricket/train_model.py --report` are **generated** from that
file, so the documentation cannot drift from the training code.

| | Model A | Model B | Model C |
|---|---|---|---|
| Question | which cricket objects, and where | what role is this track | what phase is this frame |
| Dataset | `det` | `role` | `action` |
| Labels | 8 (6 roles + BALL, STUMPS) | 6 roles | 9 bowling phases |
| Baseline | yolo11n, stock COCO weights, transferred | majority class -> multinomial logistic regression | phase prior -> logistic regression -> linear-chain CRF |
| Input | RGB frame | box geometry + motion over the whole track | hip-centred normalised keypoints + 1st/2nd differences |

Deliberate choices worth knowing:

- **No transformer by default.** With 0 labelled deliveries there is no basis for
  capacity selection. The declared baseline for B and C is the simplest thing
  that can be fit; a stronger architecture is a *recorded experiment*, not a
  default.
- **Model B's features exclude every PaceAI score.** Otherwise the role model
  would just learn to imitate the heuristic it is meant to replace, and the
  experiment would measure nothing.
- **`models/bowler_batsman_yolo11n.pt` is not reused.** It was trained on
  model-derived auto-labels, so its reported accuracy is self-agreement and is
  never used as a result anywhere in this project.
- **`UNKNOWN` is a real label**, not a missing value. A person visible but
  unassignable, or a phase the annotator could not determine, is recorded as
  `UNKNOWN` rather than guessed.

### The guards

Training refuses to start unless all three pass:

| Guard | Requirement |
|---|---|
| `human_labels_present` | at least one **train** row attributable to a human |
| `held_out_split_nonempty` | `test` holds at least one sample |
| `label_support_per_class` | every class has >= 2 human train samples |

```bash
python scripts/cricket/train_model.py --report   # human-readable readiness
python scripts/cricket/train_model.py --model B --dry-run
python scripts/cricket/train_model.py --model B          # trains, or refuses
```

The dry run for Model B today exits non-zero and says why. That refusal is the
feature: it is what stops a first annotation session from silently producing a
model scored on its own training data.

---

## 4. Evaluation

```bash
# observed behaviour of the current pipeline on real videos
python scripts/cricket/baseline.py --limit 8

# the A/B/C/D experiment
python scripts/cricket/experiments.py --plan
python scripts/cricket/experiments.py --run --limit 8
```

### What the baseline actually measured

Running the shipping pipeline over 8 of the 72 registered videos:

| Observation | Value |
|---|---|
| Clips where a bowler was auto-confirmed | 1 / 8 (12.5%, 95% CI 2.2-47.1%) |
| Clips where the subject was verified | 1 / 8 |
| Clips with at least one identity switch | **8 / 8** |
| Mean identity switches per clip | 1.125 |
| Refusal reasons | `ambiguous_margin` x6, `low_evidence` x1, confirmed x1 |

These are *behaviour*, not accuracy. They say the selector abstains most of the
time and that identity is unstable on every clip tried. They say nothing about
whether the one confirmed clip is correct -- that needs a human.

### The metrics

All twelve are defined in `src/cricket_understanding/metrics.py`, and all twelve
are `NOT MEASURED` today. They split into three groups:

**Bowler identity** (needs a human to say which track is the bowler)
`bowler_top1_accuracy`, `bowler_precision`, `bowler_recall`, `bowler_f1`,
`false_selection_rate`, `bowler_abstention_rate`, `identity_switches`,
`track_continuity`.

`false_selection_rate` is the one that matters most operationally: a wrong
bowler means every downstream biomechanical number describes the wrong human.
Abstention is scored as a false negative rather than ignored, because refusing
to answer is a decision with consequences.

**Delivery** `delivery_detection_accuracy` (one-to-one match on temporal
IoU >= 0.5), `release_frame_error_frames` (MAE in frames, matched deliveries
only).

**Phases** `phase_accuracy`, plus `phase_macro_f1` -- because a model that always
predicts `RUN_UP` can look accurate on a run-up-heavy corpus.

Every rate carries a Wilson interval; F1 and MAE carry a bootstrap interval.
Below 5 paired samples **no interval is printed at all** and the point estimate
is marked as too small to characterise.

---

## 5. The A/B/C/D experiment

| Arm | Definition | Runnable today |
|---|---|---|
| **A** | current selection logic (`tracking.select_bowler_track_with_meta`) | yes |
| **B** | A + lower-half spatial prior | yes |
| **C** | B + cricket role model (Model B) | **no** -- blocked by `model_B` |
| **D** | C + temporal bowling-action model (Model C) | **no** -- blocked by `model_C` |

A and B run on **one** tracking pass, so both arms see identical candidates and
any difference is attributable to the prior rather than to detector noise. Arm B
is implemented by patching the single ranking function, so arm A's score floor,
margin gate, identity stitch, and switch counter are the unmodified shipping
code on the same call path.

### What the first run showed

The prior changed the selected track in **8 of 8** clips and raised
auto-confirmations from 1/8 to 4/8.

**That is not evidence that the prior helps.** Confirmation is the selector
grading its own homework. A prior that pushes a low-standing batsman over the
line would raise exactly this count while making selection *worse*. The number
that settles it is `bowler_top1_accuracy` against a human, which is
`NOT MEASURED` for all four arms.

A known weakness, stated rather than tuned away: the prior's bonus is up to
0.50, while arm A's cricket-evidence scores for these clips sit around 0.3-0.6.
The prior is therefore large enough to outweigh the evidence term and decide the
winner by itself. The weight was deliberately left alone -- shrinking it until
the arm stopped flipping selections would be threshold tuning against an
unlabelled set, which is precisely the failure mode this project is built to
avoid. The honest sequence is: label a holdout, then pre-register a weight
chosen on the validation split only.

There is **no composite score** and no declared winner anywhere in the report.
Arms are compared metric by metric, and a verdict says `INCONCLUSIVE` whenever
the intervals overlap -- claiming a winner on overlapping intervals is reading
noise as signal.

---

## 6. Rules this codebase enforces

These are machine-checked, not conventions:

1. **No metric without human ground truth.** `compute_all` returns
   `NOT MEASURED`, never `0.0`. A metric that reports zero because nobody
   labelled anything looks exactly like a real result, which is why the status
   is a separate field.
2. **Model-derived labels are never ground truth.**
   `metrics.is_human_annotator` rejects auto/model/yolo/pseudo ids at scoring
   time, and the same check guards Model B's training. A rejected annotation is
   named in the metric's `reason`, not silently dropped.
3. **No test-set tuning.** `assert_no_test_selection` raises
   `ProtocolViolationError` if an arm is selected on `test` or a test video id.
4. **Model selection happens on `val`.** Once a holdout exists, a
   `SEPARATED` verdict is only a decision to confirm; the reported number is the
   one read from `test` exactly once.
5. **Uncertainty is printed or omitted, never implied.**
6. **Guards refuse; they do not warn.** A task that fails a guard is `NOT READY`
   and the trainer exits non-zero.

---

## 7. The order of operations from here

1. Annotate the 12 test videos first. They are the scarcest resource and the
   only thing standing between this project and a real number.
2. Annotate at least 48 train videos so all three guards can pass.
3. `python scripts/cricket/generate_datasets.py`
4. `python scripts/cricket/train_model.py --model B` -- now exits 0.
5. `python scripts/cricket/experiments.py --run --limit 72` -- arms C and D
   become runnable; the A-vs-B question gets answered against humans.
6. Re-run `baseline.py` and report the test-split numbers.

Steps 3-6 are already implemented and will run the moment step 1 lands. Nothing
in this repository needs to be written to make them work.

---

## 8. Files

| Path | Purpose |
|---|---|
| `tools/annotate_cricket.py` | annotation GUI |
| `src/cricket_understanding/schema.py` | annotation schema + provenance rules |
| `src/cricket_understanding/qc.py` | annotation quality control |
| `src/cricket_understanding/splits.py` | leakage-safe grouped splits |
| `src/cricket_understanding/generators.py` | dataset generators (det/role/action/track/delivery) |
| `src/cricket_understanding/tasks.py` | Models A/B/C + readiness guards |
| `src/cricket_understanding/metrics.py` | the twelve metrics + arm comparison |
| `scripts/cricket/train_model.py` | guarded training entry point |
| `scripts/cricket/baseline.py` | observed-baseline harness |
| `scripts/cricket/experiments.py` | A/B/C/D experiment driver |
| `evaluation/cricket_task_readiness.json` | generated Model A/B/C status |
| `evaluation/cricket_understanding_baseline.md` | generated baseline report |
| `evaluation/cricket_experiments.md` | generated experiment report |
| `tests/test_cricket_understanding.py` | pipeline tests |
| `tests/test_cricket_experiments.py` | training/metrics/experiment tests |
