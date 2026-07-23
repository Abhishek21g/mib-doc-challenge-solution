# Evaluation and Release Harness

This development-only harness wraps the official evaluator and keeps labels,
PDFs, split assignments, case-level reports, and calibration samples outside the
submitted runtime image. Reports are explicitly labeled as local engineering
benchmarks, not official leaderboard scores.

## Honest split

Generate a deterministic, adjudication-stratified tuning/calibration/release
manifest to an external working directory:

```bash
python3 scripts/evaluation_harness.py split \
  --truth data/train_labels.csv \
  --seed mib-public-v1 \
  --label-output-dir /tmp/mib-evaluation/labels \
  --output /tmp/mib-evaluation/splits.json
```

The three partitions are disjoint. A calibration or release result must declare
which split(s) were used for tuning, and the harness rejects self-evaluation.
If any cases were inspected before freezing a revised split, pass their CSV via
`--forced-tuning-manifest`. The splitter pins them to tuning before assigning
fresh calibration and release cases, preventing accidental holdout reuse.

## Official score and breakdowns

Run the real pipeline for one manifest partition while recording development-
only policy traces and calibration samples:

```bash
python3 scripts/evaluation_harness.py run-benchmark \
  --input-dir data/train \
  --truth data/train_labels.csv \
  --split-manifest /tmp/mib-evaluation/splits.json \
  --split-role release \
  --split-name public-release-v1 \
  --predictions /tmp/mib-evaluation/predictions.jsonl \
  --samples /tmp/mib-evaluation/release-samples.jsonl \
  --run-report /tmp/mib-evaluation/runtime.json
```

The instrumented runner uses the same renderer, extractor, resolver, policy
engine, pinned artifacts, four-worker limit, and canonical writer as the
submission. Its trace/sample output stays outside the runtime image.

Then wrap the official evaluator:

```bash
python3 scripts/evaluation_harness.py evaluate \
  --truth /tmp/mib-evaluation/release_labels.csv \
  --submission /tmp/mib-evaluation/predictions.jsonl \
  --split-name public-release-v1 \
  --split-role release \
  --tuned-on public-tuning-v1 \
  --runtime-seconds 123.4 \
  --peak-memory-mib 512 \
  --output /tmp/mib-evaluation/candidate-report.json
```

The report contains the official total, extraction, classification, and
calibration scores; missing/invalid rows; catastrophic false approvals;
per-field match rates; classification, damage, difficulty, and adversarial
groups; and runtime/memory measurements. Use `--coverage full_training` for the
reproducible 1,000-case public training benchmark instead of a small subset.

## False-approval gate

```bash
python3 scripts/evaluation_harness.py gate \
  --baseline-report /tmp/mib-evaluation/baseline-report.json \
  --candidate-report /tmp/mib-evaluation/candidate-report.json \
  --output /tmp/mib-evaluation/release-decision.json
```

The gate blocks results measured on tuning data, mismatched comparison splits,
missing required measurements, detected identity leakage, and any increase in
catastrophic false approvals. It also identifies newly regressed golden and
adversarial cases before adoption.

## Pinned artifacts

`fit-calibration` fits pool-adjacent-violators isotonic regression from an
external JSONL file containing `case_id`, `split_name`, `raw_signal`, and the
boolean `correct` target. It also requires the external split manifest and
rejects samples outside its calibration partition or overlapping its tuning
partition. Case IDs are used only to prove separation and sample uniqueness and
are removed from the published artifact. `validate-exceptions` admits only
strict `DENIED`/`NEEDS_REVIEW` rules with trusted visible support, held-out
support, split separation, no identity key, and no false-approval regression.

Only the resulting `confidence_calibration.json` and `policy_exceptions.json`
may be copied into `mib_pipeline/artifacts/`. The leakage scanner rejects case
IDs, filenames, PDF/file hashes, and identity lookup keys before publication.

## Separate evaluation image

Build the optional development environment with:

```bash
docker build -f evaluation/Dockerfile -t mib-evaluation .
```

The image contains tooling, schemas, and runtime artifact validators, but no
`data/` directory. Mount labels and predictions read-only and reports writable.
The submitted root `Dockerfile` never copies `devtools/`, the evaluation CLI, or
evaluation reports.

The frozen WO-8 protocol and aggregate gate result are recorded in
[`evaluation/RELEASE_EVIDENCE.md`](evaluation/RELEASE_EVIDENCE.md).
