# WO-8 Release Evidence

These are local engineering benchmark results, not official leaderboard
scores. All labels, split manifests, per-case reports, traces, and calibration
samples remained outside the repository and the submitted runtime image.

## Honest split protocol

- Source: 1,000 public training cases, stratified by adjudication.
- Frozen roles: 700 tuning, 150 calibration, and 150 release cases.
- Seed label: `public-stratified-v2-forced-diagnostic-tuning`.
- Thirty cases inspected during the initial OCR diagnosis were explicitly
  forced into tuning before fresh calibration and release assignments were
  generated.
- The final extraction and policy logic was frozen before calibration and
  release evaluation.
- Confidence was fit only on the 150 calibration cases with deterministic
  pool-adjacent-violators isotonic regression.
- The final candidate was evaluated once on the fresh release partition.

## Release result

The baseline was executed from committed revision `026daee`. The candidate and
baseline used the same 150-case release partition and the same official
evaluator wrapper.

| Measurement | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Total score | 45.46 | 100.11 | +54.65 |
| Extraction | 9.59 | 34.36 | +24.77 |
| Classification | 35.87 | 52.40 | +16.53 |
| Calibration | 0.00 | 13.36 | +13.36 |
| Catastrophic false approvals | 0 | 0 | 0 |
| Answered cases | 150 | 150 | 0 |
| Invalid rows | 0 | 0 | 0 |
| Runtime | 160.08 s | 164.77 s | +4.69 s |
| Peak memory | 398.3 MiB | 370.9 MiB | -27.4 MiB |

Release gate decision: **PASS**.

The gate also confirmed honest split metadata, matching case coverage,
non-zero runtime and memory measurements, official-evaluator metrics, and no
case identity leakage in either pinned runtime artifact.
