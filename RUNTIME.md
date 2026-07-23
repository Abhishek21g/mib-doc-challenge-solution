# Offline Runtime

The repository root is a buildable, CPU-only Docker submission. It
implements the evaluator's exact two-argument boundary:

```text
<input_pdf_dir> <output_predictions_path>
```

The runtime renders every PDF page to visible pixels, runs bounded Tesseract OCR,
links evidence to the active case and applicant, resolves fields through the
published six-level precedence hierarchy, and applies deterministic policy.
When and only when a scored primary output field is internally
`FieldState.UNKNOWN`, a separate RapidOCR pass may read the already-rendered
pages and fill that output value. A visibly resolved literal `unknown` is not
an internal gap and remains immutable. RapidOCR is resolved independently; it
does not add evidence to the primary case or cause policy to be re-run.
Missing, contested, illegible, or untrusted-only decision evidence is routed to
`NEEDS_REVIEW`; `APPROVED` is emitted only after the stricter approval bar is
cleared. An offline, versioned isotonic map first calibrates a policy-derived
decision signal into the probability that the emitted adjudication is correct;
it never uses OCR confidence as the calibration target. The canonical writer
emits exactly the twelve submission fields, rejects duplicate case IDs, and
writes atomically.

The final recovery boundary is deliberately narrow and identity-free. Frozen
evidence-semantic guards may repair a genuinely unresolved output value or a
low-confidence `NEEDS_REVIEW` decision only when the visible primary and RapidOCR
readings meet explicit provenance, agreement, and safety requirements. Denial
rules take priority over approval recovery. The guards do not contain case IDs,
applicant names, filenames, sponsor IDs, or home-world values, and any RapidOCR
exception returns the untouched primary prediction. As the outermost stage, a
pinned linear map recalibrates confidence on low-confidence `NEEDS_REVIEW` rows
using only the emitted decision, bounded input confidence, generic unknown-field
indicators, generic risk presence, and fee category. Its typed boundary cannot
change any of the other eleven fields.

## Build from a clean checkout

```bash
docker build -t mib-submission .
```

The image uses an exact Python patch release. Runtime Python dependencies are
listed in `requirements.lock` and installed with `--no-deps --require-hashes`.
The complete CPython 3.12 wheel closure is pinned for Linux x86_64 and ARM64.
`opencv-python-headless` intentionally supplies `cv2` instead of RapidOCR's
GUI-enabled `opencv-python` metadata dependency; for this reason `pip check`
reports a known distribution-name mismatch even though the runtime import is
satisfied. Do not remove `--no-deps`, or pip will install both variants.

`rapidocr==3.9.2` contains its three default CPU ONNX models and the recognition
dictionary. `RapidOcrEngine` passes the installed model directory as a string
for the pinned OmegaConf 2.0.0 runtime and sets ONNX Runtime intra- and inter-op
threads to one. One engine is created lazily per batch worker thread. Tesseract
5 and its English/OSD data are installed at image-build time from
version-pinned Debian packages. Runtime installation or downloads are never
used.

RapidOCR/PaddleOCR licenses, exact model hashes, upstream provenance, and the
Baidu model attribution are copied into `/app/third_party_licenses`; other
wheel licenses and notices remain in the installed package tree. See
`third_party_licenses/README.md` before redistributing an image.

## Run with the scoring constraints

Create a dedicated output directory before starting the container:

```bash
mkdir -p /tmp/mib-output
docker run --rm \
  --network none \
  --cpus 4 \
  --memory 8g \
  --pids-limit 512 \
  --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=2g \
  --mount type=bind,src="$PWD/data/train",dst=/input,readonly \
  --mount type=bind,src="/tmp/mib-output",dst=/output \
  mib-submission /input /output/predictions.jsonl
```

The entrypoint rejects missing or extra arguments. It reads only the supplied
input directory and writes only the exact supplied output path. The image runs
as an unprivileged user, sets `/tmp` as its temporary and home directory, and
does not require a writable container root.

## Offline and resource guarantees

- No runtime package installation, model download, API key, network call, or
  external service is used. RapidOCR reads only models embedded in its pinned
  wheel.
- The image is CPU-only. Common numeric and tokenization thread pools are capped
  at four threads, and the application worker limit can never exceed four.
- Any scratch files remain below `/tmp`.
- Final predictions are written to the caller-provided output mount.
- The submitted image is designed for the evaluator's 4-vCPU, 8-GiB RAM,
  512-PID, and 2-GiB `/tmp` limits.

## Verify locally

Run the host-side contract tests:

```bash
python3 -m unittest discover -s tests -v
```

The RapidOCR recovery tests use injected fakes, so the host test suite can run
without installing the heavyweight OCR closure. Image verification must also
exercise both target architectures, including the offline/read-only boundary:

```bash
docker run --rm --network none --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=2g \
  mib-submission /input /output/predictions.jsonl
```

When Docker is available, run the image with the full command above. A
successful run exits with status zero and creates canonical policy predictions
for independently processed PDFs. A technically unreadable case is isolated
rather than aborting the batch. An empty input directory produces an empty
`predictions.jsonl`.
