# Active-speaker local lab runtime

The additive `active_speaker_source_worker.py` entry point measures complete
raw uploads with one resident AVA model, pinned YuNet tracking, the unchanged
two-view scorer, and no review video. It is explicitly throughput-only, not a
v2 observation or crop authority. The default supplied-track handler is unchanged.
Select the source worker only through an owned temporary benchmark endpoint's
command override. Its strict batch is limited to three distinct sources, two
raw hours, and 2400 seconds of execution; partial coverage never receives a full
source-hour denominator. Model/runtime sources are inherited from the pinned
Starforge closure; the new measurement entry point is bound by this worker's
release commit and immutable image. No production endpoint is changed.

This directory is an isolated LR-ASD feasibility runtime for Starforge Visual
Director. It is not imported by Studio, Audio-Worker, Render2, or the existing
Visual Director planner. Its successful result is a timestamped face-track
observation ledger plus an annotated review video. It grants **no crop or
production authority**.

The runtime deliberately does not vendor or download LR-ASD source, LR-ASD
weights, YuNet weights, or input media. Every executable external artifact must
be supplied through an absolute path and accompanied by its exact SHA-256. The
LR-ASD checkout must additionally be clean for the four source files that this
adapter executes and resolve to the exact requested Git commit.

## Pinned first-test closure

- Docker base: local cached `audio-worker:holy-grale-695b212`
- Docker base image ID:
  `sha256:973f44ba32c211a01f527a008fe8ec31bfc91a8c706d0e49e2ee2eb3f4b83690`
- LR-ASD commit: `1b6dcd2d8fc2895683de6508ec6294ec47d388ca`
- LR-ASD executed-source SHA-256 at that checkout:
  `89e4de74949aba7457b8254206885ea0646338c9d91a1e8556dbd3aebabd4eda`
- TalkSet checkpoint SHA-256:
  `6b4ef53694e874e96cf630198dc479c78aebb3993bbf166aee3d926dfe7d9342`
- AVA checkpoint SHA-256:
  `85e6c77fc981595234790d1e128ebb60352d37726b2445e0ef8891e2512fe9e3`
- `opencv-python-headless==4.11.0.86`
- `python-speech-features==0.6`

The source closure hash covers only the files dynamically imported by the
adapter: `loss.py`, `model/Classifier.py`, `model/Encoder.py`, and
`model/Model.py`. Check the exact local identity before running:

```bash
python src/tools/starforge/visual-director/active-speaker/active_speaker_runtime.py \
  source-identity \
  --lrasd-root /absolute/path/to/LR-ASD \
  --lrasd-revision 1b6dcd2d8fc2895683de6508ec6294ec47d388ca
```

## Hermetic contract tests

The pure contract tests require only Python. The clock tests additionally use
local FFmpeg to generate real `+500ms` and `-500ms` stream-origin fixtures;
they do not use the model or any source video:

```bash
python -m unittest discover \
  -s src/tools/starforge/visual-director/active-speaker/tests \
  -p 'test_*.py' -v
```

## Build the local image

The base image must already exist locally. The build installs only the two
explicit pinned dependencies and copies no model assets:

```bash
docker build \
  --build-arg AUDIO_WORKER_BASE_IMAGE_ID=sha256:973f44ba32c211a01f527a008fe8ec31bfc91a8c706d0e49e2ee2eb3f4b83690 \
  -t starforge-active-speaker:local-v1 \
  src/tools/starforge/visual-director/active-speaker
```

The image defaults to the unprivileged numeric identity `65532:65532`. For
local bind-mounted output, the documented `--user "$(id -u):$(id -g)"`
override keeps generated evidence owned and readable by the host account.

## Run one diagnostic

Inspect the input once to choose explicit stream indexes; the runtime never
guesses them:

```bash
ffprobe -v error -show_entries stream=index,codec_type,codec_name \
  -of json /absolute/path/to/input.mp4
```

Then run with all roots mounted read-only and a fresh, nonexistent output
directory. Substitute the real YuNet and media hashes:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v /absolute/path/to/LR-ASD:/inputs/lrasd:ro \
  -v /absolute/path/to/yunet.onnx:/inputs/yunet.onnx:ro \
  -v /absolute/path/to/input.mp4:/inputs/input.mp4:ro \
  -v /absolute/path/to/original-source.mp4:/inputs/source.mp4:ro \
  -v /absolute/path/to/output-parent:/outputs \
  starforge-active-speaker:local-v1 run \
  --base-image-id sha256:973f44ba32c211a01f527a008fe8ec31bfc91a8c706d0e49e2ee2eb3f4b83690 \
  --lrasd-root /inputs/lrasd \
  --lrasd-revision 1b6dcd2d8fc2895683de6508ec6294ec47d388ca \
  --lrasd-source-sha256 89e4de74949aba7457b8254206885ea0646338c9d91a1e8556dbd3aebabd4eda \
  --checkpoint /inputs/lrasd/weight/finetuning_TalkSet.model \
  --checkpoint-sha256 6b4ef53694e874e96cf630198dc479c78aebb3993bbf166aee3d926dfe7d9342 \
  --yunet /inputs/yunet.onnx \
  --yunet-sha256 REPLACE_WITH_EXACT_64_LOWERCASE_HEX \
  --input-video /inputs/input.mp4 \
  --input-sha256 REPLACE_WITH_EXACT_64_LOWERCASE_HEX \
  --source-video /inputs/source.mp4 \
  --source-video-sha256 REPLACE_WITH_ORIGINAL_SOURCE_SHA256 \
  --source-interval-start-us REPLACE_WITH_EXACT_START_MICROSECONDS \
  --source-interval-end-us REPLACE_WITH_EXACT_END_MICROSECONDS \
  --video-stream-index 0 \
  --audio-stream-index 1 \
  --output-dir /outputs/attempt-001 \
  --device auto
```

The selected audio and video stream `start_pts` values and their first decoded
frame timestamps must agree, then are preserved relative to video frame zero.
Positive audio offsets are filled with exact sample-counted silence; negative
offsets are trimmed by an exact sample count. Missing or contradictory origin
metadata fails closed instead of independently rebasing both streams to zero.

Each attempt uses a new output directory. `result.json` and `failure.json` are
published atomically and no-clobber. Success is published only after the review
video has been decoded again and its 25fps video clock, 16kHz mono audio clock,
dimensions, codecs, frame count, duration, and bounded AAC tail are validated.
The receipt identity binds the base image ID, Dockerfile, dependency lock,
license, runtime source, tracking/preprocessing policies, prepared input hash,
original source hash, and selected source interval.

The runtime never converts missing or malformed evidence into “no speaker.” The
annotated video shows boxes and uncalibrated `rawSpeakingScore` values only.
Those values are class-1 logits averaged across the explicit ordered context
set `[1, 2, 3, 4, 5, 6]`; they are not probabilities or crop decisions.

The receipt-bound detector defaults are YuNet score threshold `0.7` and a
maximum in-shot tracking gap of `15` frames. They match the maintained face
detector's production threshold and remain explicit CLI overrides for sealed
experiments.

LR-ASD is copyright (c) 2025 Liao Junhua and licensed under MIT. The exact
upstream license text is preserved in `LR-ASD-LICENSE.txt`. This runtime does
not redistribute the upstream source or checkpoints.

## Supplied-track mirror-invariant v2

V2 is additive: the v1 Dockerfile, Python entry point, schema, source closure,
and one-view scoring path remain byte-identical. V2 has its own entry point and
image definition. It accepts only a complete, authenticated canonical-track
manifest; it has no YuNet argument and its media adapter rejects any attempt to
invoke detection or tracking.

The manifest schema is
`starforge-active-speaker-supplied-tracks-v2`. Its exact root keys are:

```json
{
  "clock": {},
  "clockIdentity": "sha256:...",
  "contentIdentity": "sha256:...",
  "producer": {},
  "schemaVersion": "starforge-active-speaker-supplied-tracks-v2",
  "status": "COMPLETE",
  "tracks": []
}
```

`clock` uses the same canonical clock projection published by v1: audio,
explicit input streams and origins, prepared-input bytes/hash, contiguous
half-open shots, original-source interval/bytes/hash, and exact `25/1` video
dimensions and decoded-frame count. `producer` binds the canonical producer
kind, strict YuNet/no-fallback model and threshold, tracking policy,
runtime/source closure, and an explicit `0..frameCount-1` processed-frame PTS
ledger. Tracks are ordered by `trackId`, remain inside one shot, and contain
contiguous dense frames using the existing `TrackGeometry.as_json()` shape.
`clockIdentity` hashes `clock`; `contentIdentity` hashes the complete manifest
after removing only `contentIdentity`. The separately supplied file SHA-256
authenticates the exact UTF-8 JSON bytes.

`producer.geometryLineage` is explicit. Direct canonical observations use
`BASE_OBSERVED` with producer kind `starforge-canonical-face-tracks-v1` and
must bind the exact sealed v1 observation bytes, independent file SHA-256, and
all five recomputed v1 identities:

```json
{
  "inputSha256": "...",
  "kind": "BASE_OBSERVED",
  "sourceObservation": {
    "bytes": 162177,
    "identities": {
      "clockIdentity": "sha256:...",
      "modelIdentity": "sha256:...",
      "observationIdentity": "sha256:...",
      "runIdentity": "sha256:...",
      "runtimeIdentity": "sha256:..."
    },
    "schemaVersion": "starforge-active-speaker-observation-v1",
    "sha256": "..."
  }
}
```

The v1 result is not trusted merely because those identities are copied into a
manifest. Every base and derived v2 run must mount that result separately and
pass its independently computed SHA-256. The loader strict-keys the full v1
envelope and recomputes its clock, model, observation, runtime, and run
identities. The base manifest must then exactly reproduce its clock, tracks,
tracking policy, YuNet detector/no-fallback record, runtime identity, and
runtime source closure.

A mechanically flipped presentation uses
producer kind `starforge-horizontal-mirror-face-tracks-v1` plus lineage kind
`HORIZONTAL_MIRROR_DERIVED`, and binds the base manifest file
SHA-256/content identity, base input SHA-256, derived input SHA-256, and the
frozen `x1 = width - source.x2`, `x2 = width - source.x1`, unchanged-y and
topology-preservation policy. Derived runs must additionally mount the exact
base manifest and pass `--lineage-source-tracks` plus its independently
computed `--lineage-source-tracks-sha256`. A derived detector record is
explicitly `AUTHENTICATED_SOURCE_MANIFEST` evidence and must match the base
manifest's YuNet model and tracking policy, so it cannot claim YuNet ran on the
flipped presentation. The runtime authenticates both files
and the same exact `sourceObservation` root, plus every preserved ID, frame,
shot, PTS, observation flag, and transformed
box. It therefore never misrepresents transformed boxes as a second YuNet
detection pass. Its producer runtime/source closure identifies the transform
producer; the base manifest retains the closure of the detector/tracker pass.
The v1 result, supplied manifest, and (for derived inputs) base manifest are
all authenticated before inference and re-read, re-hashed, re-parsed, and
cross-validated immediately after inference before any success receipt can be
published.

The pinned LR-ASD commit has no `model/__init__.py`; `model` is an implicit
namespace package and executes no initializer bytes. V2 authenticates the four
actual source files, requires that initializer to remain absent, requires the
namespace search path to resolve only to the mounted `model/` directory, and
verifies every imported module origin. The checkout must not contain shadow
`model.py` or executable `__pycache__` bytecode. Inspect this closure with
bytecode writing disabled:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  src/tools/starforge/visual-director/active-speaker/active_speaker_runtime_v2.py \
  source-identity-v2 \
  --lrasd-root /absolute/path/to/LR-ASD \
  --lrasd-revision 1b6dcd2d8fc2895683de6508ec6294ec47d388ca
```

Build the isolated v2 image without changing the v1 tag:

```bash
docker build \
  -f src/tools/starforge/visual-director/active-speaker/Dockerfile.v2 \
  --build-arg AUDIO_WORKER_BASE_IMAGE_ID=sha256:973f44ba32c211a01f527a008fe8ec31bfc91a8c706d0e49e2ee2eb3f4b83690 \
  -t starforge-active-speaker:checkpoint31-authenticated-lineage \
  src/tools/starforge/visual-director/active-speaker
```

Run with the supplied manifest and every input mounted read-only. The output
parent is the only writable bind; the container filesystem stays read-only:

```bash
docker run --rm \
  --network none \
  --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=256m \
  --user "$(id -u):$(id -g)" \
  -v /absolute/path/to/LR-ASD:/inputs/lrasd:ro \
  -v /absolute/path/to/checkpoint.model:/inputs/checkpoint.model:ro \
  -v /absolute/path/to/v1-result.json:/inputs/v1-result.json:ro \
  -v /absolute/path/to/tracks.json:/inputs/tracks.json:ro \
  -v /absolute/path/to/input.mp4:/inputs/input.mp4:ro \
  -v /absolute/path/to/original.mp4:/inputs/source.mp4:ro \
  -v /absolute/path/to/output-parent:/outputs \
  starforge-active-speaker:checkpoint31-authenticated-lineage run-supplied-v2 \
  --base-image-id sha256:973f44ba32c211a01f527a008fe8ec31bfc91a8c706d0e49e2ee2eb3f4b83690 \
  --lrasd-root /inputs/lrasd \
  --lrasd-revision 1b6dcd2d8fc2895683de6508ec6294ec47d388ca \
  --lrasd-source-sha256 REPLACE_WITH_V2_SOURCE_SHA256 \
  --checkpoint /inputs/checkpoint.model \
  --checkpoint-sha256 REPLACE_WITH_CHECKPOINT_SHA256 \
  --base-observation-result /inputs/v1-result.json \
  --base-observation-result-sha256 REPLACE_WITH_V1_RESULT_FILE_SHA256 \
  --supplied-tracks /inputs/tracks.json \
  --supplied-tracks-sha256 REPLACE_WITH_MANIFEST_FILE_SHA256 \
  --input-video /inputs/input.mp4 \
  --input-sha256 REPLACE_WITH_INPUT_SHA256 \
  --source-video /inputs/source.mp4 \
  --source-video-sha256 REPLACE_WITH_SOURCE_SHA256 \
  --source-interval-start-us REPLACE_WITH_START \
  --source-interval-end-us REPLACE_WITH_END \
  --video-stream-index 0 \
  --audio-stream-index 1 \
  --output-dir /outputs/attempt-v2-001 \
  --device cpu
```

For a `HORIZONTAL_MIRROR_DERIVED` manifest, also mount its exact base manifest
and pass `--lineage-source-tracks` and
`--lineage-source-tracks-sha256`. Omitting either half of that pair, supplying
a different base, or presenting a v1 result whose independently pinned hash or
recomputed identities differ is a contract failure, never a no-face result.

Each supplied crop is scored sequentially in canonical and contiguous
horizontal-mirror form with identical audio and the unchanged LR-ASD contexts
`[1,2,3,4,5,6]`. Every score sample preserves both component logits and their
exact `math.fsum` arithmetic mean. V2 uses an edge-padded 13-frame geometry
median and floor-left/ceil-right horizontal crop bounds; both policies are
receipt-bound and exactly mirror-equivariant even for fractional boxes at track
boundaries. Either-view failure fails the whole run.
The output remains diagnostic-only with `cropAuthority: NONE`.

## Bounded RunPod adapter

`Dockerfile.runpod` packages that exact v2 runtime as an additive serverless
worker. The adapter owns only transport and subprocess supervision; it cannot
change face tracks, scores, thresholds, crop authority, or the authenticated
v2 receipt. Build its executable test stage and runtime image with:

```bash
docker build --target test \
  -f src/tools/starforge/visual-director/active-speaker/Dockerfile.runpod \
  -t starforge-active-speaker:runpod-test \
  src/tools/starforge/visual-director/active-speaker

docker build \
  -f src/tools/starforge/visual-director/active-speaker/Dockerfile.runpod \
  -t starforge-active-speaker:runpod \
  src/tools/starforge/visual-director/active-speaker
```

The endpoint must set `STARFORGE_ARTIFACT_HOSTS` to a comma-separated exact
hostname allowlist. Each request uses schema
`starforge-active-speaker-runpod-request-v1` and supplies four HTTP(S)
artifacts as `{ "bytes", "sha256", "url" }`: prepared input, original source,
canonical supplied tracks, and the independently authenticated base
observation. The remaining required fields are checkpoint (`AVA` or
`TALKSET`), exact source interval in microseconds, explicit audio/video stream
indexes, a 1–900 second subprocess deadline, and output mode
(`METRICS_ONLY` or `FULL_RESULT`). Unknown keys fail closed.

Downloads happen once, in parallel, with exact byte/hash checks. The handler
then launches the v2 runtime as a fresh CUDA-only process group and kills the
whole group on deadline. Its compact response separates raw-source duration,
25 fps face-track seconds, two-view workload, download time, runtime stage
time, handler time, and immutable worker/runtime identities. The authenticated
score ledger is always returned, including canonical, horizontal-mirror, and
exact-mean logits for the requested AVA or TalkSet checkpoint; `FULL_RESULT`
adds the rest of the v2 receipt. Those fields are measurement evidence only;
serverless cost must still come from the RunPod job and billing receipts, never
from handler wall time alone.
