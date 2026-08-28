# Audio Specialist — RunPod Worker

GPU-accelerated audio analysis worker for [Web2Labs Studio](https://www.web2labs.com). Combines **Faster-Whisper** transcription with **CLAP** audio-text similarity scoring in a single RunPod serverless call.

One upload, two signals: transcript + audio understanding. v2.

## What it does

1. **Faster-Whisper** — Speech-to-text with word-level timing and per-word confidence (probability)
2. **Wav2vec2 forced alignment** (optional, `force_align: true`) — re-times each word against the actual audio waveform (~30-50ms accuracy vs Whisper's ~100-300ms) and adds NP-SBV2 silence-run boundaries (`onset_start` / `offset_end`) for cut-friendly timing
3. **CLAP** (optional) — Scores audio against natural language queries ("loud explosions", "excited reactions", "dramatic music") and returns per-second affine cosine similarities. They are bounded signals, not calibrated event probabilities.
4. **pyannote speaker-diarization-3.1** (experimental, `diarize: true`) —
   emits anonymous speaker turns and word-ordinal attribution as a separate
   sidecar. It never rewrites Whisper/NP-SBV2 word geometry.
5. **NVIDIA Parakeet TDT 0.6B v3** (experimental,
   `asr_backend: "parakeet"`) — fast punctuation-aware ASR for 25 European
   languages. It is explicit opt-in; the default and every draft path remain
   Whisper.
6. **SaT 3L small punctuation probes** (experimental,
   `sat_punctuation_probe` / `sat_punctuation_batch_probe`) — accept one
   exact source-bound XLM-R token window or one bounded arrival batch of up to
   eight windows and return diagnostic terminal-boundary probabilities. They
   cannot rewrite transcript words or geometry and are never loaded by normal
   audio requests.

All models run on the same GPU, sharing the audio file. CLAP overlaps the CTranslate2 Whisper phase, then joins before wav2vec2 alignment or pyannote inference so the heavy PyTorch activation peaks do not stack. Forced alignment adds ~30-50% of the Whisper wall time.

Whisper models stay **resident** once loaded (multi-model residency): a request for `small` no longer evicts `large-v3`, so mixed traffic (Studio chunks + tools presets + the `medium` fallback) avoids model-reload churn. A resident Whisper model is evicted only when a load fails with classified resource exhaustion; authentication, artifact, and network failures leave healthy models intact.

Cold construction of CLAP, wav2vec2, pyannote, Parakeet, and SaT is serialized
through one process-wide lock. Current Transformers versions construct models inside a
temporary meta-device context that patches PyTorch process globals; allowing a
torchaudio model to start in a parallel thread can otherwise leave that model
with unmaterialized meta parameters. Only first-load construction and device
transfer are serialized—resident-model inference remains concurrent. Wav2vec2
retries one clean serialized construction if a bundle still returns meta
tensors, then fails closed rather than publishing or inventing weights.

The image starts an asynchronous CLAP model and audio-decoder warmup by default
(`AUDIO_WORKER_PRELOAD=clap`) so RunPod registration is not blocked by model
transfer. The variable accepts a comma-separated opt-in list (`clap`,
`alignment`, `diarization`, `parakeet`, `sat`, or `whisper:<model>`). CLAP uses
64-window microbatches by default; `CLAP_MICROBATCH_SIZE` may lower that initial
ceiling for a smaller GPU, and runtime OOM recovery still halves it
automatically.

## Models

Pre-downloaded at exact reviewed revisions into the image (every accepted Whisper model is runtime-offline):
- **Whisper base** — public API default
- **Whisper large-v3** — web2labs primary transcription model
- **Whisper medium** — web2labs fallback + tools quality preset
- **Whisper small** — tools fast preset
- **Whisper turbo** — hub CI test
- **CLAP** (`laion/larger_clap_music_and_speech`) — audio-text similarity
- **Wav2vec2** (`WAV2VEC2_ASR_LARGE_LV60K_960H`) — CTC forced alignment (English)
- **Parakeet TDT 0.6B v3**
  (`nvidia/parakeet-tdt-0.6b-v3@7c35754d…`) — opt-in multilingual ASR probe
- **SaT 3L small**
  (`segment-any-text/sat-3l-sm@137da054…`) with pinned XLM-R tokenizer —
  opt-in source-window and maximum-eight-window arrival-batch probe
- **pyannote speaker diarization 3.1** — the pipeline, segmentation model, and
  WeSpeaker PyTorch checkpoint are pinned independently. Because the
  repositories are gated, a deployable image is built with the `hf_token`
  BuildKit secret. Runtime loading is cache-only and never reaches the Hub.

Only `base`, `small`, `medium`, `large-v3`, and `turbo` are accepted. Adding a
Whisper model requires adding and baking its immutable revision first; requests
never trigger a mutable Hugging Face download.

## Input

| Input | Type | Description |
|---|---|---|
| `audio` | str | HTTP(S) URL to an audio file, streamed with retries and a 64 MiB ceiling. Signed credentials/path/query/fragment data is redacted from failures and logs. |
| `audio_base64` | str | Strict Base64-encoded audio file, at most 64 MiB decoded. |
| `span_stream` | dict | Holy Grale streaming mode. Final mode: `{mode:"final", spans:[{index,audio,start_sec}]}` with 1–64 unique spans. Draft mode: `{mode:"draft", next_url, poll_ms?, budget_sec?, idle_timeout_sec?}`. Draft warmup mode: `{mode:"draft_warmup", model?}` (defaults to `turbo`). Mutually exclusive with `audio`/`audio_base64`; yields results via `/stream`. |
| `sat_punctuation_probe` | dict | Explicit diagnostic-only SaT window request. Mutually exclusive with audio, `span_stream`, and `clap_queries`; see the exact contract below. |
| `sat_punctuation_batch_probe` | dict | Explicit diagnostic-only SaT arrival batch with one to eight source windows. Mutually exclusive with the single-window probe, audio, `span_stream`, and `clap_queries`. |
| `model` | str | Whisper model. Default: `"base"` |
| `asr_backend` | str | `"whisper"` (default) or the explicit experimental `"parakeet"` path. Parakeet currently supports classic/final jobs only and rejects CLAP, forced alignment, diarization, translation, and VAD instead of silently ignoring them. |
| `transcription` | str | Output format: `"plain_text"`, `"formatted_text"`, `"srt"`, `"vtt"`. Default: `"plain_text"` |
| `translate` | bool | Translate to English. Default: `false` |
| `language` | str | Language code, or `null` for auto-detection. Default: `null` |
| `word_timestamps` | bool | Include per-word timestamps and probability. Default: `false` |
| `force_align` | bool | Re-time supported-language `word_timestamps` via wav2vec2 CTC alignment and add per-word `onset_start`/`offset_end` evidence. Requires `word_timestamps: true`. The current model supports English only; unsupported languages fail soft with an explicit status. Default: `false` |
| `diarize` | bool | Experimental speaker diarization sidecar. Requires `word_timestamps: true` for word attribution. Default: `false` |
| `diarize_min_speakers` | int | Optional minimum speaker hint from 0–64. `0` means automatic. |
| `diarize_max_speakers` | int | Optional maximum speaker hint from 0–64. `0` means automatic. |
| `enable_vad` | bool | Enable Silero VAD to filter non-speech. Default: `false` |
| `clap_queries` | dict | CLAP query dict `{name: "description"}` with 1–256 entries and bounded names/descriptions. If omitted, CLAP scoring is skipped. |
| `temperature` | float | Sampling temperature. Default: `0` |
| `best_of` | int | Candidates when sampling with non-zero temperature. Default: `5` |
| `beam_size` | int | Beam search width. Default: `5` |
| `patience` | float | Beam decoding patience. Default: `1.0` |
| `length_penalty` | float | Token length penalty. Default: `1.0` |
| `suppress_tokens` | str | Token IDs to suppress. Default: `"-1"` |
| `initial_prompt` | str | Prompt text for the first window. Default: `null` |
| `condition_on_previous_text` | bool | Feed previous output as prompt. Default: `true` |
| `temperature_increment_on_fallback` | float | Temperature increment on failure. Default: `0.2` |
| `compression_ratio_threshold` | float | Compression ratio threshold. Default: `2.4` |
| `logprob_threshold` | float | Average log probability threshold. Default: `-1.0` |
| `no_speech_threshold` | float | No-speech probability threshold. Default: `0.6` |

## Output

### Whisper segments (always returned)

```json
{
  "segments": [
    {
      "id": 0, "start": 0.0, "end": 5.2,
      "text": " Four score and seven years ago...",
      "avg_logprob": -0.12, "compression_ratio": 1.68, "no_speech_prob": 0.05
    }
  ],
  "detected_language": "en",
  "transcription": "Four score and seven years ago..."
}
```

### Word timestamps (when `word_timestamps: true`)

```json
{
  "word_timestamps": [
    { "word": "Four", "start": 0.0, "end": 0.3, "probability": 0.98 },
    { "word": "score", "start": 0.3, "end": 0.6, "probability": 0.95 }
  ]
}
```

### Parakeet probe (experimental)

Parakeet is selected only by sending `"asr_backend": "parakeet"`. The image
contains the exact model revision and runtime inference is
`local_files_only`. CUDA loads use the exact fp16 route retained by the
five-language float32 control; CPU loads remain float32. The selected and
actual runtime dtypes are returned in `asr_backend_evidence`. Its native token
durations are deterministically
coalesced into word timestamps, but they are **not** NP-SBV2 or Natural
Landing cut authority:

```json
{
  "input": {
    "audio": "https://example.com/german.wav",
    "asr_backend": "parakeet",
    "language": "de",
    "word_timestamps": true
  }
}
```

The response includes `asr_backend_evidence` with the model revision,
framework version, supported-language list, language evidence, measured
inference time, and
`timestamp_authority: "DIRECTIONAL_NOT_NP_SBV2"`. Parakeet does not expose a
per-word probability through this route, so `probability` is `null`; the
worker never invents confidence. Unsupported declared languages fail with a
request to route the source to Whisper.

### SaT punctuation window and arrival-batch probes (experimental)

The SaT path is selected only by sending `sat_punctuation_probe` or
`sat_punctuation_batch_probe`; normal Whisper and Parakeet jobs never load it.
Every request must bind the exact
model/tokenizer revisions, threshold `0.65`, lowercase source SHA-256, one
bounded token window, its token SHA-256, and strictly ordered absolute word
terminal anchors. Complete windows must be exactly 510 tokens on the
64-token source grid. Terminal tails may be shorter.

The response includes exact snapshot manifests, actual device/dtype, runtime,
input/window identity, and a row per terminal anchor with the raw boundary
probability and strict-threshold `PERIOD`/`NONE` label. Language is
caller-asserted and explicitly not model-verified. Every authority flag is
false: this endpoint is evidence, not transcript, geometry, Natural Landing,
NP-SBV2, cut, or production authority. Invalid identity fails closed.
The image pins `wtpsplit==2.2.1`, its import-order dependency
`skops==0.14.0`, and `transformers==5.9.0`; all three versions are checked at
probe load.

`sat_punctuation_batch_probe` carries the same source, language, candidate,
token-hash, and absolute-anchor walls for one to eight windows. Complete
windows must be consecutive on the 64-token source grid. At most one
provisional terminal tail is allowed and it must be last. Terminal ordinals
must be unique across the batch. The worker pads to eight rows and performs
one model forward pass; each returned window retains the exact single-window
identity, while the response adds a deterministic batch identity. All
authority flags remain false.

With `force_align: true`, supported-language words are re-timed against the
audio and each aligned word additionally carries NP-SBV2 silence-run
boundaries. The render layer can cut anywhere in `[onset_start, start]` or
`[end, offset_end]` without slicing mid-phoneme. The current
`WAV2VEC2_ASR_LARGE_LV60K_960H` model is English-only. Unsupported languages
keep their exact Whisper geometry and return
`alignment.status: "UNSUPPORTED_LANGUAGE"` rather than a false alignment
claim.

Every attempted alignment returns a typed top-level summary. Authority remains
per word because numbers, symbols, edge words, or failed CTC chunks may retain
Whisper timing:

```json
{
  "word_timestamps_aligned": true,
  "alignment": {
    "schema_version": "w2l-forced-alignment-v1",
    "status": "PARTIAL",
    "model_id": "torchaudio/WAV2VEC2_ASR_LARGE_LV60K_960H",
    "detected_language": "en",
    "supported_languages": ["en"],
    "total_words": 2,
    "aligned_words": 1,
    "fallback_words": 1,
    "aligned_word_fraction": 0.5,
    "per_word_authority": true,
    "transcript_geometry_mutated": false
  },
  "word_timestamps": [
    { "word": "Four", "start": 0.05, "end": 0.31, "probability": 0.98,
      "onset_start": 0.01, "offset_end": 0.38,
      "alignment_status": "ALIGNED_SUPPORTED",
      "alignment_authority": true },
    { "word": "2026", "start": 0.40, "end": 0.72, "probability": 0.96,
      "alignment_status": "FALLBACK_UNALIGNED",
      "alignment_authority": false,
      "alignment_reason": "NO_MODEL_VOCABULARY" }
  ]
}
```

Words the aligner cannot handle keep their original Whisper timing and have no
`onset_start`/`offset_end`. An alignment load or inference failure is fail-soft:
the already-valid Whisper transcript is returned with
`alignment.status: "FAILED"` and `word_timestamps_aligned: false`.

### Speaker diarization sidecar (experimental)

With `diarize: true`, the worker additionally emits
`speaker_diarization`. The artifact explicitly declares
`boundary_authority: false`, keeps speaker identity chunk-local, and refers to
words by their unchanged ordinal in `word_timestamps`:

```json
{
  "speaker_diarization": {
    "schema_version": "w2l-speaker-diarization-v1",
    "status": "COMPLETED",
    "quality_status": "PARTIAL",
    "model_load_policy": "BAKED_CACHE_ONLY",
    "identity_scope": "CHUNK_LOCAL_UNSTABLE",
    "boundary_authority": false,
    "transcript_geometry_mutated": false,
    "speaker_count": 2,
    "word_count": 12,
    "attribution_count": 11,
    "attribution_fraction": 0.9167,
    "coverage_fraction": 0.9167,
    "turns": [
      { "start_sec": 0.1, "end_sec": 2.3, "speaker_id": "SPEAKER_00" }
    ],
    "word_attributions": [
      {
        "word_index": 0,
        "status": "ATTRIBUTED",
        "speaker_id": "SPEAKER_00",
        "coverage_fraction": 0.98,
        "confidence": 0.98,
        "overlap": false,
        "sequential_handoff": false,
        "candidate_speaker_ids": ["SPEAKER_00"]
      }
    ]
  }
}
```

`confidence` is retained as a compatibility alias for temporal
`coverage_fraction`; it is not a calibrated speaker probability. True
simultaneous speech returns v1-compatible word `status: "UNKNOWN"`, additive
`attribution_reason: "AMBIGUOUS_OVERLAP"`, no forced `speaker_id`, and
per-candidate coverage evidence. A sequential A→B handoff in one word is
reported separately and is not mislabeled as overlap. Top-level v1 `status`
remains `COMPLETED` or `FAILED`; additive `quality_status` is `COMPLETED`,
`PARTIAL`, or `EMPTY_OUTPUT`. Zero turns when words need attribution are
`FAILED` with `DIARIZATION_EMPTY_OUTPUT`. Final span-stream sidecars also carry
`timebase: "SPAN_RELATIVE_SECONDS"`, `span_index`, and `span_start_sec`.

The speaker-diarization-3.1 and segmentation-3.0 model conditions must be
accepted by the token used during image build. The token is not persisted and
is not required at runtime. A missing/incompatible baked artifact returns a
`FAILED` sidecar while preserving the successful transcript.

### CLAP scores (when `clap_queries` provided)

```json
{
  "clap_scores": {
    "schema_version": "w2l-clap-scores-v2",
    "status": "COMPLETED",
    "error_code": null,
    "retryable": false,
    "scores": {
      "action": [0.52, 0.48, 0.91, 0.87, ...],
      "reaction": [0.31, 0.29, 0.72, 0.68, ...]
    },
    "duration": 120.5,
    "window_count": 121,
    "final_window_valid_seconds": 0.5,
    "model": "laion/larger_clap_music_and_speech",
    "model_revision": "195c3a3e68faebb3e2088b9a79e79b43ddbda76b",
    "device": "cuda",
    "windowSize": 1.0,
    "score_semantics": {
      "source": "cosine_similarity",
      "calibration": "affine_cosine_not_probability"
    },
    "batching": {
      "configured_microbatch_size": 64,
      "effective_microbatch_size": 64,
      "oom_retries": 0
    }
  }
}
```

Each query gets a per-second array of bounded affine cosine similarities. The
scores are useful for relative ranking and class-specific calibration, but are
not event probabilities and should not share one absolute threshold across all
prompts. The pinned Transformers 5.9 preprocessing explicitly uses CLAP's
`repeatpad` audio strategy; text embeddings are cached by bounded query-set
identity, text prompts use longest-in-microbatch padding under a fixed token
cap, and CUDA/cuBLAS/cuDNN allocation failures halve audio or text microbatches
down to one item. Use the signal for:
- Content-type-specific highlight detection (gunfire for gaming, applause for talks)
- Audio energy profiling with prompt-specific calibration
- Open-vocabulary audio event detection

CLAP is fail-soft with respect to a valid Whisper transcript. If scoring fails,
`clap_scores` remains `null` for backward compatibility and `clap_diagnostics`
contains the same v2 status/error/retryability/timing metadata without a score
matrix.

## Example

```json
{
  "input": {
    "audio": "https://example.com/chunk_000.wav",
    "model": "large-v3",
    "word_timestamps": true,
    "enable_vad": true,
    "clap_queries": {
      "action": "loud explosions and gunfire",
      "reaction": "excited shouting and screaming",
      "music": "dramatic orchestral music"
    }
  }
}
```

## Errors

A failed audio download fails the job with a classifiable signature instead of
an opaque `FileNotFoundError` from inside faster-whisper:

```json
{
  "error": "MEDIA_FETCH_FAILED: could not fetch audio from https://example.com",
  "code": "MEDIA_FETCH_FAILED",
  "stage": "classic_download",
  "retryable": true
}
```

The web2labs server recognizes `MEDIA_FETCH_FAILED` and skips its
model-fallback retry (re-running a doomed download with a smaller model
wouldn't help). The SDK already retries the download 3× with backoff before
this fires. The response exposes only the URL origin; credentials, path, query,
and fragment are never echoed.

## Backwards compatibility

Existing callers that omit `asr_backend` get the same behavior as before:
Whisper-only transcription. CLAP and `force_align` remain opt-in. Parakeet is
not eligible for draft jobs and has no automatic production routing.

### Holy Grale span-stream mode

For Studio's streaming-first pipeline, callers may send a span-stream job
instead of a single `audio` URL.

Final-tier mode transcribes ready spans with the same quality settings as the
classic Studio path:

```json
{
  "input": {
    "span_stream": {
      "mode": "final",
      "spans": [
        { "index": 0, "audio": "https://example.com/span_000.wav", "start_sec": 0 },
        { "index": 1, "audio": "https://example.com/span_001.wav", "start_sec": 120 }
      ]
    },
    "model": "large-v3",
    "word_timestamps": true,
    "force_align": true
  }
}
```

The handler yields one normal transcription result per span, with `span_index`
and `start_sec` added. `return_aggregate_stream` is enabled, so clients can use
`/stream` for incremental span results while `/run` and `/runsync` can still
receive the aggregate list. One span download or inference failure yields a
typed `event: "span_error"` with `failed_span_index` and does not discard later
ready spans. Recoverable events intentionally use `message` rather than a
top-level `error`, because RunPod 1.8.2 treats `error` as terminal. Each
downloaded span is deleted before its result is yielded; the job no longer
retains every span until terminal cleanup.

Draft-tier mode is the Holy Grale ticker probe path. It polls a growing
`next_url` endpoint for small audio segments, transcribes each available segment
with Whisper `turbo` using beam size 1, skips CLAP and forced alignment, and
yields ticker batches:

```json
{
  "input": {
    "span_stream": {
      "mode": "draft",
      "next_url": "https://example.com/next-audio?after=0",
      "poll_ms": 500,
      "budget_sec": 480,
      "idle_timeout_sec": 30
    },
    "language": "en"
  }
}
```

Each segment yield has:

```json
{
  "mode": "draft",
  "event": "segment",
  "yield_index": 0,
  "cursor": "1",
  "next_url": "https://example.com/next-audio?after=1",
  "start_sec": 0,
  "end_sec": 10,
  "words": [{ "word": "Hello", "start": 0.1, "end": 0.4, "probability": 0.91 }],
  "transcription": "Hello...",
  "timing": {
    "job_elapsed_ms": 2400,
    "poll_index": 0,
    "poll_ms": 14,
    "audio_download_ms": 130,
    "model_warmup_ms": 900,
    "model_warmup_wait_ms": 0,
    "prediction_ms": 820
  }
}
```

When the draft job reaches EOF, idle timeout, or the budget, it emits a terminal
control yield with `event:"closed"` and `reason:"eof" | "idle_timeout" |
"budget_exhausted"`. The server can chain a successor draft job from the returned
`cursor` and `next_url`.

Draft warmup mode loads the draft ASR model without polling audio, so Studio can
hide cold-start latency during draft creation/upload:

```json
{
  "input": {
    "span_stream": {
      "mode": "draft_warmup",
      "model": "turbo"
    },
    "model": "turbo",
    "word_timestamps": false
  }
}
```

It yields a single control event:

```json
{
  "mode": "draft_warmup",
  "event": "warmed",
  "model": "turbo",
  "yield_index": 0,
  "timing": {
    "job_elapsed_ms": 900,
    "model_warmup_ms": 900
  }
}
```

## Build and verification

The Dockerfile pins its CUDA base digest, the full reviewed Python dependency
resolution in `builder/constraints.txt`, and reviewed Hub revisions. To include
the gated diarization artifacts without persisting a
token in image metadata, expose the token only as a BuildKit secret:

```bash
docker build --build-arg GATED_MODELS_AVAILABLE=true \
  --build-arg AUDIO_WORKER_BUILD_SHA="$(git rev-parse HEAD)" \
  --secret id=hf_token,env=HF_TOKEN -t audio-worker:holy-grale .
docker run --rm --entrypoint python audio-worker:holy-grale \
  -m unittest discover -v -s / -p 'test_*.py'
docker run --rm --network none -e AUDIO_WORKER_REAL_MODEL_SMOKE=1 \
  --entrypoint python audio-worker:holy-grale \
  -m unittest -v test_diarizer_real_model
docker run --rm --network none -e AUDIO_WORKER_REAL_MODEL_SMOKE=1 \
  --entrypoint python audio-worker:holy-grale \
  -m unittest -v test_clap_real_model
```

`CI | Holy Grale source gate` runs the compile and deployment-tool unit gates
on GitHub-hosted infrastructure for every pull request and trusted branch push.
It deliberately does **not** build the image. GitHub's standard hosted runner
has only 14 GB of SSD storage, while this model-baking build needs substantially
more. The former `runs-on: DO` push jobs had no registered runner and therefore
sat queued until GitHub's 24-hour timeout instead of building anything.

An image release is now an explicit, two-stage operation through
`Release | Build and publish Holy Grale image`:

1. Enter an exact 40-character commit that is already on `holy-grale`.
2. The hosted preflight re-runs the source gates and refuses unless the
   repository variable `AUDIO_WORKER_BUILDER_READY` is exactly `true`.
3. A registered `[self-hosted, linux, x64, audio-worker-builder]` runner must
   prove that Docker,
   Buildx, and at least 100 GiB under Docker's storage root are available.
4. BuildKit and all test containers are capped at 16 CPUs and 64 GiB RAM so
   the shared Studio host retains operational headroom. The named persistent
   BuildKit builder keeps its private local cache between releases.
5. The runner builds the exact commit locally and runs the full in-image suite,
   offline diarizer smoke, and offline CLAP smoke.
6. Only a completely green image is pushed to GHCR. The workflow records the
   resulting immutable `repository@sha256:...` reference in a 90-day release
   receipt. Publishing never deploys the endpoint.

Keep `AUDIO_WORKER_BUILDER_READY` unset or `false` until a builder is actually
registered and has passed the documented bootstrap checks. Do not turn the
variable into a ceremonial bypass: it is the fail-fast replacement for the old
24-hour queue. The complete builder lifecycle and release procedure live in the
[Web2Labs operator runbook](https://git.web2labs.dev/web2labs/web2labs/-/blob/main/docs/operations/runpod-audio-worker.md).

Runtime sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` globally after
the model-baking layer, preventing background Hub conversion checks as well as
ordinary loader downloads. The image label and startup log carry the exact
`AUDIO_WORKER_BUILD_SHA`. Secret-bearing BuildKit layers are never exported to
the GitHub Actions cache; the persistent runner's private local cache is the
only build cache used by that job.

Test-endpoint rollout is a separate, fail-closed operation and is dry-run by
default. RunPod endpoint `worker-audio-expert` (`dx99xymo20v3o9`) is bound to
REST v1 template `worker-audio-expert-template` (`3aapcopikw`). The helper
accepts only an immutable digest in the locked GHCR repository. It verifies the
endpoint, template, exclusive endpoint-to-template binding, and registry
credential `GitHub All` (`cmnhowndh00b5l707vr072ars`) before it can patch
anything:

```bash
python scripts/deploy_holy_grale_test.py \
  --image ghcr.io/vinlow/worker-audio-specialist@sha256:<digest>
# Copy current_image from this fresh dry-run, then apply only with all locks:
python scripts/deploy_holy_grale_test.py \
  --image ghcr.io/vinlow/worker-audio-specialist@sha256:<digest> \
  --apply \
  --expected-current-image ghcr.io/vinlow/worker-audio-specialist@sha256:<current> \
  --confirm-endpoint dx99xymo20v3o9
```

The equivalent GitHub entry point is `Deploy | Holy Grale test endpoint` in the
shared `Production` environment. The environment is a credential scope, not a
claim that this locked test endpoint is production. It always performs the
read-only plan first. Applying
requires the immutable release image, the release's source commit, the exact
freshly observed current image, the literal endpoint ID, and the explicit
`apply` checkbox. The secret-bearing job checks out deployment tooling only
from the trusted `holy-grale` branch; it never executes arbitrary supplied code.

On apply, the only REST v1 template PATCH fields are `imageName` and
`containerRegistryAuthId`. The helper then re-reads the topology and requires
all preserved template configuration—including the environment object—to be
unchanged. `updated` confirms the template mutation, not completion of worker
replacement or a successful application-level canary. The helper never targets
a production endpoint and never prints environment values or API response
bodies.

If a completed template update later serves a worker from the retained old
digest, do not gamble with application retries. Re-trigger the rolling release
for the exact already-current immutable image. The dedicated flag refuses a
different target image and keeps every ordinary topology, optimistic-lock, and
endpoint-confirmation wall:

```bash
python scripts/deploy_holy_grale_test.py \
  --image ghcr.io/vinlow/worker-audio-specialist@sha256:<current>
python scripts/deploy_holy_grale_test.py \
  --image ghcr.io/vinlow/worker-audio-specialist@sha256:<current> \
  --apply \
  --force-rolling-release \
  --expected-current-image ghcr.io/vinlow/worker-audio-specialist@sha256:<current> \
  --confirm-endpoint dx99xymo20v3o9
```

`rolling-release-triggered` proves only that the guarded PATCH completed. Wait
for new-image worker evidence and run the feature-specific canary before using
the endpoint for a larger experiment.

## Based on

Fork of [runpod-workers/worker-faster_whisper](https://github.com/runpod-workers/worker-faster_whisper) with per-word probability from [Vinlow/worker-faster_whisper-probability](https://github.com/Vinlow/worker-faster_whisper-probability).
