# Audio Specialist — RunPod Worker

GPU-accelerated audio analysis worker for [Web2Labs Studio](https://www.web2labs.com). Combines **Faster-Whisper** transcription with **CLAP** audio-text similarity scoring in a single RunPod serverless call.

One upload, two signals: transcript + audio understanding. v2.

## What it does

1. **Faster-Whisper** — Speech-to-text with word-level timing and per-word confidence (probability)
2. **Wav2vec2 forced alignment** (optional, `force_align: true`) — re-times each word against the actual audio waveform (~30-50ms accuracy vs Whisper's ~100-300ms) and adds NP-SBV2 silence-run boundaries (`onset_start` / `offset_end`) for cut-friendly timing
3. **CLAP** (optional) — Scores audio against natural language queries ("loud explosions", "excited reactions", "dramatic music") and returns per-second relevance scores
4. **pyannote speaker-diarization-3.1** (experimental, `diarize: true`) —
   emits anonymous speaker turns and word-ordinal attribution as a separate
   sidecar. It never rewrites Whisper/NP-SBV2 word geometry.
5. **NVIDIA Parakeet TDT 0.6B v3** (experimental,
   `asr_backend: "parakeet"`) — fast punctuation-aware ASR for 25 European
   languages. It is explicit opt-in; the default and every draft path remain
   Whisper.

All models run on the same GPU, sharing the audio file. CLAP runs **concurrently with transcription** (near-zero wall-time overhead; serially it added ~5s per 2-minute chunk); forced alignment adds ~30-50% of the Whisper wall time.

Whisper models stay **resident** once loaded (multi-model residency): a request for `small` no longer evicts `large-v3`, so mixed traffic (Studio chunks + tools presets + the `medium` fallback) causes no model-reload churn. The full production set fits in ~9GB alongside CLAP + wav2vec2; on VRAM pressure the least-recently-used model is evicted and the load retried.

Cold construction of CLAP, wav2vec2, pyannote, and Parakeet is serialized
through one process-wide lock. Current Transformers versions construct models inside a
temporary meta-device context that patches PyTorch process globals; allowing a
torchaudio model to start in a parallel thread can otherwise leave that model
with unmaterialized meta parameters. Only first-load construction and device
transfer are serialized—resident-model inference remains concurrent. Wav2vec2
retries one clean serialized construction if a bundle still returns meta
tensors, then fails closed rather than publishing or inventing weights.

## Models

Pre-downloaded into the image (instant cold start — every model a production code path requests):
- **Whisper large-v3** — web2labs primary transcription model
- **Whisper medium** — web2labs fallback + tools quality preset
- **Whisper small** — tools fast preset
- **Whisper turbo** — hub CI test
- **CLAP** (`laion/larger_clap_music_and_speech`) — audio-text similarity
- **Wav2vec2** (`WAV2VEC2_ASR_LARGE_LV60K_960H`) — CTC forced alignment (English)
- **Parakeet TDT 0.6B v3**
  (`nvidia/parakeet-tdt-0.6b-v3@7c35754d…`) — opt-in multilingual ASR probe

Other Whisper sizes in `AVAILABLE_MODELS` work too but download from HuggingFace on first request.

## Input

| Input | Type | Description |
|---|---|---|
| `audio` | str | URL to audio file |
| `audio_base64` | str | Base64-encoded audio file |
| `span_stream` | dict | Holy Grale streaming mode. Final mode: `{mode:"final", spans:[{index,audio,start_sec}]}`. Draft mode: `{mode:"draft", next_url, poll_ms?, budget_sec?, idle_timeout_sec?}`. Draft warmup mode: `{mode:"draft_warmup", model?}`. Mutually exclusive with `audio`/`audio_base64`; yields results via `/stream`. |
| `model` | str | Whisper model. Default: `"base"` |
| `asr_backend` | str | `"whisper"` (default) or the explicit experimental `"parakeet"` path. Parakeet currently supports classic/final jobs only and rejects CLAP, forced alignment, diarization, translation, and VAD instead of silently ignoring them. |
| `transcription` | str | Output format: `"plain_text"`, `"formatted_text"`, `"srt"`, `"vtt"`. Default: `"plain_text"` |
| `translate` | bool | Translate to English. Default: `false` |
| `language` | str | Language code, or `null` for auto-detection. Default: `null` |
| `word_timestamps` | bool | Include per-word timestamps and probability. Default: `false` |
| `force_align` | bool | Re-time supported-language `word_timestamps` via wav2vec2 CTC alignment and add per-word `onset_start`/`offset_end` evidence. Requires `word_timestamps: true`. The current model supports English only; unsupported languages fail soft with an explicit status. Default: `false` |
| `diarize` | bool | Experimental speaker diarization sidecar. Requires `word_timestamps: true` for word attribution. Default: `false` |
| `diarize_min_speakers` | int | Optional minimum speaker hint. `0` means automatic. |
| `diarize_max_speakers` | int | Optional maximum speaker hint. `0` means automatic. |
| `enable_vad` | bool | Enable Silero VAD to filter non-speech. Default: `false` |
| `clap_queries` | dict | CLAP query dict `{name: "description"}`. If omitted, CLAP scoring is skipped. |
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
`local_files_only`. Its native token durations are deterministically
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
    "identity_scope": "CHUNK_LOCAL_UNSTABLE",
    "boundary_authority": false,
    "transcript_geometry_mutated": false,
    "speaker_count": 2,
    "turns": [
      { "start_sec": 0.1, "end_sec": 2.3, "speaker_id": "SPEAKER_00" }
    ],
    "word_attributions": [
      {
        "word_index": 0,
        "status": "ATTRIBUTED",
        "speaker_id": "SPEAKER_00",
        "confidence": 0.98,
        "overlap": false
      }
    ]
  }
}
```

`HUGGINGFACE_TOKEN` (or `HF_TOKEN`) must be available to the runtime and the
speaker-diarization-3.1 plus segmentation-3.0 model conditions must have been
accepted. A missing model/token returns a `FAILED` sidecar while preserving
the successful transcript.

### CLAP scores (when `clap_queries` provided)

```json
{
  "clap_scores": {
    "scores": {
      "action": [0.52, 0.48, 0.91, 0.87, ...],
      "reaction": [0.31, 0.29, 0.72, 0.68, ...]
    },
    "duration": 120.5,
    "model": "laion/larger_clap_music_and_speech",
    "device": "cuda",
    "windowSize": 1.0
  }
}
```

Each query gets a per-second array of relevance scores (0-1). Use these for:
- Content-type-specific highlight detection (gunfire for gaming, applause for talks)
- Audio energy profiling without manual threshold tuning
- Open-vocabulary audio event detection

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

```
MEDIA_FETCH_FAILED: could not download audio from <url>
```

The web2labs server recognizes `MEDIA_FETCH_FAILED` and skips its
model-fallback retry (re-running a doomed download with a smaller model
wouldn't help). The SDK already retries the download 3× with backoff before
this fires.

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
receive the aggregate list.

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

## Based on

Fork of [runpod-workers/worker-faster_whisper](https://github.com/runpod-workers/worker-faster_whisper) with per-word probability from [Vinlow/worker-faster_whisper-probability](https://github.com/Vinlow/worker-faster_whisper-probability).
