# Audio Specialist — RunPod Worker

GPU-accelerated audio analysis worker for [Web2Labs Studio](https://www.web2labs.com). Combines **Faster-Whisper** transcription with **CLAP** audio-text similarity scoring in a single RunPod serverless call.

One upload, two signals: transcript + audio understanding. v2.

## What it does

1. **Faster-Whisper** — Speech-to-text with word-level timing and per-word confidence (probability)
2. **CLAP** (optional) — Scores audio against natural language queries ("loud explosions", "excited reactions", "dramatic music") and returns per-second relevance scores

Both models run on the same GPU, sharing the audio file. CLAP adds ~5s per 2-minute chunk — negligible overhead on top of Whisper.

## Models

Pre-downloaded (instant cold start):
- **Whisper large-v3** — highest quality transcription
- **Whisper distil-large-v3.5** — near large-v3 quality, ~5x faster
- **Whisper turbo** — fastest, good for draft transcriptions
- **CLAP** (`laion/larger_clap_music_and_speech`) — audio-text similarity

## Input

| Input | Type | Description |
|---|---|---|
| `audio` | str | URL to audio file |
| `audio_base64` | str | Base64-encoded audio file |
| `model` | str | Whisper model. Default: `"base"` |
| `transcription` | str | Output format: `"plain_text"`, `"formatted_text"`, `"srt"`, `"vtt"`. Default: `"plain_text"` |
| `translate` | bool | Translate to English. Default: `false` |
| `language` | str | Language code, or `null` for auto-detection. Default: `null` |
| `word_timestamps` | bool | Include per-word timestamps and probability. Default: `false` |
| `enable_vad` | bool | Enable Silero VAD to filter non-speech. Default: `false` |
| `clap_queries` | dict | CLAP query dict `{name: "description"}`. If omitted, CLAP scoring is skipped. |
| `temperature` | float | Sampling temperature. Default: `0` |
| `best_of` | int | Candidates when sampling with non-zero temperature. Default: `5` |
| `beam_size` | int | Beam search width. Default: `5` |
| `patience` | float | Beam decoding patience. Default: `1.0` |
| `length_penalty` | float | Token length penalty. Default: `0` |
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

## Backwards compatibility

Existing callers that don't send `clap_queries` get the same behavior as before — Whisper-only transcription. CLAP is purely additive.

## Based on

Fork of [runpod-workers/worker-faster_whisper](https://github.com/runpod-workers/worker-faster_whisper) with per-word probability from [Vinlow/worker-faster_whisper-probability](https://github.com/Vinlow/worker-faster_whisper-probability).
