"""Experimental NVIDIA Parakeet TDT transcription sidecar.

The production-compatible worker contract remains Whisper by default.  This
module is loaded only when a request explicitly selects ``asr_backend:
"parakeet"``.  The model and framework versions are pinned and downloaded at
image-build time; runtime inference is local-cache-only.

Parakeet exposes token durations, not the NP-SBV2 onset/offset authority used
by Studio cuts.  We deterministically coalesce those tokens into directional
word geometry and label that limitation in the response instead of pretending
the timestamps are cut authority.
"""

import re
import threading
import time
from typing import Iterable, Mapping, Optional, Sequence

from hf_auth import hf_from_pretrained_kwargs
from model_load_lock import serialized_model_load


PARAKEET_SCHEMA_VERSION = "w2l-parakeet-tdt-probe-v2"
PARAKEET_MODEL_DTYPE_SELECTION = "cuda-float16-cpu-float32"
PARAKEET_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
PARAKEET_MODEL_REVISION = "7c35754d166cca382ad1e53e68b01e7c575f3a1d"
PARAKEET_TRANSFORMERS_VERSION = "5.9.0"
PARAKEET_SAMPLE_RATE = 16000
PARAKEET_SUPPORTED_LANGUAGES = frozenset(
    {
        "bg",
        "hr",
        "cs",
        "da",
        "nl",
        "en",
        "et",
        "fi",
        "fr",
        "de",
        "el",
        "hu",
        "it",
        "lv",
        "lt",
        "mt",
        "pl",
        "pt",
        "ro",
        "sk",
        "sl",
        "es",
        "sv",
        "ru",
        "uk",
    }
)

_CONTROL_TOKEN_RE = re.compile(r"<\|([^|>]+)\|>")
_TERMINAL_PUNCTUATION_RE = re.compile(r"[.!?…][\"'’”»)]*$")
_PUNCTUATION_ONLY_RE = re.compile(r"^[^\w\s]+$", re.UNICODE)
_MAX_DIRECTIONAL_SEGMENT_SECONDS = 30.0


def normalize_language_code(language_code: Optional[str]) -> Optional[str]:
    """Normalize a BCP-47-ish language code to its primary subtag."""
    if not isinstance(language_code, str):
        return None
    normalized = language_code.strip().lower().replace("_", "-")
    if not normalized:
        return None
    return normalized.split("-", 1)[0]


def supports_language(language_code: Optional[str]) -> bool:
    return normalize_language_code(language_code) in PARAKEET_SUPPORTED_LANGUAGES


def extract_model_language(
    token_ids: Sequence[int],
    tokenizer,
) -> tuple[Optional[str], list[str]]:
    """Extract supported language control tokens from a generated sequence."""
    raw_tokens = tokenizer.convert_ids_to_tokens(list(token_ids))
    observed = []
    for token in raw_tokens:
        if not isinstance(token, str):
            continue
        match = _CONTROL_TOKEN_RE.fullmatch(token)
        if not match:
            continue
        language = normalize_language_code(match.group(1))
        if language in PARAKEET_SUPPORTED_LANGUAGES and language not in observed:
            observed.append(language)
    return (observed[0] if observed else None), observed


def strip_control_tokens(text: str) -> str:
    """Remove model control markers while preserving transcript punctuation."""
    return re.sub(r"\s+", " ", _CONTROL_TOKEN_RE.sub("", text)).strip()


def _finish_word(words: list[dict], current: Optional[dict]) -> None:
    if current is None:
        return
    if not current["word"].strip():
        return
    words.append(current)


def token_timestamps_to_words(
    token_timestamps: Iterable[Mapping],
) -> list[dict]:
    """Coalesce decoded token-duration geometry into directional words.

    The Transformers Parakeet processor emits incrementally decoded token
    chunks.  A leading whitespace starts a new word, punctuation attaches to
    the preceding word, and subword pieces extend the current word.  We retain
    the native min/max token geometry and explicitly withhold probability and
    cut authority.
    """
    words: list[dict] = []
    current: Optional[dict] = None

    for entry in token_timestamps:
        token = entry.get("token")
        if not isinstance(token, str) or not token:
            continue
        if _CONTROL_TOKEN_RE.fullmatch(token):
            continue

        try:
            start = float(entry["start"])
            end = float(entry["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < 0 or end < start:
            continue

        parts = re.findall(r"\s+|[^\s]+", token)
        pending_space = False
        for part in parts:
            if part.isspace():
                pending_space = True
                continue

            starts_new_word = pending_space or current is None
            punctuation_only = bool(_PUNCTUATION_ONLY_RE.fullmatch(part))
            if punctuation_only and current is not None and not pending_space:
                current["word"] += part
                current["end"] = max(current["end"], end)
                continue

            if starts_new_word:
                _finish_word(words, current)
                prefix = " " if pending_space and words else ""
                current = {
                    "word": f"{prefix}{part}",
                    "start": start,
                    "end": end,
                    "probability": None,
                    "timestamp_source": "PARAKEET_NATIVE_TOKEN_DURATION",
                    "timestamp_authority": "DIRECTIONAL_NOT_NP_SBV2",
                }
            else:
                current["word"] += part
                current["end"] = max(current["end"], end)
            pending_space = False

    _finish_word(words, current)
    return words


def words_to_directional_segments(
    words: Sequence[Mapping],
    fallback_text: str,
    audio_duration_seconds: float,
) -> list[dict]:
    """Create sentence-ish compatibility segments without inventing timing."""
    if not words:
        if not fallback_text:
            return []
        return [
            {
                "id": 0,
                "seek": 0,
                "start": 0.0,
                "end": float(audio_duration_seconds),
                "text": fallback_text,
                "tokens": [],
                "temperature": None,
                "avg_logprob": None,
                "compression_ratio": None,
                "no_speech_prob": None,
                "timestamp_authority": "AUDIO_EXTENT_FALLBACK_NOT_WORD_AUTHORITY",
            }
        ]

    segments = []
    current_words = []
    segment_start = None
    for word in words:
        if segment_start is None:
            segment_start = float(word["start"])
        current_words.append(word)
        segment_end = float(word["end"])
        text = "".join(str(item["word"]) for item in current_words).strip()
        terminal = bool(_TERMINAL_PUNCTUATION_RE.search(text))
        duration_limit = (
            segment_end - segment_start >= _MAX_DIRECTIONAL_SEGMENT_SECONDS
        )
        if not terminal and not duration_limit:
            continue
        segments.append(
            {
                "id": len(segments),
                "seek": 0,
                "start": segment_start,
                "end": segment_end,
                "text": text,
                "tokens": [],
                "temperature": None,
                "avg_logprob": None,
                "compression_ratio": None,
                "no_speech_prob": None,
                "timestamp_authority": "PARAKEET_NATIVE_TOKEN_DURATION_DIRECTIONAL",
            }
        )
        current_words = []
        segment_start = None

    if current_words:
        segments.append(
            {
                "id": len(segments),
                "seek": 0,
                "start": float(current_words[0]["start"]),
                "end": float(current_words[-1]["end"]),
                "text": "".join(
                    str(item["word"]) for item in current_words
                ).strip(),
                "tokens": [],
                "temperature": None,
                "avg_logprob": None,
                "compression_ratio": None,
                "no_speech_prob": None,
                "timestamp_authority": "PARAKEET_NATIVE_TOKEN_DURATION_DIRECTIONAL",
            }
        )
    return segments


class ParakeetTranscriber:
    """Lazy, serialized Parakeet loader with single-flight GPU inference."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self._setup_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def setup(self):
        """Load the pinned model from the image-local Hugging Face cache."""
        if self.model is not None:
            return
        with self._setup_lock:
            if self.model is not None:
                return
            with serialized_model_load("parakeet-tdt-transcriber"):
                if self.model is not None:
                    return
                import torch
                import transformers
                from transformers import AutoModelForTDT, AutoProcessor

                if transformers.__version__ != PARAKEET_TRANSFORMERS_VERSION:
                    raise RuntimeError(
                        "Parakeet runtime requires transformers "
                        f"{PARAKEET_TRANSFORMERS_VERSION}, found "
                        f"{transformers.__version__}"
                    )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = (
                    torch.float16
                    if device == "cuda"
                    else torch.float32
                )
                pretrained_kwargs = {
                    "revision": PARAKEET_MODEL_REVISION,
                    "local_files_only": True,
                    **hf_from_pretrained_kwargs(),
                }
                print(
                    "[Parakeet] Loading pinned model "
                    f"{PARAKEET_MODEL_ID}@{PARAKEET_MODEL_REVISION} on {device}",
                    flush=True,
                )
                processor = AutoProcessor.from_pretrained(
                    PARAKEET_MODEL_ID,
                    **pretrained_kwargs,
                )
                model = AutoModelForTDT.from_pretrained(
                    PARAKEET_MODEL_ID,
                    # The exact five-language worker control found no
                    # normalized quality repair from model-declared float32,
                    # while warm execution was slower. Preserve that control
                    # in receipts and use the evidenced CUDA fp16 route.
                    dtype=dtype,
                    **pretrained_kwargs,
                )
                model = model.to(device).eval()
                self.processor = processor
                self.model = model
                self.device = device
                print("[Parakeet] Model loaded", flush=True)

    def transcribe(
        self,
        audio_path: str,
        *,
        language_hint: Optional[str] = None,
        include_word_timestamps: bool = True,
    ) -> dict:
        """Transcribe one source and return the worker compatibility shape."""
        normalized_hint = normalize_language_code(language_hint)
        if normalized_hint is not None and normalized_hint not in PARAKEET_SUPPORTED_LANGUAGES:
            raise ValueError(
                "Parakeet does not support requested language "
                f"{normalized_hint!r}; route this source to Whisper"
            )

        with self._inference_lock:
            self.setup()
            import librosa
            import torch

            started = time.perf_counter()
            waveform, _ = librosa.load(
                audio_path,
                sr=PARAKEET_SAMPLE_RATE,
                mono=True,
            )
            audio_duration_seconds = len(waveform) / PARAKEET_SAMPLE_RATE
            inputs = self.processor(
                [waveform],
                sampling_rate=PARAKEET_SAMPLE_RATE,
                return_tensors="pt",
            )
            inputs = inputs.to(device=self.device, dtype=self.model.dtype)
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    return_dict_in_generate=True,
                )
            if getattr(output, "durations", None) is None:
                raise RuntimeError(
                    "Parakeet generation returned no token durations"
                )
            decoded, timestamp_batches = self.processor.decode(
                output.sequences,
                durations=output.durations,
                skip_special_tokens=True,
            )
            raw_text = decoded[0] if isinstance(decoded, list) else decoded
            transcript = strip_control_tokens(str(raw_text))
            token_timestamps = (
                timestamp_batches[0]
                if timestamp_batches
                and isinstance(timestamp_batches[0], list)
                else timestamp_batches
            )
            words = token_timestamps_to_words(token_timestamps or [])
            segments = words_to_directional_segments(
                words,
                transcript,
                audio_duration_seconds,
            )

            sequence_ids = output.sequences[0].detach().cpu().tolist()
            model_language, observed_languages = extract_model_language(
                sequence_ids,
                self.processor.tokenizer,
            )
            if model_language is not None:
                detected_language = model_language
                language_authority = "PARAKEET_GENERATED_CONTROL_TOKEN"
            elif normalized_hint is not None:
                detected_language = normalized_hint
                language_authority = "CALLER_HINT_MODEL_DETECTION_MISSING"
            else:
                detected_language = None
                language_authority = "MISSING"
            if (
                model_language is not None
                and normalized_hint is not None
                and model_language != normalized_hint
            ):
                language_status = "MODEL_HINT_MISMATCH"
            elif model_language is not None:
                language_status = "MODEL_DETECTED"
            elif normalized_hint is not None:
                language_status = "HINT_ONLY"
            else:
                language_status = "UNKNOWN"

            results = {
                "segments": segments,
                "detected_language": detected_language,
                "transcription": transcript,
                "translation": None,
                "device": self.device,
                "model": PARAKEET_MODEL_ID,
                "asr_backend": "parakeet",
                "word_timestamps_aligned": False,
                "alignment": {
                    "status": "PARAKEET_NATIVE_DIRECTIONAL",
                    "per_word_authority": False,
                    "np_sbv2_authority": False,
                    "natural_landing_authority": False,
                    "transcript_geometry_mutated": False,
                },
                "asr_backend_evidence": {
                    "schema_version": PARAKEET_SCHEMA_VERSION,
                    "backend": "parakeet",
                    "model_id": PARAKEET_MODEL_ID,
                    "model_revision": PARAKEET_MODEL_REVISION,
                    "framework": "transformers",
                    "framework_version": PARAKEET_TRANSFORMERS_VERSION,
                    "model_dtype": str(self.model.dtype).removeprefix(
                        "torch."
                    ),
                    "model_dtype_selection": (
                        PARAKEET_MODEL_DTYPE_SELECTION
                    ),
                    "supported_languages": sorted(
                        PARAKEET_SUPPORTED_LANGUAGES
                    ),
                    "language_hint": normalized_hint,
                    "model_detected_language": model_language,
                    "observed_model_languages": observed_languages,
                    "language_authority": language_authority,
                    "language_status": language_status,
                    "timestamp_source": "PARAKEET_NATIVE_TOKEN_DURATION",
                    "timestamp_authority": "DIRECTIONAL_NOT_NP_SBV2",
                    "word_probability_authority": False,
                    "audio_duration_seconds": audio_duration_seconds,
                    "inference_ms": int(
                        (time.perf_counter() - started) * 1000
                    ),
                    "production_routing_changed": False,
                },
            }
            if include_word_timestamps:
                results["word_timestamps"] = words
            return results
