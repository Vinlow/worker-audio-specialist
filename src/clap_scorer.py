"""CLAP audio-text similarity scoring with a fail-soft wire contract.

The scorer keeps the model resident, embeds non-overlapping one-second audio
windows, and returns the legacy affine-cosine scores used by Studio.  Runtime
loads are deliberately offline and revision-pinned: the worker image is
responsible for pre-populating the Hugging Face cache.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from hf_auth import hf_from_pretrained_kwargs
from model_manifest import CLAP_MODEL_ID, CLAP_MODEL_REVISION
from model_load_lock import serialized_model_load


CLAP_RESPONSE_SCHEMA = "w2l-clap-scores-v2"

WINDOW_SIZE = 1.0
SAMPLE_RATE = 48000

MAX_QUERY_COUNT = 256
MAX_QUERY_NAME_CHARS = 256
MAX_QUERY_TEXT_CHARS = 2048
MAX_TOTAL_QUERY_TEXT_CHARS = MAX_QUERY_COUNT * MAX_QUERY_TEXT_CHARS

DEFAULT_MICROBATCH_SIZE = 64
MIN_MICROBATCH_SIZE = 1
MAX_MICROBATCH_SIZE = 256
# Kept as a compatibility alias for code/tests that imported the old constant.
MAX_BATCH_SIZE = DEFAULT_MICROBATCH_SIZE
MICROBATCH_ENV = "CLAP_MICROBATCH_SIZE"
LEGACY_MICROBATCH_ENV = "CLAP_BATCH_SIZE"

DEFAULT_TEXT_MICROBATCH_SIZE = 32
MIN_TEXT_MICROBATCH_SIZE = 1
MAX_TEXT_MICROBATCH_SIZE = 64
TEXT_MICROBATCH_ENV = "CLAP_TEXT_MICROBATCH_SIZE"
DEFAULT_TEXT_MAX_LENGTH = 77
MAX_REASONABLE_TEXT_MODEL_LENGTH = 4096

TEXT_EMBEDDING_CACHE_SIZE = 16


class ClapScoringError(RuntimeError):
    """Expected scoring failure carrying stable API error metadata."""

    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class _CudaOOMRetry(RuntimeError):
    """Internal OOM signal raised only after batch tensors are released."""


class ClapScorer:
    """Lazy CLAP scorer with bounded caches and adaptive GPU microbatches."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.lock = threading.Lock()
        self._audio_decoder_warmed = False
        self._text_embedding_cache: OrderedDict[Tuple[str, ...], Any] = (
            OrderedDict()
        )

    @staticmethod
    def _new_timing() -> Dict[str, float]:
        return {
            "total_seconds": 0.0,
            "model_load_seconds": 0.0,
            "audio_load_seconds": 0.0,
            "text_embedding_seconds": 0.0,
            "audio_preprocessing_seconds": 0.0,
            "audio_embedding_seconds": 0.0,
        }

    def _preprocessing_diagnostics(self) -> Dict[str, Any]:
        return {
            "sample_rate_hz": SAMPLE_RATE,
            "window_seconds": WINDOW_SIZE,
            "window_hop_seconds": WINDOW_SIZE,
            "final_window_padding": "zero_to_one_second",
            "audio_processor_padding_argument": "repeatpad",
            "checkpoint_padding_strategy": "repeatpad",
            "text_padding": "longest_per_microbatch",
            "text_truncation": True,
            "text_max_length_tokens": self._text_max_length(),
        }

    @staticmethod
    def _score_semantics() -> Dict[str, Any]:
        return {
            "source": "cosine_similarity",
            "transform": "clip((cosine + 1) / 2, 0, 1)",
            "range": [0.0, 1.0],
            "rounding_decimals": 4,
            "calibration": "affine_cosine_not_probability",
        }

    @staticmethod
    def _configured_microbatch_size() -> int:
        raw = os.environ.get(MICROBATCH_ENV)
        if raw is None:
            raw = os.environ.get(LEGACY_MICROBATCH_ENV)
        if raw is None:
            return max(
                MIN_MICROBATCH_SIZE,
                min(MAX_MICROBATCH_SIZE, int(MAX_BATCH_SIZE)),
            )
        try:
            requested = int(raw)
        except (TypeError, ValueError):
            requested = DEFAULT_MICROBATCH_SIZE
        return max(MIN_MICROBATCH_SIZE, min(MAX_MICROBATCH_SIZE, requested))

    @staticmethod
    def _configured_text_microbatch_size() -> int:
        raw = os.environ.get(TEXT_MICROBATCH_ENV)
        if raw is None:
            return DEFAULT_TEXT_MICROBATCH_SIZE
        try:
            requested = int(raw)
        except (TypeError, ValueError):
            requested = DEFAULT_TEXT_MICROBATCH_SIZE
        return max(
            MIN_TEXT_MICROBATCH_SIZE,
            min(MAX_TEXT_MICROBATCH_SIZE, requested),
        )

    def _text_max_length(self) -> int:
        candidates = []
        tokenizer = getattr(self.processor, "tokenizer", None)
        tokenizer_limit = getattr(tokenizer, "model_max_length", None)
        text_config = getattr(
            getattr(self.model, "config", None),
            "text_config",
            None,
        )
        model_limit = getattr(text_config, "max_position_embeddings", None)
        for candidate in (tokenizer_limit, model_limit):
            if (
                isinstance(candidate, int)
                and 0 < candidate <= MAX_REASONABLE_TEXT_MODEL_LENGTH
            ):
                candidates.append(candidate)
        return min(candidates) if candidates else DEFAULT_TEXT_MAX_LENGTH

    @staticmethod
    def _validate_queries(queries: Any) -> Tuple[list[str], list[str]]:
        if not isinstance(queries, dict):
            raise ClapScoringError(
                "INVALID_QUERIES_TYPE",
                "queries must be a dictionary of name to text",
            )
        if not queries:
            raise ClapScoringError(
                "EMPTY_QUERIES",
                "at least one CLAP query is required",
            )
        if len(queries) > MAX_QUERY_COUNT:
            raise ClapScoringError(
                "TOO_MANY_QUERIES",
                f"query count exceeds the maximum of {MAX_QUERY_COUNT}",
            )

        names: list[str] = []
        texts: list[str] = []
        total_text_chars = 0
        for name, text in queries.items():
            if not isinstance(name, str) or not name.strip():
                raise ClapScoringError(
                    "INVALID_QUERY_NAME",
                    "query names must be non-empty strings",
                )
            if len(name) > MAX_QUERY_NAME_CHARS:
                raise ClapScoringError(
                    "QUERY_NAME_TOO_LONG",
                    f"query names may contain at most {MAX_QUERY_NAME_CHARS} characters",
                )
            if not isinstance(text, str) or not text.strip():
                raise ClapScoringError(
                    "INVALID_QUERY_TEXT",
                    f"query {name!r} must have non-empty text",
                )
            normalized_text = text.strip()
            if len(normalized_text) > MAX_QUERY_TEXT_CHARS:
                raise ClapScoringError(
                    "QUERY_TEXT_TOO_LONG",
                    f"query text may contain at most {MAX_QUERY_TEXT_CHARS} characters",
                )
            total_text_chars += len(normalized_text)
            if total_text_chars > MAX_TOTAL_QUERY_TEXT_CHARS:
                raise ClapScoringError(
                    "QUERY_TEXT_BUDGET_EXCEEDED",
                    "combined query text exceeds the request limit",
                )
            names.append(name)
            texts.append(normalized_text)
        return names, texts

    def _ensure_loaded(self) -> None:
        """Load the exact image-baked CLAP snapshot once."""
        if self.model is not None:
            return

        with serialized_model_load("clap-scorer"):
            if self.model is not None:
                return

            import torch
            from transformers import ClapModel, ClapProcessor

            print(
                "[ClapScorer] Loading pinned CLAP model: "
                f"{CLAP_MODEL_ID}@{CLAP_MODEL_REVISION}"
            )
            pretrained_kwargs = hf_from_pretrained_kwargs()
            pretrained_kwargs.update(
                {
                    "revision": CLAP_MODEL_REVISION,
                    "local_files_only": True,
                }
            )
            # Keep objects local until construction and device transfer both
            # succeed. A cold-load failure must never publish a partial scorer.
            processor = ClapProcessor.from_pretrained(
                CLAP_MODEL_ID,
                **pretrained_kwargs,
            )
            model = ClapModel.from_pretrained(
                CLAP_MODEL_ID,
                **pretrained_kwargs,
            )
            model.eval()

            if torch.cuda.is_available():
                model = model.to("cuda")
                device = "cuda"
                print(
                    "[ClapScorer] Model loaded on GPU "
                    f"({torch.cuda.get_device_name(0)})"
                )
            else:
                device = "cpu"
                print("[ClapScorer] Model loaded on CPU (GPU not available)")

            # ``model`` is the readiness flag used by the lock-free fast path.
            # Publish it last so no concurrent caller can observe a model with
            # a missing processor/device or stale embedding cache.
            self.processor = processor
            self.device = device
            self._text_embedding_cache.clear()
            self.model = model

    def warmup(self) -> None:
        """Deserialize CLAP and pay librosa's lazy decoder cost before dispatch."""
        with self.lock:
            self._ensure_loaded()
            if self._audio_decoder_warmed:
                return

            # librosa lazily imports its decoder/resampler stack on the first
            # call.  On the exact image this costs roughly seven seconds, even
            # though subsequent decodes take milliseconds.  Exercise the same
            # in-memory WAV/resampling path during the asynchronous worker
            # warmup so the first paid job does not absorb that cold penalty.
            import io
            import wave

            import librosa

            started = time.perf_counter()
            wav_bytes = io.BytesIO()
            with wave.open(wav_bytes, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16_000)
                wav_file.writeframes(b"\x00\x00" * 160)
            wav_bytes.seek(0)
            librosa.load(wav_bytes, sr=SAMPLE_RATE, mono=True)
            self._audio_decoder_warmed = True
            print(
                "[ClapScorer] Audio decoder warmup complete "
                f"({time.perf_counter() - started:.3f}s)"
            )

    @staticmethod
    def _unwrap_features(output: Any) -> Any:
        pooler_output = getattr(output, "pooler_output", None)
        if pooler_output is not None:
            return pooler_output
        if isinstance(output, (tuple, list)):
            # BaseModelOutputWithPooling tuple order is
            # (last_hidden_state, pooler_output, ...). Older feature helpers
            # sometimes return a one-element tuple containing embeddings.
            candidates = output[1:2] + output[0:1]
            for candidate in candidates:
                if candidate is not None and hasattr(candidate, "norm"):
                    return candidate
            raise ClapScoringError(
                "INVALID_MODEL_OUTPUT",
                "CLAP feature output tuple has no embedding tensor",
                retryable=True,
            )
        return output

    @staticmethod
    def _move_inputs(inputs: Mapping[str, Any], device: str) -> Dict[str, Any]:
        return {name: tensor.to(device) for name, tensor in inputs.items()}

    @staticmethod
    def _is_cuda_oom(error: BaseException, torch_module: Any) -> bool:
        cuda_namespace = getattr(torch_module, "cuda", None)
        oom_type = getattr(cuda_namespace, "OutOfMemoryError", None)
        if isinstance(oom_type, type) and isinstance(error, oom_type):
            return True
        message = str(error).lower()
        return any(
            signature in message
            for signature in (
                "cuda out of memory",
                "cuda error: out of memory",
                "cublas_status_alloc_failed",
                "cudnn_status_alloc_failed",
                "cudaerror_memoryallocation",
                "failed to allocate",
                "not enough memory",
            )
        )

    @staticmethod
    def _empty_cuda_cache(torch_module: Any) -> None:
        cuda_namespace = getattr(torch_module, "cuda", None)
        empty_cache = getattr(cuda_namespace, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()

    def _embed_text_batch(
        self,
        torch_module: Any,
        query_texts: Sequence[str],
        text_max_length: int,
    ) -> Any:
        text_inputs = None
        text_output = None
        text_embeddings = None
        try:
            text_inputs = self.processor(
                text=list(query_texts),
                text_kwargs={
                    # Pad only to the longest prompt in this bounded
                    # microbatch. The reviewed tokenizer cap is 512 tokens;
                    # padding every ordinary catalog prompt to that full cap
                    # wastes quadratic attention work and VRAM.
                    "padding": "longest",
                    "truncation": True,
                    "max_length": text_max_length,
                    "return_tensors": "pt",
                },
            )
            text_inputs = self._move_inputs(text_inputs, self.device)
            with torch_module.inference_mode():
                text_output = self.model.get_text_features(
                    **text_inputs,
                    return_dict=True,
                )
                text_embeddings = self._unwrap_features(text_output)
                return text_embeddings / text_embeddings.norm(
                    dim=-1,
                    keepdim=True,
                )
        except Exception as error:
            if not self._is_cuda_oom(error, torch_module):
                raise
            text_embeddings = None
            text_output = None
            text_inputs = None
            raise _CudaOOMRetry(str(error)) from None

    def _get_text_embeddings(
        self,
        torch_module: Any,
        query_texts: Sequence[str],
        diagnostics: MutableMapping[str, Any],
    ) -> Any:
        cache_key = tuple(query_texts)
        cached = self._text_embedding_cache.get(cache_key)
        if cached is not None:
            self._text_embedding_cache.move_to_end(cache_key)
            diagnostics["text_cache"]["hit"] = True
            diagnostics["text_cache"]["entries"] = len(
                self._text_embedding_cache
            )
            return cached

        text_max_length = self._text_max_length()
        microbatch_size = int(
            diagnostics["batching"]["configured_text_microbatch_size"]
        )
        embedding_batches = []
        next_query = 0
        while next_query < len(query_texts):
            batch_size = min(microbatch_size, len(query_texts) - next_query)
            batch_texts = query_texts[next_query : next_query + batch_size]
            try:
                embeddings = self._embed_text_batch(
                    torch_module,
                    batch_texts,
                    text_max_length,
                )
            except Exception as error:
                if not isinstance(error, _CudaOOMRetry) and not self._is_cuda_oom(
                    error,
                    torch_module,
                ):
                    raise
                self._empty_cuda_cache(torch_module)
                if batch_size <= MIN_TEXT_MICROBATCH_SIZE:
                    raise ClapScoringError(
                        "CUDA_OUT_OF_MEMORY",
                        "CLAP text inference exhausted CUDA memory at batch size 1",
                        retryable=True,
                    ) from error
                microbatch_size = max(
                    MIN_TEXT_MICROBATCH_SIZE,
                    batch_size // 2,
                )
                diagnostics["batching"][
                    "effective_text_microbatch_size"
                ] = microbatch_size
                diagnostics["batching"]["text_oom_retries"] += 1
                continue
            embedding_batches.append(embeddings)
            next_query += batch_size

        if len(embedding_batches) == 1:
            text_embeddings = embedding_batches[0]
        else:
            text_embeddings = torch_module.cat(
                embedding_batches,
                dim=0,
            )

        self._text_embedding_cache[cache_key] = text_embeddings
        self._text_embedding_cache.move_to_end(cache_key)
        while len(self._text_embedding_cache) > TEXT_EMBEDDING_CACHE_SIZE:
            self._text_embedding_cache.popitem(last=False)
        diagnostics["text_cache"]["entries"] = len(
            self._text_embedding_cache
        )
        return text_embeddings

    @staticmethod
    def _validate_audio(waveform: Any, sample_rate: Any) -> np.ndarray:
        try:
            normalized = np.asarray(waveform)
        except Exception as error:
            raise ClapScoringError(
                "INVALID_AUDIO",
                "decoded audio could not be represented as a numeric array",
            ) from error
        if normalized.ndim != 1 or normalized.size == 0:
            raise ClapScoringError(
                "EMPTY_AUDIO",
                "decoded audio must contain at least one mono sample",
            )
        if not np.issubdtype(normalized.dtype, np.number):
            raise ClapScoringError(
                "INVALID_AUDIO",
                "decoded audio samples must be numeric",
            )
        if not np.isfinite(normalized).all():
            raise ClapScoringError(
                "NONFINITE_AUDIO",
                "decoded audio contains NaN or infinite samples",
            )
        try:
            finite_sample_rate = float(sample_rate)
        except (TypeError, ValueError) as error:
            raise ClapScoringError(
                "INVALID_SAMPLE_RATE",
                "decoded audio sample rate is invalid",
            ) from error
        if not np.isfinite(finite_sample_rate) or finite_sample_rate <= 0:
            raise ClapScoringError(
                "INVALID_SAMPLE_RATE",
                "decoded audio sample rate must be finite and positive",
            )
        if finite_sample_rate != SAMPLE_RATE:
            raise ClapScoringError(
                "UNEXPECTED_SAMPLE_RATE",
                f"decoded audio must be resampled to {SAMPLE_RATE} Hz",
            )
        return normalized

    def _embed_audio_batch(
        self,
        torch_module: Any,
        batch_chunks: Sequence[np.ndarray],
        text_embeddings: Any,
    ) -> np.ndarray:
        # Never use boolean ``padding=True`` here. Transformers 5.9 routes the
        # explicit CLAP strategy through structured audio kwargs.
        audio_inputs = None
        audio_output = None
        audio_embeddings = None
        try:
            audio_inputs = self.processor(
                audio=list(batch_chunks),
                audio_kwargs={
                    "sampling_rate": SAMPLE_RATE,
                    "padding": "repeatpad",
                    "return_tensors": "pt",
                },
            )
            audio_inputs = self._move_inputs(audio_inputs, self.device)
            with torch_module.inference_mode():
                audio_output = self.model.get_audio_features(
                    **audio_inputs,
                    return_dict=True,
                )
                audio_embeddings = self._unwrap_features(audio_output)
                audio_embeddings = audio_embeddings / audio_embeddings.norm(
                    dim=-1,
                    keepdim=True,
                )
                return (audio_embeddings @ text_embeddings.T).cpu().numpy()
        except Exception as error:
            if not self._is_cuda_oom(error, torch_module):
                raise
            # A caught PyTorch exception retains its traceback frame. Clear
            # tensor locals before signaling the caller so empty_cache() can
            # actually release the failed attempt's allocator blocks.
            audio_embeddings = None
            audio_output = None
            audio_inputs = None
            raise _CudaOOMRetry(str(error)) from None

    def score(self, wav_path: Any, queries: Any) -> Dict[str, Any]:
        """Score audio, returning a stable success or failure dictionary.

        No ordinary model, input, preprocessing, or inference exception escapes
        this method. Legacy success consumers retain ``scores``, ``duration``,
        ``model``, ``device``, and ``windowSize`` unchanged.
        """
        started = time.perf_counter()
        timing = self._new_timing()
        microbatch_size = DEFAULT_MICROBATCH_SIZE
        text_microbatch_size = DEFAULT_TEXT_MICROBATCH_SIZE
        diagnostics: Dict[str, Any] = {
            "duration": None,
            "window_count": None,
            "final_window_valid_seconds": None,
            "batching": {
                "configured_microbatch_size": microbatch_size,
                "effective_microbatch_size": microbatch_size,
                "oom_retries": 0,
                "configured_text_microbatch_size": text_microbatch_size,
                "effective_text_microbatch_size": text_microbatch_size,
                "text_oom_retries": 0,
            },
            "text_cache": {
                "hit": False,
                "entries": len(self._text_embedding_cache),
                "capacity": TEXT_EMBEDDING_CACHE_SIZE,
            },
        }

        try:
            microbatch_size = self._configured_microbatch_size()
            diagnostics["batching"][
                "configured_microbatch_size"
            ] = microbatch_size
            diagnostics["batching"][
                "effective_microbatch_size"
            ] = microbatch_size
            text_microbatch_size = self._configured_text_microbatch_size()
            diagnostics["batching"][
                "configured_text_microbatch_size"
            ] = text_microbatch_size
            diagnostics["batching"][
                "effective_text_microbatch_size"
            ] = text_microbatch_size
            query_names, query_texts = self._validate_queries(queries)
            with self.lock:
                load_started = time.perf_counter()
                try:
                    self._ensure_loaded()
                except Exception as error:
                    raise ClapScoringError(
                        "MODEL_LOAD_FAILED",
                        str(error) or type(error).__name__,
                        retryable=True,
                    ) from error
                finally:
                    timing["model_load_seconds"] += (
                        time.perf_counter() - load_started
                    )

                scores = self._score_batched(
                    wav_path,
                    query_names,
                    query_texts,
                    timing=timing,
                    diagnostics=diagnostics,
                )
            timing["total_seconds"] = time.perf_counter() - started
            return self._response(
                status="COMPLETED",
                scores=scores,
                duration=diagnostics["duration"],
                timing=timing,
                diagnostics=diagnostics,
            )
        except Exception as error:
            timing["total_seconds"] = time.perf_counter() - started
            if isinstance(error, ClapScoringError):
                code = error.code
                retryable = error.retryable
            else:
                code = "CLAP_SCORING_FAILED"
                retryable = True
            try:
                error_text = str(error).replace("\n", " ").strip()
            except Exception:
                error_text = type(error).__name__
            if not error_text:
                error_text = type(error).__name__
            error_text = error_text[:300]
            try:
                print(f"[ClapScorer] Scoring failed [{code}]: {error_text}")
            except Exception:
                pass
            try:
                return self._response(
                    status="FAILED",
                    scores={},
                    duration=diagnostics["duration"],
                    timing=timing,
                    diagnostics=diagnostics,
                    error_code=code,
                    error=error_text,
                    retryable=retryable,
                )
            except Exception as response_error:
                # The response builder only operates on local primitives, but
                # keep the public fail-soft promise even if it regresses.
                return {
                    "schema_version": CLAP_RESPONSE_SCHEMA,
                    "status": "FAILED",
                    "error_code": "RESPONSE_BUILD_FAILED",
                    "error": type(response_error).__name__,
                    "retryable": True,
                    "scores": {},
                    "duration": diagnostics.get("duration"),
                    "window_count": diagnostics.get("window_count"),
                    "final_window_valid_seconds": diagnostics.get(
                        "final_window_valid_seconds"
                    ),
                    "model": CLAP_MODEL_ID,
                    "device": self.device or "unknown",
                    "windowSize": WINDOW_SIZE,
                    "model_revision": CLAP_MODEL_REVISION,
                    "timing": dict(timing),
                    "preprocessing": {
                        "sample_rate_hz": SAMPLE_RATE,
                        "window_seconds": WINDOW_SIZE,
                        "window_hop_seconds": WINDOW_SIZE,
                        "final_window_padding": "zero_to_one_second",
                        "audio_processor_padding_argument": "repeatpad",
                        "checkpoint_padding_strategy": "repeatpad",
                        "text_padding": "longest_per_microbatch",
                        "text_truncation": True,
                        "text_max_length_tokens": DEFAULT_TEXT_MAX_LENGTH,
                    },
                    "score_semantics": {
                        "source": "cosine_similarity",
                        "transform": "clip((cosine + 1) / 2, 0, 1)",
                        "range": [0.0, 1.0],
                        "rounding_decimals": 4,
                        "calibration": "affine_cosine_not_probability",
                    },
                    "batching": dict(diagnostics["batching"]),
                    "text_cache": dict(diagnostics["text_cache"]),
                }

    def _score_batched(
        self,
        wav_path: Any,
        query_names: Sequence[str],
        query_texts: Sequence[str] | None = None,
        *,
        timing: MutableMapping[str, float] | None = None,
        diagnostics: MutableMapping[str, Any] | None = None,
    ) -> Dict[str, list[float]]:
        """Embed all one-second windows using adaptive microbatches."""
        import librosa
        import torch

        # Keep the former private two-argument shape usable for local callers.
        if query_texts is None:
            if not isinstance(query_names, dict):
                raise ClapScoringError(
                    "INVALID_QUERIES_TYPE",
                    "queries must be a dictionary of name to text",
                )
            query_names, query_texts = self._validate_queries(query_names)

        timing = timing if timing is not None else self._new_timing()
        if diagnostics is None:
            configured = self._configured_microbatch_size()
            configured_text = self._configured_text_microbatch_size()
            diagnostics = {
                "duration": None,
                "window_count": None,
                "final_window_valid_seconds": None,
                "batching": {
                    "configured_microbatch_size": configured,
                    "effective_microbatch_size": configured,
                    "oom_retries": 0,
                    "configured_text_microbatch_size": configured_text,
                    "effective_text_microbatch_size": configured_text,
                    "text_oom_retries": 0,
                },
                "text_cache": {
                    "hit": False,
                    "entries": len(self._text_embedding_cache),
                    "capacity": TEXT_EMBEDDING_CACHE_SIZE,
                },
            }

        audio_load_started = time.perf_counter()
        try:
            waveform, sample_rate = librosa.load(
                wav_path,
                sr=SAMPLE_RATE,
                mono=True,
            )
        except Exception as error:
            raise ClapScoringError(
                "AUDIO_LOAD_FAILED",
                str(error) or type(error).__name__,
            ) from error
        finally:
            timing["audio_load_seconds"] += (
                time.perf_counter() - audio_load_started
            )

        waveform = self._validate_audio(waveform, sample_rate)
        duration = len(waveform) / float(SAMPLE_RATE)
        if not np.isfinite(duration) or duration <= 0:
            raise ClapScoringError(
                "INVALID_AUDIO_DURATION",
                "decoded audio duration must be finite and positive",
            )
        # Six decimals preserves the 48 kHz one-sample quantum (20.833 us), so
        # ceil(duration / windowSize) always matches the emitted score count.
        diagnostics["duration"] = round(duration, 6)
        window_samples = int(WINDOW_SIZE * SAMPLE_RATE)
        window_count = int(np.ceil(len(waveform) / window_samples))
        final_window_samples = len(waveform) % window_samples or window_samples
        diagnostics["window_count"] = window_count
        diagnostics["final_window_valid_seconds"] = round(
            final_window_samples / float(SAMPLE_RATE),
            6,
        )

        text_started = time.perf_counter()
        try:
            text_embeddings = self._get_text_embeddings(
                torch,
                query_texts,
                diagnostics,
            )
        finally:
            timing["text_embedding_seconds"] += (
                time.perf_counter() - text_started
            )

        preprocessing_started = time.perf_counter()
        chunks = []
        for index in range(window_count):
            start = index * window_samples
            end = min(start + window_samples, len(waveform))
            chunk = waveform[start:end]
            if len(chunk) < window_samples:
                chunk = np.pad(chunk, (0, window_samples - len(chunk)))
            chunks.append(chunk)
        timing["audio_preprocessing_seconds"] += (
            time.perf_counter() - preprocessing_started
        )

        all_similarities = []
        next_window = 0
        microbatch_size = int(
            diagnostics["batching"]["configured_microbatch_size"]
        )
        while next_window < len(chunks):
            batch_size = min(microbatch_size, len(chunks) - next_window)
            batch_chunks = chunks[next_window : next_window + batch_size]
            inference_started = time.perf_counter()
            try:
                similarities = self._embed_audio_batch(
                    torch,
                    batch_chunks,
                    text_embeddings,
                )
            except Exception as error:
                timing["audio_embedding_seconds"] += (
                    time.perf_counter() - inference_started
                )
                if not isinstance(error, _CudaOOMRetry) and not self._is_cuda_oom(
                    error,
                    torch,
                ):
                    raise
                self._empty_cuda_cache(torch)
                if batch_size <= MIN_MICROBATCH_SIZE:
                    raise ClapScoringError(
                        "CUDA_OUT_OF_MEMORY",
                        "CLAP inference exhausted CUDA memory at batch size 1",
                        retryable=True,
                    ) from error
                microbatch_size = max(MIN_MICROBATCH_SIZE, batch_size // 2)
                diagnostics["batching"][
                    "effective_microbatch_size"
                ] = microbatch_size
                diagnostics["batching"]["oom_retries"] += 1
                continue
            timing["audio_embedding_seconds"] += (
                time.perf_counter() - inference_started
            )
            all_similarities.append(np.asarray(similarities))
            next_window += batch_size

        all_scores = np.concatenate(all_similarities, axis=0)
        if (
            all_scores.ndim != 2
            or all_scores.shape != (len(chunks), len(query_names))
        ):
            raise ClapScoringError(
                "INVALID_MODEL_OUTPUT",
                "CLAP returned an unexpected similarity matrix shape",
                retryable=True,
            )
        if not np.isfinite(all_scores).all():
            raise ClapScoringError(
                "NONFINITE_MODEL_OUTPUT",
                "CLAP returned NaN or infinite similarities",
                retryable=True,
            )

        scores: Dict[str, list[float]] = {}
        for query_index, name in enumerate(query_names):
            raw_cosine = all_scores[:, query_index]
            relevance = np.clip((raw_cosine + 1.0) / 2.0, 0.0, 1.0)
            scores[name] = [round(float(value), 4) for value in relevance]
        return scores

    def _response(
        self,
        *,
        status: str,
        scores: Dict[str, list[float]],
        duration: float | None,
        timing: Mapping[str, float],
        diagnostics: Mapping[str, Any],
        error_code: str | None = None,
        error: str | None = None,
        retryable: bool = False,
    ) -> Dict[str, Any]:
        rounded_timing = {
            name: round(max(0.0, float(value)), 6)
            for name, value in timing.items()
        }
        return {
            "schema_version": CLAP_RESPONSE_SCHEMA,
            "status": status,
            "error_code": error_code,
            "error": error,
            "retryable": bool(retryable),
            # Backward-compatible success fields. They remain present on a
            # failure so downstream logging/serialization is also fail-soft.
            "scores": scores,
            "duration": duration,
            "window_count": diagnostics["window_count"],
            "final_window_valid_seconds": diagnostics[
                "final_window_valid_seconds"
            ],
            "model": CLAP_MODEL_ID,
            "device": self.device or "unknown",
            "windowSize": WINDOW_SIZE,
            "model_revision": CLAP_MODEL_REVISION,
            "timing": rounded_timing,
            "preprocessing": self._preprocessing_diagnostics(),
            "score_semantics": self._score_semantics(),
            "batching": dict(diagnostics["batching"]),
            "text_cache": dict(diagnostics["text_cache"]),
        }

    @staticmethod
    def is_available() -> bool:
        """Check if CLAP runtime dependencies are installed."""
        try:
            import librosa  # noqa: F401
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False
