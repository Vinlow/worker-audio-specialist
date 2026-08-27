import contextlib
import math
import os
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from clap_scorer import (
    CLAP_MODEL_ID,
    CLAP_MODEL_REVISION,
    CLAP_RESPONSE_SCHEMA,
    MAX_QUERY_COUNT,
    ClapScorer,
)


class _FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    def to(self, _device):
        return self

    def norm(self, dim=-1, keepdim=True):
        return _FakeTensor(
            np.linalg.norm(self.values, axis=dim, keepdims=keepdim)
        )

    def __truediv__(self, other):
        values = other.values if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self.values / values)

    def __matmul__(self, other):
        values = other.values if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self.values @ values)

    @property
    def T(self):
        return _FakeTensor(self.values.T)

    def cpu(self):
        return self

    def numpy(self):
        return self.values.copy()


class _FakeCuda:
    class OutOfMemoryError(RuntimeError):
        pass

    def __init__(self):
        self.empty_cache_calls = 0

    @staticmethod
    def is_available():
        return False

    @staticmethod
    def get_device_name(_index):
        return "fake-cuda"

    def empty_cache(self):
        self.empty_cache_calls += 1


class _FakeTorch:
    def __init__(self):
        self.cuda = _FakeCuda()

    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()

    @staticmethod
    def cat(tensors, dim=0):
        return _FakeTensor(
            np.concatenate([tensor.values for tensor in tensors], axis=dim)
        )


class _FakeProcessor:
    def __init__(self):
        self.calls = []
        self.tokenizer = SimpleNamespace(model_max_length=128)

    def __call__(self, **kwargs):
        if "text" in kwargs:
            texts = kwargs["text"]
            self.calls.append(
                {
                    "kind": "text",
                    "count": len(texts),
                    "keys": set(kwargs),
                    "text_kwargs": kwargs.get("text_kwargs"),
                }
            )
            vectors = []
            for index, _text in enumerate(texts):
                vectors.append([1.0, 0.0] if index % 2 == 0 else [0.0, 1.0])
            return {"input_ids": _FakeTensor(vectors)}

        chunks = kwargs["audio"]
        self.calls.append(
            {
                "kind": "audio",
                "count": len(chunks),
                "keys": set(kwargs),
                "audio_kwargs": kwargs.get("audio_kwargs"),
            }
        )
        vectors = []
        for chunk in chunks:
            marker = float(chunk[0])
            if marker > 0:
                vectors.append([1.0, 0.0])
            elif marker < 0:
                vectors.append([-1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return {"input_features": _FakeTensor(vectors)}


class _FakeModel:
    def __init__(
        self,
        *,
        oom_above=None,
        always_oom=False,
        text_oom_above=None,
        always_text_oom=False,
        output_mode="tensor",
        cuda=None,
    ):
        self.oom_above = oom_above
        self.always_oom = always_oom
        self.text_oom_above = text_oom_above
        self.always_text_oom = always_text_oom
        self.output_mode = output_mode
        self.cuda = cuda
        self.audio_batch_sizes = []
        self.text_batch_sizes = []
        self.return_dict_calls = []
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(max_position_embeddings=77)
        )

    def _output(self, tensor, kind):
        if self.output_mode == "pooler":
            return SimpleNamespace(pooler_output=tensor)
        if self.output_mode == "tuple":
            batch_size = tensor.values.shape[0]
            dummy = (
                [[0.0, 1.0]] * batch_size
                if kind == "text"
                else [[1.0, 0.0]] * batch_size
            )
            return (_FakeTensor(dummy), tensor)
        return tensor

    def get_text_features(self, input_ids, return_dict=False):
        batch_size = input_ids.values.shape[0]
        self.text_batch_sizes.append(batch_size)
        self.return_dict_calls.append(("text", return_dict))
        if self.always_text_oom or (
            self.text_oom_above is not None
            and batch_size > self.text_oom_above
        ):
            raise self.cuda.OutOfMemoryError(
                "CUDA out of memory in fake CLAP text tower"
            )
        return self._output(input_ids, "text")

    def get_audio_features(self, input_features, return_dict=False):
        batch_size = input_features.values.shape[0]
        self.audio_batch_sizes.append(batch_size)
        self.return_dict_calls.append(("audio", return_dict))
        if self.always_oom or (
            self.oom_above is not None and batch_size > self.oom_above
        ):
            raise self.cuda.OutOfMemoryError("CUDA out of memory in fake CLAP")
        return self._output(input_features, "audio")


class ClapScorerTest(unittest.TestCase):
    def setUp(self):
        self.torch = _FakeTorch()
        self.processor = _FakeProcessor()
        self.model = _FakeModel(cuda=self.torch.cuda)
        self.scorer = ClapScorer()
        self.scorer.processor = self.processor
        self.scorer.model = self.model
        self.scorer.device = "cuda"

    @staticmethod
    def _waveform(markers):
        return np.concatenate(
            [np.full(48000, marker, dtype=np.float32) for marker in markers]
        )

    @contextlib.contextmanager
    def _runtime(self, waveform):
        fake_librosa = SimpleNamespace(
            load=Mock(return_value=(waveform, 48000))
        )
        with patch.dict(
            sys.modules,
            {"torch": self.torch, "librosa": fake_librosa},
        ):
            yield fake_librosa

    def test_load_is_revision_pinned_and_runtime_offline(self):
        processor_calls = []
        model_calls = []

        class LoadingProcessor:
            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                processor_calls.append((model_id, kwargs))
                return cls()

        class LoadingModel:
            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                model_calls.append((model_id, kwargs))
                return cls()

            def eval(self):
                return self

        fake_transformers = SimpleNamespace(
            ClapModel=LoadingModel,
            ClapProcessor=LoadingProcessor,
        )
        scorer = ClapScorer()
        with patch.dict(
            sys.modules,
            {"torch": self.torch, "transformers": fake_transformers},
        ), patch(
            "clap_scorer.hf_from_pretrained_kwargs",
            return_value={"token": "not-logged"},
        ):
            scorer._ensure_loaded()

        expected = {
            "token": "not-logged",
            "revision": CLAP_MODEL_REVISION,
            "local_files_only": True,
        }
        self.assertEqual(processor_calls, [(CLAP_MODEL_ID, expected)])
        self.assertEqual(model_calls, [(CLAP_MODEL_ID, expected)])
        self.assertEqual(scorer.device, "cpu")

    def test_warmup_primes_lazy_audio_decoder_only_once(self):
        scorer = ClapScorer()
        scorer._ensure_loaded = Mock()
        fake_librosa = SimpleNamespace(
            load=Mock(return_value=(np.zeros(480, dtype=np.float32), 48000))
        )

        with patch.dict(sys.modules, {"librosa": fake_librosa}):
            scorer.warmup()
            scorer.warmup()

        self.assertEqual(scorer._ensure_loaded.call_count, 2)
        fake_librosa.load.assert_called_once()
        _audio_source, = fake_librosa.load.call_args.args
        self.assertEqual(fake_librosa.load.call_args.kwargs, {
            "sr": 48000,
            "mono": True,
        })
        self.assertTrue(scorer._audio_decoder_warmed)

    def test_model_readiness_is_published_only_after_all_state(self):
        model_visible = threading.Event()
        release_publication = threading.Event()
        snapshots = []
        errors = []

        class LoadingProcessor:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

        class LoadingModel:
            publication_marker = True

            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

            def eval(self):
                return self

        class PublishingScorer(ClapScorer):
            def __setattr__(self, name, value):
                super().__setattr__(name, value)
                if name == "model" and getattr(
                    value,
                    "publication_marker",
                    False,
                ):
                    model_visible.set()
                    release_publication.wait(timeout=2)

        scorer = PublishingScorer()
        scorer._text_embedding_cache[("stale",)] = object()
        fake_transformers = SimpleNamespace(
            ClapModel=LoadingModel,
            ClapProcessor=LoadingProcessor,
        )

        def capture(callable_):
            try:
                callable_()
            except Exception as error:
                errors.append(error)

        def observe_fast_path():
            scorer._ensure_loaded()
            snapshots.append(
                (
                    scorer.processor,
                    scorer.device,
                    len(scorer._text_embedding_cache),
                )
            )

        with patch.dict(
            sys.modules,
            {"torch": self.torch, "transformers": fake_transformers},
        ), patch(
            "clap_scorer.hf_from_pretrained_kwargs",
            return_value={},
        ):
            loader = threading.Thread(
                target=lambda: capture(scorer._ensure_loaded),
            )
            loader.start()
            try:
                self.assertTrue(model_visible.wait(timeout=2))
                observer = threading.Thread(
                    target=lambda: capture(observe_fast_path),
                )
                observer.start()
                observer.join(timeout=2)
                self.assertFalse(observer.is_alive())
            finally:
                release_publication.set()
                loader.join(timeout=2)

        self.assertFalse(loader.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(snapshots), 1)
        processor, device, cache_entries = snapshots[0]
        self.assertIsInstance(processor, LoadingProcessor)
        self.assertEqual(device, "cpu")
        self.assertEqual(cache_entries, 0)

    def test_success_contract_preserves_affine_cosine_scores(self):
        waveform = self._waveform([1.0, -1.0, 0.0])
        with self._runtime(waveform):
            result = self.scorer.score("audio.wav", {"event": "an event"})

        self.assertEqual(result["schema_version"], CLAP_RESPONSE_SCHEMA)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertIsNone(result["error_code"])
        self.assertIsNone(result["error"])
        self.assertFalse(result["retryable"])
        self.assertEqual(result["scores"], {"event": [1.0, 0.0, 0.5]})
        self.assertEqual(result["duration"], 3.0)
        self.assertEqual(result["model"], CLAP_MODEL_ID)
        self.assertEqual(result["device"], "cuda")
        self.assertEqual(result["windowSize"], 1.0)
        self.assertEqual(result["model_revision"], CLAP_MODEL_REVISION)
        self.assertEqual(
            result["score_semantics"]["calibration"],
            "affine_cosine_not_probability",
        )
        self.assertEqual(
            set(result["timing"]),
            {
                "total_seconds",
                "model_load_seconds",
                "audio_load_seconds",
                "text_embedding_seconds",
                "audio_preprocessing_seconds",
                "audio_embedding_seconds",
            },
        )

    def test_one_sample_tail_duration_matches_emitted_window_count(self):
        waveform = np.ones(48001, dtype=np.float32)
        with self._runtime(waveform):
            result = self.scorer.score("audio.wav", {"event": "an event"})

        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(len(result["scores"]["event"]), 2)
        self.assertEqual(result["window_count"], 2)
        self.assertEqual(
            math.ceil(result["duration"] / result["windowSize"]),
            len(result["scores"]["event"]),
        )
        self.assertEqual(result["duration"], 1.000021)
        self.assertEqual(result["final_window_valid_seconds"], 0.000021)

    def test_pooler_and_tuple_feature_outputs_are_both_safe(self):
        for output_mode in ("pooler", "tuple"):
            with self.subTest(output_mode=output_mode):
                processor = _FakeProcessor()
                model = _FakeModel(
                    output_mode=output_mode,
                    cuda=self.torch.cuda,
                )
                scorer = ClapScorer()
                scorer.processor = processor
                scorer.model = model
                scorer.device = "cuda"
                with self._runtime(self._waveform([1.0])):
                    result = scorer.score(
                        "audio.wav",
                        {"event": "an event"},
                    )

                self.assertEqual(result["status"], "COMPLETED")
                self.assertEqual(result["scores"], {"event": [1.0]})
                self.assertEqual(
                    model.return_dict_calls,
                    [("text", True), ("audio", True)],
                )

    def test_structured_processor_kwargs_pin_text_boundary_and_repeatpad(self):
        with self._runtime(self._waveform([1.0])):
            result = self.scorer.score("audio.wav", {"event": "an event"})

        text_call = next(call for call in self.processor.calls if call["kind"] == "text")
        audio_call = next(call for call in self.processor.calls if call["kind"] == "audio")
        self.assertNotIn("padding", text_call["keys"])
        self.assertEqual(
            text_call["text_kwargs"],
            {
                "padding": "longest",
                "truncation": True,
                "max_length": 77,
                "return_tensors": "pt",
            },
        )
        self.assertNotIn("padding", audio_call["keys"])
        self.assertNotIn("sampling_rate", audio_call["keys"])
        self.assertEqual(
            audio_call["audio_kwargs"],
            {
                "sampling_rate": 48000,
                "padding": "repeatpad",
                "return_tensors": "pt",
            },
        )
        self.assertEqual(
            result["preprocessing"]["audio_processor_padding_argument"],
            "repeatpad",
        )
        self.assertEqual(
            result["preprocessing"]["checkpoint_padding_strategy"],
            "repeatpad",
        )
        self.assertEqual(
            result["preprocessing"]["text_max_length_tokens"],
            77,
        )
        self.assertEqual(
            self.model.return_dict_calls,
            [("text", True), ("audio", True)],
        )

    def test_real_catalog_size_is_accepted_and_query_count_is_bounded(self):
        names, texts = self.scorer._validate_queries(
            {f"query-{index}": f"sound {index}" for index in range(159)}
        )
        self.assertEqual(len(names), 159)
        self.assertEqual(len(texts), 159)

        result = self.scorer.score(
            "unused.wav",
            {
                f"query-{index}": f"sound {index}"
                for index in range(MAX_QUERY_COUNT + 1)
            },
        )
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error_code"], "TOO_MANY_QUERIES")
        self.assertFalse(result["retryable"])

    def test_2048_character_emoji_query_is_safely_token_bounded(self):
        boundary_text = "🧭" * 2048
        with self._runtime(self._waveform([1.0])):
            result = self.scorer.score(
                "audio.wav",
                {"emoji-boundary": boundary_text},
            )

        self.assertEqual(result["status"], "COMPLETED")
        text_call = next(
            call for call in self.processor.calls if call["kind"] == "text"
        )
        self.assertEqual(text_call["text_kwargs"]["max_length"], 77)
        self.assertEqual(text_call["text_kwargs"]["padding"], "longest")
        self.assertTrue(text_call["text_kwargs"]["truncation"])

        too_long = self.scorer.score(
            "unused.wav",
            {"emoji-over-boundary": "🧭" * 2049},
        )
        self.assertEqual(too_long["status"], "FAILED")
        self.assertEqual(too_long["error_code"], "QUERY_TEXT_TOO_LONG")

    def test_256_max_length_queries_are_bounded_text_batches(self):
        queries = {
            f"query-{index}": "🎧" * 2048
            for index in range(MAX_QUERY_COUNT)
        }
        with patch.dict(os.environ, {}, clear=True), self._runtime(
            self._waveform([1.0])
        ):
            result = self.scorer.score("audio.wav", queries)

        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(len(result["scores"]), MAX_QUERY_COUNT)
        self.assertEqual(self.model.text_batch_sizes, [32] * 8)
        text_calls = [
            call for call in self.processor.calls if call["kind"] == "text"
        ]
        self.assertEqual(len(text_calls), 8)
        self.assertTrue(
            all(call["text_kwargs"]["max_length"] == 77 for call in text_calls)
        )

    def test_malformed_queries_fail_before_model_or_audio_work(self):
        cases = [
            (None, "INVALID_QUERIES_TYPE"),
            ([], "INVALID_QUERIES_TYPE"),
            ({}, "EMPTY_QUERIES"),
            ({" ": "sound"}, "INVALID_QUERY_NAME"),
            ({"event": "  "}, "INVALID_QUERY_TEXT"),
            ({1: "sound"}, "INVALID_QUERY_NAME"),
            ({"event": object()}, "INVALID_QUERY_TEXT"),
        ]
        self.scorer.model = None
        with patch.object(
            self.scorer,
            "_ensure_loaded",
            side_effect=AssertionError("model load must not run"),
        ) as ensure_loaded:
            for queries, code in cases:
                with self.subTest(code=code):
                    result = self.scorer.score("unused.wav", queries)
                    self.assertEqual(result["status"], "FAILED")
                    self.assertEqual(result["error_code"], code)
                    self.assertFalse(result["retryable"])
        ensure_loaded.assert_not_called()

    def test_empty_and_nonfinite_audio_return_structured_failures(self):
        cases = [
            (np.asarray([], dtype=np.float32), "EMPTY_AUDIO"),
            (np.asarray([np.nan], dtype=np.float32), "NONFINITE_AUDIO"),
            (np.asarray([np.inf], dtype=np.float32), "NONFINITE_AUDIO"),
        ]
        for waveform, code in cases:
            with self.subTest(code=code), self._runtime(waveform):
                result = self.scorer.score(
                    "audio.wav",
                    {"event": "an event"},
                )
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["error_code"], code)
            self.assertEqual(result["scores"], {})
            self.assertIn("duration", result)
            self.assertIn("timing", result)
            self.assertIn("preprocessing", result)
            self.assertIn("score_semantics", result)

    def test_text_embeddings_are_lru_cached_by_query_text_not_name(self):
        waveform = self._waveform([1.0])
        with self._runtime(waveform):
            first = self.scorer.score("audio.wav", {"first-name": "same text"})
            second = self.scorer.score("audio.wav", {"other-name": "same text"})

        text_calls = [call for call in self.processor.calls if call["kind"] == "text"]
        self.assertEqual(len(text_calls), 1)
        self.assertFalse(first["text_cache"]["hit"])
        self.assertTrue(second["text_cache"]["hit"])
        self.assertEqual(second["scores"], {"other-name": [1.0]})

    def test_text_embedding_lru_has_a_hard_capacity(self):
        waveform = self._waveform([1.0])
        with patch("clap_scorer.TEXT_EMBEDDING_CACHE_SIZE", 2), self._runtime(waveform):
            self.scorer.score("audio.wav", {"a": "text-a"})
            self.scorer.score("audio.wav", {"b": "text-b"})
            self.scorer.score("audio.wav", {"c": "text-c"})
            self.assertEqual(len(self.scorer._text_embedding_cache), 2)
            self.assertNotIn(("text-a",), self.scorer._text_embedding_cache)
            self.scorer.score("audio.wav", {"a": "text-a"})

        text_calls = [call for call in self.processor.calls if call["kind"] == "text"]
        self.assertEqual(len(text_calls), 4)

    def test_cuda_oom_halves_and_retries_the_same_windows(self):
        self.model = _FakeModel(oom_above=2, cuda=self.torch.cuda)
        self.scorer.model = self.model
        waveform = self._waveform([1.0] * 8)
        with patch.dict(os.environ, {"CLAP_MICROBATCH_SIZE": "8"}, clear=False), self._runtime(waveform):
            result = self.scorer.score("audio.wav", {"event": "an event"})

        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(len(result["scores"]["event"]), 8)
        self.assertEqual(self.model.audio_batch_sizes, [8, 4, 2, 2, 2, 2])
        self.assertEqual(self.torch.cuda.empty_cache_calls, 2)
        self.assertEqual(result["batching"]["configured_microbatch_size"], 8)
        self.assertEqual(result["batching"]["effective_microbatch_size"], 2)
        self.assertEqual(result["batching"]["oom_retries"], 2)

    def test_cublas_allocation_failure_also_halves_and_retries(self):
        original_get_audio_features = self.model.get_audio_features
        calls = 0

        def fail_first_allocation(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError(
                    "CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"
                )
            return original_get_audio_features(*args, **kwargs)

        self.model.get_audio_features = fail_first_allocation
        with patch.dict(
            os.environ,
            {"CLAP_MICROBATCH_SIZE": "2"},
            clear=False,
        ), self._runtime(self._waveform([1.0, 1.0])):
            result = self.scorer.score(
                "audio.wav",
                {"event": "an event"},
            )

        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["batching"]["oom_retries"], 1)
        self.assertEqual(result["batching"]["effective_microbatch_size"], 1)
        self.assertEqual(self.torch.cuda.empty_cache_calls, 1)

    def test_text_cuda_oom_halves_and_retries_the_same_queries(self):
        self.model = _FakeModel(text_oom_above=2, cuda=self.torch.cuda)
        self.scorer.model = self.model
        queries = {f"query-{index}": "x" * 2048 for index in range(8)}
        with patch.dict(
            os.environ,
            {"CLAP_TEXT_MICROBATCH_SIZE": "8"},
            clear=False,
        ), self._runtime(self._waveform([1.0])):
            result = self.scorer.score("audio.wav", queries)

        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(len(result["scores"]), 8)
        self.assertEqual(self.model.text_batch_sizes, [8, 4, 2, 2, 2, 2])
        self.assertEqual(self.torch.cuda.empty_cache_calls, 2)
        self.assertEqual(
            result["batching"]["configured_text_microbatch_size"],
            8,
        )
        self.assertEqual(
            result["batching"]["effective_text_microbatch_size"],
            2,
        )
        self.assertEqual(result["batching"]["text_oom_retries"], 2)

    def test_text_cuda_oom_at_one_uses_stable_retryable_error_code(self):
        self.model = _FakeModel(
            always_text_oom=True,
            cuda=self.torch.cuda,
        )
        self.scorer.model = self.model
        with patch.dict(
            os.environ,
            {"CLAP_TEXT_MICROBATCH_SIZE": "1"},
            clear=False,
        ), self._runtime(self._waveform([1.0])):
            result = self.scorer.score(
                "audio.wav",
                {"event": "x" * 2048},
            )

        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error_code"], "CUDA_OUT_OF_MEMORY")
        self.assertTrue(result["retryable"])
        self.assertEqual(self.torch.cuda.empty_cache_calls, 1)

    def test_cuda_oom_at_one_is_retryable_structured_failure(self):
        self.model = _FakeModel(always_oom=True, cuda=self.torch.cuda)
        self.scorer.model = self.model
        with patch.dict(os.environ, {"CLAP_MICROBATCH_SIZE": "1"}, clear=False), self._runtime(self._waveform([1.0])):
            result = self.scorer.score("audio.wav", {"event": "an event"})

        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error_code"], "CUDA_OUT_OF_MEMORY")
        self.assertTrue(result["retryable"])
        self.assertEqual(self.torch.cuda.empty_cache_calls, 1)

    def test_microbatch_environment_is_defaulted_and_bounded(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.scorer._configured_microbatch_size(), 64)
        with patch.dict(os.environ, {"CLAP_MICROBATCH_SIZE": "0"}, clear=True):
            self.assertEqual(self.scorer._configured_microbatch_size(), 1)
        with patch.dict(os.environ, {"CLAP_MICROBATCH_SIZE": "9999"}, clear=True):
            self.assertEqual(self.scorer._configured_microbatch_size(), 256)
        with patch.dict(os.environ, {"CLAP_MICROBATCH_SIZE": "invalid"}, clear=True):
            self.assertEqual(self.scorer._configured_microbatch_size(), 64)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                self.scorer._configured_text_microbatch_size(),
                32,
            )
        with patch.dict(
            os.environ,
            {"CLAP_TEXT_MICROBATCH_SIZE": "9999"},
            clear=True,
        ):
            self.assertEqual(
                self.scorer._configured_text_microbatch_size(),
                64,
            )

    def test_unexpected_exception_never_escapes_score(self):
        with patch.object(
            self.scorer,
            "_score_batched",
            side_effect=RuntimeError("surprise\nwith details"),
        ):
            result = self.scorer.score(
                "audio.wav",
                {"event": "an event"},
            )

        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error_code"], "CLAP_SCORING_FAILED")
        self.assertTrue(result["retryable"])
        self.assertEqual(result["error"], "surprise with details")
        self.assertEqual(result["scores"], {})
        self.assertEqual(result["model"], CLAP_MODEL_ID)
        self.assertEqual(result["device"], "cuda")
        self.assertEqual(result["windowSize"], 1.0)

    def test_success_and_failure_share_a_stable_top_level_schema(self):
        with self._runtime(self._waveform([1.0])):
            success = self.scorer.score(
                "audio.wav",
                {"event": "an event"},
            )
        with patch.object(
            self.scorer,
            "_score_batched",
            side_effect=RuntimeError("inference failed"),
        ):
            failure = self.scorer.score(
                "audio.wav",
                {"event": "an event"},
            )

        self.assertEqual(set(success), set(failure))
        self.assertEqual(
            {
                "schema_version",
                "status",
                "error_code",
                "error",
                "retryable",
                "scores",
                "duration",
                "window_count",
                "final_window_valid_seconds",
                "model",
                "device",
                "windowSize",
                "model_revision",
                "timing",
                "preprocessing",
                "score_semantics",
                "batching",
                "text_cache",
            },
            set(success),
        )

    def test_response_builder_regression_still_cannot_escape_score(self):
        with patch.object(
            self.scorer,
            "_score_batched",
            return_value={"event": [0.5]},
        ), patch.object(
            self.scorer,
            "_response",
            side_effect=RuntimeError("broken response builder"),
        ):
            result = self.scorer.score(
                "audio.wav",
                {"event": "an event"},
            )

        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error_code"], "RESPONSE_BUILD_FAILED")
        self.assertTrue(result["retryable"])
        self.assertEqual(result["schema_version"], CLAP_RESPONSE_SCHEMA)


if __name__ == "__main__":
    unittest.main()
