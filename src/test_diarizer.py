import copy
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

import yaml

from diarizer import (
    DEFAULT_MODEL_ID,
    SpeakerDiarizer,
    _canonicalize_turns,
    _load_reviewed_pipeline,
    _pin_pipeline_dependencies,
    build_diarization_sidecar,
)


def _fake_torch(cuda_available=False):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
    )
    torch.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            matmul=types.SimpleNamespace(allow_tf32=False),
        ),
        cudnn=types.SimpleNamespace(
            allow_tf32=True,
            deterministic=True,
            benchmark=False,
        ),
    )
    torch.serialization = types.SimpleNamespace(
        add_safe_globals=lambda _globals: None,
    )
    torch.device = lambda device: device
    return torch


def _annotation(*turns):
    return [
        (types.SimpleNamespace(start=start, end=end), speaker)
        for start, end, speaker in turns
    ]


class DiarizationSidecarTest(unittest.TestCase):
    def test_canonicalizes_labels_by_first_appearance(self):
        turns, mapping = _canonicalize_turns(
            [
                (2.0, 3.0, "B"),
                (0.0, 1.0, "A"),
                (3.0, 4.0, "A"),
            ]
        )

        self.assertEqual(mapping, {"A": "SPEAKER_00", "B": "SPEAKER_01"})
        self.assertEqual(
            [turn["speaker_id"] for turn in turns],
            ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"],
        )

    def test_canonicalization_tracks_discarded_invalid_turns_additively(self):
        class OverflowingFloat:
            def __float__(self):
                raise OverflowError("out of range")

        stats = {}
        turns, _mapping = _canonicalize_turns(
            [
                (0.0, 1.0, "A"),
                (1.0, 0.0, "B"),
                (float("nan"), 2.0, "C"),
                (2.0, 3.0, ""),
                (OverflowingFloat(), 4.0, "D"),
            ],
            stats=stats,
        )

        self.assertEqual(len(turns), 1)
        self.assertEqual(stats["discarded_invalid_turn_count"], 4)

    def test_default_model_uses_reviewed_revision_but_env_can_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(SpeakerDiarizer().model_id, DEFAULT_MODEL_ID)
        self.assertEqual(
            DEFAULT_MODEL_ID,
            "pyannote/speaker-diarization-3.1"
            "@84fd25912480287da0247647c3d2b4853cb3ee5d",
        )
        with patch.dict(
            os.environ,
            {"PYANNOTE_DIARIZATION_MODEL": "local-review-override"},
            clear=False,
        ):
            self.assertEqual(
                SpeakerDiarizer().model_id,
                "local-review-override",
            )

    def test_builds_attributions_without_mutating_np_sbv2_words(self):
        words = [
            {
                "word": "Hello",
                "start": 0.1,
                "end": 0.4,
                "onset_start": 0.05,
                "offset_end": 0.45,
            },
            {
                "word": "there",
                "start": 0.5,
                "end": 0.8,
                "onset_start": 0.45,
                "offset_end": 0.9,
            },
        ]
        original = copy.deepcopy(words)
        sidecar = build_diarization_sidecar(
            words,
            [
                {
                    "start_sec": 0.0,
                    "end_sec": 0.48,
                    "speaker_id": "SPEAKER_00",
                },
                {
                    "start_sec": 0.48,
                    "end_sec": 1.0,
                    "speaker_id": "SPEAKER_01",
                },
            ],
        )

        self.assertEqual(words, original)
        self.assertFalse(sidecar["transcript_geometry_mutated"])
        self.assertFalse(sidecar["boundary_authority"])
        self.assertEqual(
            [item["speaker_id"] for item in sidecar["word_attributions"]],
            ["SPEAKER_00", "SPEAKER_01"],
        )
        self.assertEqual(sidecar["status"], "COMPLETED")
        self.assertEqual(sidecar["quality_status"], "COMPLETED")
        self.assertEqual(sidecar["model"], DEFAULT_MODEL_ID)
        self.assertEqual(sidecar["model_load_policy"], "BAKED_CACHE_ONLY")
        self.assertEqual(
            sidecar["model_dependencies"],
            {
                "segmentation": (
                    "pyannote/segmentation-3.0"
                    "@e66f3d3b9eb0873085418a7b813d3b369bf160bb"
                ),
                "embedding": (
                    "pyannote/wespeaker-voxceleb-resnet34-LM"
                    "@837717ddb9ff5507820346191109dc79c958d614"
                ),
            },
        )
        self.assertEqual(sidecar["attribution_count"], 2)
        for attribution in sidecar["word_attributions"]:
            self.assertEqual(
                attribution["confidence"],
                attribution["coverage_fraction"],
            )

    def test_simultaneous_overlap_is_ambiguous_without_forced_winner(self):
        sidecar = build_diarization_sidecar(
            [{"word": "wow", "start": 1.0, "end": 1.5}],
            [
                {
                    "start_sec": 0.9,
                    "end_sec": 1.5,
                    "speaker_id": "SPEAKER_00",
                },
                {
                    "start_sec": 1.2,
                    "end_sec": 1.6,
                    "speaker_id": "SPEAKER_01",
                },
            ],
            [
                {
                    "start_sec": 0.9,
                    "end_sec": 1.5,
                    "speaker_id": "SPEAKER_00",
                }
            ],
        )

        attribution = sidecar["word_attributions"][0]
        self.assertEqual(attribution["status"], "UNKNOWN")
        self.assertEqual(
            attribution["attribution_reason"],
            "AMBIGUOUS_OVERLAP",
        )
        self.assertIsNone(attribution["speaker_id"])
        self.assertTrue(attribution["overlap"])
        self.assertFalse(attribution["sequential_handoff"])
        self.assertEqual(
            attribution["candidate_speaker_ids"],
            ["SPEAKER_00", "SPEAKER_01"],
        )
        self.assertEqual(
            {
                candidate["speaker_id"]: candidate["coverage_fraction"]
                for candidate in attribution["candidate_speakers"]
            },
            {"SPEAKER_00": 1.0, "SPEAKER_01": 0.6},
        )
        self.assertEqual(attribution["simultaneous_overlap_fraction"], 0.6)
        self.assertEqual(sidecar["status"], "COMPLETED")
        self.assertEqual(sidecar["quality_status"], "PARTIAL")
        self.assertEqual(sidecar["ambiguity_count"], 1)
        self.assertEqual(sidecar["unknown_count"], 0)

    def test_sequential_handoff_is_not_simultaneous_overlap(self):
        sidecar = build_diarization_sidecar(
            [{"word": "hello", "start": 1.0, "end": 1.5}],
            [
                {
                    "start_sec": 0.9,
                    "end_sec": 1.3,
                    "speaker_id": "SPEAKER_00",
                },
                {
                    "start_sec": 1.3,
                    "end_sec": 1.7,
                    "speaker_id": "SPEAKER_01",
                },
            ],
        )

        attribution = sidecar["word_attributions"][0]
        self.assertEqual(attribution["status"], "ATTRIBUTED")
        self.assertEqual(attribution["speaker_id"], "SPEAKER_00")
        self.assertFalse(attribution["overlap"])
        self.assertTrue(attribution["sequential_handoff"])
        self.assertEqual(attribution["simultaneous_overlap_fraction"], 0.0)
        self.assertEqual(sidecar["status"], "COMPLETED")
        self.assertEqual(sidecar["quality_status"], "COMPLETED")

    def test_exclusive_winner_is_always_present_in_candidates(self):
        sidecar = build_diarization_sidecar(
            [{"word": "hello", "start": 0.0, "end": 0.5}],
            [
                {
                    "start_sec": 0.0,
                    "end_sec": 0.2,
                    "speaker_id": "SPEAKER_00",
                }
            ],
            [
                {
                    "start_sec": 0.0,
                    "end_sec": 0.5,
                    "speaker_id": "SPEAKER_01",
                }
            ],
        )

        attribution = sidecar["word_attributions"][0]
        self.assertEqual(attribution["speaker_id"], "SPEAKER_01")
        self.assertIn(
            attribution["speaker_id"],
            attribution["candidate_speaker_ids"],
        )
        self.assertEqual(
            {
                candidate["speaker_id"]
                for candidate in attribution["candidate_speakers"]
            },
            {"SPEAKER_00", "SPEAKER_01"},
        )

    def test_empty_turn_output_is_not_reported_as_completed(self):
        sidecar = build_diarization_sidecar(
            [{"word": "hello", "start": 0.0, "end": 0.4}],
            [],
        )

        self.assertEqual(sidecar["status"], "FAILED")
        self.assertEqual(sidecar["quality_status"], "EMPTY_OUTPUT")
        self.assertEqual(sidecar["error_code"], "DIARIZATION_EMPTY_OUTPUT")
        self.assertEqual(sidecar["attribution_count"], 0)
        self.assertEqual(sidecar["unknown_count"], 1)
        self.assertEqual(sidecar["coverage_fraction"], 0.0)

    def test_turn_only_empty_result_has_no_failed_word_contract(self):
        sidecar = build_diarization_sidecar([], [])

        self.assertEqual(sidecar["status"], "COMPLETED")
        self.assertEqual(sidecar["quality_status"], "EMPTY_OUTPUT")
        self.assertNotIn("error_code", sidecar)

    def test_pipeline_empty_output_fails_when_words_need_attribution(self):
        class EmptyPipeline:
            def __call__(self, _audio_path, **_kwargs):
                return []

        diarizer = SpeakerDiarizer()
        diarizer.pipeline = EmptyPipeline()
        diarizer.device = "cpu"
        fake_torch = _fake_torch()
        with patch.dict(
            sys.modules,
            {"torch": fake_torch},
        ), patch("builtins.print") as print_mock:
            sidecar = diarizer.diarize(
                "unused.wav",
                [{"word": "hello", "start": 0.0, "end": 0.4}],
            )

        self.assertEqual(sidecar["status"], "FAILED")
        self.assertEqual(sidecar["quality_status"], "EMPTY_OUTPUT")
        self.assertEqual(sidecar["error_code"], "DIARIZATION_EMPTY_OUTPUT")
        self.assertIn("request_failed", repr(print_mock.call_args_list))

    def test_partial_word_coverage_has_counts_and_fractions(self):
        sidecar = build_diarization_sidecar(
            [
                {"word": "covered", "start": 0.0, "end": 0.4},
                {"word": "gap", "start": 1.0, "end": 1.4},
            ],
            [
                {
                    "start_sec": 0.0,
                    "end_sec": 0.4,
                    "speaker_id": "SPEAKER_00",
                }
            ],
        )

        self.assertEqual(sidecar["status"], "COMPLETED")
        self.assertEqual(sidecar["quality_status"], "PARTIAL")
        self.assertEqual(sidecar["attribution_count"], 1)
        self.assertEqual(sidecar["ambiguity_count"], 0)
        self.assertEqual(sidecar["unknown_count"], 1)
        self.assertEqual(sidecar["attribution_fraction"], 0.5)
        self.assertEqual(sidecar["unknown_fraction"], 0.5)
        self.assertEqual(sidecar["coverage_fraction"], 0.5)

    def test_unknown_when_no_turn_overlaps_word(self):
        sidecar = build_diarization_sidecar(
            [{"word": "late", "start": 10.0, "end": 10.4}],
            [
                {
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                    "speaker_id": "SPEAKER_00",
                }
            ],
        )

        self.assertEqual(sidecar["word_attributions"][0]["status"], "UNKNOWN")
        self.assertIsNone(sidecar["word_attributions"][0]["speaker_id"])

    def test_rejects_inverted_speaker_hints_before_model_load(self):
        sidecar = SpeakerDiarizer().diarize(
            "unused.wav",
            [],
            min_speakers=3,
            max_speakers=2,
        )

        self.assertEqual(sidecar["status"], "FAILED")
        self.assertIn("DIARIZATION_MIN_SPEAKERS_EXCEEDS_MAX", sidecar["error"])

    def test_concurrent_pipeline_inference_is_single_flight(self):
        entered = threading.Event()
        release = threading.Event()
        state_lock = threading.Lock()
        state = {"active": 0, "calls": 0, "max_active": 0}

        class BlockingPipeline:
            def __call__(self, _audio_path, **_kwargs):
                with state_lock:
                    state["active"] += 1
                    state["calls"] += 1
                    state["max_active"] = max(
                        state["max_active"],
                        state["active"],
                    )
                    call_number = state["calls"]
                if call_number == 1:
                    entered.set()
                    if not release.wait(timeout=2):
                        raise RuntimeError("test coordination timeout")
                with state_lock:
                    state["active"] -= 1
                return _annotation((0.0, 1.0, "A"))

        diarizer = SpeakerDiarizer()
        diarizer.pipeline = BlockingPipeline()
        diarizer.device = "cpu"
        fake_torch = _fake_torch()
        results = []

        def run():
            results.append(
                diarizer.diarize(
                    "unused.wav",
                    [{"word": "hi", "start": 0.1, "end": 0.3}],
                )
            )

        with patch.dict(sys.modules, {"torch": fake_torch}):
            first = threading.Thread(target=run)
            second = threading.Thread(target=run)
            first.start()
            self.assertTrue(entered.wait(timeout=2))
            second.start()
            time.sleep(0.05)
            with state_lock:
                self.assertEqual(state["calls"], 1)
            release.set()
            first.join(timeout=2)
            second.join(timeout=2)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(state["calls"], 2)
        self.assertEqual(state["max_active"], 1)
        self.assertEqual(
            [result["status"] for result in results],
            ["COMPLETED", "COMPLETED"],
        )

    def test_pipeline_precision_flags_are_restored_after_inference(self):
        fake_torch = _fake_torch()
        original = {
            "matmul_allow_tf32": (
                fake_torch.backends.cuda.matmul.allow_tf32
            ),
            "cudnn_allow_tf32": fake_torch.backends.cudnn.allow_tf32,
            "deterministic": fake_torch.backends.cudnn.deterministic,
            "benchmark": fake_torch.backends.cudnn.benchmark,
        }

        class MutatingPipeline:
            def __call__(self, _audio_path, **_kwargs):
                fake_torch.backends.cuda.matmul.allow_tf32 = True
                fake_torch.backends.cudnn.allow_tf32 = False
                fake_torch.backends.cudnn.deterministic = False
                fake_torch.backends.cudnn.benchmark = True
                return _annotation((0.0, 1.0, "A"))

        diarizer = SpeakerDiarizer()
        diarizer.pipeline = MutatingPipeline()
        diarizer.device = "cpu"
        with patch.dict(sys.modules, {"torch": fake_torch}):
            sidecar = diarizer.diarize(
                "unused.wav",
                [{"word": "hi", "start": 0.1, "end": 0.3}],
            )

        self.assertEqual(sidecar["status"], "COMPLETED")
        self.assertEqual(
            fake_torch.backends.cuda.matmul.allow_tf32,
            original["matmul_allow_tf32"],
        )
        self.assertEqual(
            fake_torch.backends.cudnn.allow_tf32,
            original["cudnn_allow_tf32"],
        )
        self.assertEqual(
            fake_torch.backends.cudnn.deterministic,
            original["deterministic"],
        )
        self.assertEqual(
            fake_torch.backends.cudnn.benchmark,
            original["benchmark"],
        )

    def test_inference_failure_uses_stable_path_free_error_taxonomy(self):
        leaked_detail = "/private/customer.wav token=hf_do_not_log"

        class FailingPipeline:
            def __call__(self, _audio_path, **_kwargs):
                raise RuntimeError(leaked_detail)

        diarizer = SpeakerDiarizer()
        diarizer.pipeline = FailingPipeline()
        diarizer.device = "cpu"
        fake_torch = _fake_torch()
        with patch.dict(
            sys.modules,
            {"torch": fake_torch},
        ), patch("builtins.print") as print_mock:
            sidecar = diarizer.diarize(
                "/private/customer.wav",
                [{"word": "safe", "start": 0.0, "end": 0.2}],
            )

        self.assertEqual(sidecar["status"], "FAILED")
        self.assertEqual(sidecar["stage"], "INFERENCE")
        self.assertEqual(
            sidecar["error_code"],
            "DIARIZATION_INFERENCE_FAILED",
        )
        self.assertEqual(sidecar["error"], sidecar["error_code"])
        self.assertNotIn(leaked_detail, repr(sidecar))
        self.assertNotIn(leaked_detail, repr(print_mock.call_args_list))
        self.assertEqual(
            set(sidecar["timing"]),
            {
                "load_sec",
                "inference_wait_sec",
                "inference_sec",
                "processing_sec",
            },
        )

    def test_load_failure_uses_stable_path_free_error_taxonomy(self):
        leaked_detail = "/private/model-cache token=hf_do_not_log"
        diarizer = SpeakerDiarizer(model_id="test/model")
        fake_torch = _fake_torch()
        with patch.dict(
            sys.modules,
            {"torch": fake_torch},
        ), patch.object(
            diarizer,
            "setup",
            side_effect=RuntimeError(leaked_detail),
        ), patch("builtins.print") as print_mock:
            sidecar = diarizer.diarize("unused.wav", [])

        self.assertEqual(sidecar["status"], "FAILED")
        self.assertEqual(sidecar["stage"], "LOAD")
        self.assertEqual(
            sidecar["error_code"],
            "DIARIZATION_MODEL_LOAD_FAILED",
        )
        self.assertNotIn(leaked_detail, repr(sidecar))
        self.assertNotIn(leaked_detail, repr(print_mock.call_args_list))

    def test_reviewed_pipeline_rejects_unexpected_internal_reference(self):
        config = {
            "pipeline": {
                "params": {
                    "segmentation": "unreviewed/segmentation",
                    "embedding": (
                        "pyannote/wespeaker-voxceleb-resnet34-LM"
                    ),
                }
            }
        }

        with self.assertRaisesRegex(
            RuntimeError,
            "PYANNOTE_PIPELINE_DEPENDENCY_MISMATCH",
        ):
            _pin_pipeline_dependencies(
                config,
                "/cache/pyannote/segmentation/pytorch_model.bin",
                "/cache/pyannote/embedding/pytorch_model.bin",
            )

    def test_reviewed_cache_miss_never_retries_with_network_enabled(self):
        calls = []

        def hf_hub_download(**kwargs):
            calls.append(kwargs)
            raise FileNotFoundError("reviewed artifact missing from cache")

        fake_huggingface = types.ModuleType("huggingface_hub")
        fake_huggingface.hf_hub_download = hf_hub_download

        with patch.dict(
            sys.modules,
            {"huggingface_hub": fake_huggingface},
        ):
            with self.assertRaises(FileNotFoundError):
                _load_reviewed_pipeline(object(), None)

        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["local_files_only"])
        self.assertNotIn("token", calls[0])

    def test_reviewed_pipeline_loads_exact_baked_cache_without_token(self):
        pipeline_source = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="pyannote-pipeline-",
            suffix=".yaml",
            delete=False,
        )
        segmentation_source = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="pyannote-segmentation-",
            suffix="-pytorch_model.bin",
            delete=False,
        )
        embedding_source = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="pyannote-wespeaker-",
            suffix="-pytorch_model.bin",
            delete=False,
        )
        try:
            yaml.safe_dump(
                {
                    "version": "3.1.0",
                    "pipeline": {
                        "name": "fake.SpeakerDiarization",
                        "params": {
                            "segmentation": "pyannote/segmentation-3.0",
                            "embedding": (
                                "pyannote/"
                                "wespeaker-voxceleb-resnet34-LM"
                            ),
                        },
                    },
                },
                pipeline_source,
            )
            pipeline_source.close()
            segmentation_source.write(b"reviewed-segmentation")
            segmentation_source.close()
            embedding_source.write(b"reviewed-embedding")
            embedding_source.close()

            download_calls = []
            captured = {}
            cached_artifacts = {
                (
                    "pyannote/speaker-diarization-3.1",
                    "config.yaml",
                ): pipeline_source.name,
                (
                    "pyannote/segmentation-3.0",
                    "pytorch_model.bin",
                ): segmentation_source.name,
                (
                    "pyannote/wespeaker-voxceleb-resnet34-LM",
                    "pytorch_model.bin",
                ): embedding_source.name,
            }

            def hf_hub_download(**kwargs):
                download_calls.append(kwargs)
                if kwargs.get("local_files_only") is not True:
                    raise AssertionError("network fallback attempted")
                return cached_artifacts[
                    (kwargs["repo_id"], kwargs["filename"])
                ]

            class FakePipeline:
                @classmethod
                def from_pretrained(cls, config_path, **kwargs):
                    captured["config_path"] = config_path
                    captured["config_mode"] = (
                        os.stat(config_path).st_mode & 0o777
                    )
                    with open(config_path, "r", encoding="utf-8") as source:
                        captured["config"] = yaml.safe_load(source)
                    captured["kwargs"] = kwargs
                    return cls()

                def to(self, _device):
                    return self

            fake_huggingface = types.ModuleType("huggingface_hub")
            fake_huggingface.hf_hub_download = hf_hub_download
            fake_pyannote = types.ModuleType("pyannote")
            fake_audio = types.ModuleType("pyannote.audio")
            fake_core = types.ModuleType("pyannote.audio.core")
            fake_task = types.ModuleType("pyannote.audio.core.task")
            fake_audio.Pipeline = FakePipeline
            fake_task.Problem = type("Problem", (), {})
            fake_task.Resolution = type("Resolution", (), {})
            fake_task.Specifications = type("Specifications", (), {})
            fake_torch = _fake_torch()
            fake_torch_version = types.ModuleType("torch.torch_version")
            fake_torch_version.TorchVersion = type("TorchVersion", (), {})

            with patch.dict(
                os.environ,
                {},
                clear=True,
            ), patch.dict(
                sys.modules,
                {
                    "huggingface_hub": fake_huggingface,
                    "pyannote": fake_pyannote,
                    "pyannote.audio": fake_audio,
                    "pyannote.audio.core": fake_core,
                    "pyannote.audio.core.task": fake_task,
                    "torch": fake_torch,
                    "torch.torch_version": fake_torch_version,
                },
            ), patch(
                "diarizer.install_legacy_use_auth_token_compat",
                return_value=None,
            ):
                diarizer = SpeakerDiarizer()
                diarizer.setup("cpu")

            params = captured["config"]["pipeline"]["params"]
            self.assertEqual(
                params["segmentation"],
                segmentation_source.name,
            )
            self.assertEqual(
                params["embedding"],
                embedding_source.name,
            )
            self.assertIn("pyannote", params["segmentation"])
            self.assertIn("pyannote", params["embedding"])
            self.assertEqual(captured["config_mode"], 0o600)
            self.assertEqual(
                captured["kwargs"],
                {"use_auth_token": None},
            )
            self.assertEqual(
                [
                    (
                        call["repo_id"],
                        call["filename"],
                        call["revision"],
                        call["local_files_only"],
                        "token" in call,
                    )
                    for call in download_calls
                ],
                [
                    (
                        "pyannote/speaker-diarization-3.1",
                        "config.yaml",
                        "84fd25912480287da0247647c3d2b4853cb3ee5d",
                        True,
                        False,
                    ),
                    (
                        "pyannote/segmentation-3.0",
                        "pytorch_model.bin",
                        "e66f3d3b9eb0873085418a7b813d3b369bf160bb",
                        True,
                        False,
                    ),
                    (
                        "pyannote/wespeaker-voxceleb-resnet34-LM",
                        "pytorch_model.bin",
                        "837717ddb9ff5507820346191109dc79c958d614",
                        True,
                        False,
                    ),
                ],
            )
            self.assertFalse(os.path.exists(captured["config_path"]))
        finally:
            pipeline_source.close()
            segmentation_source.close()
            embedding_source.close()
            for path in (
                pipeline_source.name,
                segmentation_source.name,
                embedding_source.name,
            ):
                if os.path.exists(path):
                    os.unlink(path)

    def test_concurrent_setup_publishes_one_complete_pipeline(self):
        entered = threading.Event()
        release = threading.Event()
        load_calls = []
        errors = []

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                load_calls.append((_args, _kwargs))
                entered.set()
                if not release.wait(timeout=2):
                    raise RuntimeError("test coordination timeout")
                return cls()

            def to(self, _device):
                return self

        fake_pyannote = types.ModuleType("pyannote")
        fake_audio = types.ModuleType("pyannote.audio")
        fake_core = types.ModuleType("pyannote.audio.core")
        fake_task = types.ModuleType("pyannote.audio.core.task")
        fake_audio.Pipeline = FakePipeline
        fake_task.Problem = type("Problem", (), {})
        fake_task.Resolution = type("Resolution", (), {})
        fake_task.Specifications = type("Specifications", (), {})
        fake_torch = _fake_torch()
        fake_torch_version = types.ModuleType("torch.torch_version")
        fake_torch_version.TorchVersion = type("TorchVersion", (), {})

        diarizer = SpeakerDiarizer(model_id="test/concurrent-pipeline")

        def setup():
            try:
                diarizer.setup("cpu")
            except Exception as exc:
                errors.append(exc)

        with patch.dict(
            os.environ,
            {"HUGGINGFACE_TOKEN": "test-token"},
            clear=False,
        ), patch.dict(
            sys.modules,
            {
                "pyannote": fake_pyannote,
                "pyannote.audio": fake_audio,
                "pyannote.audio.core": fake_core,
                "pyannote.audio.core.task": fake_task,
                "torch": fake_torch,
                "torch.torch_version": fake_torch_version,
            },
        ), patch(
            "diarizer.install_legacy_use_auth_token_compat",
            return_value=None,
        ):
            first = threading.Thread(target=setup)
            second = threading.Thread(target=setup)
            first.start()
            self.assertTrue(entered.wait(timeout=2))
            second.start()
            release.set()
            first.join(timeout=2)
            second.join(timeout=2)

        self.assertEqual(errors, [])
        self.assertEqual(
            load_calls,
            [(("test/concurrent-pipeline",), {"token": "test-token"})],
        )
        self.assertIsNotNone(diarizer.pipeline)
        self.assertEqual(diarizer.device, "cpu")


if __name__ == "__main__":
    unittest.main()
