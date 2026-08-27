import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from aligner import MAX_META_LOAD_ATTEMPTS, Wav2Vec2Aligner
from clap_scorer import ClapScorer


class _FakeWavModel:
    def __init__(self, *, is_meta=False):
        self.is_meta = is_meta
        self.to_calls = []

    def parameters(self):
        return iter([SimpleNamespace(is_meta=self.is_meta)])

    def to(self, device):
        self.to_calls.append(device)
        return self

    def eval(self):
        return self


class ModelLoadHardeningTest(unittest.TestCase):
    def test_wav2vec_retries_one_meta_bundle_then_publishes_hydrated_model(self):
        meta_model = _FakeWavModel(is_meta=True)
        hydrated_model = _FakeWavModel()
        aligner = Wav2Vec2Aligner()

        with patch(
            "aligner.BUNDLE.get_model",
            side_effect=[meta_model, hydrated_model],
        ) as get_model:
            aligner.setup("cpu")

        self.assertEqual(get_model.call_count, 2)
        self.assertIs(aligner.model, hydrated_model)
        self.assertEqual(hydrated_model.to_calls, ["cpu"])
        self.assertEqual(aligner.device, "cpu")
        self.assertIsNotNone(aligner.labels)

    def test_wav2vec_fails_closed_after_bounded_meta_retries(self):
        aligner = Wav2Vec2Aligner()
        meta_models = [
            _FakeWavModel(is_meta=True)
            for _ in range(MAX_META_LOAD_ATTEMPTS)
        ]

        with patch(
            "aligner.BUNDLE.get_model",
            side_effect=meta_models,
        ) as get_model:
            with self.assertRaisesRegex(
                RuntimeError,
                f"after {MAX_META_LOAD_ATTEMPTS} serialized attempts",
            ):
                aligner.setup("cpu")

        self.assertEqual(get_model.call_count, MAX_META_LOAD_ATTEMPTS)
        self.assertIsNone(aligner.model)
        self.assertIsNone(aligner.device)
        self.assertIsNone(aligner.labels)
        self.assertIsNone(aligner.label_to_idx)
        self.assertIsNone(aligner.sample_rate)
        self.assertIsNone(aligner.word_separator_idx)

    def test_clap_and_wav2vec_cold_construction_cannot_overlap(self):
        clap_entered = threading.Event()
        release_clap = threading.Event()
        wav_entered = threading.Event()
        errors = []

        class FakeProcessor:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

        class FakeClapModel:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                # Exercise the same transient meta-device construction state
                # without depending on Transformers' optional `accelerate`
                # integration.  Transformers 5.x no longer exports its old
                # helper, while PyTorch's native context preserves the load
                # boundary this regression is meant to cover.
                with torch.device("meta"):
                    clap_entered.set()
                    if not release_clap.wait(timeout=2):
                        raise RuntimeError("test coordination timeout")
                return cls()

            def eval(self):
                return self

        def load_wav_model(**_kwargs):
            wav_entered.set()
            return torch.nn.Linear(4, 2)

        scorer = ClapScorer()
        aligner = Wav2Vec2Aligner()

        def capture(callable_):
            try:
                callable_()
            except Exception as exc:
                errors.append(exc)

        fake_transformers = SimpleNamespace(
            ClapModel=FakeClapModel,
            ClapProcessor=FakeProcessor,
        )

        with patch.dict(
            sys.modules,
            {"transformers": fake_transformers},
        ), patch(
            "clap_scorer.hf_from_pretrained_kwargs",
            return_value={},
        ), patch(
            "torch.cuda.is_available",
            return_value=False,
        ), patch(
            "aligner.BUNDLE.get_model",
            side_effect=load_wav_model,
        ):
            clap_thread = threading.Thread(
                target=lambda: capture(scorer._ensure_loaded),
            )
            aligner_thread = threading.Thread(
                target=lambda: capture(lambda: aligner.setup("cpu")),
            )
            clap_thread.start()
            self.assertTrue(clap_entered.wait(timeout=2))
            aligner_thread.start()

            # Wav2vec is waiting at the shared cold-load boundary rather than
            # constructing inside CLAP's transient PyTorch initialization.
            time.sleep(0.05)
            self.assertFalse(wav_entered.is_set())

            release_clap.set()
            clap_thread.join(timeout=2)
            aligner_thread.join(timeout=2)

        self.assertFalse(clap_thread.is_alive())
        self.assertFalse(aligner_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(wav_entered.is_set())
        self.assertIsNotNone(scorer.model)
        self.assertIsNotNone(aligner.model)
        self.assertFalse(any(param.is_meta for param in aligner.model.parameters()))


if __name__ == "__main__":
    unittest.main()
