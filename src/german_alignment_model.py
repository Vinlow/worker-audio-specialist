"""Pinned German CTC artifact and lexical projection; no English-model fallback."""
import json
import unicodedata
from pathlib import Path


class GermanAlignmentModel:
    repository = "jonatasgrosman/wav2vec2-large-xlsr-53-german"
    revision = "4b8a02957378d0f2da2ef74091156b032c485a89"
    identity = f"{repository}@{revision}"
    sample_rate = 16000
    labels = ("<pad>", "<s>", "</s>", "<unk>", "|", "'", "-",
              *tuple("abcdefghijklmnopqrstuvwxyzäíóöü"))
    files = ("config.json", "preprocessor_config.json", "vocab.json",
             "pytorch_model.bin", "README.md")

    @classmethod
    def normalize_word(cls, text):
        """Project spelling for CTC only; never rewrite the returned ASR word.

        NFC keeps decomposed umlauts, and casefold maps ß to ss. Unknown letters
        or digits invalidate the whole word instead of aligning a partial word
        and falsely granting its omitted phonemes acoustic authority.
        """
        text = unicodedata.normalize("NFC", text).casefold().replace("’", "'")
        result = []
        for char in text:
            if char in cls.labels[5:]:
                result.append(char)
            elif char.isalpha() or char.isnumeric():
                return ""
        return "".join(result) if any(char.isalpha() for char in result) else ""

    @classmethod
    def load(cls):
        # Transformers construction is lazy; the image's skops preloader has
        # already run before this is called by Predictor or the image builder.
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        options = {"revision": cls.revision, "local_files_only": True}
        vocabulary_path = hf_hub_download(cls.repository, "vocab.json", **options)
        vocabulary = json.loads(Path(vocabulary_path).read_text(encoding="utf-8"))
        if vocabulary != {label: index for index, label in enumerate(cls.labels)}:
            raise ValueError("Pinned German CTC vocabulary does not match its blank/token contract")
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(cls.repository, **options)
        if (extractor.sampling_rate != cls.sample_rate or not extractor.do_normalize
                or not extractor.return_attention_mask):
            raise ValueError("Pinned German acoustic preprocessing contract changed")
        with torch.device("cpu"):
            model = Wav2Vec2ForCTC.from_pretrained(cls.repository, **options)
        if any(parameter.is_meta for parameter in model.parameters()):
            raise RuntimeError("German alignment model contains unhydrated meta tensors")
        if model.config.pad_token_id != 0 or model.config.vocab_size != len(cls.labels):
            raise ValueError("German CTC model blank or emission vocabulary changed")
        return model, extractor
