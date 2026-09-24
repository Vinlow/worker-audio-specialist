"""German acoustics with the maintained CTC windows, stitching and authority."""
import torch

from aligner import Wav2Vec2Aligner
from german_alignment_model import GermanAlignmentModel


class GermanCtcAcoustics(torch.nn.Module):
    """Adapt HF preprocessing/logits to the torchaudio aligner's emission port."""

    def __init__(self, model, extractor):
        super().__init__()
        self.model = model
        self.extractor = extractor

    def forward(self, waveform):
        if waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError("German alignment expects one mono acoustic window")
        inputs = self.extractor(
            waveform[0].detach().cpu().numpy(),
            sampling_rate=GermanAlignmentModel.sample_rate,
            return_tensors="pt",
        ).to(waveform.device)
        return self.model(**inputs).logits, None


class GermanWav2Vec2Aligner(Wav2Vec2Aligner):
    model_id = GermanAlignmentModel.identity
    supported_languages = frozenset({"de"})

    @staticmethod
    def normalize_word(word):
        return GermanAlignmentModel.normalize_word(word)

    def _load_components(self):
        model, extractor = GermanAlignmentModel.load()
        return (GermanCtcAcoustics(model, extractor),
                GermanAlignmentModel.labels, GermanAlignmentModel.sample_rate)
