"""
CLAP audio-text similarity scoring module.

Uses CLAP (Contrastive Language-Audio Pretraining) to compute per-second
similarity scores between audio and natural language queries.

The model is loaded once and cached for subsequent requests.
Runs on GPU when available, CPU as fallback.
"""

import os
import threading
import numpy as np

CLAP_MODEL_ID = "laion/larger_clap_music_and_speech"

# Point HF cache at network volume
MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/models")
HF_CACHE = os.path.join(MODEL_DIR, "huggingface")
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE)
WINDOW_SIZE = 1.0  # Score per 1-second window
SAMPLE_RATE = 48000


class ClapScorer:
    """CLAP audio scorer with lazy model loading and GPU support."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.lock = threading.Lock()

    def _ensure_loaded(self):
        """Load CLAP model on first use, move to GPU if available."""
        if self.model is not None:
            return

        import torch
        from transformers import ClapModel, ClapProcessor

        print(f"[ClapScorer] Loading CLAP model: {CLAP_MODEL_ID}")
        self.processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
        self.model = ClapModel.from_pretrained(CLAP_MODEL_ID)
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            self.device = "cuda"
            print(f"[ClapScorer] Model loaded on GPU ({torch.cuda.get_device_name(0)})")
        else:
            self.device = "cpu"
            print("[ClapScorer] Model loaded on CPU")

    def score(self, wav_path, queries):
        """
        Score audio against text queries using CLAP similarity.

        Args:
            wav_path: Path to WAV file (any sample rate, will be resampled)
            queries: Dict of {name: "natural language description"}

        Returns:
            Dict with per-second relevance scores per query, duration, model info.
            Returns None on failure.
        """
        with self.lock:
            try:
                self._ensure_loaded()
                return self._score_internal(wav_path, queries)
            except Exception as e:
                print(f"[ClapScorer] Scoring failed: {e}")
                return None

    def _score_internal(self, wav_path, queries):
        import torch
        import librosa

        # Load audio
        waveform, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        duration = len(waveform) / sr

        query_names = list(queries.keys())
        query_texts = list(queries.values())

        # Pre-compute text embeddings (batch all queries at once)
        text_inputs = self.processor(text=query_texts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        with torch.no_grad():
            text_out = self.model.get_text_features(**text_inputs)
            text_embeds = text_out.pooler_output if hasattr(text_out, "pooler_output") else text_out
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Score each 1-second window
        window_samples = int(WINDOW_SIZE * SAMPLE_RATE)
        n_windows = int(np.ceil(len(waveform) / window_samples))

        scores = {name: [] for name in query_names}

        for i in range(n_windows):
            start = i * window_samples
            end = min(start + window_samples, len(waveform))
            chunk = waveform[start:end]

            # Pad short chunks
            if len(chunk) < window_samples:
                chunk = np.pad(chunk, (0, window_samples - len(chunk)))

            # Get audio embedding
            audio_inputs = self.processor(audio=[chunk], sampling_rate=SAMPLE_RATE, return_tensors="pt")
            audio_inputs = {k: v.to(self.device) for k, v in audio_inputs.items()}

            with torch.no_grad():
                audio_out = self.model.get_audio_features(**audio_inputs)
                audio_embed = audio_out.pooler_output if hasattr(audio_out, "pooler_output") else audio_out
                audio_embed = audio_embed / audio_embed.norm(dim=-1, keepdim=True)

            # Compute cosine similarity with each query
            similarities = (audio_embed @ text_embeds.T).squeeze(0).cpu().numpy()

            # Map from cosine similarity [-1, 1] to relevance [0, 1]
            for j, name in enumerate(query_names):
                sim = similarities[j] if similarities.ndim > 0 else float(similarities)
                relevance = float(max(0, (sim + 1) / 2))
                scores[name].append(round(relevance, 4))

        return {
            "scores": scores,
            "duration": round(duration, 2),
            "model": CLAP_MODEL_ID,
            "device": self.device,
            "windowSize": WINDOW_SIZE,
        }

    @staticmethod
    def is_available():
        """Check if CLAP dependencies are installed."""
        try:
            import transformers
            import librosa
            return True
        except ImportError:
            return False
