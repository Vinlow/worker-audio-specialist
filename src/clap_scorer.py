"""
CLAP audio-text similarity scoring module.

Uses CLAP (Contrastive Language-Audio Pretraining) to compute per-second
similarity scores between audio and natural language queries.

The model is loaded once and cached for subsequent requests.
Runs on GPU when available, CPU as fallback.

Performance: all audio windows are batched into a single GPU forward pass,
so scoring 120 windows takes ~2-3s on GPU instead of ~50s sequentially.
"""

import threading
import numpy as np

CLAP_MODEL_ID = "laion/larger_clap_music_and_speech"
WINDOW_SIZE = 1.0  # Score per 1-second window
SAMPLE_RATE = 48000

# Max windows per batch to avoid OOM on long audio.
# 300 windows = 5 minutes of audio. Beyond that, we chunk batches.
MAX_BATCH_SIZE = 300


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
            print("[ClapScorer] Model loaded on CPU (GPU not available)")

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
                return self._score_batched(wav_path, queries)
            except Exception as e:
                print(f"[ClapScorer] Scoring failed: {e}")
                import traceback
                traceback.print_exc()
                return None

    def _score_batched(self, wav_path, queries):
        """Batch all audio windows into a single GPU forward pass."""
        import torch
        import librosa

        # Load audio
        waveform, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        duration = len(waveform) / sr

        query_names = list(queries.keys())
        query_texts = list(queries.values())

        # Pre-compute text embeddings (small batch, very fast)
        text_inputs = self.processor(text=query_texts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        with torch.no_grad():
            text_out = self.model.get_text_features(**text_inputs)
            text_embeds = text_out if not hasattr(text_out, "pooler_output") else text_out.pooler_output
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Slice audio into 1-second windows
        window_samples = int(WINDOW_SIZE * SAMPLE_RATE)
        n_windows = int(np.ceil(len(waveform) / window_samples))

        chunks = []
        for i in range(n_windows):
            start = i * window_samples
            end = min(start + window_samples, len(waveform))
            chunk = waveform[start:end]
            if len(chunk) < window_samples:
                chunk = np.pad(chunk, (0, window_samples - len(chunk)))
            chunks.append(chunk)

        # Batch all windows through CLAP in one (or few) forward passes
        all_similarities = []

        for batch_start in range(0, len(chunks), MAX_BATCH_SIZE):
            batch_chunks = chunks[batch_start:batch_start + MAX_BATCH_SIZE]

            audio_inputs = self.processor(
                audio=batch_chunks,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            audio_inputs = {k: v.to(self.device) for k, v in audio_inputs.items()}

            with torch.no_grad():
                audio_out = self.model.get_audio_features(**audio_inputs)
                audio_embeds = audio_out if not hasattr(audio_out, "pooler_output") else audio_out.pooler_output
                audio_embeds = audio_embeds / audio_embeds.norm(dim=-1, keepdim=True)

                # (batch, queries) cosine similarity matrix
                sims = (audio_embeds @ text_embeds.T).cpu().numpy()
                all_similarities.append(sims)

        # Concatenate batches and map to per-query score lists
        all_sims = np.concatenate(all_similarities, axis=0)  # (n_windows, n_queries)

        scores = {}
        for j, name in enumerate(query_names):
            # Map cosine similarity [-1, 1] → relevance [0, 1]
            raw = all_sims[:, j]
            relevance = np.clip((raw + 1) / 2, 0, 1)
            scores[name] = [round(float(v), 4) for v in relevance]

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
