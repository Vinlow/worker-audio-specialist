"""Bake the German CTC model after existing model layers; runtime stays offline."""
import skops.io as _skops_io  # noqa: F401 -- required Transformers import order
from huggingface_hub import snapshot_download

from german_alignment_model import GermanAlignmentModel


if __name__ == "__main__":
    snapshot_download(
        repo_id=GermanAlignmentModel.repository,
        revision=GermanAlignmentModel.revision,
        allow_patterns=list(GermanAlignmentModel.files),
    )
    model, extractor = GermanAlignmentModel.load()
    print(f"Baked and constructed German CTC: {GermanAlignmentModel.identity}")
