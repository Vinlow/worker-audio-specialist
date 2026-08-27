"""Immutable model identities shared by build-time and runtime loaders."""


# These are the only Whisper variants accepted by the Web2Labs worker. Every
# entry is downloaded while building the image and reopened offline at runtime.
# Adding a model to the API therefore requires adding its reviewed Hub commit
# here; an explicit request must never turn into a mutable request-time fetch.
WHISPER_MODEL_REVISIONS = {
    "base": "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66",
    "large-v3": "edaa852ec7e145841d8ffdb056a99866b5f0a478",
    "medium": "08e178d48790749d25932bbc082711ddcfdfbc4f",
    "small": "536b0662742c02347bc0e980a01041f333bce120",
    "turbo": "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
}


CLAP_MODEL_ID = "laion/larger_clap_music_and_speech"
CLAP_MODEL_REVISION = "195c3a3e68faebb3e2088b9a79e79b43ddbda76b"


WAV2VEC2_CHECKPOINT_FILENAME = (
    "wav2vec2_fairseq_large_lv60k_asr_ls960.pth"
)
WAV2VEC2_CHECKPOINT_SHA256 = (
    "7a88965716fbd598a595209bf45c1210a18a6935cfb0cf53527fc986c5543ac7"
)


PYANNOTE_PIPELINE_REPO = "pyannote/speaker-diarization-3.1"
PYANNOTE_PIPELINE_REVISION = "84fd25912480287da0247647c3d2b4853cb3ee5d"
PYANNOTE_SEGMENTATION_REPO = "pyannote/segmentation-3.0"
PYANNOTE_SEGMENTATION_REVISION = "e66f3d3b9eb0873085418a7b813d3b369bf160bb"
PYANNOTE_EMBEDDING_REPO = "pyannote/wespeaker-voxceleb-resnet34-LM"
PYANNOTE_EMBEDDING_REVISION = "837717ddb9ff5507820346191109dc79c958d614"

# Exact files used by the offline pyannote 3.3.2 loader. Keep this manifest
# shared with the image builder so a runtime reference cannot drift away from
# the artifact that was actually baked.
PYANNOTE_SNAPSHOTS = {
    PYANNOTE_PIPELINE_REPO: {
        "revision": PYANNOTE_PIPELINE_REVISION,
        "allow_patterns": ("config.yaml",),
    },
    PYANNOTE_SEGMENTATION_REPO: {
        "revision": PYANNOTE_SEGMENTATION_REVISION,
        "allow_patterns": ("config.yaml", "pytorch_model.bin"),
    },
    PYANNOTE_EMBEDDING_REPO: {
        "revision": PYANNOTE_EMBEDDING_REVISION,
        "allow_patterns": ("config.yaml", "pytorch_model.bin"),
    },
}
