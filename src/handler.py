# RunPod Hub expects a file named handler.py
# This re-exports the actual handler from rp_handler.py
from rp_handler import run_whisper_job as handler  # noqa: F401
