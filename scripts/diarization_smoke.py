"""Run the opt-in diarization sidecar against one local audio file.

This is a development-only evidence helper. It loads a Hugging Face token from
an explicitly supplied dotenv file, never prints it, and writes a source-bound
JSON receipt suitable for comparing single- and multi-speaker controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from diarizer import SpeakerDiarizer  # noqa: E402


class DiarizationSmoke:
    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("--audio", required=True, type=Path)
        parser.add_argument("--output", required=True, type=Path)
        parser.add_argument("--dotenv", type=Path)
        parser.add_argument("--hf-home", type=Path)
        parser.add_argument("--min-speakers", type=int)
        parser.add_argument("--max-speakers", type=int)
        return parser.parse_args()

    @classmethod
    def run(cls) -> int:
        args = cls._parse_args()
        audio = args.audio.resolve()
        output = args.output.resolve()
        if not audio.is_file():
            raise FileNotFoundError(audio)

        if args.dotenv:
            load_dotenv(args.dotenv.resolve(), override=False)
        if args.hf_home:
            os.environ["HF_HOME"] = str(args.hf_home.resolve())
        os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")

        sidecar = SpeakerDiarizer().diarize(
            str(audio),
            [],
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )
        receipt: Dict[str, Any] = {
            "schema_version": "w2l-diarization-smoke-receipt-v1",
            "source": {
                "path": str(audio),
                "sha256": cls._sha256(audio),
                "bytes": audio.stat().st_size,
            },
            "sidecar": sidecar,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        print(
            json.dumps(
                {
                    "status": sidecar.get("status"),
                    "speaker_count": sidecar.get("speaker_count"),
                    "turn_count": len(sidecar.get("turns", [])),
                    "processing_sec": sidecar.get("processing_sec"),
                    "output": str(output),
                }
            ),
            flush=True,
        )
        return 0 if sidecar.get("status") == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(DiarizationSmoke.run())
