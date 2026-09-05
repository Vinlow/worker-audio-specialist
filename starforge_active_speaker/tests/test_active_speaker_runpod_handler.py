import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import sys

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

import active_speaker_runpod_handler as handler  # noqa: E402


def artifact(name: str, payload: bytes = b"exact-bytes") -> dict[str, object]:
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "url": f"https://example.test/{name}",
    }


def valid_payload() -> dict[str, object]:
    return {
        "schemaVersion": handler.REQUEST_SCHEMA_VERSION,
        "inputVideo": artifact("input.mp4"),
        "sourceVideo": artifact("source.mp4"),
        "suppliedTracks": artifact("tracks.json"),
        "baseObservation": artifact("observation.json"),
        "checkpoint": "AVA",
        "sourceIntervalStartUs": 10_000_000,
        "sourceIntervalEndUs": 23_000_000,
        "videoStreamIndex": 0,
        "audioStreamIndex": 1,
        "deadlineSeconds": 300,
        "outputMode": "METRICS_ONLY",
    }


class FakeResponse:
    def __init__(self, body: bytes, content_length: int | None = None) -> None:
        self.body = body
        self.offset = 0
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            result = self.body[self.offset :]
            self.offset = len(self.body)
            return result
        result = self.body[self.offset : self.offset + size]
        self.offset += len(result)
        return result


class FakeProcess:
    def __init__(self, returncode: int = 0, *, timeout: bool = False) -> None:
        self.returncode = returncode
        self.pid = 4321
        self.timeout = timeout
        self.calls = 0

    def communicate(self, timeout=None):
        self.calls += 1
        if self.timeout and self.calls == 1:
            raise subprocess.TimeoutExpired("runtime", timeout)
        return ("/tmp/result.json\n", "")


class ActiveSpeakerRunpodHandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.worker = handler.ActiveSpeakerRunpodWorker(
            runtime_path=Path("/runtime.py"),
            lrasd_root=Path("/lrasd"),
            python_executable="python",
            allowed_artifact_hosts={"example.test"},
        )

    def test_request_contract_is_strict_and_normalizes_enums(self):
        payload = valid_payload()
        payload["checkpoint"] = "talkset"
        payload["outputMode"] = "full_result"
        validated = self.worker._validate_payload(payload)
        self.assertEqual(validated["checkpoint"], "TALKSET")
        self.assertEqual(validated["outputMode"], "FULL_RESULT")

        with self.assertRaisesRegex(handler.ActiveSpeakerWorkerError, "keys differ"):
            self.worker._validate_payload({**payload, "unexpected": True})

    def test_request_rejects_credentialed_or_non_http_urls(self):
        for url in (
            "file:///tmp/input.mp4",
            "https://user:secret@example.test/input.mp4",
            "https://example.test/input.mp4#fragment",
        ):
            payload = valid_payload()
            payload["inputVideo"] = {**artifact("input.mp4"), "url": url}
            with self.assertRaises(handler.ActiveSpeakerWorkerError):
                self.worker._validate_payload(payload)

    def test_request_rejects_artifact_hosts_outside_worker_allowlist(self):
        payload = valid_payload()
        payload["inputVideo"] = {
            **artifact("input.mp4"),
            "url": "https://metadata.internal/input.mp4",
        }
        with self.assertRaisesRegex(handler.ActiveSpeakerWorkerError, "allowlisted"):
            self.worker._validate_payload(payload)

    def test_worker_refuses_requests_when_host_allowlist_is_not_configured(self):
        worker = handler.ActiveSpeakerRunpodWorker(allowed_artifact_hosts=set())
        with self.assertRaisesRegex(handler.ActiveSpeakerWorkerError, "must be configured"):
            worker._validate_payload(valid_payload())

    def test_request_rejects_invalid_bounds_and_interval(self):
        for overrides in (
            {"deadlineSeconds": 901},
            {"sourceIntervalEndUs": 10_000_000},
            {"videoStreamIndex": -1},
        ):
            payload = {**valid_payload(), **overrides}
            with self.assertRaises(handler.ActiveSpeakerWorkerError):
                self.worker._validate_payload(payload)

    def test_download_is_single_attempt_and_authenticates_bytes(self):
        payload = b"exact-bytes"
        request = handler.ArtifactRequest(
            "fixture",
            "https://example.test/fixture.json",
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            100,
            ".json",
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            handler.urllib.request,
            "urlopen",
            return_value=FakeResponse(payload, len(payload)),
        ) as urlopen:
            result = handler._download_once(request, Path(directory), 5)
            self.assertEqual(result.path.read_bytes(), payload)
            self.assertEqual(urlopen.call_count, 1)

    def test_download_rejects_advertised_size_before_reading(self):
        payload = b"exact-bytes"
        request = handler.ArtifactRequest(
            "fixture",
            "https://example.test/fixture.json",
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            100,
            ".json",
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            handler.urllib.request,
            "urlopen",
            return_value=FakeResponse(payload, len(payload) + 1),
        ):
            with self.assertRaisesRegex(handler.ActiveSpeakerWorkerError, "advertised"):
                handler._download_once(request, Path(directory), 5)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_supervisor_kills_the_process_group_on_deadline(self):
        payload = self.worker._validate_payload(valid_payload())
        downloads = self._download_records(payload)
        process = FakeProcess(timeout=True)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            handler.subprocess, "Popen", return_value=process
        ), mock.patch.object(handler.os, "killpg") as killpg:
            with self.assertRaisesRegex(handler.ActiveSpeakerWorkerError, "DEADLINE_EXCEEDED"):
                self.worker._run_scoring(payload, downloads, Path(directory))
        killpg.assert_called_once_with(process.pid, handler.signal.SIGKILL)
        self.assertEqual(process.calls, 2)

    def test_runtime_must_authenticate_cuda_and_checkpoint31_identity(self):
        for runtime_identity, device, expected_code in (
            ("sha256:" + "0" * 64, "cuda", "RUNTIME_IDENTITY_MISMATCH"),
            (handler.EXPECTED_RUNTIME_IDENTITY, "cpu", "CUDA_REQUIRED"),
        ):
            with self.subTest(expected_code=expected_code):
                payload = self.worker._validate_payload(valid_payload())
                downloads = self._download_records(payload)
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "attempt"
                    output.mkdir()
                    (output / "result.json").write_text(
                        json.dumps(
                            {
                                "identities": {"runtimeIdentity": runtime_identity},
                                "measurements": {"model": {"device": device}},
                            }
                        ),
                        encoding="utf-8",
                    )
                    with mock.patch.object(
                        handler.subprocess, "Popen", return_value=FakeProcess()
                    ):
                        with self.assertRaises(handler.ActiveSpeakerWorkerError) as raised:
                            self.worker._run_scoring(payload, downloads, Path(directory))
                self.assertEqual(raised.exception.code, expected_code)

    def test_metrics_response_separates_raw_duration_and_scored_face_time(self):
        payload = self.worker._validate_payload(valid_payload())
        downloads = self._download_records(payload)
        result = {
            "_resultBytes": 123,
            "_resultSha256": "f" * 64,
            "identities": {"runtimeIdentity": handler.EXPECTED_RUNTIME_IDENTITY},
            "measurements": {
                "model": {"device": "cuda"},
                "scoredTrackCount": 8,
                "scoredTrackFrameCount": 527,
                "stageMilliseconds": {"lrasdTwoViewInference": 42.0},
            },
            "scoreLedger": [{"trackId": "track-1"}],
        }
        response = self.worker._response(
            payload=payload,
            result=result,
            downloads=downloads,
            process_elapsed_ms=100.0,
            total_elapsed_ms=125.0,
        )
        self.assertEqual(response["workload"]["rawSourceSeconds"], 13.0)
        self.assertEqual(response["workload"]["scoredTrackSecondsAt25Fps"], 21.08)
        self.assertEqual(response["workload"]["viewCount"], 2)
        self.assertEqual(response["result"]["scoreLedger"], result["scoreLedger"])
        self.assertNotIn("runtimeResult", response)

    @staticmethod
    def _download_records(payload):
        return {
            key: handler.DownloadedArtifact(payload[key], Path(f"/{key}"), 1.0)
            for key in ("inputVideo", "sourceVideo", "suppliedTracks", "baseObservation")
        }


if __name__ == "__main__":
    unittest.main()
