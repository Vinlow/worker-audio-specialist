"""Bounded RunPod adapter for the authenticated active-speaker v2 runtime.

The handler deliberately owns transport and process supervision only.  The
frozen scoring, media, lineage, and receipt contracts remain in
``active_speaker_runtime_v2.py`` and are executed in a fresh subprocess for
every request.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REQUEST_SCHEMA_VERSION = "starforge-active-speaker-runpod-request-v1"
RESPONSE_SCHEMA_VERSION = "starforge-active-speaker-runpod-response-v1"
BASE_IMAGE_ID = "sha256:973f44ba32c211a01f527a008fe8ec31bfc91a8c706d0e49e2ee2eb3f4b83690"
LRASD_REVISION = "1b6dcd2d8fc2895683de6508ec6294ec47d388ca"
LRASD_SOURCE_SHA256 = "89e4de74949aba7457b8254206885ea0646338c9d91a1e8556dbd3aebabd4eda"
EXPECTED_RUNTIME_IDENTITY = "sha256:24e157e4139dccc1db2c10f0185fa59aa4f361f830c0fa62ed3cd65c47c1b801"
CHECKPOINTS = {
    "AVA": {
        "path": "/opt/starforge-active-speaker/lrasd/weight/pretrain_AVA.model",
        "sha256": "85e6c77fc981595234790d1e128ebb60352d37726b2445e0ef8891e2512fe9e3",
    },
    "TALKSET": {
        "path": "/opt/starforge-active-speaker/lrasd/weight/finetuning_TalkSet.model",
        "sha256": "6b4ef53694e874e96cf630198dc479c78aebb3993bbf166aee3d926dfe7d9342",
    },
}
RUNTIME_PATH = Path("/opt/starforge-active-speaker/active_speaker_runtime_v2.py")
LRASD_ROOT = Path("/opt/starforge-active-speaker/lrasd")
MAX_URL_CHARS = 8_192
MAX_INPUT_BYTES = 2 * 1024 * 1024 * 1024
MAX_MANIFEST_BYTES = 32 * 1024 * 1024
MAX_OBSERVATION_BYTES = 32 * 1024 * 1024
MAX_DEADLINE_SECONDS = 900
MAX_FAILURE_TEXT_CHARS = 2_000
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ActiveSpeakerWorkerError(RuntimeError):
    """A fail-closed request, transport, or supervised-runtime failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class ArtifactRequest:
    name: str
    url: str
    sha256: str
    size: int
    maximum_size: int
    suffix: str


@dataclass(frozen=True)
class DownloadedArtifact:
    request: ArtifactRequest
    path: Path
    elapsed_ms: float


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_mapping(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ActiveSpeakerWorkerError("INVALID_REQUEST", f"{label} must be an object")
    received = set(value)
    if received != expected:
        raise ActiveSpeakerWorkerError(
            "INVALID_REQUEST",
            f"{label} keys differ: expected {sorted(expected)}, received {sorted(received)}",
        )
    return value


def _exact_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ActiveSpeakerWorkerError("INVALID_REQUEST", f"{label} must be a non-empty string")
    return value


def _exact_integer(value: Any, label: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ActiveSpeakerWorkerError("INVALID_REQUEST", f"{label} must be an integer")
    if value < minimum or value > maximum:
        raise ActiveSpeakerWorkerError(
            "INVALID_REQUEST", f"{label} must be between {minimum} and {maximum}"
        )
    return value


def _artifact_request(
    value: Any,
    *,
    name: str,
    maximum_size: int,
    suffix: str,
    allowed_hosts: frozenset[str],
) -> ArtifactRequest:
    artifact = _strict_mapping(value, {"bytes", "sha256", "url"}, name)
    url = _exact_string(artifact["url"], f"{name}.url")
    if len(url) > MAX_URL_CHARS or any(ord(character) < 32 for character in url):
        raise ActiveSpeakerWorkerError("INVALID_REQUEST", f"{name}.url is malformed or too long")
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise ActiveSpeakerWorkerError("INVALID_REQUEST", f"{name}.url is malformed") from error
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ActiveSpeakerWorkerError("INVALID_REQUEST", f"{name}.url must use HTTP or HTTPS")
    if parsed.username is not None or parsed.password is not None or parsed.fragment:
        raise ActiveSpeakerWorkerError(
            "INVALID_REQUEST", f"{name}.url must not contain credentials or a fragment"
        )
    if parsed.hostname.lower() not in allowed_hosts:
        raise ActiveSpeakerWorkerError(
            "ARTIFACT_HOST_FORBIDDEN", f"{name}.url host is not allowlisted"
        )
    if port is not None and not 0 < port <= 65_535:
        raise ActiveSpeakerWorkerError("INVALID_REQUEST", f"{name}.url has an invalid port")
    digest = _exact_string(artifact["sha256"], f"{name}.sha256")
    if HASH_PATTERN.fullmatch(digest) is None:
        raise ActiveSpeakerWorkerError(
            "INVALID_REQUEST", f"{name}.sha256 must be lowercase hexadecimal"
        )
    size = _exact_integer(
        artifact["bytes"], f"{name}.bytes", minimum=1, maximum=maximum_size
    )
    return ArtifactRequest(name, url, digest, size, maximum_size, suffix)


def _download_once(request: ArtifactRequest, directory: Path, timeout_seconds: int) -> DownloadedArtifact:
    started = time.monotonic()
    target = directory / f"{request.name}{request.suffix}"
    digest = hashlib.sha256()
    received = 0
    url_request = urllib.request.Request(
        request.url,
        headers={
            "Accept": "application/octet-stream,application/json,video/*",
            "User-Agent": "web2labs-starforge-active-speaker/1",
        },
    )
    try:
        with urllib.request.urlopen(url_request, timeout=timeout_seconds) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    advertised = int(content_length)
                except ValueError as error:
                    raise ActiveSpeakerWorkerError(
                        "DOWNLOAD_FAILED", f"{request.name} has an invalid Content-Length"
                    ) from error
                if advertised != request.size:
                    raise ActiveSpeakerWorkerError(
                        "ARTIFACT_SIZE_MISMATCH",
                        f"{request.name} advertised {advertised} bytes; expected {request.size}",
                    )
            with target.open("xb") as output:
                while True:
                    chunk = response.read(min(1024 * 1024, request.size - received + 1))
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > request.size or received > request.maximum_size:
                        raise ActiveSpeakerWorkerError(
                            "ARTIFACT_TOO_LARGE", f"{request.name} exceeds its bound"
                        )
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
    except ActiveSpeakerWorkerError:
        target.unlink(missing_ok=True)
        raise
    except Exception as error:
        target.unlink(missing_ok=True)
        raise ActiveSpeakerWorkerError(
            "DOWNLOAD_FAILED", f"{request.name} could not be fetched"
        ) from error
    if received != request.size:
        target.unlink(missing_ok=True)
        raise ActiveSpeakerWorkerError(
            "ARTIFACT_SIZE_MISMATCH",
            f"{request.name} received {received} bytes; expected {request.size}",
        )
    resolved_digest = digest.hexdigest()
    if resolved_digest != request.sha256:
        target.unlink(missing_ok=True)
        raise ActiveSpeakerWorkerError(
            "ARTIFACT_HASH_MISMATCH", f"{request.name} SHA-256 differs"
        )
    return DownloadedArtifact(request, target, (time.monotonic() - started) * 1000)


class ActiveSpeakerRunpodWorker:
    """Validate one request, fetch exact artifacts, and supervise v2 scoring."""

    def __init__(
        self,
        *,
        runtime_path: Path = RUNTIME_PATH,
        lrasd_root: Path = LRASD_ROOT,
        python_executable: str = sys.executable,
        allowed_artifact_hosts: set[str] | frozenset[str] | None = None,
    ) -> None:
        self.runtime_path = runtime_path
        self.lrasd_root = lrasd_root
        self.python_executable = python_executable
        configured_hosts = (
            allowed_artifact_hosts
            if allowed_artifact_hosts is not None
            else {
                host.strip().lower()
                for host in os.environ.get("STARFORGE_ARTIFACT_HOSTS", "").split(",")
                if host.strip()
            }
        )
        self.allowed_artifact_hosts = frozenset(configured_hosts)

    def handle(self, job: Any) -> dict[str, Any]:
        started = time.monotonic()
        if not isinstance(job, Mapping):
            raise ActiveSpeakerWorkerError("INVALID_REQUEST", "job must be an object")
        payload = self._validate_payload(job.get("input"))
        job_id = str(job.get("id") or "unknown")
        safe_job_id = re.sub(r"[^A-Za-z0-9._-]", "_", job_id)[:80] or "unknown"
        temporary_root = Path(tempfile.mkdtemp(prefix=f"starforge-asd-{safe_job_id}-"))
        try:
            downloads = self._download_inputs(payload, temporary_root)
            process_started = time.monotonic()
            result = self._run_scoring(payload, downloads, temporary_root)
            process_elapsed_ms = (time.monotonic() - process_started) * 1000
            return self._response(
                payload=payload,
                result=result,
                downloads=downloads,
                process_elapsed_ms=process_elapsed_ms,
                total_elapsed_ms=(time.monotonic() - started) * 1000,
            )
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)

    def _validate_payload(self, value: Any) -> dict[str, Any]:
        if not self.allowed_artifact_hosts:
            raise ActiveSpeakerWorkerError(
                "WORKER_MISCONFIGURED", "STARFORGE_ARTIFACT_HOSTS must be configured"
            )
        payload = _strict_mapping(
            value,
            {
                "audioStreamIndex",
                "baseObservation",
                "checkpoint",
                "deadlineSeconds",
                "inputVideo",
                "outputMode",
                "schemaVersion",
                "sourceIntervalEndUs",
                "sourceIntervalStartUs",
                "sourceVideo",
                "suppliedTracks",
                "videoStreamIndex",
            },
            "input",
        )
        if payload["schemaVersion"] != REQUEST_SCHEMA_VERSION:
            raise ActiveSpeakerWorkerError("INVALID_REQUEST", "unsupported schemaVersion")
        checkpoint = _exact_string(payload["checkpoint"], "checkpoint").upper()
        if checkpoint not in CHECKPOINTS:
            raise ActiveSpeakerWorkerError("INVALID_REQUEST", "checkpoint must be AVA or TALKSET")
        output_mode = _exact_string(payload["outputMode"], "outputMode").upper()
        if output_mode not in {"FULL_RESULT", "METRICS_ONLY"}:
            raise ActiveSpeakerWorkerError(
                "INVALID_REQUEST", "outputMode must be FULL_RESULT or METRICS_ONLY"
            )
        start_us = _exact_integer(
            payload["sourceIntervalStartUs"],
            "sourceIntervalStartUs",
            minimum=0,
            maximum=2**53 - 1,
        )
        end_us = _exact_integer(
            payload["sourceIntervalEndUs"],
            "sourceIntervalEndUs",
            minimum=1,
            maximum=2**53 - 1,
        )
        if end_us <= start_us:
            raise ActiveSpeakerWorkerError(
                "INVALID_REQUEST", "sourceIntervalEndUs must exceed sourceIntervalStartUs"
            )
        return {
            "schemaVersion": REQUEST_SCHEMA_VERSION,
            "inputVideo": _artifact_request(
                payload["inputVideo"],
                name="input-video",
                maximum_size=MAX_INPUT_BYTES,
                suffix=".mp4",
                allowed_hosts=self.allowed_artifact_hosts,
            ),
            "sourceVideo": _artifact_request(
                payload["sourceVideo"],
                name="source-video",
                maximum_size=MAX_INPUT_BYTES,
                suffix=".mp4",
                allowed_hosts=self.allowed_artifact_hosts,
            ),
            "suppliedTracks": _artifact_request(
                payload["suppliedTracks"],
                name="supplied-tracks",
                maximum_size=MAX_MANIFEST_BYTES,
                suffix=".json",
                allowed_hosts=self.allowed_artifact_hosts,
            ),
            "baseObservation": _artifact_request(
                payload["baseObservation"],
                name="base-observation",
                maximum_size=MAX_OBSERVATION_BYTES,
                suffix=".json",
                allowed_hosts=self.allowed_artifact_hosts,
            ),
            "checkpoint": checkpoint,
            "sourceIntervalStartUs": start_us,
            "sourceIntervalEndUs": end_us,
            "videoStreamIndex": _exact_integer(
                payload["videoStreamIndex"], "videoStreamIndex", minimum=0, maximum=64
            ),
            "audioStreamIndex": _exact_integer(
                payload["audioStreamIndex"], "audioStreamIndex", minimum=0, maximum=64
            ),
            "deadlineSeconds": _exact_integer(
                payload["deadlineSeconds"],
                "deadlineSeconds",
                minimum=1,
                maximum=MAX_DEADLINE_SECONDS,
            ),
            "outputMode": output_mode,
        }

    def _download_inputs(
        self, payload: Mapping[str, Any], directory: Path
    ) -> dict[str, DownloadedArtifact]:
        artifact_keys = ("inputVideo", "sourceVideo", "suppliedTracks", "baseObservation")
        timeout_seconds = min(120, int(payload["deadlineSeconds"]))
        with ThreadPoolExecutor(max_workers=len(artifact_keys)) as executor:
            futures = {
                key: executor.submit(_download_once, payload[key], directory, timeout_seconds)
                for key in artifact_keys
            }
            return {key: futures[key].result() for key in artifact_keys}

    def _run_scoring(
        self,
        payload: Mapping[str, Any],
        downloads: Mapping[str, DownloadedArtifact],
        directory: Path,
    ) -> Mapping[str, Any]:
        checkpoint = CHECKPOINTS[str(payload["checkpoint"])]
        output_directory = directory / "attempt"
        command = [
            self.python_executable,
            str(self.runtime_path),
            "run-supplied-v2",
            "--base-image-id",
            BASE_IMAGE_ID,
            "--lrasd-root",
            str(self.lrasd_root),
            "--lrasd-revision",
            LRASD_REVISION,
            "--lrasd-source-sha256",
            LRASD_SOURCE_SHA256,
            "--checkpoint",
            checkpoint["path"],
            "--checkpoint-sha256",
            checkpoint["sha256"],
            "--supplied-tracks",
            str(downloads["suppliedTracks"].path),
            "--supplied-tracks-sha256",
            downloads["suppliedTracks"].request.sha256,
            "--base-observation-result",
            str(downloads["baseObservation"].path),
            "--base-observation-result-sha256",
            downloads["baseObservation"].request.sha256,
            "--input-video",
            str(downloads["inputVideo"].path),
            "--input-sha256",
            downloads["inputVideo"].request.sha256,
            "--source-video",
            str(downloads["sourceVideo"].path),
            "--source-video-sha256",
            downloads["sourceVideo"].request.sha256,
            "--source-interval-start-us",
            str(payload["sourceIntervalStartUs"]),
            "--source-interval-end-us",
            str(payload["sourceIntervalEndUs"]),
            "--video-stream-index",
            str(payload["videoStreamIndex"]),
            "--audio-stream-index",
            str(payload["audioStreamIndex"]),
            "--output-dir",
            str(output_directory),
            "--device",
            "cuda",
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        try:
            stdout, stderr = process.communicate(timeout=int(payload["deadlineSeconds"]))
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            raise ActiveSpeakerWorkerError(
                "DEADLINE_EXCEEDED", "active-speaker subprocess exceeded its deadline"
            ) from error
        if process.returncode != 0:
            failure_text = (stderr or stdout or "no diagnostic output").strip()
            raise ActiveSpeakerWorkerError(
                "RUNTIME_FAILED", failure_text[-MAX_FAILURE_TEXT_CHARS:]
            )
        result_path = output_directory / "result.json"
        try:
            result_bytes = result_path.read_bytes()
            result = json.loads(result_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ActiveSpeakerWorkerError(
                "RUNTIME_RESULT_INVALID", "active-speaker result could not be authenticated"
            ) from error
        if not isinstance(result, Mapping):
            raise ActiveSpeakerWorkerError("RUNTIME_RESULT_INVALID", "result must be an object")
        identities = result.get("identities")
        measurements = result.get("measurements")
        if not isinstance(identities, Mapping) or not isinstance(measurements, Mapping):
            raise ActiveSpeakerWorkerError(
                "RUNTIME_RESULT_INVALID", "result lacks identities or measurements"
            )
        if identities.get("runtimeIdentity") != EXPECTED_RUNTIME_IDENTITY:
            raise ActiveSpeakerWorkerError(
                "RUNTIME_IDENTITY_MISMATCH", "runtime identity differs from checkpoint 31"
            )
        model = measurements.get("model")
        if not isinstance(model, Mapping) or model.get("device") != "cuda":
            raise ActiveSpeakerWorkerError(
                "CUDA_REQUIRED", "runtime did not authenticate CUDA execution"
            )
        return {**result, "_resultBytes": len(result_bytes), "_resultSha256": _sha256_bytes(result_bytes)}

    def _response(
        self,
        *,
        payload: Mapping[str, Any],
        result: Mapping[str, Any],
        downloads: Mapping[str, DownloadedArtifact],
        process_elapsed_ms: float,
        total_elapsed_ms: float,
    ) -> dict[str, Any]:
        result_without_private = {key: value for key, value in result.items() if not key.startswith("_")}
        measurements = result_without_private["measurements"]
        track_frames = int(measurements["scoredTrackFrameCount"])
        source_seconds = (
            int(payload["sourceIntervalEndUs"]) - int(payload["sourceIntervalStartUs"])
        ) / 1_000_000
        response: dict[str, Any] = {
            "schemaVersion": RESPONSE_SCHEMA_VERSION,
            "status": "COMPLETED",
            "authority": "DIAGNOSTIC_OBSERVATION_ONLY",
            "cropAuthority": "NONE",
            "worker": {
                "imageReleaseSha": os.environ.get(
                    "STARFORGE_ACTIVE_SPEAKER_RELEASE_SHA", "unknown"
                ),
                "starforgeSourceSha": os.environ.get(
                    "STARFORGE_ACTIVE_SPEAKER_SOURCE_SHA", "unknown"
                ),
                "baseAudioWorkerBuildSha": os.environ.get(
                    "AUDIO_WORKER_BUILD_SHA", "unknown"
                ),
                "baseImageId": BASE_IMAGE_ID,
                "runtimeIdentity": result_without_private["identities"]["runtimeIdentity"],
                "lrasdRevision": LRASD_REVISION,
                "lrasdSourceSha256": LRASD_SOURCE_SHA256,
            },
            "workload": {
                "checkpoint": payload["checkpoint"],
                "rawSourceSeconds": source_seconds,
                "preparedInputBytes": downloads["inputVideo"].request.size,
                "originalSourceBytes": downloads["sourceVideo"].request.size,
                "scoredTrackCount": int(measurements["scoredTrackCount"]),
                "scoredTrackFrames": track_frames,
                "scoredTrackSecondsAt25Fps": track_frames / 25,
                "viewCount": 2,
            },
            "timing": {
                "downloadMilliseconds": {
                    key: round(download.elapsed_ms, 3) for key, download in downloads.items()
                },
                "runtimeProcessMilliseconds": round(process_elapsed_ms, 3),
                "totalHandlerMilliseconds": round(total_elapsed_ms, 3),
                "runtimeStagesMilliseconds": measurements["stageMilliseconds"],
            },
            "result": {
                "bytes": result["_resultBytes"],
                "sha256": result["_resultSha256"],
                "identities": result_without_private["identities"],
                "scoreLedger": result_without_private["scoreLedger"],
                "scoreLedgerSha256": _sha256_bytes(
                    _canonical_json(result_without_private["scoreLedger"])
                ),
            },
        }
        response["responseIdentity"] = "sha256:" + _sha256_bytes(_canonical_json(response))
        if payload["outputMode"] == "FULL_RESULT":
            response["runtimeResult"] = result_without_private
        return response


_WORKER = ActiveSpeakerRunpodWorker()


def handler(job: Any) -> dict[str, Any]:
    """RunPod handler entry point."""
    return _WORKER.handle(job)


def main() -> None:
    import runpod

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
