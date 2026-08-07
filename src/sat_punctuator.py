"""Experimental source-token-bound SaT punctuation window probe.

Ordinary worker requests do not touch this component. The caller must send one
bounded, already-tokenized source window plus exact model and tokenizer
identity. The probe returns diagnostic terminal-boundary probabilities only;
it never owns transcript text, word geometry, Natural Landing, or cuts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import struct
import threading
import time

from model_load_lock import serialized_model_load


SAT_REQUEST_SCHEMA_VERSION = "w2l-sat-punctuation-window-request-v1"
SAT_RESPONSE_SCHEMA_VERSION = "w2l-sat-punctuation-window-probe-v1"
SAT_BATCH_REQUEST_SCHEMA_VERSION = (
    "w2l-sat-punctuation-batch-request-v1"
)
SAT_BATCH_RESPONSE_SCHEMA_VERSION = (
    "w2l-sat-punctuation-batch-probe-v1"
)
SAT_MODEL_ID = "segment-any-text/sat-3l-sm"
SAT_MODEL_REVISION = "137da054051ad9f1eac42025f758db4ac9f22535"
SAT_TOKENIZER_ID = "FacebookAI/xlm-roberta-base"
SAT_TOKENIZER_REVISION = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
SAT_WTPSPLIT_VERSION = "2.2.1"
SAT_SKOPS_VERSION = "0.14.0"
SAT_TRANSFORMERS_VERSION = "5.9.0"
SAT_BOUNDARY_THRESHOLD = 0.65
SAT_MAX_LENGTH_TOKENS = 512
SAT_CONTENT_WINDOW_TOKENS = 510
SAT_STRIDE_TOKENS = 64
SAT_PADDED_BATCH_SIZE = 8
SAT_LAUNCH_LANGUAGES = frozenset({"en", "de", "fr", "es", "pt", "it"})
SAT_SOURCE_CONTRACT_KIND = (
    "w2l-worker-audio-specialist-sat-punctuation-source-contract-v2"
)
SAT_SOURCE_CONTRACT_ROUTE = "sat_punctuation_batch_probe"
SAT_SOURCE_CONTRACT_FILES = {
    "src/predict.py": "predict.py",
    "src/rp_handler.py": "rp_handler.py",
    "src/sat_punctuator.py": "sat_punctuator.py",
}
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def token_bytes(token_ids) -> bytes:
    payload = bytearray()
    for token_id in token_ids:
        if (
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
            or token_id > 0xFFFFFFFF
        ):
            raise ValueError(f"invalid tokenizer id: {token_id!r}")
        payload.extend(struct.pack(">I", token_id))
    return bytes(payload)


def token_sha256(token_ids) -> str:
    return hashlib.sha256(token_bytes(token_ids)).hexdigest()


def normalize_language(value) -> str:
    if not isinstance(value, str):
        raise ValueError("SaT language must be a string")
    normalized = value.strip().lower().replace("_", "-").split("-", 1)[0]
    if normalized not in SAT_LAUNCH_LANGUAGES:
        raise ValueError(
            f"SaT punctuation probe does not support language {normalized!r}"
        )
    return normalized


def validate_probe_request(value) -> dict:
    if not isinstance(value, dict):
        raise ValueError("sat_punctuation_probe must be an object")
    if value.get("schemaVersion") != SAT_REQUEST_SCHEMA_VERSION:
        raise ValueError("unsupported SaT punctuation request schema")

    source_fingerprint = value.get("sourceFingerprint")
    if (
        not isinstance(source_fingerprint, str)
        or not SHA256_HEX.fullmatch(source_fingerprint)
    ):
        raise ValueError("SaT source fingerprint must be lowercase sha256")
    language = normalize_language(value.get("language"))

    candidate = value.get("candidate")
    expected_candidate = {
        "modelId": SAT_MODEL_ID,
        "modelRevision": SAT_MODEL_REVISION,
        "tokenizerId": SAT_TOKENIZER_ID,
        "tokenizerRevision": SAT_TOKENIZER_REVISION,
        "boundaryThreshold": SAT_BOUNDARY_THRESHOLD,
    }
    if candidate != expected_candidate:
        raise ValueError("SaT punctuation candidate identity drifted")

    window = value.get("window")
    if not isinstance(window, dict):
        raise ValueError("SaT punctuation request has no window")
    start_token = window.get("startToken")
    end_token = window.get("endTokenExclusive")
    terminal_tail = window.get("terminalTail")
    if (
        not isinstance(start_token, int)
        or isinstance(start_token, bool)
        or start_token < 0
        or not isinstance(end_token, int)
        or isinstance(end_token, bool)
        or end_token <= start_token
        or not isinstance(terminal_tail, bool)
    ):
        raise ValueError("SaT punctuation window geometry is invalid")

    input_token_ids = value.get("inputTokenIds")
    if (
        not isinstance(input_token_ids, list)
        or not input_token_ids
        or len(input_token_ids) > SAT_CONTENT_WINDOW_TOKENS
    ):
        raise ValueError("SaT punctuation token window is empty or oversized")
    # Validate the original values before int conversion so True cannot become
    # token id 1.
    token_bytes(input_token_ids)
    normalized_ids = [int(token_id) for token_id in input_token_ids]
    if end_token - start_token != len(normalized_ids):
        raise ValueError("SaT punctuation token/window cardinality drifted")
    if (
        not terminal_tail
        and (
            len(normalized_ids) != SAT_CONTENT_WINDOW_TOKENS
            or start_token % SAT_STRIDE_TOKENS != 0
        )
    ):
        raise ValueError(
            "SaT complete window must be 510 tokens on the 64-token grid"
        )
    input_token_sha = value.get("inputTokenSha256")
    if input_token_sha != token_sha256(normalized_ids):
        raise ValueError("SaT punctuation input token identity drifted")

    anchors = value.get("terminalAnchors")
    if not isinstance(anchors, list) or not anchors:
        raise ValueError(
            "SaT punctuation terminal anchors must be a non-empty array"
        )
    normalized_anchors = []
    observed_ordinals = set()
    previous_ordinal = -1
    previous_terminal = start_token - 1
    for anchor in anchors:
        if not isinstance(anchor, dict):
            raise ValueError("SaT punctuation terminal anchor is not an object")
        ordinal = anchor.get("ordinal")
        terminal = anchor.get("terminalTokenIndex")
        if (
            not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or ordinal < 0
            or ordinal <= previous_ordinal
            or ordinal in observed_ordinals
            or not isinstance(terminal, int)
            or isinstance(terminal, bool)
            or terminal <= previous_terminal
            or not start_token <= terminal < end_token
        ):
            raise ValueError("SaT punctuation terminal anchor is invalid")
        observed_ordinals.add(ordinal)
        previous_ordinal = ordinal
        previous_terminal = terminal
        normalized_anchors.append(
            {
                "ordinal": ordinal,
                "terminalTokenIndex": terminal,
            }
        )

    return {
        "schemaVersion": SAT_REQUEST_SCHEMA_VERSION,
        "sourceFingerprint": source_fingerprint,
        "language": language,
        "candidate": expected_candidate,
        "window": {
            "startToken": start_token,
            "endTokenExclusive": end_token,
            "terminalTail": terminal_tail,
        },
        "inputTokenIds": normalized_ids,
        "inputTokenSha256": input_token_sha,
        "terminalAnchors": normalized_anchors,
    }


def validate_batch_request(value) -> dict:
    if not isinstance(value, dict):
        raise ValueError("sat_punctuation_batch_probe must be an object")
    if value.get("schemaVersion") != SAT_BATCH_REQUEST_SCHEMA_VERSION:
        raise ValueError("unsupported SaT punctuation batch request schema")
    windows = value.get("windows")
    if (
        not isinstance(windows, list)
        or not windows
        or len(windows) > SAT_PADDED_BATCH_SIZE
    ):
        raise ValueError(
            "SaT punctuation batch must contain between one and "
            f"{SAT_PADDED_BATCH_SIZE} windows"
        )

    normalized = []
    previous_start = -1
    previous_complete_start = None
    observed_ordinals = set()
    previous_global_ordinal = -1
    terminal_tail_count = 0
    for batch_index, window_payload in enumerate(windows):
        if not isinstance(window_payload, dict):
            raise ValueError(
                f"SaT punctuation batch window {batch_index} is not an object"
            )
        unexpected_keys = set(window_payload) - {
            "window",
            "inputTokenIds",
            "inputTokenSha256",
            "terminalAnchors",
        }
        if unexpected_keys:
            raise ValueError(
                "SaT punctuation batch window contains unexpected keys: "
                + ", ".join(sorted(str(key) for key in unexpected_keys))
            )
        single = validate_probe_request(
            {
                **window_payload,
                "schemaVersion": SAT_REQUEST_SCHEMA_VERSION,
                "sourceFingerprint": value.get("sourceFingerprint"),
                "language": value.get("language"),
                "candidate": value.get("candidate"),
            }
        )
        start = single["window"]["startToken"]
        if start <= previous_start:
            raise ValueError(
                "SaT punctuation batch windows must have strictly "
                "increasing start tokens"
            )
        previous_start = start
        is_tail = single["window"]["terminalTail"]
        if is_tail:
            terminal_tail_count += 1
            if batch_index != len(windows) - 1:
                raise ValueError(
                    "SaT punctuation terminal tail must be the final "
                    "batch window"
                )
        else:
            if (
                previous_complete_start is not None
                and start != previous_complete_start + SAT_STRIDE_TOKENS
            ):
                raise ValueError(
                    "SaT punctuation complete batch windows must be "
                    "consecutive on the source grid"
                )
            previous_complete_start = start
        for anchor in single["terminalAnchors"]:
            ordinal = anchor["ordinal"]
            if ordinal in observed_ordinals:
                raise ValueError(
                    "SaT punctuation batch contains a duplicate "
                    f"terminal ordinal: {ordinal}"
                )
            if ordinal <= previous_global_ordinal:
                raise ValueError(
                    "SaT punctuation batch terminal ordinals must follow "
                    "global source order"
                )
            observed_ordinals.add(ordinal)
            previous_global_ordinal = ordinal
        normalized.append(single)
    if terminal_tail_count > 1:
        raise ValueError(
            "SaT punctuation batch accepts at most one terminal tail"
        )

    return {
        "schemaVersion": SAT_BATCH_REQUEST_SCHEMA_VERSION,
        "sourceFingerprint": normalized[0]["sourceFingerprint"],
        "language": normalized[0]["language"],
        "candidate": normalized[0]["candidate"],
        "windows": normalized,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_contract_id(source_root=None) -> str:
    """Bind the deployed SaT route to the exact executable source bytes.

    The manifest deliberately excludes a Git SHA: embedding a commit that
    contains its own expected commit is recursive. The caller pins this
    content ID, which covers the route, schemas, and every source file that
    can construct or serve the probe.
    """
    root = (
        Path(source_root)
        if source_root is not None
        else Path(__file__).resolve().parent
    )
    body = {
        "kind": SAT_SOURCE_CONTRACT_KIND,
        "route": SAT_SOURCE_CONTRACT_ROUTE,
        "requestSchema": SAT_BATCH_REQUEST_SCHEMA_VERSION,
        "batchResponseSchema": SAT_BATCH_RESPONSE_SCHEMA_VERSION,
        "windowResponseSchema": SAT_RESPONSE_SCHEMA_VERSION,
        "sourceBlobs": {
            logical_path: sha256_file(root / runtime_path)
            for logical_path, runtime_path in sorted(
                SAT_SOURCE_CONTRACT_FILES.items()
            )
        },
    }
    return "sha256:" + hashlib.sha256(
        canonical_json(body).encode("utf-8")
    ).hexdigest()


def snapshot_identity(path: Path, allowed_paths) -> dict:
    files = []
    for relative_path in sorted(allowed_paths):
        file_path = path / relative_path
        if not file_path.is_file():
            continue
        files.append(
            {
                "path": relative_path,
                "bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    if not files:
        raise ValueError(f"SaT snapshot has no expected files: {path}")
    return {
        "files": files,
        "fileCount": len(files),
        "totalBytes": sum(item["bytes"] for item in files),
        "manifestSha256": hashlib.sha256(
            canonical_json(files).encode("utf-8")
        ).hexdigest(),
    }


class SaTPunctuator:
    """Lazy, serialized SaT loader with single-flight window inference."""

    def __init__(self):
        self.sat = None
        self.device = None
        self.snapshot = None
        self.model_dtype = None
        self.load_seconds = None
        self.source_contract_id = source_contract_id()
        self._setup_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def setup(self):
        if self.sat is not None:
            return
        with self._setup_lock:
            if self.sat is not None:
                return
            with serialized_model_load("sat-punctuation-probe"):
                if self.sat is not None:
                    return
                import torch
                import skops
                import wtpsplit
                import transformers
                from huggingface_hub import snapshot_download
                from wtpsplit import SaT

                if skops.__version__ != SAT_SKOPS_VERSION:
                    raise RuntimeError(
                        "SaT runtime requires skops "
                        f"{SAT_SKOPS_VERSION}, found {skops.__version__}"
                    )
                if wtpsplit.__version__ != SAT_WTPSPLIT_VERSION:
                    raise RuntimeError(
                        "SaT runtime requires wtpsplit "
                        f"{SAT_WTPSPLIT_VERSION}, found {wtpsplit.__version__}"
                    )
                if transformers.__version__ != SAT_TRANSFORMERS_VERSION:
                    raise RuntimeError(
                        "SaT runtime requires transformers "
                        f"{SAT_TRANSFORMERS_VERSION}, found "
                        f"{transformers.__version__}"
                    )
                started = time.perf_counter()
                model_snapshot = Path(
                    snapshot_download(
                        repo_id=SAT_MODEL_ID,
                        revision=SAT_MODEL_REVISION,
                        allow_patterns=["config.json", "model.safetensors"],
                        local_files_only=True,
                    )
                )
                tokenizer_snapshot = Path(
                    snapshot_download(
                        repo_id=SAT_TOKENIZER_ID,
                        revision=SAT_TOKENIZER_REVISION,
                        allow_patterns=[
                            "config.json",
                            "sentencepiece.bpe.model",
                            "special_tokens_map.json",
                            "tokenizer.json",
                            "tokenizer_config.json",
                        ],
                        local_files_only=True,
                    )
                )
                sat = SaT(
                    str(model_snapshot),
                    tokenizer_name_or_path=str(tokenizer_snapshot),
                    from_pretrained_kwargs={"local_files_only": True},
                )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                sat.model.to(device)
                sat.model.eval()
                model_dtype = str(next(sat.model.model.parameters()).dtype)
                self.snapshot = {
                    "model": snapshot_identity(
                        model_snapshot,
                        ["config.json", "model.safetensors"],
                    ),
                    "tokenizer": snapshot_identity(
                        tokenizer_snapshot,
                        [
                            "config.json",
                            "sentencepiece.bpe.model",
                            "special_tokens_map.json",
                            "tokenizer.json",
                            "tokenizer_config.json",
                        ],
                    ),
                }
                self.sat = sat
                self.device = device
                self.model_dtype = model_dtype
                self.load_seconds = time.perf_counter() - started
                print(
                    "[SaTPunctuation] Loaded pinned model "
                    f"{SAT_MODEL_ID}@{SAT_MODEL_REVISION} on {device}",
                    flush=True,
                )

    def _authority(self) -> dict:
        return {
            "punctuation": False,
            "transcript": False,
            "wordGeometry": False,
            "naturalLanding": False,
            "npSbv2": False,
            "cut": False,
            "production": False,
        }

    def _implementation(self) -> dict:
        return {
            "wtpsplitVersion": SAT_WTPSPLIT_VERSION,
            "skopsVersion": SAT_SKOPS_VERSION,
            "transformersVersion": SAT_TRANSFORMERS_VERSION,
            "snapshot": self.snapshot,
            "languageAuthority": "CALLER_ASSERTED_NOT_MODEL_VERIFIED",
            "sourceContractId": self.source_contract_id,
        }

    def _infer_normalized_windows(self, normalized_windows):
        if (
            not normalized_windows
            or len(normalized_windows) > SAT_PADDED_BATCH_SIZE
        ):
            raise ValueError(
                "SaT inference requires between one and "
                f"{SAT_PADDED_BATCH_SIZE} normalized windows"
            )
        with self._inference_lock:
            self.setup()
            import numpy as np
            import torch
            from wtpsplit.utils import Constants

            cls_token_id = self.sat.tokenizer.cls_token_id
            sep_token_id = self.sat.tokenizer.sep_token_id
            pad_token_id = self.sat.tokenizer.pad_token_id
            if None in (cls_token_id, sep_token_id, pad_token_id):
                raise RuntimeError("SaT tokenizer is missing CLS, SEP, or PAD")

            batch_input_ids = np.full(
                (SAT_PADDED_BATCH_SIZE, SAT_MAX_LENGTH_TOKENS),
                int(pad_token_id),
                dtype=np.int64,
            )
            batch_attention_mask = np.zeros(
                (SAT_PADDED_BATCH_SIZE, SAT_MAX_LENGTH_TOKENS),
                dtype=np.float32,
            )
            for batch_index, normalized in enumerate(normalized_windows):
                input_token_ids = normalized["inputTokenIds"]
                if max(input_token_ids) >= self.sat.tokenizer.vocab_size:
                    raise ValueError(
                        "SaT punctuation token id exceeds vocabulary"
                    )
                batch_input_ids[batch_index, 0] = int(cls_token_id)
                batch_input_ids[
                    batch_index,
                    1 : 1 + len(input_token_ids),
                ] = np.asarray(input_token_ids, dtype=np.int64)
                batch_input_ids[
                    batch_index,
                    1 + len(input_token_ids),
                ] = int(sep_token_id)
                batch_attention_mask[
                    batch_index,
                    : len(input_token_ids) + 2,
                ] = 1.0

            if self.device == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
            started = time.perf_counter()
            logits = self.sat.model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
            )["logits"]
            if self.device == "cuda":
                torch.cuda.synchronize(self.device)
            inference_seconds = time.perf_counter() - started

            window_results = []
            for batch_index, normalized in enumerate(normalized_windows):
                input_token_ids = normalized["inputTokenIds"]
                token_logits = logits[
                    batch_index,
                    1 : 1 + len(input_token_ids),
                    Constants.NEWLINE_INDEX,
                ].astype(np.float64)
                probabilities = 1.0 / (
                    1.0 + np.exp(-np.clip(token_logits, -30.0, 30.0))
                )
                start_token = normalized["window"]["startToken"]
                rows = []
                for anchor in normalized["terminalAnchors"]:
                    local_index = (
                        anchor["terminalTokenIndex"] - start_token
                    )
                    probability = float(probabilities[local_index])
                    rows.append(
                        {
                            **anchor,
                            "localTokenIndex": local_index,
                            "terminalProbability": probability,
                            "rawModelLabel": (
                                "PERIOD"
                                if probability > SAT_BOUNDARY_THRESHOLD
                                else "NONE"
                            ),
                        }
                    )

                identity_body = {
                    "schemaVersion": SAT_RESPONSE_SCHEMA_VERSION,
                    "sourceFingerprint": normalized[
                        "sourceFingerprint"
                    ],
                    "language": normalized["language"],
                    "candidate": normalized["candidate"],
                    "window": normalized["window"],
                    "inputTokenSha256": normalized[
                        "inputTokenSha256"
                    ],
                    "terminalAnchors": normalized["terminalAnchors"],
                }
                window_results.append(
                    {
                        **identity_body,
                        "windowIdentity": "sha256:"
                        + hashlib.sha256(
                            canonical_json(identity_body).encode("utf-8")
                        ).hexdigest(),
                        "rows": rows,
                    }
                )

            runtime = {
                "device": self.device,
                "modelDtype": self.model_dtype,
                "loadSeconds": self.load_seconds,
                "inferenceSeconds": inference_seconds,
                "cudaPeakAllocatedBytes": (
                    int(torch.cuda.max_memory_allocated(self.device))
                    if self.device == "cuda"
                    else None
                ),
                "inputWindowCount": len(normalized_windows),
                "paddedBatchSize": SAT_PADDED_BATCH_SIZE,
                "maxLengthTokens": SAT_MAX_LENGTH_TOKENS,
            }
            return window_results, runtime

    def infer_window(self, request) -> dict:
        normalized = validate_probe_request(request)
        windows, runtime = self._infer_normalized_windows([normalized])
        return {
            **windows[0],
            "runtime": runtime,
            "implementation": self._implementation(),
            "authority": self._authority(),
        }

    def infer_batch(self, request) -> dict:
        normalized = validate_batch_request(request)
        windows, runtime = self._infer_normalized_windows(
            normalized["windows"]
        )
        identity_body = {
            "schemaVersion": SAT_BATCH_RESPONSE_SCHEMA_VERSION,
            "sourceFingerprint": normalized["sourceFingerprint"],
            "language": normalized["language"],
            "candidate": normalized["candidate"],
            "windowIdentities": [
                window["windowIdentity"]
                for window in windows
            ],
        }
        return {
            **identity_body,
            "batchIdentity": "sha256:"
            + hashlib.sha256(
                canonical_json(identity_body).encode("utf-8")
            ).hexdigest(),
            "windows": windows,
            "runtime": runtime,
            "implementation": self._implementation(),
            "authority": self._authority(),
        }
