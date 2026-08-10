#!/usr/bin/env python3
"""Safely roll an immutable holy-grale image onto the RunPod test endpoint.

The script is deliberately dry-run by default. It updates the endpoint-bound
template through RunPod's REST API only after verifying every fixed identity,
name, registry credential, and exclusive endpoint/template binding below.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any


RUNPOD_API_BASE = "https://rest.runpod.io/v1"
RUNPOD_API_KEY_ENV = "RUNPOD_API_KEY"

TEMPLATE_ID = "3aapcopikw"
TEMPLATE_NAME = "worker-audio-expert-template"
ENDPOINT_ID = "dx99xymo20v3o9"
ENDPOINT_NAME = "worker-audio-expert"
REGISTRY_AUTH_ID = "cmnhowndh00b5l707vr072ars"
REGISTRY_AUTH_NAME = "GitHub All"
TEMPLATE_READ_PATH = (
    f"/templates/{TEMPLATE_ID}?includeEndpointBoundTemplates=true"
)

IMAGE_REPOSITORY = "ghcr.io/vinlow/worker-audio-specialist"
IMAGE_PATTERN = re.compile(
    rf"^{re.escape(IMAGE_REPOSITORY)}@sha256:[0-9a-f]{{64}}$"
)
MAX_RESPONSE_BYTES = 1024 * 1024

# Values compared before and after a mutation. Secret environment values are
# compared in memory but are never included in output or exception messages.
PRESERVED_TEMPLATE_FIELDS = (
    "category",
    "containerDiskInGb",
    "dockerEntrypoint",
    "dockerStartCmd",
    "env",
    "isPublic",
    "isRunpod",
    "isServerless",
    "name",
    "ports",
    "readme",
    "startSsh",
    "volumeInGb",
    "volumeMountPath",
)


class DeploymentError(RuntimeError):
    """A safe-to-print deployment validation or API error."""


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Prevent an authorization header from following a redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


RequestJson = Callable[[str, str, Mapping[str, Any] | None], Any]


class RunPodClient:
    """Minimal client for the official RunPod REST resources used here."""

    def __init__(self, api_key: str, *, timeout_seconds: float = 20.0):
        if not isinstance(api_key, str) or not api_key.strip():
            raise DeploymentError(
                f"{RUNPOD_API_KEY_ENV} must be set in the environment"
            )
        self._api_key = api_key.strip()
        self._timeout_seconds = timeout_seconds
        self._opener = urllib.request.build_opener(_NoRedirectHandler())

    def request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        if not path.startswith("/") or "://" in path:
            raise DeploymentError("refusing a non-relative RunPod API path")
        encoded_payload = None
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        if payload is not None:
            encoded_payload = json.dumps(payload, separators=(",", ":")).encode(
                "utf-8"
            )
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            RUNPOD_API_BASE + path,
            data=encoded_payload,
            headers=headers,
            method=method,
        )
        try:
            with self._opener.open(
                request,
                timeout=self._timeout_seconds,
            ) as response:
                content_length = response.headers.get("Content-Length")
                if content_length:
                    try:
                        if int(content_length) > MAX_RESPONSE_BYTES:
                            raise DeploymentError(
                                "RunPod response exceeded the safety limit"
                            )
                    except ValueError as error:
                        raise DeploymentError(
                            "RunPod returned an invalid Content-Length"
                        ) from error
                body = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            # Never echo the response body: validation errors can contain
            # submitted fields and template environment values.
            raise DeploymentError(
                f"RunPod {method} {path} returned HTTP {error.code}"
            ) from None
        except urllib.error.URLError:
            raise DeploymentError(
                f"RunPod {method} {path} could not be reached"
            ) from None

        if len(body) > MAX_RESPONSE_BYTES:
            raise DeploymentError("RunPod response exceeded the safety limit")
        try:
            return json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise DeploymentError(
                f"RunPod {method} {path} returned invalid JSON"
            ) from None


class HolyGraleTestDeployer:
    """Fail-closed deployment coordinator for exactly one test endpoint."""

    def __init__(self, request_json: RequestJson):
        self._request_json = request_json

    @staticmethod
    def validate_image(image: str) -> str:
        if not isinstance(image, str) or not IMAGE_PATTERN.fullmatch(image):
            raise DeploymentError(
                "image must be exactly "
                f"{IMAGE_REPOSITORY}@sha256:<64 lowercase hex characters>"
            )
        return image

    @staticmethod
    def _require_mapping(payload: Any, resource: str) -> Mapping[str, Any]:
        if not isinstance(payload, dict):
            raise DeploymentError(f"RunPod returned an invalid {resource}")
        return payload

    @staticmethod
    def _assert_field(
        payload: Mapping[str, Any],
        field: str,
        expected: Any,
        resource: str,
    ) -> None:
        if payload.get(field) != expected:
            raise DeploymentError(
                f"{resource} {field} did not match the deployment lock"
            )

    @classmethod
    def _assert_template(cls, payload: Any) -> Mapping[str, Any]:
        template = cls._require_mapping(payload, "template")
        cls._assert_field(template, "id", TEMPLATE_ID, "template")
        cls._assert_field(template, "name", TEMPLATE_NAME, "template")
        cls._assert_field(template, "isServerless", True, "template")
        if not isinstance(template.get("env"), dict):
            raise DeploymentError("template env was not an object")
        return template

    @classmethod
    def _assert_endpoint(cls, payload: Any) -> Mapping[str, Any]:
        endpoint = cls._require_mapping(payload, "endpoint")
        cls._assert_field(endpoint, "id", ENDPOINT_ID, "endpoint")
        cls._assert_field(endpoint, "name", ENDPOINT_NAME, "endpoint")
        cls._assert_field(endpoint, "templateId", TEMPLATE_ID, "endpoint")
        return endpoint

    @classmethod
    def _assert_registry_auth(cls, payload: Any) -> Mapping[str, Any]:
        registry = cls._require_mapping(payload, "registry auth")
        cls._assert_field(
            registry,
            "id",
            REGISTRY_AUTH_ID,
            "registry auth",
        )
        cls._assert_field(
            registry,
            "name",
            REGISTRY_AUTH_NAME,
            "registry auth",
        )
        return registry

    @staticmethod
    def _assert_exclusive_binding(payload: Any) -> None:
        if not isinstance(payload, list):
            raise DeploymentError("RunPod returned an invalid endpoint list")
        bound_endpoint_ids = []
        for endpoint in payload:
            if not isinstance(endpoint, dict):
                raise DeploymentError(
                    "RunPod endpoint list contained an invalid entry"
                )
            if endpoint.get("templateId") == TEMPLATE_ID:
                endpoint_id = endpoint.get("id")
                if not isinstance(endpoint_id, str):
                    raise DeploymentError(
                        "template binding contained an invalid endpoint ID"
                    )
                bound_endpoint_ids.append(endpoint_id)
        if bound_endpoint_ids != [ENDPOINT_ID]:
            raise DeploymentError(
                "test template is not bound exclusively to the locked endpoint"
            )

    @staticmethod
    def _configuration_snapshot(
        template: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            field: copy.deepcopy(template[field])
            for field in PRESERVED_TEMPLATE_FIELDS
            if field in template
        }

    @classmethod
    def _assert_configuration_preserved(
        cls,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
    ) -> None:
        before_snapshot = cls._configuration_snapshot(before)
        after_snapshot = cls._configuration_snapshot(after)
        if before_snapshot == after_snapshot:
            return
        changed_fields = sorted(
            field
            for field in set(before_snapshot) | set(after_snapshot)
            if before_snapshot.get(field) != after_snapshot.get(field)
        )
        # Field names are safe; values may contain secrets and stay private.
        raise DeploymentError(
            "template configuration changed unexpectedly: "
            + ", ".join(changed_fields)
        )

    def _read_and_validate(
        self,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        endpoint = self._assert_endpoint(
            self._request_json("GET", f"/endpoints/{ENDPOINT_ID}", None)
        )
        template = self._assert_template(
            self._request_json("GET", TEMPLATE_READ_PATH, None)
        )
        self._assert_registry_auth(
            self._request_json(
                "GET",
                f"/containerregistryauth/{REGISTRY_AUTH_ID}",
                None,
            )
        )
        self._assert_exclusive_binding(
            self._request_json("GET", "/endpoints", None)
        )
        return endpoint, template

    def deploy(self, image: str, *, apply: bool = False) -> dict[str, Any]:
        image = self.validate_image(image)
        _, template_before = self._read_and_validate()
        already_current = (
            template_before.get("imageName") == image
            and template_before.get("containerRegistryAuthId")
            == REGISTRY_AUTH_ID
        )
        summary = {
            "mode": "apply" if apply else "dry-run",
            "status": "already-current" if already_current else "planned",
            "endpoint_id": ENDPOINT_ID,
            "endpoint_name": ENDPOINT_NAME,
            "template_id": TEMPLATE_ID,
            "template_name": TEMPLATE_NAME,
            "current_image": template_before.get("imageName"),
            "target_image": image,
            "registry_auth_id": REGISTRY_AUTH_ID,
            "configuration_preserved": True,
            "environment_variable_count": len(template_before["env"]),
        }
        if not apply or already_current:
            return summary

        # PATCH is intentionally partial. Omitting env and every other
        # template field is what preserves them under the official REST API.
        self._request_json(
            "PATCH",
            f"/templates/{TEMPLATE_ID}",
            {
                "imageName": image,
                "containerRegistryAuthId": REGISTRY_AUTH_ID,
            },
        )

        _, template_after = self._read_and_validate()
        self._assert_field(
            template_after,
            "imageName",
            image,
            "template",
        )
        self._assert_field(
            template_after,
            "containerRegistryAuthId",
            REGISTRY_AUTH_ID,
            "template",
        )
        self._assert_configuration_preserved(template_before, template_after)
        summary["status"] = "updated"
        summary["current_image"] = image
        summary["environment_variable_count"] = len(template_after["env"])
        return summary


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or deploy an immutable image to the locked RunPod "
            "holy-grale test endpoint. Defaults to dry-run."
        )
    )
    parser.add_argument(
        "--image",
        required=True,
        help=(
            f"exact immutable image: {IMAGE_REPOSITORY}"
            "@sha256:<64 lowercase hex characters>"
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the template PATCH after all deployment locks pass",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        api_key = os.environ.get(RUNPOD_API_KEY_ENV, "")
        client = RunPodClient(api_key)
        summary = HolyGraleTestDeployer(client.request_json).deploy(
            args.image,
            apply=args.apply,
        )
    except DeploymentError as error:
        print(f"deployment refused: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    if not args.apply and summary["status"] != "already-current":
        print("dry-run only; pass --apply to mutate the locked test template")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
