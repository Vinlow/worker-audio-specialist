import copy
import io
import json
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import Mock, patch

from scripts import deploy_holy_grale_test as deploy


VALID_IMAGE = (
    "ghcr.io/vinlow/worker-audio-specialist@sha256:" + "a" * 64
)
OLD_IMAGE = "registry.runpod.net/worker:test"
SECRET_VALUE = "do-not-print-this-secret"


class FakeRunPod:
    def __init__(self):
        self.calls = []
        self.endpoint = {
            "id": deploy.ENDPOINT_ID,
            "name": deploy.ENDPOINT_NAME,
            "type": "QUEUE",
            "requestUrls": {
                "run": f"https://api.runpod.ai/v2/{deploy.ENDPOINT_ID}/run"
            },
            "image": OLD_IMAGE,
            "args": "",
            "disk": 5,
            "ports": [],
            "env": {"HF_TOKEN": SECRET_VALUE},
            "registry": None,
            "gpu": {
                "count": 1,
                "pools": ["AMPERE_24", "AMPERE_48", "ADA_24"],
            },
            "workers": {"idleTimeout": 45, "max": 30, "min": 0},
            "scaling": {"queueDelay": 4, "type": "QUEUE_DELAY"},
            "dataCenterIds": [],
            "networkVolumes": [],
            "timeout": 60_000_000,
            "flashboot": "FLASHBOOT",
            "createdAt": "2026-08-01T12:00:00Z",
            "endpointVersion": 24,
            "rollout": {"percent": 100, "state": "READY"},
            "release": {"id": "release-before"},
        }
        self.registry = {
            "id": deploy.REGISTRY_AUTH_ID,
            "name": deploy.REGISTRY_AUTH_NAME,
        }
        self.drift_env_after_patch = False

    def request_json(self, method, path, payload=None):
        self.calls.append((method, path, copy.deepcopy(payload)))
        if method == "GET" and path == deploy.ENDPOINT_PATH:
            return copy.deepcopy(self.endpoint)
        if method == "GET" and path == deploy.REGISTRY_AUTH_PATH:
            return copy.deepcopy(self.registry)
        if method == "PATCH" and path == deploy.ENDPOINT_PATH:
            self.endpoint.update(payload)
            self.endpoint["endpointVersion"] += 1
            self.endpoint["rollout"] = {"percent": 0, "state": "ROLLING"}
            self.endpoint["release"] = {"id": "release-after"}
            if self.drift_env_after_patch:
                self.endpoint["env"] = {"HF_TOKEN": "changed-secret"}
            return copy.deepcopy(self.endpoint)
        raise AssertionError(f"unexpected request: {method} {path}")


class HolyGraleTestDeployerTest(unittest.TestCase):
    def setUp(self):
        self.api = FakeRunPod()
        self.deployer = deploy.HolyGraleTestDeployer(self.api.request_json)

    def test_rejects_tags_and_wrong_repositories_before_network_access(self):
        invalid_images = (
            "ghcr.io/vinlow/worker-audio-specialist:latest",
            "ghcr.io/other/worker-audio-specialist@sha256:" + "a" * 64,
            "ghcr.io/vinlow/worker-audio-specialist@sha256:" + "A" * 64,
            "ghcr.io/vinlow/worker-audio-specialist@sha256:abc",
        )
        for image in invalid_images:
            with self.subTest(image=image):
                with self.assertRaises(deploy.DeploymentError):
                    self.deployer.deploy(image)
        self.assertEqual(self.api.calls, [])

    def test_default_dry_run_performs_only_v2_reads_and_hides_env(self):
        summary = self.deployer.deploy(VALID_IMAGE)

        self.assertEqual(summary["mode"], "dry-run")
        self.assertEqual(summary["status"], "planned")
        self.assertEqual(summary["environment_variable_count"], 1)
        self.assertNotIn("env", summary)
        self.assertNotIn(SECRET_VALUE, json.dumps(summary))
        self.assertEqual(
            self.api.calls,
            [
                ("GET", deploy.ENDPOINT_PATH, None),
                ("GET", deploy.REGISTRY_AUTH_PATH, None),
            ],
        )
        self.assertEqual(self.api.endpoint["image"], OLD_IMAGE)
        self.assertEqual(deploy.RUNPOD_API_BASE, "https://api.runpod.io/v2")

    def test_endpoint_identity_locks_refuse_before_mutation(self):
        for field, value in (
            ("id", "some-other-endpoint"),
            ("name", "production-worker"),
        ):
            with self.subTest(field=field):
                api = FakeRunPod()
                api.endpoint[field] = value
                deployer = deploy.HolyGraleTestDeployer(api.request_json)
                with self.assertRaises(deploy.DeploymentError):
                    deployer.deploy(VALID_IMAGE, apply=True)
                self.assertFalse(
                    any(method == "PATCH" for method, _, _ in api.calls)
                )

    def test_registry_identity_lock_refuses_before_mutation(self):
        for field, value in (
            ("id", "some-other-registry"),
            ("name", "Unknown credentials"),
        ):
            with self.subTest(field=field):
                api = FakeRunPod()
                api.registry[field] = value
                deployer = deploy.HolyGraleTestDeployer(api.request_json)
                with self.assertRaises(deploy.DeploymentError):
                    deployer.deploy(VALID_IMAGE, apply=True)
                self.assertFalse(
                    any(method == "PATCH" for method, _, _ in api.calls)
                )

    def test_invalid_inline_configuration_refuses_before_mutation(self):
        for field, value in (
            ("image", None),
            ("env", [("SECRET", SECRET_VALUE)]),
            ("registry", {"id": deploy.REGISTRY_AUTH_ID}),
        ):
            with self.subTest(field=field):
                api = FakeRunPod()
                api.endpoint[field] = value
                deployer = deploy.HolyGraleTestDeployer(api.request_json)
                with self.assertRaises(deploy.DeploymentError):
                    deployer.deploy(VALID_IMAGE, apply=True)
                self.assertFalse(
                    any(method == "PATCH" for method, _, _ in api.calls)
                )

    def test_apply_patches_only_inline_image_and_registry_then_revalidates(self):
        before = copy.deepcopy(self.api.endpoint)

        summary = self.deployer.deploy(VALID_IMAGE, apply=True)

        patch_calls = [call for call in self.api.calls if call[0] == "PATCH"]
        self.assertEqual(
            patch_calls,
            [
                (
                    "PATCH",
                    deploy.ENDPOINT_PATH,
                    {
                        "image": VALID_IMAGE,
                        "registry": deploy.REGISTRY_AUTH_ID,
                    },
                )
            ],
        )
        self.assertEqual(summary["status"], "updated")
        self.assertEqual(
            deploy.HolyGraleTestDeployer._configuration_snapshot(
                self.api.endpoint
            ),
            deploy.HolyGraleTestDeployer._configuration_snapshot(before),
        )
        self.assertEqual(self.api.endpoint["env"], before["env"])
        self.assertEqual(self.api.endpoint["workers"], before["workers"])
        self.assertEqual(self.api.endpoint["endpointVersion"], 25)
        self.assertNotEqual(self.api.endpoint["rollout"], before["rollout"])
        self.assertNotEqual(self.api.endpoint["release"], before["release"])

    def test_apply_is_idempotent_when_exact_image_and_auth_are_current(self):
        self.api.endpoint["image"] = VALID_IMAGE
        self.api.endpoint["registry"] = deploy.REGISTRY_AUTH_ID

        summary = self.deployer.deploy(VALID_IMAGE, apply=True)

        self.assertEqual(summary["status"], "already-current")
        self.assertFalse(
            any(method == "PATCH" for method, _, _ in self.api.calls)
        )

    def test_post_update_drift_names_field_without_exposing_secret(self):
        self.api.drift_env_after_patch = True

        with self.assertRaises(deploy.DeploymentError) as raised:
            self.deployer.deploy(VALID_IMAGE, apply=True)

        message = str(raised.exception)
        self.assertIn("env", message)
        self.assertNotIn(SECRET_VALUE, message)
        self.assertNotIn("changed-secret", message)

    def test_cli_is_dry_run_without_apply_flag(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        fake_client = Mock()
        fake_client.request_json = self.api.request_json
        with patch.dict(
            "os.environ", {deploy.RUNPOD_API_KEY_ENV: "api-secret"}
        ), patch.object(
            deploy,
            "RunPodClient",
            return_value=fake_client,
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = deploy.main(["--image", VALID_IMAGE])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertIn("dry-run only", stdout.getvalue())
        self.assertNotIn("api-secret", stdout.getvalue())
        self.assertNotIn(SECRET_VALUE, stdout.getvalue())
        self.assertFalse(
            any(method == "PATCH" for method, _, _ in self.api.calls)
        )


class RunPodClientTest(unittest.TestCase):
    def test_uses_v2_base_and_refuses_absolute_paths(self):
        client = deploy.RunPodClient("api-secret")

        with self.assertRaises(deploy.DeploymentError):
            client.request_json("GET", "https://attacker.example/endpoint")

    def test_http_error_never_includes_api_response_body(self):
        client = deploy.RunPodClient("api-secret")
        response_body = json.dumps(
            {"env": {"HF_TOKEN": SECRET_VALUE}, "detail": "private body"}
        ).encode("utf-8")
        http_error = urllib.error.HTTPError(
            url=deploy.RUNPOD_API_BASE + deploy.ENDPOINT_PATH,
            code=422,
            msg="Unprocessable Entity",
            hdrs={},
            fp=io.BytesIO(response_body),
        )

        with patch.object(client._opener, "open", side_effect=http_error):
            with self.assertRaises(deploy.DeploymentError) as raised:
                client.request_json(
                    "PATCH",
                    deploy.ENDPOINT_PATH,
                    {"image": VALID_IMAGE},
                )

        message = str(raised.exception)
        self.assertIn("HTTP 422", message)
        self.assertNotIn(SECRET_VALUE, message)
        self.assertNotIn("private body", message)


if __name__ == "__main__":
    unittest.main()
