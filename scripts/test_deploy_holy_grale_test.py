import copy
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from scripts import deploy_holy_grale_test as deploy


VALID_IMAGE = (
    "ghcr.io/vinlow/worker-audio-specialist@sha256:"
    + "a" * 64
)
OLD_IMAGE = "registry.runpod.net/worker:test"
SECRET_VALUE = "do-not-print-this-secret"


class FakeRunPod:
    def __init__(self):
        self.calls = []
        self.endpoint = {
            "id": deploy.ENDPOINT_ID,
            "name": deploy.ENDPOINT_NAME,
            "templateId": deploy.TEMPLATE_ID,
            "workersMin": 0,
        }
        self.template = {
            "id": deploy.TEMPLATE_ID,
            "name": deploy.TEMPLATE_NAME,
            "imageName": OLD_IMAGE,
            "containerRegistryAuthId": "",
            "isServerless": True,
            "category": "NVIDIA",
            "containerDiskInGb": 50,
            "env": {"HF_TOKEN": SECRET_VALUE},
            "readme": "kept",
            "startSsh": False,
            "volumeMountPath": "/workspace",
        }
        self.registry = {
            "id": deploy.REGISTRY_AUTH_ID,
            "name": deploy.REGISTRY_AUTH_NAME,
        }
        self.endpoints = [self.endpoint]
        self.drift_env_after_patch = False

    def request_json(self, method, path, payload=None):
        self.calls.append((method, path, copy.deepcopy(payload)))
        if method == "GET" and path == f"/endpoints/{deploy.ENDPOINT_ID}":
            return copy.deepcopy(self.endpoint)
        if method == "GET" and path == deploy.TEMPLATE_READ_PATH:
            return copy.deepcopy(self.template)
        if method == "GET" and path == (
            f"/containerregistryauth/{deploy.REGISTRY_AUTH_ID}"
        ):
            return copy.deepcopy(self.registry)
        if method == "GET" and path == "/endpoints":
            return copy.deepcopy(self.endpoints)
        if method == "PATCH" and path == f"/templates/{deploy.TEMPLATE_ID}":
            self.template.update(payload)
            if self.drift_env_after_patch:
                self.template["env"] = {"HF_TOKEN": "changed-secret"}
            return copy.deepcopy(self.template)
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

    def test_default_dry_run_performs_only_reads_and_never_exposes_env(self):
        summary = self.deployer.deploy(VALID_IMAGE)

        self.assertEqual(summary["mode"], "dry-run")
        self.assertEqual(summary["status"], "planned")
        self.assertEqual(summary["environment_variable_count"], 1)
        self.assertNotIn("env", summary)
        self.assertNotIn(SECRET_VALUE, json.dumps(summary))
        self.assertTrue(all(method == "GET" for method, _, _ in self.api.calls))
        self.assertEqual(self.api.template["imageName"], OLD_IMAGE)

    def test_wrong_name_or_binding_refuses_before_mutation(self):
        for field, value in (
            ("name", "production-worker"),
            ("templateId", "some-other-template"),
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

    def test_template_and_registry_identity_locks_refuse_mutation(self):
        cases = (
            ("template", "id", "some-other-template"),
            ("template", "name", "production-template"),
            ("registry", "id", "some-other-registry"),
            ("registry", "name", "Unknown credentials"),
        )
        for resource, field, value in cases:
            with self.subTest(resource=resource, field=field):
                api = FakeRunPod()
                getattr(api, resource)[field] = value
                deployer = deploy.HolyGraleTestDeployer(api.request_json)
                with self.assertRaises(deploy.DeploymentError):
                    deployer.deploy(VALID_IMAGE, apply=True)
                self.assertFalse(
                    any(method == "PATCH" for method, _, _ in api.calls)
                )

    def test_shared_template_refuses_before_mutation(self):
        self.api.endpoints.append(
            {
                "id": "unexpected-production-endpoint",
                "name": "must-not-roll",
                "templateId": deploy.TEMPLATE_ID,
            }
        )

        with self.assertRaisesRegex(
            deploy.DeploymentError,
            "not bound exclusively",
        ):
            self.deployer.deploy(VALID_IMAGE, apply=True)
        self.assertFalse(
            any(method == "PATCH" for method, _, _ in self.api.calls)
        )

    def test_apply_patches_only_image_and_registry_then_revalidates(self):
        before = copy.deepcopy(self.api.template)

        summary = self.deployer.deploy(VALID_IMAGE, apply=True)

        patch_calls = [
            call for call in self.api.calls if call[0] == "PATCH"
        ]
        self.assertEqual(
            patch_calls,
            [
                (
                    "PATCH",
                    f"/templates/{deploy.TEMPLATE_ID}",
                    {
                        "imageName": VALID_IMAGE,
                        "containerRegistryAuthId": deploy.REGISTRY_AUTH_ID,
                    },
                )
            ],
        )
        self.assertFalse(
            any(
                method != "GET" and path.startswith("/endpoints")
                for method, path, _ in self.api.calls
            )
        )
        self.assertEqual(summary["status"], "updated")
        self.assertEqual(self.api.template["env"], before["env"])
        self.assertEqual(self.api.template["containerDiskInGb"], 50)
        self.assertEqual(self.api.template["readme"], "kept")

    def test_apply_is_idempotent_when_exact_image_and_auth_are_current(self):
        self.api.template["imageName"] = VALID_IMAGE
        self.api.template["containerRegistryAuthId"] = deploy.REGISTRY_AUTH_ID

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
        fake_client = unittest.mock.Mock()
        fake_client.request_json = self.api.request_json
        with patch.dict("os.environ", {deploy.RUNPOD_API_KEY_ENV: "api-secret"}), patch.object(
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


if __name__ == "__main__":
    unittest.main()
