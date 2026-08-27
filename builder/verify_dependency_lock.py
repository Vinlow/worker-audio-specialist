"""Fail the image build when its installed Python graph drifts from review."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from packaging.utils import canonicalize_name


# These distributions are installed outside requirements.txt/constraints.txt.
# Their exact versions are still audited: the CUDA family comes from the
# Dockerfile's pinned torch wheel, while the rest comes from the pinned base
# image plus its apt layer. Any movement therefore stops the build for review.
REVIEWED_EXTERNAL_DISTRIBUTIONS = {
    "blinker": "1.4",
    "dbus-python": "1.2.18",
    "distro": "1.7.0",
    "httplib2": "0.20.2",
    "importlib-metadata": "4.6.4",
    "jeepney": "0.7.1",
    "keyring": "23.5.0",
    "launchpadlib": "1.10.16",
    "lazr-restfulclient": "0.14.4",
    "lazr-uri": "1.0.6",
    "more-itertools": "8.10.0",
    "nvidia-cublas-cu12": "12.8.3.14",
    "nvidia-cuda-cupti-cu12": "12.8.57",
    "nvidia-cuda-nvrtc-cu12": "12.8.61",
    "nvidia-cuda-runtime-cu12": "12.8.57",
    "nvidia-cudnn-cu12": "9.7.1.26",
    "nvidia-cufft-cu12": "11.3.3.41",
    "nvidia-cufile-cu12": "1.13.0.11",
    "nvidia-curand-cu12": "10.3.9.55",
    "nvidia-cusolver-cu12": "11.7.2.55",
    "nvidia-cusparse-cu12": "12.5.7.53",
    "nvidia-cusparselt-cu12": "0.6.3",
    "nvidia-nccl-cu12": "2.26.2",
    "nvidia-nvjitlink-cu12": "12.8.61",
    "nvidia-nvtx-cu12": "12.8.55",
    "oauthlib": "3.2.0",
    "pip": "26.2.1",
    "pycairo": "1.20.1",
    "pygobject": "3.42.1",
    "pyjwt": "2.3.0",
    "python-apt": "2.4.0+ubuntu4.1",
    "secretstorage": "3.3.1",
    "torch": "2.7.1+cu128",
    "torchaudio": "2.7.1+cu128",
    "triton": "3.3.1",
    "wadllib": "1.3.6",
    "wheel": "0.37.1",
    "zipp": "1.0.0",
}


def read_exact_constraints(path: Path) -> dict[str, str]:
    constraints: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise RuntimeError(
                f"constraint line {line_number} is not an exact pin"
            )
        name, version = (part.strip() for part in line.split("==", 1))
        canonical_name = canonicalize_name(name)
        if not canonical_name or not version:
            raise RuntimeError(
                f"constraint line {line_number} is invalid"
            )
        if canonical_name in constraints:
            raise RuntimeError(f"duplicate constraint for {canonical_name}")
        constraints[canonical_name] = version
    return constraints


def installed_distributions() -> dict[str, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=json"],
        check=True,
        capture_output=True,
        text=True,
    )
    packages = json.loads(completed.stdout)
    return {
        canonicalize_name(package["name"]): package["version"]
        for package in packages
    }


def verify(constraints_path: Path) -> None:
    expected = read_exact_constraints(constraints_path)
    for name, version in REVIEWED_EXTERNAL_DISTRIBUTIONS.items():
        canonical_name = canonicalize_name(name)
        if canonical_name in expected:
            raise RuntimeError(
                f"external distribution {canonical_name} is also constrained"
            )
        expected[canonical_name] = version

    installed = installed_distributions()
    missing = sorted(set(expected) - set(installed))
    unexpected = sorted(set(installed) - set(expected))
    mismatched = sorted(
        name
        for name in set(expected) & set(installed)
        if expected[name] != installed[name]
    )
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        if mismatched:
            details.append(
                "mismatched="
                + ",".join(
                    f"{name}:{installed[name]}!={expected[name]}"
                    for name in mismatched
                )
            )
        raise RuntimeError("dependency lock drift: " + "; ".join(details))

    print(
        "Dependency lock verified: "
        f"{len(installed)} installed distributions match review."
    )


if __name__ == "__main__":
    verify(Path(sys.argv[1] if len(sys.argv) > 1 else "/constraints.txt"))
