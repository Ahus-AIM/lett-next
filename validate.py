from __future__ import annotations

import json
import argparse
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUTS = REPO_ROOT / "data/validation_npz/validation_public_npz"
DEFAULT_OUTPUTS = REPO_ROOT / "lett_next_outputs"
DEFAULT_TARBALL = REPO_ROOT / "artifacts/lett-next.tar.gz"
DEFAULT_IMAGE_TAG = "lett-next:latest"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the LETT-NeXt Docker submission image.")
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--tarball", type=Path, default=DEFAULT_TARBALL)
    parser.add_argument("--image-tag", default=DEFAULT_IMAGE_TAG)
    return parser.parse_args()


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = _parse_args()
    inputs = args.inputs if args.inputs.is_absolute() else REPO_ROOT / args.inputs
    outputs = args.outputs if args.outputs.is_absolute() else REPO_ROOT / args.outputs
    tarball = args.tarball if args.tarball.is_absolute() else REPO_ROOT / args.tarball
    _require(inputs)
    _require(tarball)
    if outputs.exists():
        shutil.rmtree(outputs)
    outputs.mkdir(parents=True)

    _run(["docker", "load", "-i", str(tarball)])
    _run(
        [
            "docker",
            "container",
            "run",
            "-m",
            "8G",
            "--name",
            "lett-next",
            "--rm",
            "-e",
            "TASK2_KEEP_RUN_METADATA=1",
            "-v",
            f"{inputs}:/workspace/inputs/",
            "-v",
            f"{outputs}:/workspace/outputs/",
            args.image_tag,
            "/bin/bash",
            "-c",
            "sh predict.sh",
        ]
    )
    summary = json.loads((outputs / "prediction_summary.json").read_text(encoding="utf-8"))
    stats = {
        "case_count": summary.get("case_count"),
        "failed_case_count": summary.get("failed_case_count"),
        "timeout_case_count": summary.get("timeout_case_count"),
        "val_score": summary.get("val_score"),
        "val_dsc": summary.get("val_dsc"),
        "val_nsd": summary.get("val_nsd"),
        "mean_total_seconds": summary.get("mean_total_seconds"),
        "max_total_seconds": summary.get("max_total_seconds"),
        "max_rss_mb": summary.get("max_rss_mb"),
        "mean_cpu_memory_time_mb_seconds": summary.get("mean_cpu_memory_time_mb_seconds"),
        "total_cpu_memory_time_mb_seconds": summary.get("total_cpu_memory_time_mb_seconds"),
    }
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
