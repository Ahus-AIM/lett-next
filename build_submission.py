from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
IMAGE_TAG = "lett-next:latest"
TARBALL = REPO_ROOT / "artifacts/lett-next.tar.gz"
CHECKPOINT = REPO_ROOT / "artifacts/checkpoint.pt"
PREDICT_SH = REPO_ROOT / "predict.sh"
DOCKERFILE = REPO_ROOT / "Dockerfile"


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def _verify_predict_defaults() -> None:
    text = PREDICT_SH.read_text(encoding="utf-8")
    required = [
        'TASK2_THRESHOLD="${TASK2_THRESHOLD:-0.35}"',
        'TASK2_AUTOZOOM_ENABLE="${TASK2_AUTOZOOM_ENABLE:-1}"',
        'TASK2_AUTOZOOM_MAX_PASSES="${TASK2_AUTOZOOM_MAX_PASSES:-2}"',
        'TASK2_AUTOZOOM_MIN_PASSES="${TASK2_AUTOZOOM_MIN_PASSES:-1}"',
        'TASK2_AUTOZOOM_GROWTH_ZYX="${TASK2_AUTOZOOM_GROWTH_ZYX:-1.25,1.15,1.15}"',
        'TASK2_AUTOZOOM_SCALE_MODE="${TASK2_AUTOZOOM_SCALE_MODE:-adaptive}"',
        'TASK2_AUTOZOOM_REFINE_DISABLE="${TASK2_AUTOZOOM_REFINE_DISABLE:-1}"',
    ]
    missing = [item for item in required if item not in text]
    if missing:
        raise ValueError(f"predict.sh defaults do not match the submitted adaptive AutoZoom settings: {missing}")


def main() -> None:
    _require(CHECKPOINT)
    _require(DOCKERFILE)
    _require(PREDICT_SH)
    _verify_predict_defaults()

    subprocess.run(["docker", "build", "-f", "Dockerfile", "-t", IMAGE_TAG, "."], cwd=REPO_ROOT, check=True)
    TARBALL.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(TARBALL, "wb") as output:
        subprocess.run(["docker", "save", IMAGE_TAG], cwd=REPO_ROOT, stdout=output, check=True)

    print(
        json.dumps(
            {
                "image": IMAGE_TAG,
                "tarball": str(TARBALL.relative_to(REPO_ROOT)),
                "checkpoint": str(CHECKPOINT.relative_to(REPO_ROOT)),
                "autozoom": "adaptive-needed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
