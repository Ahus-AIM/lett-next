from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUTS = REPO_ROOT / "data/validation_npz/validation_public_npz"
DEFAULT_OUTPUTS = REPO_ROOT / "results/submission_validation"
DEFAULT_CHECKPOINT = REPO_ROOT / "artifacts/checkpoint.pt"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a checkpoint with the same CPU submission path used by the Docker image."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--time-limit-seconds", type=float, default=60.0)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output-format", choices=("nii", "npz", "both"), default="nii")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    from lett_next.submission_validation import run_checkpoint_submission_validation

    args = _parse_args()
    summary = run_checkpoint_submission_validation(
        repo_root=REPO_ROOT,
        checkpoint_path=_resolve(args.checkpoint),
        inputs_dir=_resolve(args.inputs),
        outputs_dir=_resolve(args.outputs),
        threshold=args.threshold,
        threads=args.threads,
        time_limit_seconds=args.time_limit_seconds,
        max_cases=args.max_cases,
        output_format=args.output_format,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
