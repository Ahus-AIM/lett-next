from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "configs/lett_next.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_thresholds(value: str | None) -> tuple[float, ...] | None:
    if value in (None, ""):
        return None
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to evaluate")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Config path; retained for CLI compatibility")
    parser.add_argument("--inputs", type=Path, default=None, help="Validation NPZ input directory")
    parser.add_argument("--thresholds", type=str, default=None, help="Comma-separated thresholds")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--time-limit-seconds", type=float, default=60.0)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output-format", choices=("nii", "npz", "both"), default="nii")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    from lett_next import recipe
    from lett_next.submission_validation import run_submission_threshold_sweep

    args = _parse_args()
    checkpoint_path = _resolve(args.checkpoint)
    output_path = _resolve(args.output) if args.output is not None else checkpoint_path.parent / "threshold_sweep.json"
    thresholds = _parse_thresholds(args.thresholds) or recipe.DEFAULT_THRESHOLDS
    outputs_root = output_path.parent / f"{output_path.stem}_outputs"
    payload = run_submission_threshold_sweep(
        repo_root=REPO_ROOT,
        checkpoint_path=checkpoint_path,
        outputs_root=outputs_root,
        inputs_dir=None if args.inputs is None else _resolve(args.inputs),
        thresholds=thresholds,
        threads=args.threads,
        time_limit_seconds=args.time_limit_seconds,
        max_cases=args.max_cases,
        output_format=args.output_format,
    )
    payload |= {
        "checkpoint": str(checkpoint_path),
        "config": str(_resolve(args.config)),
        "outputs_root": str(outputs_root),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
