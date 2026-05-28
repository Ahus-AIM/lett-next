from __future__ import annotations

from pathlib import Path

from . import recipe
from .submission_cpu import run_submission_prediction


def default_validation_inputs(repo_root: Path) -> Path:
    public_npz = repo_root / "data/validation_npz/validation_public_npz"
    return public_npz if public_npz.exists() else repo_root / "data/validation_npz"


def resolve_validation_inputs(repo_root: Path, inputs_dir: Path | None = None) -> Path:
    if inputs_dir is None:
        return default_validation_inputs(repo_root)
    public_npz = inputs_dir / "validation_public_npz"
    return public_npz if public_npz.exists() else inputs_dir


def threshold_label(threshold: float) -> str:
    return f"{float(threshold):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def run_checkpoint_submission_validation(
    *,
    repo_root: Path,
    checkpoint_path: Path,
    outputs_dir: Path,
    inputs_dir: Path | None = None,
    threshold: float | None = None,
    threads: int = 12,
    time_limit_seconds: float = 60.0,
    max_cases: int | None = None,
    output_format: str = "nii",
) -> dict[str, object]:
    return run_submission_prediction(
        inputs_dir=resolve_validation_inputs(repo_root, inputs_dir),
        outputs_dir=outputs_dir,
        checkpoint_path=checkpoint_path,
        model_name=recipe.MODEL_NAME,
        crop_size_zyx=recipe.CROP_SIZE_ZYX,
        prompt_sigma=recipe.PROMPT_SIGMA,
        threshold=recipe.SUBMISSION_THRESHOLD if threshold is None else float(threshold),
        deep_supervision=recipe.DEEP_SUPERVISION,
        allow_untrained_smoke=False,
        output_format=output_format,
        threads=threads,
        time_limit_seconds=time_limit_seconds,
        max_cases=max_cases,
        **recipe.SUBMISSION_AUTOZOOM_KWARGS,
    )


def run_submission_threshold_sweep(
    *,
    repo_root: Path,
    checkpoint_path: Path,
    outputs_root: Path,
    thresholds: tuple[float, ...],
    inputs_dir: Path | None = None,
    threads: int = 12,
    time_limit_seconds: float = 60.0,
    max_cases: int | None = None,
    output_format: str = "nii",
) -> dict[str, object]:
    summaries: dict[str, dict[str, object]] = {}
    best_threshold = None
    best_score = None
    for threshold in thresholds:
        name = f"{float(threshold):.3f}".rstrip("0").rstrip(".")
        summary = run_checkpoint_submission_validation(
            repo_root=repo_root,
            checkpoint_path=checkpoint_path,
            outputs_dir=outputs_root / f"threshold_{threshold_label(threshold)}",
            inputs_dir=inputs_dir,
            threshold=float(threshold),
            threads=threads,
            time_limit_seconds=time_limit_seconds,
            max_cases=max_cases,
            output_format=output_format,
        )
        row = {
            "threshold": float(threshold),
            "val_dsc": float(summary.get("val_dsc", 0.0)),
            "val_nsd": float(summary.get("val_nsd", 0.0)),
            "val_score": float(summary.get("val_score", 0.0)),
            "case_count": int(summary.get("case_count", 0)),
            "failed_case_count": int(summary.get("failed_case_count", 0)),
            "timeout_case_count": int(summary.get("timeout_case_count", 0)),
            "outputs_dir": str(outputs_root / f"threshold_{threshold_label(threshold)}"),
        }
        summaries[name] = row
        if best_score is None or float(row["val_score"]) > best_score:
            best_score = float(row["val_score"])
            best_threshold = float(threshold)
    return {
        "best_threshold": best_threshold,
        "best_score": best_score,
        "thresholds": summaries,
        "validation_path": "submission",
    }
