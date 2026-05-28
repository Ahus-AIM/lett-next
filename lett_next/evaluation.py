from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from . import recipe
from .config import ExperimentConfig
from .data import LEGACY_RECIST_RESAMPLING_MODE, get_recist_target_mask, load_case
from .inference import predict_case
from .metrics import aggregate_metrics, compute_case_metrics
from .pants_aux import PantsAuxClassMap


def load_case_for_config(record, config: ExperimentConfig, *, aux_class_map: PantsAuxClassMap | None = None):
    return load_case(
        record,
        target_spacing_xyz=config.target_spacing_xyz,
        resampling_mode=LEGACY_RECIST_RESAMPLING_MODE,
        require_mask=False,
        require_recist_target=False,
        normalization_stats=config.normalization_stats,
        case_cache_dir=config.case_cache_dir,
        pants_aux_class_map=aux_class_map,
    )


def evaluate_cases(
    model: torch.nn.Module,
    records,
    config: ExperimentConfig,
    device: str,
    threshold: float,
    aux_class_map: PantsAuxClassMap | None = None,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    cpu_seconds: list[float] = []
    model.eval()
    with torch.no_grad():
        for record in records:
            case = load_case_for_config(record, config, aux_class_map=aux_class_map)
            target_mask = get_recist_target_mask(case)
            prediction, elapsed, _ = predict_case(
                model=model,
                case=case,
                crop_size_zyx=config.crop_size_zyx,
                prompt_sigma=config.prompt_sigma,
                threshold=float(threshold),
                device=device,
                apply_postprocess=True,
                prompt_mode=recipe.PRIMARY_PROMPT_MODE,
                **recipe.POSTPROCESS_KWARGS,
                **recipe.VALIDATION_AUTOZOOM_KWARGS,
            )
            metrics = {"dsc": 0.0, "nsd": 0.0}
            if target_mask is not None:
                metrics = compute_case_metrics(target_mask, prediction, case.spacing_xyz)
            rows.append({"case_id": case.case_id, **metrics})
            cpu_seconds.append(float(elapsed))
    aggregate = aggregate_metrics([{"dsc": row["dsc"], "nsd": row["nsd"]} for row in rows])
    aggregate["cpu_seconds_per_case"] = float(np.mean(cpu_seconds)) if cpu_seconds else 0.0
    return aggregate, rows


def evaluate_thresholds(
    model: torch.nn.Module,
    records,
    config: ExperimentConfig,
    device: str,
    thresholds: Iterable[float],
    aux_class_map: PantsAuxClassMap | None = None,
) -> dict[str, object]:
    summaries: dict[str, dict[str, float]] = {}
    best_threshold = None
    best_score = None
    for threshold in thresholds:
        metrics, _ = evaluate_cases(
            model=model,
            records=records,
            config=config,
            device=device,
            threshold=float(threshold),
            aux_class_map=aux_class_map,
        )
        name = f"{float(threshold):.3f}".rstrip("0").rstrip(".")
        summaries[name] = {
            "threshold": float(threshold),
            "val_dsc": float(metrics["val_dsc"]),
            "val_nsd": float(metrics["val_nsd"]),
            "val_score": float(metrics["val_score"]),
            "cpu_seconds_per_case": float(metrics["cpu_seconds_per_case"]),
        }
        if best_score is None or summaries[name]["val_score"] > best_score:
            best_score = summaries[name]["val_score"]
            best_threshold = float(threshold)
    return {
        "best_threshold": best_threshold,
        "best_score": best_score,
        "thresholds": summaries,
    }
