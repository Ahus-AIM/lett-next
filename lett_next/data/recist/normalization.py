from __future__ import annotations

import json
import sys
import time

import numpy as np

from .records import CaseRecord, NormalizationStats


def _coerce_normalization_stats(stats: NormalizationStats | dict[str, object] | None) -> NormalizationStats | None:
    if stats is None:
        return None
    if isinstance(stats, NormalizationStats):
        return stats
    return NormalizationStats(
        mode=str(stats.get("mode", "dataset_stats")),
        clip_low=float(stats["clip_low"]),
        clip_high=float(stats["clip_high"]),
        mean=float(stats["mean"]),
        std=float(stats["std"]),
        source_case_count=int(stats.get("source_case_count", 0)),
        source_voxel_count=int(stats.get("source_voxel_count", 0)),
    )


def compute_ct_normalization_stats(
    records: list[CaseRecord],
    *,
    target_spacing_xyz: tuple[float, float, float] | None = None,
    resampling_mode: str = "standard",
    max_voxels_per_case: int = 20000,
    seed: int = 13,
    use_mask_values: bool = True,
    progress_label: str | None = None,
    progress_every_cases: int = 25,
    progress_every_seconds: float = 30.0,
) -> NormalizationStats:
    rng = np.random.default_rng(seed)
    sampled_values: list[np.ndarray] = []
    source_case_count = 0
    source_voxel_count = 0
    total_cases = len(records)
    started = time.monotonic()
    last_progress_report = started
    last_loading_report = started - float(progress_every_seconds)

    def _progress_payload(
        event: str,
        *,
        completed_cases: int,
        record: CaseRecord | None = None,
    ) -> dict[str, object]:
        elapsed = max(time.monotonic() - started, 1e-6)
        rate = float(completed_cases / elapsed) if completed_cases > 0 else 0.0
        remaining = max(total_cases - completed_cases, 0)
        eta_seconds = float(remaining / rate) if rate > 0.0 else None
        payload: dict[str, object] = {
            "event": event,
            "label": str(progress_label),
            "completed_cases": int(completed_cases),
            "total_cases": int(total_cases),
            "elapsed_seconds": round(float(elapsed), 1),
            "cases_per_second": round(rate, 4),
            "eta_seconds": None if eta_seconds is None else round(float(eta_seconds), 1),
            "source_case_count": int(source_case_count),
            "source_voxel_count": int(source_voxel_count),
        }
        if record is not None:
            payload["case_id"] = record.case_id
            payload["source"] = record.source
        return payload

    def _print_progress(payload: dict[str, object]) -> None:
        print(json.dumps(payload), file=sys.stderr, flush=True)

    if progress_label is not None:
        _print_progress(
            {
                "event": "normalization_stats_start",
                "label": str(progress_label),
                "total_cases": int(total_cases),
                "max_voxels_per_case": int(max_voxels_per_case),
                "resampling_mode": str(resampling_mode),
                "use_mask_values": bool(use_mask_values),
                "target_spacing_xyz": (
                    None
                    if target_spacing_xyz is None
                    else [float(value) for value in target_spacing_xyz]
                ),
            }
        )

    for index, record in enumerate(records, start=1):
        now = time.monotonic()
        if progress_label is not None and (
            index == 1 or now - last_loading_report >= float(progress_every_seconds)
        ):
            _print_progress(
                _progress_payload(
                    "normalization_stats_loading_case",
                    completed_cases=index - 1,
                    record=record,
                )
            )
            last_loading_report = now
        from .case_loader import load_case

        case = load_case(
            record,
            target_spacing_xyz=target_spacing_xyz,
            resampling_mode=resampling_mode,
            validate_integrity=True,
            require_mask=False,
            require_recist_target=False,
            load_mask=bool(use_mask_values),
        )
        image = np.asarray(case.image, dtype=np.float32)
        if use_mask_values and case.mask is not None and np.any(case.mask > 0):
            values = image[np.asarray(case.mask) > 0]
        else:
            nonzero = image[np.asarray(image) != 0]
            values = nonzero if nonzero.size > 0 else image.reshape(-1)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        if values.size > int(max_voxels_per_case):
            selected = rng.choice(values.size, size=int(max_voxels_per_case), replace=False)
            values = values[selected]
        sampled_values.append(values)
        source_case_count += 1
        source_voxel_count += int(values.size)
        now = time.monotonic()
        if progress_label is not None and (
            index == total_cases
            or index % max(int(progress_every_cases), 1) == 0
            or now - last_progress_report >= float(progress_every_seconds)
        ):
            _print_progress(
                _progress_payload(
                    "normalization_stats_progress",
                    completed_cases=index,
                    record=record,
                )
            )
            last_progress_report = now
    if not sampled_values:
        raise ValueError("Cannot compute normalization stats: no finite image voxels found")
    all_values = np.concatenate(sampled_values).astype(np.float32, copy=False)
    clip_low, clip_high = np.percentile(all_values, [0.5, 99.5])
    clipped = np.clip(all_values, float(clip_low), float(clip_high))
    std = float(clipped.std())
    if std < 1e-6:
        std = 1.0
    return NormalizationStats(
        mode="dataset_stats",
        clip_low=float(clip_low),
        clip_high=float(clip_high),
        mean=float(clipped.mean()),
        std=std,
        source_case_count=int(source_case_count),
        source_voxel_count=int(source_voxel_count),
    )


def normalize_image(
    image: np.ndarray,
    normalization_stats: NormalizationStats | dict[str, object] | None = None,
    normalization_mode: str = "legacy_crop_zscore",
) -> np.ndarray:
    if normalization_mode in {"dataset_stats", "fixed_hu"}:
        stats = _coerce_normalization_stats(normalization_stats)
        if stats is None:
            raise ValueError(f"normalization_stats must be provided when normalization_mode={normalization_mode}")
        clipped = np.clip(np.asarray(image, dtype=np.float32), float(stats.clip_low), float(stats.clip_high))
        std = float(stats.std) if float(stats.std) >= 1e-6 else 1.0
        return ((clipped - float(stats.mean)) / std).astype(np.float32)
    if normalization_mode not in {"legacy_crop_zscore", "crop_zscore", "legacy"}:
        raise ValueError(f"Unsupported normalization_mode: {normalization_mode}")
    low, high = np.percentile(image, [1.0, 99.0])
    image = np.clip(image, low, high)
    mean = float(image.mean())
    std = float(image.std())
    if std < 1e-6:
        std = 1.0
    return ((image - mean) / std).astype(np.float32)
