from __future__ import annotations

import argparse
import json
import os
import resource
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

from .data import IMAGE_KEYS, SPACING_KEYS, _find_first_present_key
from .data import _resample_loaded_case_recist
from .inference import predict_case
from .metrics import aggregate_metrics, compute_binary_stats, compute_labeled_case_metrics
from .models import build_model


ENDPOINT_KEYS = (
    "recist_endpoints_xyz",
    "recist_endpoints",
    "diameter_endpoints_xyz",
    "endpoints_xyz",
)
SLICE_INDEX_KEYS = ("recist_slice_index", "slice_index", "diameter_slice_index")


@dataclass(slots=True)
class PromptSpec:
    label_value: int
    endpoints_xyz: np.ndarray
    slice_index: int
    marker_voxels: int


@dataclass(slots=True)
class LoadedSubmissionCase:
    case_id: str
    image: np.ndarray
    spacing_xyz: np.ndarray
    direction: np.ndarray | None
    origin: np.ndarray | None
    prompts: list[PromptSpec]
    ground_truth: np.ndarray | None
    normalization_stats: dict[str, object] | None = None


def _parse_tuple_3(raw: str) -> tuple[int, int, int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"Expected three comma-separated integers, got {raw!r}")
    return values[0], values[1], values[2]


def _parse_float_tuple_3(raw: str) -> tuple[float, float, float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"Expected three comma-separated floats, got {raw!r}")
    return values[0], values[1], values[2]


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _current_rss_mb() -> float:
    statm_path = Path("/proc/self/statm")
    try:
        resident_pages = int(statm_path.read_text(encoding="utf-8").split()[1])
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return float(resident_pages * page_size / (1024.0 * 1024.0))
    except (OSError, IndexError, ValueError):
        return _rss_mb()


class MemoryTimeSampler:
    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._samples: list[tuple[float, float]] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._record_sample()
        self._thread.start()

    def stop(self) -> dict[str, float]:
        self._stop.set()
        self._thread.join(timeout=max(self.interval_seconds * 4.0, 0.2))
        self._record_sample()
        samples = sorted(self._samples, key=lambda sample: sample[0])
        if len(samples) < 2:
            rss_mb = samples[0][1] if samples else _current_rss_mb()
            return {"memory_time_mb_seconds": 0.0, "sample_peak_rss_mb": float(rss_mb)}
        area = 0.0
        for (time_a, rss_a), (time_b, rss_b) in zip(samples[:-1], samples[1:], strict=True):
            area += max(float(time_b - time_a), 0.0) * 0.5 * float(rss_a + rss_b)
        return {
            "memory_time_mb_seconds": float(area),
            "sample_peak_rss_mb": float(max(rss for _, rss in samples)),
        }

    def _record_sample(self) -> None:
        self._samples.append((time.perf_counter(), _current_rss_mb()))

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._record_sample()


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _farthest_points_yx(points_yx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_yx, dtype=np.float32)
    if len(points) == 0:
        raise ValueError("Cannot derive RECIST endpoints from an empty marker")
    if len(points) == 1:
        return points[0], points[0]
    best_start = points[0]
    best_end = points[0]
    best_distance = -1.0
    chunk_size = 4096
    for start in range(0, len(points), chunk_size):
        chunk = points[start : start + chunk_size]
        squared = np.sum((chunk[:, None, :] - points[None, :, :]) ** 2, axis=-1)
        chunk_index, point_index = np.unravel_index(int(np.argmax(squared)), squared.shape)
        distance = float(squared[chunk_index, point_index])
        if distance > best_distance:
            best_distance = distance
            best_start = chunk[chunk_index]
            best_end = points[point_index]
    return best_start, best_end


def _derive_prompt_from_marker(marker_zyx: np.ndarray, label_value: int) -> PromptSpec:
    marker = np.asarray(marker_zyx) > 0
    points = np.argwhere(marker)
    if len(points) == 0:
        raise ValueError(f"RECIST marker label {label_value} is empty")
    slices, counts = np.unique(points[:, 0], return_counts=True)
    slice_index = int(slices[np.argmax(counts)])
    slice_points_yx = points[points[:, 0] == slice_index][:, 1:3]
    start_yx, end_yx = _farthest_points_yx(slice_points_yx)
    endpoints_xyz = np.asarray(
        [
            [float(start_yx[1]), float(start_yx[0]), float(slice_index)],
            [float(end_yx[1]), float(end_yx[0]), float(slice_index)],
        ],
        dtype=np.float32,
    )
    return PromptSpec(
        label_value=int(label_value),
        endpoints_xyz=endpoints_xyz,
        slice_index=slice_index,
        marker_voxels=int(marker.sum()),
    )


def _prompt_specs_from_npz(data: np.lib.npyio.NpzFile) -> list[PromptSpec]:
    keys = set(data.files)
    endpoint_key = _find_first_present_key(keys, ENDPOINT_KEYS)
    if endpoint_key is not None:
        endpoints = np.asarray(data[endpoint_key], dtype=np.float32).reshape(2, 3)
        slice_key = _find_first_present_key(keys, SLICE_INDEX_KEYS)
        if slice_key is None:
            slice_index = int(round(float(endpoints[:, 2].mean())))
        else:
            slice_index = int(np.asarray(data[slice_key]).item())
        return [PromptSpec(label_value=1, endpoints_xyz=endpoints, slice_index=slice_index, marker_voxels=0)]

    if "recist" not in keys:
        raise KeyError("Input NPZ must contain either RECIST endpoints or a recist marker array")
    recist = np.asarray(data["recist"])
    labels = [int(value) for value in np.unique(recist).tolist() if int(value) > 0]
    if not labels:
        raise ValueError("RECIST marker array does not contain any positive labels")
    return [_derive_prompt_from_marker(recist == label_value, label_value) for label_value in labels]


def _load_submission_npz(path: Path) -> LoadedSubmissionCase:
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        image_key = _find_first_present_key(keys, IMAGE_KEYS)
        if image_key is None:
            raise KeyError(f"Could not find image array in {path}")
        image = np.asarray(data[image_key])
        if not np.issubdtype(image.dtype, np.number) or image.dtype.itemsize > np.dtype(np.float32).itemsize:
            image = np.asarray(image, dtype=np.float32)
        spacing_key = _find_first_present_key(keys, SPACING_KEYS)
        spacing = np.asarray(data[spacing_key], dtype=np.float32).reshape(-1)[:3] if spacing_key else np.ones(3, dtype=np.float32)
        direction = np.asarray(data["direction"], dtype=np.float64).reshape(3, 3) if "direction" in keys else None
        origin = np.asarray(data["origin"], dtype=np.float64).reshape(3) if "origin" in keys else None
        ground_truth = np.asarray(data["gts"], dtype=np.uint16) if "gts" in keys else None
        return LoadedSubmissionCase(
            case_id=path.stem,
            image=image,
            spacing_xyz=spacing,
            direction=direction,
            origin=origin,
            prompts=_prompt_specs_from_npz(data),
            ground_truth=None if ground_truth is None else ground_truth.astype(np.uint16),
        )


def _build_case_namespace(case: LoadedSubmissionCase, prompt: PromptSpec) -> SimpleNamespace:
    return SimpleNamespace(
        case_id=case.case_id,
        image=case.image,
        raw_mask=None,
        mask=None,
        spacing_xyz=case.spacing_xyz,
        recist_endpoints_xyz=prompt.endpoints_xyz,
        recist_slice_index=prompt.slice_index,
        prompt_source=f"recist_label_{prompt.label_value}",
        recist_target_mask=None,
        recist_target_selection=None,
        normalization_stats=case.normalization_stats,
    )


def _model_settings_from_checkpoint(checkpoint_path: Path | None) -> dict[str, object]:
    if checkpoint_path is None:
        return {}
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = state.get("config_snapshot") or state.get("config") or {}
    if not isinstance(config, dict):
        config = {}
    return {
        "state": state,
        "model_name": state.get("model_name") or config.get("model_name"),
        "deep_supervision": bool(config.get("deep_supervision", False)),
        "crop_size_zyx": config.get("crop_size_zyx"),
        "prompt_sigma": config.get("prompt_sigma"),
        "threshold": config.get("threshold"),
        "target_spacing_xyz": config.get("target_spacing_xyz"),
        "normalization_stats": config.get("normalization_stats"),
    }


def _resize_prediction_to_shape(prediction: np.ndarray, output_shape_zyx: tuple[int, int, int]) -> np.ndarray:
    source = np.asarray(prediction, dtype=np.uint8)
    output_shape = tuple(int(dim) for dim in output_shape_zyx)
    if tuple(source.shape) == output_shape:
        return source
    zoom = tuple(float(out_dim) / float(max(in_dim, 1)) for out_dim, in_dim in zip(output_shape, source.shape, strict=True))
    resized = ndimage.zoom(source, zoom=zoom, order=0)
    restored = np.zeros(output_shape, dtype=np.uint8)
    common = tuple(slice(0, min(int(dst), int(src))) for dst, src in zip(output_shape, resized.shape, strict=True))
    restored[common] = resized[common]
    return restored


def _resampled_case_for_submission(
    case: LoadedSubmissionCase,
    target_spacing_xyz: tuple[float, float, float] | None,
) -> LoadedSubmissionCase:
    if target_spacing_xyz is None:
        return case
    if np.allclose(case.spacing_xyz.astype(np.float32), np.asarray(target_spacing_xyz, dtype=np.float32), atol=1e-6):
        return case
    if not case.prompts:
        return case
    first_prompt = case.prompts[0]
    image, _, spacing_xyz, endpoints_xyz, slice_index = _resample_loaded_case_recist(
        image=case.image,
        raw_mask=None,
        spacing_xyz=case.spacing_xyz,
        recist_endpoints_xyz=first_prompt.endpoints_xyz,
        recist_slice_index=first_prompt.slice_index,
        target_spacing_xyz=target_spacing_xyz,
    )
    scale_xyz = case.spacing_xyz.astype(np.float32) / np.asarray(target_spacing_xyz, dtype=np.float32)
    max_xyz = np.asarray([image.shape[2] - 1, image.shape[1] - 1, image.shape[0] - 1], dtype=np.float32)
    prompts: list[PromptSpec] = []
    for index, prompt in enumerate(case.prompts):
        if index == 0:
            resampled_endpoints = endpoints_xyz
            resampled_slice_index = slice_index
        else:
            resampled_endpoints = np.clip(prompt.endpoints_xyz * scale_xyz[None, :], 0.0, max_xyz[None, :])
            resampled_slice_index = int(np.clip(round(float(prompt.slice_index) * float(scale_xyz[2])), 0, image.shape[0] - 1))
        prompts.append(
            PromptSpec(
                label_value=prompt.label_value,
                endpoints_xyz=np.asarray(resampled_endpoints, dtype=np.float32),
                slice_index=int(resampled_slice_index),
                marker_voxels=prompt.marker_voxels,
            )
        )
    return LoadedSubmissionCase(
        case_id=case.case_id,
        image=np.asarray(image, dtype=np.float32),
        spacing_xyz=np.asarray(spacing_xyz, dtype=np.float32),
        direction=case.direction,
        origin=case.origin,
        prompts=prompts,
        ground_truth=case.ground_truth,
        normalization_stats=case.normalization_stats,
    )


def _load_model(
    checkpoint_path: Path | None,
    *,
    model_name: str,
    deep_supervision: bool,
) -> torch.nn.Module:
    model = build_model(model_name, in_channels=3, deep_supervision=deep_supervision)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_state = state["model_state"]
        if all(str(key).startswith("module.") for key in model_state):
            model_state = {str(key).removeprefix("module."): value for key, value in model_state.items()}
        incompatible = model.load_state_dict(model_state, strict=False)
        unexpected = [key for key in incompatible.unexpected_keys if not str(key).startswith("aux_out.")]
        if incompatible.missing_keys or unexpected:
            raise RuntimeError(
                "Checkpoint is incompatible with the CPU submission model: "
                f"missing_keys={list(incompatible.missing_keys)} unexpected_keys={unexpected}"
            )
    model.to("cpu")
    model.eval()
    return model


def _save_prediction_npz(path: Path, case: LoadedSubmissionCase, prediction: np.ndarray) -> None:
    payload: dict[str, Any] = {
        "prediction": prediction.astype(np.uint16, copy=False),
        "spacing": case.spacing_xyz.astype(np.float32),
    }
    if case.direction is not None:
        payload["direction"] = case.direction
    if case.origin is not None:
        payload["origin"] = case.origin
    np.savez_compressed(path, **payload)


def _save_prediction_nifti(path: Path, case: LoadedSubmissionCase, prediction: np.ndarray) -> None:
    affine = np.eye(4, dtype=np.float64)
    direction = np.eye(3, dtype=np.float64) if case.direction is None else case.direction.astype(np.float64)
    spacing = case.spacing_xyz.astype(np.float64)
    affine[:3, :3] = direction @ np.diag(spacing)
    if case.origin is not None:
        affine[:3, 3] = case.origin.astype(np.float64)
    image_xyz = prediction.astype(np.uint16, copy=False).transpose(2, 1, 0)
    nib.save(nib.Nifti1Image(image_xyz, affine), str(path))


def _write_outputs(
    output_dir: Path,
    case: LoadedSubmissionCase,
    prediction: np.ndarray,
    output_format: str,
) -> list[str]:
    output_paths: list[str] = []
    if output_format in {"npz", "both"}:
        npz_path = output_dir / f"{case.case_id}_prediction.npz"
        _save_prediction_npz(npz_path, case, prediction)
        output_paths.append(str(npz_path))
    if output_format in {"nii", "both"}:
        nii_path = output_dir / f"{case.case_id}.nii.gz"
        _save_prediction_nifti(nii_path, case, prediction)
        output_paths.append(str(nii_path))
    return output_paths


def _predict_one_case(
    model: torch.nn.Module,
    path: Path,
    output_dir: Path,
    *,
    crop_size_zyx: tuple[int, int, int],
    prompt_sigma: float,
    threshold: float,
    output_format: str,
    time_limit_seconds: float,
    autozoom_enable: bool = False,
    autozoom_max_passes: int = 3,
    autozoom_min_passes: int = 1,
    autozoom_growth_zyx: tuple[float, float, float] = (1.5, 1.25, 1.25),
    autozoom_max_zoom_zyx: tuple[float, float, float] = (4.0, 2.0, 2.0),
    autozoom_scale_mode: str = "fixed",
    autozoom_refine_enable: bool = True,
    autozoom_refine_max_boxes: int = 4,
    normalization_stats: dict[str, object] | None = None,
    target_spacing_xyz: tuple[float, float, float] | None = None,
) -> dict[str, object]:
    total_start = time.perf_counter()
    load_start = time.perf_counter()
    case = _load_submission_npz(path)
    case.normalization_stats = normalization_stats
    inference_case = _resampled_case_for_submission(case, target_spacing_xyz)
    inference_case.normalization_stats = normalization_stats
    load_seconds = time.perf_counter() - load_start

    prediction = np.zeros(case.image.shape, dtype=np.uint16)
    prompt_rows: list[dict[str, object]] = []
    forward_seconds_total = 0.0
    memory_sampler = MemoryTimeSampler()
    memory_sampler.start()
    try:
        postprocess_start = time.perf_counter()
        for prompt in inference_case.prompts:
            prompt_start = time.perf_counter()
            prompt_prediction_resampled, forward_seconds, _ = predict_case(
                model=model,
                case=_build_case_namespace(inference_case, prompt),
                crop_size_zyx=crop_size_zyx,
                prompt_sigma=prompt_sigma,
                threshold=threshold,
                device="cpu",
                prompt_mode="full",
                apply_postprocess=True,
                autozoom_enable=autozoom_enable,
                autozoom_max_passes=autozoom_max_passes,
                autozoom_min_passes=autozoom_min_passes,
                autozoom_growth_zyx=autozoom_growth_zyx,
                autozoom_max_zoom_zyx=autozoom_max_zoom_zyx,
                autozoom_scale_mode=autozoom_scale_mode,
                autozoom_refine_enable=autozoom_refine_enable,
                autozoom_refine_max_boxes=autozoom_refine_max_boxes,
            )
            prompt_prediction = _resize_prediction_to_shape(prompt_prediction_resampled, tuple(int(dim) for dim in case.image.shape))
            prompt_elapsed = time.perf_counter() - prompt_start
            forward_seconds_total += float(forward_seconds)
            prompt_mask = prompt_prediction > 0
            overlap_voxels = int(np.logical_and(prediction > 0, prompt_mask).sum())
            new_prompt_voxels = np.logical_and(prediction == 0, prompt_mask)
            prediction[new_prompt_voxels] = np.uint16(prompt.label_value)
            prompt_rows.append(
                {
                    "label_value": prompt.label_value,
                    "marker_voxels": prompt.marker_voxels,
                    "forward_seconds": round(float(forward_seconds), 6),
                    "prompt_total_seconds": round(float(prompt_elapsed), 6),
                    "inference_predicted_voxels": int(prompt_prediction_resampled.sum()),
                    "predicted_voxels": int(prompt_prediction.sum()),
                    "assigned_voxels": int(new_prompt_voxels.sum()),
                    "overlap_voxels": overlap_voxels,
                }
            )
            del prompt_prediction_resampled, prompt_prediction, prompt_mask, new_prompt_voxels
        postprocess_seconds = time.perf_counter() - postprocess_start - forward_seconds_total

        save_start = time.perf_counter()
        output_paths = _write_outputs(output_dir, case, prediction, output_format)
        save_seconds = time.perf_counter() - save_start

        total_seconds = time.perf_counter() - total_start
    finally:
        memory_stats = memory_sampler.stop()
    row: dict[str, object] = {
        "case_id": case.case_id,
        "input_path": str(path),
        "output_paths": output_paths,
        "status": "ok" if total_seconds <= time_limit_seconds else "timeout",
        "shape_zyx": list(case.image.shape),
        "inference_shape_zyx": list(inference_case.image.shape),
        "spacing_xyz": case.spacing_xyz.tolist(),
        "inference_spacing_xyz": inference_case.spacing_xyz.tolist(),
        "prompt_count": len(case.prompts),
        "prompts": prompt_rows,
        "load_seconds": round(float(load_seconds), 6),
        "forward_seconds": round(float(forward_seconds_total), 6),
        "postprocess_seconds": round(float(max(postprocess_seconds, 0.0)), 6),
        "save_seconds": round(float(save_seconds), 6),
        "total_seconds": round(float(total_seconds), 6),
        "time_limit_seconds": round(float(time_limit_seconds), 6),
        "rss_mb": round(_rss_mb(), 3),
        "sample_peak_rss_mb": round(float(memory_stats["sample_peak_rss_mb"]), 3),
        "cpu_memory_time_mb_seconds": round(float(memory_stats["memory_time_mb_seconds"]), 6),
        "predicted_voxels": int(np.count_nonzero(prediction)),
    }
    if case.ground_truth is not None:
        row |= {f"binary_{key}": round(float(value), 6) for key, value in compute_binary_stats(case.ground_truth, prediction).items()}
        labeled_metrics = compute_labeled_case_metrics(
            case.ground_truth,
            prediction,
            case.spacing_xyz,
            [prompt.label_value for prompt in case.prompts],
        )
        row |= {
            "dsc": round(float(labeled_metrics["dsc"]), 6),
            "nsd": round(float(labeled_metrics["nsd"]), 6),
            "lesion_count": int(labeled_metrics["lesion_count"]),
            "lesion_metrics": [
                {
                    "label_value": int(metric["label_value"]),
                    "dsc": round(float(metric["dsc"]), 6),
                    "nsd": round(float(metric["nsd"]), 6),
                    "gt_voxels": int(metric["gt_voxels"]),
                    "pred_voxels": int(metric["pred_voxels"]),
                }
                for metric in labeled_metrics["lesions"]
            ],
        }
    return row


def run_submission_prediction(
    *,
    inputs_dir: Path,
    outputs_dir: Path,
    checkpoint_path: Path | None,
    model_name: str,
    crop_size_zyx: tuple[int, int, int],
    prompt_sigma: float,
    threshold: float,
    deep_supervision: bool,
    allow_untrained_smoke: bool,
    output_format: str,
    threads: int,
    time_limit_seconds: float,
    autozoom_enable: bool = False,
    autozoom_max_passes: int = 3,
    autozoom_min_passes: int = 1,
    autozoom_growth_zyx: tuple[float, float, float] = (1.5, 1.25, 1.25),
    autozoom_max_zoom_zyx: tuple[float, float, float] = (4.0, 2.0, 2.0),
    autozoom_scale_mode: str = "fixed",
    autozoom_refine_enable: bool = True,
    autozoom_refine_max_boxes: int = 4,
    max_cases: int | None = None,
) -> dict[str, object]:
    if checkpoint_path is not None and not checkpoint_path.exists():
        if not allow_untrained_smoke:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint_path = None
    if checkpoint_path is None and not allow_untrained_smoke:
        raise FileNotFoundError("No checkpoint provided. Set TASK2_ALLOW_UNTRAINED_SMOKE=1 only for plumbing tests.")
    if output_format not in {"nii", "npz", "both"}:
        raise ValueError(f"Unsupported output format: {output_format}")

    torch.set_num_threads(max(int(threads), 1))
    checkpoint_settings = _model_settings_from_checkpoint(checkpoint_path)
    resolved_model_name = str(checkpoint_settings.get("model_name") or model_name)
    resolved_deep_supervision = bool(checkpoint_settings.get("deep_supervision", deep_supervision))
    resolved_crop = checkpoint_settings.get("crop_size_zyx") or crop_size_zyx
    resolved_crop_size_zyx = tuple(int(value) for value in resolved_crop)  # type: ignore[arg-type]
    resolved_prompt_sigma = float(checkpoint_settings.get("prompt_sigma") or prompt_sigma)
    checkpoint_threshold = checkpoint_settings.get("threshold")
    resolved_threshold = float(threshold)
    raw_target_spacing = checkpoint_settings.get("target_spacing_xyz")
    resolved_target_spacing_xyz = (
        tuple(float(value) for value in raw_target_spacing)  # type: ignore[union-attr]
        if raw_target_spacing is not None
        else None
    )
    resolved_normalization_stats = checkpoint_settings.get("normalization_stats")
    if not isinstance(resolved_normalization_stats, dict):
        resolved_normalization_stats = None

    outputs_dir.mkdir(parents=True, exist_ok=True)
    input_paths = sorted(path for path in inputs_dir.rglob("*.npz") if path.is_file())
    if max_cases is not None:
        input_paths = input_paths[: int(max_cases)]
    if not input_paths:
        raise FileNotFoundError(f"No .npz inputs found under {inputs_dir}")

    model = _load_model(
        checkpoint_path,
        model_name=resolved_model_name,
        deep_supervision=resolved_deep_supervision,
    )

    log_path = outputs_dir / "prediction_log.jsonl"
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        header = {
            "event": "start",
            "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
            "untrained_smoke": checkpoint_path is None,
            "model_name": resolved_model_name,
            "crop_size_zyx": list(resolved_crop_size_zyx),
            "prompt_sigma": resolved_prompt_sigma,
            "threshold": resolved_threshold,
            "checkpoint_threshold": checkpoint_threshold,
            "threshold_source": "cli_or_env",
            "target_spacing_xyz": resolved_target_spacing_xyz,
            "threads": torch.get_num_threads(),
            "output_format": output_format,
            "input_count": len(input_paths),
            "autozoom_enable": autozoom_enable,
            "autozoom_max_passes": autozoom_max_passes,
            "autozoom_min_passes": autozoom_min_passes,
            "autozoom_growth_zyx": list(autozoom_growth_zyx),
            "autozoom_max_zoom_zyx": list(autozoom_max_zoom_zyx),
            "autozoom_scale_mode": str(autozoom_scale_mode),
            "autozoom_refine_enable": autozoom_refine_enable,
            "autozoom_refine_max_boxes": autozoom_refine_max_boxes,
            "normalization_stats": resolved_normalization_stats,
        }
        handle.write(json.dumps(header, sort_keys=True, default=_json_default) + "\n")
        handle.flush()
        for index, input_path in enumerate(input_paths, start=1):
            try:
                row = _predict_one_case(
                    model,
                    input_path,
                    outputs_dir,
                    crop_size_zyx=resolved_crop_size_zyx,
                    prompt_sigma=resolved_prompt_sigma,
                    threshold=resolved_threshold,
                    output_format=output_format,
                    time_limit_seconds=time_limit_seconds,
                    autozoom_enable=autozoom_enable,
                    autozoom_max_passes=autozoom_max_passes,
                    autozoom_min_passes=autozoom_min_passes,
                    autozoom_growth_zyx=autozoom_growth_zyx,
                    autozoom_max_zoom_zyx=autozoom_max_zoom_zyx,
                    autozoom_scale_mode=autozoom_scale_mode,
                    autozoom_refine_enable=autozoom_refine_enable,
                    autozoom_refine_max_boxes=autozoom_refine_max_boxes,
                    normalization_stats=resolved_normalization_stats,
                    target_spacing_xyz=resolved_target_spacing_xyz,
                )
            except Exception as error:
                row = {
                    "case_id": input_path.stem,
                    "input_path": str(input_path),
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "rss_mb": round(_rss_mb(), 3),
                }
            row["event"] = "case"
            row["case_index"] = index
            row["case_count"] = len(input_paths)
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")
            handle.flush()

    failed = [row for row in rows if row.get("status") != "ok"]
    summary = {
        "event": "summary",
        "case_count": len(rows),
        "failed_case_count": len(failed),
        "timeout_case_count": sum(1 for row in rows if row.get("status") == "timeout"),
        "max_total_seconds": max((float(row.get("total_seconds", 0.0)) for row in rows), default=0.0),
        "mean_total_seconds": float(np.mean([float(row.get("total_seconds", 0.0)) for row in rows])) if rows else 0.0,
        "max_rss_mb": max((float(row.get("rss_mb", 0.0)) for row in rows), default=0.0),
        "total_cpu_memory_time_mb_seconds": float(
            np.sum([float(row.get("cpu_memory_time_mb_seconds", 0.0)) for row in rows])
        )
        if rows
        else 0.0,
        "mean_cpu_memory_time_mb_seconds": float(
            np.mean([float(row.get("cpu_memory_time_mb_seconds", 0.0)) for row in rows])
        )
        if rows
        else 0.0,
        "total_seconds": time.perf_counter() - started,
        "log_path": str(log_path),
    }
    metric_rows = [
        {"dsc": float(row["dsc"]), "nsd": float(row["nsd"])}
        for row in rows
        if "dsc" in row and "nsd" in row
    ]
    if metric_rows:
        summary |= aggregate_metrics(metric_rows)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, sort_keys=True, default=_json_default) + "\n")
    (outputs_dir / "prediction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    if failed:
        raise RuntimeError(f"{len(failed)} prediction case(s) failed or exceeded time limit; see {log_path}")
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-only LETT-NeXt submission predictor for NPZ mirror inputs.")
    parser.add_argument("--inputs", type=Path, default=Path(os.environ.get("TASK2_INPUT_DIR", "/workspace/inputs")))
    parser.add_argument("--outputs", type=Path, default=Path(os.environ.get("TASK2_OUTPUT_DIR", "/workspace/outputs")))
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("TASK2_CHECKPOINT", "/workspace/model/checkpoint.pt")))
    parser.add_argument("--model-name", default=os.environ.get("TASK2_MODEL_NAME", "mednextv2_f32"))
    parser.add_argument("--crop-size-zyx", type=_parse_tuple_3, default=_parse_tuple_3(os.environ.get("TASK2_CROP_SIZE_ZYX", "128,160,160")))
    parser.add_argument("--prompt-sigma", type=float, default=float(os.environ.get("TASK2_PROMPT_SIGMA", "2.0")))
    parser.add_argument("--threshold", type=float, default=float(os.environ.get("TASK2_THRESHOLD", "0.5")))
    parser.add_argument("--threads", type=int, default=int(os.environ.get("TASK2_CPU_THREADS", os.environ.get("OMP_NUM_THREADS", "12"))))
    parser.add_argument("--time-limit-seconds", type=float, default=float(os.environ.get("TASK2_TIME_LIMIT_SECONDS", "60")))
    parser.add_argument("--max-cases", type=int, default=None if os.environ.get("TASK2_MAX_CASES") in (None, "") else int(os.environ["TASK2_MAX_CASES"]))
    parser.add_argument("--output-format", choices=("nii", "npz", "both"), default=os.environ.get("TASK2_OUTPUT_FORMAT", "nii"))
    parser.add_argument("--deep-supervision", action="store_true", default=_bool_env("TASK2_DEEP_SUPERVISION", False))
    parser.add_argument("--allow-untrained-smoke", action="store_true", default=_bool_env("TASK2_ALLOW_UNTRAINED_SMOKE", False))
    parser.add_argument("--autozoom-enable", action="store_true", default=_bool_env("TASK2_AUTOZOOM_ENABLE", False))
    parser.add_argument("--autozoom-max-passes", type=int, default=int(os.environ.get("TASK2_AUTOZOOM_MAX_PASSES", "3")))
    parser.add_argument("--autozoom-min-passes", type=int, default=int(os.environ.get("TASK2_AUTOZOOM_MIN_PASSES", "1")))
    parser.add_argument(
        "--autozoom-growth-zyx",
        type=_parse_float_tuple_3,
        default=_parse_float_tuple_3(os.environ.get("TASK2_AUTOZOOM_GROWTH_ZYX", "1.5,1.25,1.25")),
    )
    parser.add_argument(
        "--autozoom-max-zoom-zyx",
        type=_parse_float_tuple_3,
        default=_parse_float_tuple_3(os.environ.get("TASK2_AUTOZOOM_MAX_ZOOM_ZYX", "4.0,2.0,2.0")),
    )
    parser.add_argument(
        "--autozoom-scale-mode",
        choices=("fixed", "adaptive"),
        default=os.environ.get("TASK2_AUTOZOOM_SCALE_MODE", "fixed"),
    )
    parser.add_argument("--autozoom-refine-disable", action="store_true", default=_bool_env("TASK2_AUTOZOOM_REFINE_DISABLE", False))
    parser.add_argument("--autozoom-refine-max-boxes", type=int, default=int(os.environ.get("TASK2_AUTOZOOM_REFINE_MAX_BOXES", "4")))
    return parser.parse_args(argv)


def _console_summary(summary: dict[str, object]) -> dict[str, object]:
    keys = (
        "case_count",
        "failed_case_count",
        "timeout_case_count",
        "max_total_seconds",
        "mean_total_seconds",
        "total_seconds",
        "max_rss_mb",
        "mean_cpu_memory_time_mb_seconds",
        "total_cpu_memory_time_mb_seconds",
    )
    return {key: summary[key] for key in keys if key in summary}


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    checkpoint_path = args.checkpoint
    if str(checkpoint_path) in {"", "none", "None"}:
        checkpoint_path = None
    summary = run_submission_prediction(
        inputs_dir=args.inputs,
        outputs_dir=args.outputs,
        checkpoint_path=checkpoint_path,
        model_name=args.model_name,
        crop_size_zyx=args.crop_size_zyx,
        prompt_sigma=args.prompt_sigma,
        threshold=args.threshold,
        deep_supervision=args.deep_supervision,
        allow_untrained_smoke=args.allow_untrained_smoke,
        output_format=args.output_format,
        threads=args.threads,
        time_limit_seconds=args.time_limit_seconds,
        autozoom_enable=args.autozoom_enable,
        autozoom_max_passes=args.autozoom_max_passes,
        autozoom_min_passes=args.autozoom_min_passes,
        autozoom_growth_zyx=args.autozoom_growth_zyx,
        autozoom_max_zoom_zyx=args.autozoom_max_zoom_zyx,
        autozoom_scale_mode=args.autozoom_scale_mode,
        autozoom_refine_enable=not args.autozoom_refine_disable,
        autozoom_refine_max_boxes=args.autozoom_refine_max_boxes,
        max_cases=args.max_cases,
    )
    print(json.dumps(_console_summary(summary), indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
