from __future__ import annotations

import numpy as np

from ...prompts import build_prompt_channels
from .normalization import NormalizationStats, normalize_image


def _apply_intensity_augmentation(
    normalized: np.ndarray,
    intensity_shift: float,
    intensity_scale: float,
    noise_std: float,
) -> np.ndarray:
    if intensity_scale > 0.0:
        normalized = normalized * np.float32(
            1.0 + np.random.uniform(-float(intensity_scale), float(intensity_scale))
        )
    if intensity_shift > 0.0:
        normalized = normalized + np.float32(
            np.random.uniform(-float(intensity_shift), float(intensity_shift))
        )
    if noise_std > 0.0:
        normalized = normalized + np.random.normal(
            0.0,
            float(noise_std),
            size=normalized.shape,
        ).astype(np.float32)
    return normalized.astype(np.float32)


def perturb_recist_endpoints_for_training(
    endpoints_xyz: np.ndarray,
    shape_zyx: tuple[int, int, int],
    probability: float = 0.0,
    shift_xy: int = 0,
    endpoint_jitter_xy: int = 0,
    slice_jitter: int = 0,
    length_scale_min: float = 1.0,
    length_scale_max: float = 1.0,
) -> np.ndarray:
    endpoints = np.asarray(endpoints_xyz, dtype=np.float32).reshape(2, 3).copy()
    probability = float(np.clip(probability, 0.0, 1.0))
    if probability <= 0.0 or float(np.random.random()) >= probability:
        return endpoints

    shift_xy = max(int(shift_xy), 0)
    endpoint_jitter_xy = max(int(endpoint_jitter_xy), 0)
    slice_jitter = max(int(slice_jitter), 0)
    length_scale_min = float(length_scale_min)
    length_scale_max = float(length_scale_max)
    if length_scale_min > length_scale_max:
        raise ValueError(
            "train_recist_aug_length_scale_min must be <= train_recist_aug_length_scale_max"
        )

    if length_scale_min != 1.0 or length_scale_max != 1.0:
        scale = float(np.random.uniform(length_scale_min, length_scale_max))
        midpoint_xy = endpoints[:, :2].mean(axis=0)
        endpoints[:, :2] = midpoint_xy + (endpoints[:, :2] - midpoint_xy) * scale

    if shift_xy > 0:
        shared_shift_xy = np.asarray(
            [
                np.random.randint(-shift_xy, shift_xy + 1),
                np.random.randint(-shift_xy, shift_xy + 1),
            ],
            dtype=np.float32,
        )
        endpoints[:, :2] += shared_shift_xy[None, :]

    if endpoint_jitter_xy > 0:
        endpoint_jitter = np.random.randint(
            -endpoint_jitter_xy,
            endpoint_jitter_xy + 1,
            size=(2, 2),
        ).astype(np.float32)
        endpoints[:, :2] += endpoint_jitter

    if slice_jitter > 0:
        endpoints[:, 2] += np.float32(np.random.randint(-slice_jitter, slice_jitter + 1))

    max_xyz = np.asarray(
        [shape_zyx[2] - 1, shape_zyx[1] - 1, shape_zyx[0] - 1],
        dtype=np.float32,
    )
    return np.clip(endpoints, 0.0, max_xyz).astype(np.float32)


def build_input_channels(
    image_crop: np.ndarray,
    endpoints_xyz: np.ndarray,
    prompt_sigma: float,
    prompt_mode: str = "full",
    case_id: str | None = None,
    prompt_source: str | None = None,
    intensity_shift: float = 0.0,
    intensity_scale: float = 0.0,
    noise_std: float = 0.0,
    normalization_stats: NormalizationStats | dict[str, object] | None = None,
    normalization_mode: str = "legacy_crop_zscore",
) -> tuple[np.ndarray, np.ndarray]:
    normalized = _apply_intensity_augmentation(
        normalize_image(
            image_crop,
            normalization_stats=normalization_stats,
            normalization_mode=normalization_mode,
        ),
        intensity_shift=intensity_shift,
        intensity_scale=intensity_scale,
        noise_std=noise_std,
    )
    prompt = build_prompt_channels(
        shape_zyx=image_crop.shape,
        endpoints_xyz=endpoints_xyz,
        sigma=prompt_sigma,
        prompt_mode=prompt_mode,
        case_id=case_id,
        prompt_source=prompt_source,
    )
    channels = [normalized, prompt.line, prompt.endpoints]
    prompt_region = np.maximum(prompt.line, prompt.endpoints > 0.25).astype(np.float32)
    return np.stack(channels, axis=0).astype(np.float32), prompt_region
