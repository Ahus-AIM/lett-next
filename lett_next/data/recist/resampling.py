from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import ndimage

def resample_loaded_case_arrays(
    *,
    image: np.ndarray,
    raw_mask: np.ndarray | None,
    spacing_xyz: np.ndarray,
    recist_endpoints_xyz: np.ndarray,
    recist_slice_index: int,
    target_spacing_xyz: tuple[float, float, float],
    mode: Literal["standard", "anisotropic"],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, int]:
    source_spacing_xyz = np.asarray(spacing_xyz, dtype=np.float32)
    target_spacing_xyz_array = np.asarray(target_spacing_xyz, dtype=np.float32)
    if np.any(target_spacing_xyz_array <= 0):
        raise ValueError(f"target_spacing_xyz must be positive, got {target_spacing_xyz}")
    if np.allclose(source_spacing_xyz, target_spacing_xyz_array, atol=1e-6):
        return image, raw_mask, source_spacing_xyz, recist_endpoints_xyz, recist_slice_index
    zoom_zyx = tuple(float(v) for v in (source_spacing_xyz[::-1] / target_spacing_xyz_array[::-1]).tolist())
    if mode == "standard":
        resampled_image = ndimage.zoom(np.asarray(image, dtype=np.float32), zoom=zoom_zyx, order=1)
        resampled_raw_mask = None if raw_mask is None else ndimage.zoom(np.asarray(raw_mask), zoom=zoom_zyx, order=0)
    elif mode == "anisotropic":
        lowres_axis = _anisotropic_axis(source_spacing_xyz)
        resampled_image = _resample_array_anisotropic(
            np.asarray(image, dtype=np.float32),
            zoom_zyx,
            lowres_axis,
            inplane_order=1,
            lowres_order=0,
        )
        resampled_raw_mask = None
        if raw_mask is not None:
            resampled_raw_mask = _resample_array_anisotropic(
                np.asarray(raw_mask),
                zoom_zyx,
                lowres_axis,
                inplane_order=0,
                lowres_order=0,
            )
    else:
        raise ValueError(f"Unsupported resampling mode: {mode}")
    scale_xyz = source_spacing_xyz / target_spacing_xyz_array
    resampled_endpoints_xyz = np.asarray(recist_endpoints_xyz, dtype=np.float32) * scale_xyz[None, :]
    max_xyz = np.asarray(
        [resampled_image.shape[2] - 1, resampled_image.shape[1] - 1, resampled_image.shape[0] - 1],
        dtype=np.float32,
    )
    resampled_endpoints_xyz = np.clip(resampled_endpoints_xyz, 0.0, max_xyz[None, :])
    resampled_slice_index = int(np.clip(round(float(recist_slice_index) * float(scale_xyz[2])), 0, resampled_image.shape[0] - 1))
    return (
        np.asarray(resampled_image, dtype=np.float32),
        None if resampled_raw_mask is None else np.asarray(resampled_raw_mask),
        target_spacing_xyz_array.astype(np.float32),
        resampled_endpoints_xyz.astype(np.float32),
        resampled_slice_index,
    )


def _resample_loaded_case(
    image: np.ndarray,
    raw_mask: np.ndarray | None,
    spacing_xyz: np.ndarray,
    recist_endpoints_xyz: np.ndarray,
    recist_slice_index: int,
    target_spacing_xyz: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, int]:
    return resample_loaded_case_arrays(
        image=image,
        raw_mask=raw_mask,
        spacing_xyz=spacing_xyz,
        recist_endpoints_xyz=recist_endpoints_xyz,
        recist_slice_index=recist_slice_index,
        target_spacing_xyz=target_spacing_xyz,
        mode="standard",
    )


def _zoom_array(array: np.ndarray, zoom_zyx: tuple[float, float, float], order: int) -> np.ndarray:
    if np.allclose(np.asarray(zoom_zyx, dtype=np.float32), np.ones(3, dtype=np.float32), atol=1e-6):
        return np.asarray(array).copy()
    return ndimage.zoom(np.asarray(array), zoom=zoom_zyx, order=int(order))


def _anisotropic_axis(spacing_xyz: np.ndarray, threshold: float = 3.0) -> int | None:
    spacing_zyx = np.asarray(spacing_xyz, dtype=np.float32)[::-1]
    min_spacing = float(np.min(spacing_zyx))
    if min_spacing <= 0.0:
        return None
    axis = int(np.argmax(spacing_zyx))
    return axis if float(spacing_zyx[axis] / min_spacing) >= float(threshold) else None


def _resample_array_anisotropic(
    array: np.ndarray,
    zoom_zyx: tuple[float, float, float],
    lowres_axis: int | None,
    *,
    inplane_order: int,
    lowres_order: int,
) -> np.ndarray:
    if lowres_axis is None:
        return _zoom_array(array, zoom_zyx, order=inplane_order)
    first_zoom = [float(value) for value in zoom_zyx]
    first_zoom[lowres_axis] = 1.0
    resampled = _zoom_array(np.asarray(array), tuple(first_zoom), order=inplane_order)
    second_zoom = [1.0, 1.0, 1.0]
    second_zoom[lowres_axis] = float(zoom_zyx[lowres_axis])
    return _zoom_array(resampled, tuple(second_zoom), order=lowres_order)


def _resample_loaded_case_recist(
    image: np.ndarray,
    raw_mask: np.ndarray | None,
    spacing_xyz: np.ndarray,
    recist_endpoints_xyz: np.ndarray,
    recist_slice_index: int,
    target_spacing_xyz: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, int]:
    return resample_loaded_case_arrays(
        image=image,
        raw_mask=raw_mask,
        spacing_xyz=spacing_xyz,
        recist_endpoints_xyz=recist_endpoints_xyz,
        recist_slice_index=recist_slice_index,
        target_spacing_xyz=target_spacing_xyz,
        mode="anisotropic",
    )


def _resample_anatomy_aux_arrays(
    aux_target: np.ndarray,
    aux_valid_mask: np.ndarray,
    *,
    source_spacing_xyz: np.ndarray,
    target_spacing_xyz: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    source_spacing_xyz = np.asarray(source_spacing_xyz, dtype=np.float32)
    target_spacing_xyz_array = np.asarray(target_spacing_xyz, dtype=np.float32)
    if np.allclose(source_spacing_xyz, target_spacing_xyz_array, atol=1e-6):
        return np.asarray(aux_target, dtype=np.int16), np.asarray(aux_valid_mask, dtype=bool)
    zoom_zyx = tuple(float(v) for v in (source_spacing_xyz[::-1] / target_spacing_xyz_array[::-1]).tolist())
    lowres_axis = _anisotropic_axis(source_spacing_xyz)
    resampled_target = _resample_array_anisotropic(
        np.asarray(aux_target, dtype=np.int16),
        zoom_zyx,
        lowres_axis,
        inplane_order=0,
        lowres_order=0,
    )
    resampled_valid = _resample_array_anisotropic(
        np.asarray(aux_valid_mask, dtype=np.uint8),
        zoom_zyx,
        lowres_axis,
        inplane_order=0,
        lowres_order=0,
    )
    return np.asarray(resampled_target, dtype=np.int16), (np.asarray(resampled_valid) > 0)
