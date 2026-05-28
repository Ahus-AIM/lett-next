from __future__ import annotations

import numpy as np
from scipy import ndimage


def compute_crop_slices(
    shape_zyx: tuple[int, int, int],
    center_zyx: np.ndarray,
    crop_size_zyx: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    starts: list[int] = []
    stops: list[int] = []
    for dim, crop_dim, center_value in zip(shape_zyx, crop_size_zyx, center_zyx, strict=True):
        crop_dim = min(int(crop_dim), int(dim))
        start = int(round(float(center_value))) - crop_dim // 2
        start = max(0, min(start, dim - crop_dim))
        starts.append(start)
        stops.append(start + crop_dim)
    return tuple(slice(start, stop) for start, stop in zip(starts, stops, strict=True))  # type: ignore[return-value]


def _sample_jitter(jitter_zyx: tuple[int, int, int]) -> np.ndarray:
    return np.asarray(
        [
            np.random.randint(-int(jitter_zyx[0]), int(jitter_zyx[0]) + 1) if jitter_zyx[0] > 0 else 0,
            np.random.randint(-int(jitter_zyx[1]), int(jitter_zyx[1]) + 1) if jitter_zyx[1] > 0 else 0,
            np.random.randint(-int(jitter_zyx[2]), int(jitter_zyx[2]) + 1) if jitter_zyx[2] > 0 else 0,
        ],
        dtype=np.float32,
    )


def pad_array_to_shape(
    array: np.ndarray,
    target_shape_zyx: tuple[int, int, int],
    *,
    constant_value: int | float | bool = 0,
) -> np.ndarray:
    pad_width = []
    needs_padding = False
    for dim, target_dim in zip(array.shape, target_shape_zyx, strict=True):
        pad_after = max(int(target_dim) - int(dim), 0)
        pad_width.append((0, pad_after))
        needs_padding = needs_padding or pad_after > 0
    padded = np.pad(array, pad_width, mode="constant", constant_values=constant_value) if needs_padding else array
    return padded[tuple(slice(0, int(dim)) for dim in target_shape_zyx)]


def extract_prompt_crop(
    image: np.ndarray,
    mask: np.ndarray | None,
    endpoints_xyz: np.ndarray,
    crop_size_zyx: tuple[int, int, int],
    jitter_zyx: tuple[int, int, int] = (0, 0, 0),
) -> dict[str, object]:
    endpoints_xyz = np.asarray(endpoints_xyz, dtype=np.float32)
    midpoint_xyz = endpoints_xyz.mean(axis=0)
    center_zyx = midpoint_xyz[::-1] + _sample_jitter(jitter_zyx)
    crop_slices = compute_crop_slices(image.shape, center_zyx, crop_size_zyx)
    crop_start_xyz = np.asarray(
        [crop_slices[2].start, crop_slices[1].start, crop_slices[0].start],
        dtype=np.float32,
    )
    cropped_endpoints_xyz = endpoints_xyz - crop_start_xyz
    image_crop = image[crop_slices]
    mask_crop = None if mask is None else mask[crop_slices]
    restore_shape_zyx = tuple(int(dim) for dim in image_crop.shape)
    image_crop = pad_array_to_shape(image_crop, crop_size_zyx, constant_value=0.0)
    if mask_crop is not None:
        mask_crop = pad_array_to_shape(mask_crop, crop_size_zyx, constant_value=0)
    return {
        "image": image_crop,
        "mask": mask_crop,
        "crop_slices": crop_slices,
        "endpoints_xyz": cropped_endpoints_xyz.astype(np.float32),
        "slice_index": int(round(float(cropped_endpoints_xyz[:, 2].mean()))),
        "restore_shape_zyx": restore_shape_zyx,
    }


def _resize_crop_exact(
    image_crop: np.ndarray,
    mask_crop: np.ndarray | None,
    target_shape_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    source_shape = np.asarray(image_crop.shape, dtype=np.float32)
    target_shape = np.asarray(target_shape_zyx, dtype=np.int64)
    zoom = target_shape.astype(np.float32) / np.maximum(source_shape, 1.0)
    if np.allclose(zoom, np.ones(3, dtype=np.float32), atol=1e-6):
        return image_crop.astype(np.float32, copy=False), None if mask_crop is None else (mask_crop > 0).astype(np.uint8), zoom

    resized_image = ndimage.zoom(np.asarray(image_crop, dtype=np.float32), zoom=tuple(float(v) for v in zoom), order=1)
    resized_mask = None
    if mask_crop is not None:
        resized_mask = ndimage.zoom((np.asarray(mask_crop) > 0).astype(np.uint8), zoom=tuple(float(v) for v in zoom), order=0)

    slices = tuple(slice(0, int(dim)) for dim in target_shape.tolist())
    resized_image = resized_image[slices]
    if resized_mask is not None:
        resized_mask = resized_mask[slices]

    resized_image, resized_mask = _pad_to_shape(resized_image, resized_mask, target_shape_zyx)
    if resized_mask is not None:
        resized_mask = (resized_mask > 0).astype(np.uint8, copy=False)
        source_points = np.argwhere(np.asarray(mask_crop) > 0)
        if len(source_points) > 0:
            projected = np.floor((source_points.astype(np.float32) + 0.5) * zoom[None, :]).astype(np.int64)
            for axis, dim in enumerate(target_shape.tolist()):
                projected[:, axis] = np.clip(projected[:, axis], 0, int(dim) - 1)
            resized_mask[tuple(projected[:, axis] for axis in range(3))] = 1
    return resized_image.astype(np.float32, copy=False), resized_mask, zoom


def _endpoint_margin_voxels_zyx(
    spacing_xyz: np.ndarray | tuple[float, float, float] | None,
    margin_mm: float,
) -> np.ndarray:
    if spacing_xyz is None or margin_mm <= 0.0:
        return np.zeros(3, dtype=np.float32)
    spacing = np.maximum(np.asarray(spacing_xyz, dtype=np.float32), 1e-6)
    return (float(margin_mm) / spacing)[::-1].astype(np.float32)


def _center_with_endpoint_margin(
    shape_zyx: tuple[int, int, int],
    center_zyx: np.ndarray,
    source_size_zyx: tuple[int, int, int],
    endpoints_xyz: np.ndarray,
    spacing_xyz: np.ndarray | tuple[float, float, float] | None,
    margin_mm: float,
) -> np.ndarray:
    shape = np.asarray(shape_zyx, dtype=np.int64)
    source_size = np.asarray(source_size_zyx, dtype=np.int64)
    endpoints_zyx = np.asarray(endpoints_xyz, dtype=np.float32)[:, ::-1]
    margin_zyx = _endpoint_margin_voxels_zyx(spacing_xyz, margin_mm)
    adjusted = np.asarray(center_zyx, dtype=np.float32).copy()
    for axis in range(3):
        dim = int(shape[axis])
        window = min(int(source_size[axis]), dim)
        if dim <= 0 or window <= 0:
            continue
        preferred = int(round(float(adjusted[axis]))) - window // 2
        low_start = max(0, int(np.ceil(float(endpoints_zyx[:, axis].max() + margin_zyx[axis] - (window - 1)))))
        high_start = min(dim - window, int(np.floor(float(endpoints_zyx[:, axis].min() - margin_zyx[axis]))))
        if low_start <= high_start:
            start = max(low_start, min(preferred, high_start))
            adjusted[axis] = float(start + window // 2)
    return adjusted


def extract_multifov_recist_prompt_crop(
    image: np.ndarray,
    mask: np.ndarray,
    endpoints_xyz: np.ndarray,
    crop_size_zyx: tuple[int, int, int],
    source_fov_scale_zyx: tuple[float, float, float],
    jitter_zyx: tuple[int, int, int] = (0, 0, 0),
    ensure_endpoints_inside: bool = False,
    endpoint_margin_mm: float = 0.0,
    spacing_xyz: np.ndarray | tuple[float, float, float] | None = None,
) -> dict[str, object]:
    endpoints_xyz = np.asarray(endpoints_xyz, dtype=np.float32)
    source_size_zyx = tuple(
        max(1, int(np.ceil(float(crop_dim) * float(scale))))
        for crop_dim, scale in zip(crop_size_zyx, source_fov_scale_zyx, strict=True)
    )
    center_zyx = endpoints_xyz.mean(axis=0)[::-1] + _sample_jitter(jitter_zyx)
    if ensure_endpoints_inside:
        center_zyx = _center_with_endpoint_margin(
            tuple(int(dim) for dim in image.shape),
            center_zyx,
            source_size_zyx,
            endpoints_xyz,
            spacing_xyz,
            endpoint_margin_mm,
        )
    crop_slices = compute_crop_slices(image.shape, center_zyx, source_size_zyx)
    crop_start_xyz = np.asarray(
        [crop_slices[2].start, crop_slices[1].start, crop_slices[0].start],
        dtype=np.float32,
    )
    source_image = np.asarray(image[crop_slices], dtype=np.float32)
    source_mask = (np.asarray(mask[crop_slices]) > 0).astype(np.uint8)
    restore_shape_zyx = tuple(int(dim) for dim in source_image.shape)
    source_image, source_mask = _pad_to_shape(source_image, source_mask, source_size_zyx)
    if source_mask is None:
        raise ValueError("Multifov RECIST crop unexpectedly lost its target mask")
    resized_image, resized_mask, zoom_zyx = _resize_crop_exact(source_image, source_mask, crop_size_zyx)
    if resized_mask is None:
        raise ValueError("Multifov RECIST crop unexpectedly lost its target mask after resize")
    source_endpoints_xyz = endpoints_xyz - crop_start_xyz[None, :]
    zoom_xyz = zoom_zyx[::-1]
    resized_endpoints_xyz = source_endpoints_xyz * zoom_xyz[None, :]
    max_xyz = np.asarray([crop_size_zyx[2] - 1, crop_size_zyx[1] - 1, crop_size_zyx[0] - 1], dtype=np.float32)
    resized_endpoints_xyz = np.clip(resized_endpoints_xyz, 0.0, max_xyz[None, :])
    return {
        "image": resized_image.astype(np.float32, copy=False),
        "mask": resized_mask.astype(np.uint8, copy=False),
        "crop_slices": crop_slices,
        "endpoints_xyz": resized_endpoints_xyz.astype(np.float32),
        "slice_index": int(round(float(resized_endpoints_xyz[:, 2].mean()))),
        "restore_shape_zyx": restore_shape_zyx,
        "source_fov_scale_zyx": tuple(float(value) for value in source_fov_scale_zyx),
        "source_crop_size_zyx": source_size_zyx,
        "resize_zoom_zyx": tuple(float(value) for value in zoom_zyx.tolist()),
    }


def _pad_to_shape(
    image: np.ndarray,
    mask: np.ndarray | None,
    target_shape_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray | None]:
    image = pad_array_to_shape(image, target_shape_zyx, constant_value=0.0)
    if mask is not None:
        mask = pad_array_to_shape(mask, target_shape_zyx, constant_value=0)
    return image, mask


def _apply_zyx_flips(
    image: np.ndarray,
    mask: np.ndarray,
    endpoints_xyz: np.ndarray,
    flip_axes_zyx: tuple[bool, bool, bool],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformed_endpoints = np.asarray(endpoints_xyz, dtype=np.float32).copy()
    for axis, should_flip in enumerate(flip_axes_zyx):
        if not should_flip:
            continue
        image = np.flip(image, axis=axis).copy()
        mask = np.flip(mask, axis=axis).copy()
        xyz_axis = 2 - axis
        transformed_endpoints[:, xyz_axis] = float(image.shape[axis] - 1) - transformed_endpoints[:, xyz_axis]
    return image, mask, transformed_endpoints


def _downscale_case_for_bbox_fit(
    image: np.ndarray,
    mask: np.ndarray,
    endpoints_xyz: np.ndarray,
    scale_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if np.allclose(scale_zyx, np.ones(3, dtype=np.float32), atol=1e-6):
        return image.astype(np.float32, copy=False), (mask > 0).astype(np.uint8), endpoints_xyz.astype(np.float32)

    zoom_zyx = tuple(float(value) for value in scale_zyx.tolist())
    resized_image = ndimage.zoom(np.asarray(image, dtype=np.float32), zoom=zoom_zyx, order=1)
    resized_mask = ndimage.zoom((np.asarray(mask) > 0).astype(np.uint8), zoom=zoom_zyx, order=0)
    resized_mask = (resized_mask > 0).astype(np.uint8, copy=False)

    source_points = np.argwhere(np.asarray(mask) > 0)
    if len(source_points) > 0:
        projected = np.floor((source_points.astype(np.float32) + 0.5) * scale_zyx[None, :]).astype(np.int64)
        for axis, dim in enumerate(resized_mask.shape):
            projected[:, axis] = np.clip(projected[:, axis], 0, int(dim) - 1)
        resized_mask[tuple(projected[:, axis] for axis in range(3))] = 1

    scale_xyz = scale_zyx[::-1]
    scaled_endpoints = np.asarray(endpoints_xyz, dtype=np.float32) * scale_xyz[None, :]
    max_xyz = np.asarray(
        [resized_image.shape[2] - 1, resized_image.shape[1] - 1, resized_image.shape[0] - 1],
        dtype=np.float32,
    )
    scaled_endpoints = np.clip(scaled_endpoints, 0.0, max_xyz[None, :])
    return resized_image.astype(np.float32, copy=False), resized_mask, scaled_endpoints.astype(np.float32)


def _bbox_fit_crop_slices(
    shape_zyx: tuple[int, int, int],
    bbox_min_zyx: np.ndarray,
    bbox_max_zyx: np.ndarray,
    endpoints_xyz: np.ndarray,
    crop_size_zyx: tuple[int, int, int],
    margin_zyx: tuple[int, int, int],
    jitter_zyx: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    shape = np.asarray(shape_zyx, dtype=np.int64)
    crop_size = np.asarray(crop_size_zyx, dtype=np.int64)
    margin = np.asarray(margin_zyx, dtype=np.int64)
    region_min = np.maximum(bbox_min_zyx.astype(np.int64) - margin, 0)
    region_max = np.minimum(bbox_max_zyx.astype(np.int64) + margin, shape)
    center_zyx = np.asarray(endpoints_xyz, dtype=np.float32).mean(axis=0)[::-1] + _sample_jitter(jitter_zyx)

    starts: list[int] = []
    stops: list[int] = []
    for axis in range(3):
        dim = int(shape[axis])
        target_dim = int(crop_size[axis])
        actual_dim = min(target_dim, dim)
        if actual_dim <= 0:
            raise ValueError(f"Invalid crop dimension for axis {axis}: shape={shape_zyx}, crop={crop_size_zyx}")
        min_start = max(0, int(region_max[axis]) - actual_dim)
        max_start = min(int(region_min[axis]), dim - actual_dim)
        preferred = int(round(float(center_zyx[axis]))) - actual_dim // 2
        if min_start <= max_start:
            start = max(min_start, min(preferred, max_start))
        else:
            start = max(0, min(preferred, dim - actual_dim))
        starts.append(int(start))
        stops.append(int(start + actual_dim))
    return tuple(slice(start, stop) for start, stop in zip(starts, stops, strict=True))  # type: ignore[return-value]


def extract_recist_bbox_fit_crop(
    image: np.ndarray,
    mask: np.ndarray,
    endpoints_xyz: np.ndarray,
    crop_size_zyx: tuple[int, int, int],
    margin_zyx: tuple[int, int, int] = (2, 4, 4),
    jitter_zyx: tuple[int, int, int] = (0, 0, 0),
) -> dict[str, object]:
    binary_mask = np.asarray(mask) > 0
    points = np.argwhere(binary_mask)
    if len(points) == 0:
        raise ValueError("Cannot build RECIST bbox-fit crop from an empty target mask")

    crop_size = np.asarray(crop_size_zyx, dtype=np.float32)
    margin = np.asarray(margin_zyx, dtype=np.float32)
    bbox_min = points.min(axis=0).astype(np.int64)
    bbox_max = (points.max(axis=0) + 1).astype(np.int64)
    bbox_size = (bbox_max - bbox_min).astype(np.float32)
    requested_region = bbox_size + 2.0 * margin
    scale_zyx = np.minimum(1.0, crop_size / np.maximum(requested_region, 1.0)).astype(np.float32)

    image_array = np.asarray(image, dtype=np.float32)
    endpoints_array = np.asarray(endpoints_xyz, dtype=np.float32)
    if np.any(scale_zyx < 0.999):
        shape = np.asarray(image_array.shape, dtype=np.int64)
        source_window = np.ceil(crop_size / np.maximum(scale_zyx, 1e-6)).astype(np.int64)
        source_window = np.maximum(source_window, np.ceil(requested_region).astype(np.int64))
        source_window = np.minimum(source_window, shape)
        source_slices = _bbox_fit_crop_slices(
            tuple(int(dim) for dim in image_array.shape),
            bbox_min,
            bbox_max,
            endpoints_array,
            tuple(int(dim) for dim in source_window.tolist()),
            margin_zyx,
            (0, 0, 0),
        )
        source_start_xyz = np.asarray(
            [source_slices[2].start, source_slices[1].start, source_slices[0].start],
            dtype=np.float32,
        )
        source_image = image_array[source_slices]
        source_mask = binary_mask[source_slices].astype(np.uint8)
        source_endpoints = endpoints_array - source_start_xyz[None, :]
    else:
        source_slices = tuple(slice(0, int(dim)) for dim in image_array.shape)
        source_image = image_array
        source_mask = binary_mask.astype(np.uint8)
        source_endpoints = endpoints_array

    scaled_image, scaled_mask, scaled_endpoints = _downscale_case_for_bbox_fit(
        image=source_image,
        mask=source_mask,
        endpoints_xyz=source_endpoints,
        scale_zyx=scale_zyx,
    )
    scaled_points = np.argwhere(scaled_mask > 0)
    if len(scaled_points) == 0:
        raise ValueError("RECIST bbox-fit crop lost the target mask during scaling")

    scaled_bbox_min = scaled_points.min(axis=0).astype(np.int64)
    scaled_bbox_max = (scaled_points.max(axis=0) + 1).astype(np.int64)
    crop_slices = _bbox_fit_crop_slices(
        tuple(int(dim) for dim in scaled_image.shape),
        scaled_bbox_min,
        scaled_bbox_max,
        scaled_endpoints,
        crop_size_zyx,
        margin_zyx,
        jitter_zyx,
    )
    crop_start_xyz = np.asarray(
        [crop_slices[2].start, crop_slices[1].start, crop_slices[0].start],
        dtype=np.float32,
    )
    image_crop = scaled_image[crop_slices]
    mask_crop = scaled_mask[crop_slices]
    restore_shape_zyx = tuple(int(dim) for dim in image_crop.shape)
    image_crop, mask_crop_or_none = _pad_to_shape(image_crop, mask_crop, crop_size_zyx)
    if mask_crop_or_none is None:
        raise ValueError("RECIST bbox-fit crop unexpectedly lost its target mask")
    mask_crop = (mask_crop_or_none > 0).astype(np.uint8)

    endpoints_crop = scaled_endpoints - crop_start_xyz[None, :]
    max_xyz = np.asarray([crop_size_zyx[2] - 1, crop_size_zyx[1] - 1, crop_size_zyx[0] - 1], dtype=np.float32)
    endpoints_crop = np.clip(endpoints_crop, 0.0, max_xyz[None, :])
    target_voxels = int(np.asarray(scaled_mask > 0).sum())
    covered_voxels = int(mask_crop.sum())
    return {
        "image": image_crop.astype(np.float32, copy=False),
        "mask": mask_crop.astype(np.uint8, copy=False),
        "crop_slices": crop_slices,
        "endpoints_xyz": endpoints_crop.astype(np.float32),
        "slice_index": int(round(float(endpoints_crop[:, 2].mean()))),
        "restore_shape_zyx": restore_shape_zyx,
        "bbox_fit_scale_zyx": tuple(float(value) for value in scale_zyx.tolist()),
        "bbox_fit_rescaled": bool(np.any(scale_zyx < 0.999)),
        "bbox_fit_target_coverage": float(covered_voxels / max(target_voxels, 1)),
        "bbox_fit_original_bbox_zyx": [[int(v) for v in bbox_min.tolist()], [int(v) for v in bbox_max.tolist()]],
        "bbox_fit_source_slices_zyx": [
            [int(axis_slice.start) for axis_slice in source_slices],
            [int(axis_slice.stop) for axis_slice in source_slices],
        ],
        "bbox_fit_scaled_bbox_zyx": [
            [int(v) for v in scaled_bbox_min.tolist()],
            [int(v) for v in scaled_bbox_max.tolist()],
        ],
    }


def extract_center_crop(
    image: np.ndarray,
    raw_mask: np.ndarray | None,
    center_zyx: np.ndarray,
    crop_size_zyx: tuple[int, int, int],
    jitter_zyx: tuple[int, int, int] = (0, 0, 0),
) -> dict[str, object]:
    crop_slices = compute_crop_slices(
        image.shape,
        np.asarray(center_zyx, dtype=np.float32) + _sample_jitter(jitter_zyx),
        crop_size_zyx,
    )
    image_crop = image[crop_slices]
    raw_mask_crop = None if raw_mask is None else raw_mask[crop_slices]
    restore_shape_zyx = tuple(int(dim) for dim in image_crop.shape)
    image_crop = pad_array_to_shape(image_crop, crop_size_zyx, constant_value=0.0)
    if raw_mask_crop is not None:
        raw_mask_crop = pad_array_to_shape(raw_mask_crop, crop_size_zyx, constant_value=0)
    return {
        "image": image_crop,
        "raw_mask": raw_mask_crop,
        "crop_slices": crop_slices,
        "restore_shape_zyx": restore_shape_zyx,
    }


def restore_crop_to_full_shape(
    crop_mask: np.ndarray,
    full_shape: tuple[int, int, int],
    crop_slices: tuple[slice, slice, slice],
) -> np.ndarray:
    full_mask = np.zeros(full_shape, dtype=crop_mask.dtype)
    full_mask[crop_slices] = crop_mask
    return full_mask


def _mask_bbox_zyx(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    points = np.argwhere(np.asarray(mask) > 0)
    if len(points) == 0:
        return None
    return points.min(axis=0).astype(np.int64), (points.max(axis=0) + 1).astype(np.int64)


def _endpoints_inside_shape_with_margin(
    endpoints_xyz: np.ndarray,
    shape_zyx: tuple[int, int, int],
    spacing_xyz: np.ndarray | tuple[float, float, float] | None,
    margin_mm: float,
) -> bool:
    endpoints = np.asarray(endpoints_xyz, dtype=np.float32).reshape(2, 3)
    max_xyz = np.asarray([shape_zyx[2] - 1, shape_zyx[1] - 1, shape_zyx[0] - 1], dtype=np.float32)
    margin_xyz = _endpoint_margin_voxels_zyx(spacing_xyz, margin_mm)[::-1]
    effective_margin = np.minimum(margin_xyz, np.maximum(max_xyz, 0.0) * 0.5)
    return bool(np.all(endpoints >= effective_margin[None, :]) and np.all(endpoints <= (max_xyz - effective_margin)[None, :]))


def prompt_crop_needs_bbox_fit_tail(
    full_mask: np.ndarray,
    prompt_crop: dict[str, object],
    crop_size_zyx: tuple[int, int, int],
    spacing_xyz: np.ndarray | tuple[float, float, float] | None = None,
    endpoint_margin_mm: float = 0.0,
    require_endpoint_margin: bool = False,
) -> bool:
    full_bbox = _mask_bbox_zyx(full_mask)
    if full_bbox is None:
        return False
    bbox_min, bbox_max = full_bbox
    bbox_size = bbox_max - bbox_min
    if np.any(bbox_size > np.asarray(crop_size_zyx, dtype=np.int64)):
        return True

    full_voxels = int((np.asarray(full_mask) > 0).sum())
    crop_mask = np.asarray(prompt_crop["mask"]) > 0
    crop_voxels = int(crop_mask.sum())
    if full_voxels > 0 and float(crop_voxels) / float(full_voxels) < 0.95:
        return True

    crop_bbox = _mask_bbox_zyx(crop_mask)
    if crop_bbox is not None:
        crop_bbox_min, crop_bbox_max = crop_bbox
        if np.any(crop_bbox_min <= 0) or np.any(crop_bbox_max >= np.asarray(crop_mask.shape, dtype=np.int64)):
            return True

    margin_mm = float(endpoint_margin_mm) if require_endpoint_margin else 0.0
    if not _endpoints_inside_shape_with_margin(
        np.asarray(prompt_crop["endpoints_xyz"], dtype=np.float32),
        tuple(int(dim) for dim in np.asarray(prompt_crop["image"]).shape),
        spacing_xyz,
        margin_mm,
    ):
        return True
    return False
