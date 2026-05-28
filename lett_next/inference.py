from __future__ import annotations

import time

import numpy as np
import torch
from scipy import ndimage
from scipy.special import expit

from .data import (
    build_input_channels,
    compute_crop_slices,
    extract_prompt_crop,
    get_recist_target_mask,
    restore_crop_to_full_shape,
)
from .models import _final_logits
from .prompts import make_wrong_prompt, rasterize_line_mask, xyz_to_zyx


POSTPROCESS_MODES = {
    "prompt_component",
    "prompt_tube_component",
    "prompt_box_component",
    "prompt_hybrid_component",
}


def _recist_midpoint_zyx(endpoints_xyz: np.ndarray | None, prompt_region: np.ndarray) -> np.ndarray | None:
    if endpoints_xyz is not None:
        endpoints = np.asarray(endpoints_xyz, dtype=np.float32)
        if endpoints.shape == (2, 3):
            return xyz_to_zyx(endpoints.mean(axis=0))
    prompt_points = np.argwhere(np.asarray(prompt_region) > 0)
    if len(prompt_points) == 0:
        return None
    return prompt_points.mean(axis=0).astype(np.float32)


def _effective_postprocess_endpoints(
    shape_zyx: tuple[int, int, int],
    endpoints_xyz: np.ndarray | None,
    prompt_mode: str,
) -> np.ndarray | None:
    if endpoints_xyz is None:
        return None
    endpoints = np.asarray(endpoints_xyz, dtype=np.float32)
    if endpoints.shape != (2, 3):
        return None
    if prompt_mode == "wrong_prompt":
        return make_wrong_prompt(shape_zyx, endpoints)
    if prompt_mode == "no_prompt":
        return None
    return endpoints


def _prompt_tube_region(
    shape_zyx: tuple[int, int, int],
    endpoints_xyz: np.ndarray | None,
    radius_xy: int,
    z_thickness: int,
) -> np.ndarray:
    if endpoints_xyz is None:
        return np.zeros(shape_zyx, dtype=bool)
    return rasterize_line_mask(
        shape_zyx,
        np.asarray(endpoints_xyz, dtype=np.float32),
        radius_xy=max(int(radius_xy), 0),
        z_thickness=max(int(z_thickness), 0),
    ) > 0


def _prompt_endpoint_box_region(
    shape_zyx: tuple[int, int, int],
    endpoints_xyz: np.ndarray | None,
    margin_zyx: tuple[int, int, int],
) -> np.ndarray:
    region = np.zeros(shape_zyx, dtype=bool)
    if endpoints_xyz is None:
        return region
    endpoints = np.asarray(endpoints_xyz, dtype=np.float32)
    if endpoints.shape != (2, 3):
        return region
    points_zyx = np.stack([xyz_to_zyx(point) for point in endpoints], axis=0)
    margin = np.asarray([max(int(value), 0) for value in margin_zyx], dtype=np.int32)
    lower = np.floor(points_zyx.min(axis=0)).astype(np.int32) - margin
    upper = np.ceil(points_zyx.max(axis=0)).astype(np.int32) + margin + 1
    shape = np.asarray(shape_zyx, dtype=np.int32)
    lower = np.clip(lower, 0, shape)
    upper = np.clip(upper, 0, shape)
    if np.any(upper <= lower):
        return region
    region[
        int(lower[0]) : int(upper[0]),
        int(lower[1]) : int(upper[1]),
        int(lower[2]) : int(upper[2]),
    ] = True
    return region


def _component_distance_to_point(labeled: np.ndarray, component_id: int, point_zyx: np.ndarray | None) -> float:
    if point_zyx is None:
        return float(component_id)
    points = np.argwhere(labeled == component_id)
    if len(points) == 0:
        return float("inf")
    center = points.mean(axis=0)
    return float(np.linalg.norm(center - np.asarray(point_zyx, dtype=np.float32)))


def _select_component_id(
    labeled: np.ndarray,
    component_count: int,
    prompt_region: np.ndarray,
    *,
    mode: str,
    endpoints_xyz: np.ndarray | None,
    tube_radius_xy: int,
    tube_z_thickness: int,
    box_margin_zyx: tuple[int, int, int],
    prompt_mode: str,
) -> tuple[int | None, bool, int, int, float]:
    if component_count <= 0:
        return None, False, 0, 0, 0.0
    mode = str(mode)
    if mode not in POSTPROCESS_MODES:
        raise ValueError(f"Unsupported postprocess mode: {mode}")

    shape_zyx = tuple(int(dim) for dim in labeled.shape)
    effective_endpoints = _effective_postprocess_endpoints(shape_zyx, endpoints_xyz, prompt_mode)
    prompt_bool = np.asarray(prompt_region) > 0
    tube_region = _prompt_tube_region(shape_zyx, effective_endpoints, tube_radius_xy, tube_z_thickness)
    box_region = _prompt_endpoint_box_region(shape_zyx, effective_endpoints, box_margin_zyx)
    midpoint = _recist_midpoint_zyx(effective_endpoints, prompt_region)

    best_id: int | None = None
    best_key: tuple[float, ...] | None = None
    selected_prompt_overlap = 0
    selected_tube_overlap = 0
    selected_box_overlap = 0
    selected_distance = 0.0
    for component_id in range(1, int(component_count) + 1):
        component_mask = labeled == component_id
        prompt_overlap = int((component_mask & prompt_bool).sum())
        tube_overlap = int((component_mask & tube_region).sum())
        box_overlap = int((component_mask & box_region).sum())
        distance = _component_distance_to_point(labeled, component_id, midpoint)
        size = int(component_mask.sum())
        if mode == "prompt_component":
            key = (float(prompt_overlap), -distance, float(size), -float(component_id))
        elif mode == "prompt_tube_component":
            key = (float(tube_overlap), -distance, float(size), -float(component_id))
        elif mode == "prompt_box_component":
            key = (float(box_overlap), -distance, float(size), -float(component_id))
        else:
            key = (float(tube_overlap), float(box_overlap), -distance, float(size), -float(component_id))
        if best_key is None or key > best_key:
            best_key = key
            best_id = component_id
            selected_prompt_overlap = prompt_overlap
            selected_tube_overlap = tube_overlap
            selected_box_overlap = box_overlap
            selected_distance = distance

    if best_id is None:
        return None, False, 0, 0, 0.0
    selected_has_overlap = (
        selected_prompt_overlap > 0
        if mode == "prompt_component"
        else selected_tube_overlap > 0
        if mode == "prompt_tube_component"
        else selected_box_overlap > 0
        if mode == "prompt_box_component"
        else selected_tube_overlap > 0 or selected_box_overlap > 0
    )
    return best_id, bool(selected_has_overlap), selected_prompt_overlap, selected_box_overlap, float(selected_distance)


def _apply_tiny_island_cleanup(
    labeled: np.ndarray,
    component_count: int,
    selected_component_id: int,
    *,
    min_voxels: int,
    min_fraction: float,
) -> np.ndarray:
    selected_size = int((labeled == selected_component_id).sum())
    if selected_size <= 0:
        return (labeled > 0).astype(np.uint8)
    threshold = max(int(min_voxels), int(np.ceil(float(min_fraction) * float(selected_size))))
    keep = labeled == selected_component_id
    for component_id in range(1, int(component_count) + 1):
        if component_id == selected_component_id:
            continue
        component_mask = labeled == component_id
        if int(component_mask.sum()) >= threshold:
            keep |= component_mask
    return keep.astype(np.uint8)


def keep_prompt_component(
    mask: np.ndarray,
    prompt_region: np.ndarray,
    *,
    mode: str = "prompt_component",
    prompt_endpoints_xyz: np.ndarray | None = None,
    prompt_mode: str = "full",
    tube_radius_xy: int = 6,
    tube_z_thickness: int = 1,
    box_margin_zyx: tuple[int, int, int] = (1, 8, 8),
    tiny_island_min_voxels: int = 0,
    tiny_island_min_fraction: float = 0.0,
) -> np.ndarray:
    labeled, component_count = ndimage.label(mask > 0)
    if component_count <= 1:
        return (mask > 0).astype(np.uint8)
    cleanup_enabled = int(tiny_island_min_voxels) > 0 or float(tiny_island_min_fraction) > 0.0
    if mode != "prompt_component" or cleanup_enabled:
        selected_component_id, selected_has_overlap, _, _, _ = _select_component_id(
            labeled,
            int(component_count),
            prompt_region,
            mode=mode,
            endpoints_xyz=prompt_endpoints_xyz,
            tube_radius_xy=tube_radius_xy,
            tube_z_thickness=tube_z_thickness,
            box_margin_zyx=box_margin_zyx,
            prompt_mode=prompt_mode,
        )
        if selected_component_id is None:
            return (mask > 0).astype(np.uint8)
        if mode == "prompt_component" and not selected_has_overlap and len(np.argwhere(prompt_region > 0)) == 0:
            return (mask > 0).astype(np.uint8)
        if cleanup_enabled:
            return _apply_tiny_island_cleanup(
                labeled,
                int(component_count),
                int(selected_component_id),
                min_voxels=int(tiny_island_min_voxels),
                min_fraction=float(tiny_island_min_fraction),
            )
        return (labeled == selected_component_id).astype(np.uint8)
    best_component = None
    best_score = -1.0
    prompt_points = np.argwhere(prompt_region > 0)
    for component_id in range(1, component_count + 1):
        component_mask = labeled == component_id
        overlap = float((component_mask & (prompt_region > 0)).sum())
        if overlap > best_score:
            best_component = component_mask
            best_score = overlap
    if best_component is not None and best_score > 0:
        return best_component.astype(np.uint8)
    if len(prompt_points) == 0:
        return (mask > 0).astype(np.uint8)
    prompt_center = prompt_points.mean(axis=0)
    best_distance = None
    best_component = None
    for component_id in range(1, component_count + 1):
        component_points = np.argwhere(labeled == component_id)
        component_center = component_points.mean(axis=0)
        distance = float(np.linalg.norm(component_center - prompt_center))
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_component = labeled == component_id
    return best_component.astype(np.uint8) if best_component is not None else (mask > 0).astype(np.uint8)


def component_audit(mask: np.ndarray, prompt_region: np.ndarray) -> dict[str, object]:
    labeled, component_count = ndimage.label(mask > 0)
    components: list[dict[str, int]] = []
    for component_id in range(1, component_count + 1):
        component_mask = labeled == component_id
        components.append(
            {
                "component_id": component_id,
                "size": int(component_mask.sum()),
                "prompt_overlap": int((component_mask & (prompt_region > 0)).sum()),
            }
        )
    components.sort(key=lambda row: row["size"], reverse=True)
    return {
        "num_components": int(component_count),
        "components": components,
    }


def prepare_case_inference(
    case,
    crop_size_zyx: tuple[int, int, int],
    prompt_sigma: float,
    prompt_mode: str = "full",
    prompt_endpoints_xyz: np.ndarray | None = None,
) -> dict[str, object]:
    crop = extract_prompt_crop(
        image=case.image,
        mask=get_recist_target_mask(case),
        endpoints_xyz=case.recist_endpoints_xyz,
        crop_size_zyx=crop_size_zyx,
    )
    effective_prompt_endpoints_xyz = (
        np.asarray(crop["endpoints_xyz"], dtype=np.float32)
        if prompt_endpoints_xyz is None
        else np.asarray(prompt_endpoints_xyz, dtype=np.float32)
    )
    normalization_stats = getattr(case, "normalization_stats", None)
    normalization_mode = (
        str(normalization_stats.get("mode", "dataset_stats"))
        if isinstance(normalization_stats, dict)
        else "legacy_crop_zscore"
    )
    inputs, prompt_region = build_input_channels(
        image_crop=crop["image"],  # type: ignore[arg-type]
        endpoints_xyz=effective_prompt_endpoints_xyz,
        prompt_sigma=prompt_sigma,
        prompt_mode=prompt_mode,
        case_id=case.case_id,
        prompt_source=getattr(case, "prompt_source", "unknown"),
        normalization_stats=normalization_stats,
        normalization_mode=normalization_mode,
    )
    return {
        "crop": crop,
        "inputs": inputs,
        "prompt_region": prompt_region,
        "prompt_endpoints_xyz": effective_prompt_endpoints_xyz,
    }


def run_model_on_case_inputs(
    model: torch.nn.Module,
    prepared: dict[str, object],
    device: str,
) -> dict[str, object]:
    input_tensor = torch.from_numpy(prepared["inputs"])[None, ...].to(  # type: ignore[arg-type]
        device=device,
        dtype=torch.float32,
    )
    start = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        logits = _final_logits(model(input_tensor))[0, 0].detach().cpu().numpy()
    elapsed = time.perf_counter() - start
    probabilities = expit(logits, out=logits).astype(np.float32, copy=False)
    return {
        "probabilities": probabilities,
        "elapsed": elapsed,
    }


def decode_prediction(
    probabilities: np.ndarray,
    prompt_region: np.ndarray,
    crop_slices: tuple[slice, slice, slice],
    restore_shape_zyx: tuple[int, int, int],
    full_shape: tuple[int, int, int],
    threshold: float,
    apply_postprocess: bool,
    postprocess_mode: str = "prompt_component",
    prompt_endpoints_xyz: np.ndarray | None = None,
    prompt_mode: str = "full",
    postprocess_tube_radius_xy: int = 6,
    postprocess_tube_z_thickness: int = 1,
    postprocess_box_margin_zyx: tuple[int, int, int] = (1, 8, 8),
    postprocess_tiny_island_min_voxels: int = 0,
    postprocess_tiny_island_min_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    crop_prediction = (probabilities >= threshold).astype(np.uint8)
    if apply_postprocess:
        crop_prediction = keep_prompt_component(
            crop_prediction,
            prompt_region,
            mode=postprocess_mode,
            prompt_endpoints_xyz=prompt_endpoints_xyz,
            prompt_mode=prompt_mode,
            tube_radius_xy=postprocess_tube_radius_xy,
            tube_z_thickness=postprocess_tube_z_thickness,
            box_margin_zyx=postprocess_box_margin_zyx,
            tiny_island_min_voxels=postprocess_tiny_island_min_voxels,
            tiny_island_min_fraction=postprocess_tiny_island_min_fraction,
        )
    restore_slices = tuple(slice(0, int(dim)) for dim in restore_shape_zyx)
    crop_prediction_restore = crop_prediction[restore_slices]
    full_prediction = restore_crop_to_full_shape(
        crop_mask=crop_prediction_restore,
        full_shape=full_shape,
        crop_slices=crop_slices,
    )
    return crop_prediction, full_prediction


def _resize_zyx(array: np.ndarray, output_shape_zyx: tuple[int, int, int], order: int) -> np.ndarray:
    output_shape = tuple(int(dim) for dim in output_shape_zyx)
    source = np.asarray(array)
    if tuple(source.shape) == output_shape:
        return source.copy()
    factors = [float(out_dim) / float(max(in_dim, 1)) for out_dim, in_dim in zip(output_shape, source.shape, strict=True)]
    resized = ndimage.zoom(source, zoom=factors, order=order)
    if tuple(resized.shape) == output_shape:
        return resized.astype(source.dtype, copy=False)
    fixed = np.zeros(output_shape, dtype=resized.dtype)
    common_slices = tuple(slice(0, min(int(a), int(b))) for a, b in zip(output_shape, resized.shape, strict=True))
    fixed[common_slices] = resized[common_slices]
    return fixed.astype(source.dtype, copy=False)


def _zoomed_crop_size(
    crop_size_zyx: tuple[int, int, int],
    zoom_zyx: np.ndarray,
) -> tuple[int, int, int]:
    return tuple(
        max(int(np.ceil(float(base_dim) * float(zoom))), 1)
        for base_dim, zoom in zip(crop_size_zyx, zoom_zyx, strict=True)
    )


def _run_zoom_pass(
    model: torch.nn.Module,
    case,
    crop_size_zyx: tuple[int, int, int],
    zoom_zyx: np.ndarray,
    prompt_sigma: float,
    threshold: float,
    device: str,
    prompt_mode: str,
    apply_postprocess: bool,
    prompt_endpoints_xyz: np.ndarray | None,
    postprocess_mode: str,
    postprocess_tube_radius_xy: int,
    postprocess_tube_z_thickness: int,
    postprocess_box_margin_zyx: tuple[int, int, int],
    postprocess_tiny_island_min_voxels: int,
    postprocess_tiny_island_min_fraction: float,
) -> dict[str, object]:
    zoom_crop_size = _zoomed_crop_size(crop_size_zyx, zoom_zyx)
    crop = extract_prompt_crop(
        image=case.image,
        mask=get_recist_target_mask(case),
        endpoints_xyz=case.recist_endpoints_xyz,
        crop_size_zyx=zoom_crop_size,
    )
    image_crop = np.asarray(crop["image"], dtype=np.float32)
    network_image = _resize_zyx(image_crop, crop_size_zyx, order=1).astype(np.float32)
    effective_prompt_endpoints_xyz = (
        np.asarray(crop["endpoints_xyz"], dtype=np.float32)
        if prompt_endpoints_xyz is None
        else np.asarray(prompt_endpoints_xyz, dtype=np.float32)
    )
    scale_zyx = np.asarray(crop_size_zyx, dtype=np.float32) / np.asarray(image_crop.shape, dtype=np.float32)
    scale_xyz = np.asarray([scale_zyx[2], scale_zyx[1], scale_zyx[0]], dtype=np.float32)
    network_endpoints_xyz = effective_prompt_endpoints_xyz * scale_xyz
    normalization_stats = getattr(case, "normalization_stats", None)
    normalization_mode = (
        str(normalization_stats.get("mode", "dataset_stats"))
        if isinstance(normalization_stats, dict)
        else "legacy_crop_zscore"
    )
    inputs, prompt_region = build_input_channels(
        image_crop=network_image,
        endpoints_xyz=network_endpoints_xyz,
        prompt_sigma=prompt_sigma,
        prompt_mode=prompt_mode,
        case_id=case.case_id,
        prompt_source=getattr(case, "prompt_source", "unknown"),
        normalization_stats=normalization_stats,
        normalization_mode=normalization_mode,
    )
    outputs = run_model_on_case_inputs(
        model=model,
        prepared={"inputs": inputs},
        device=device,
    )
    restore_shape_zyx = tuple(int(dim) for dim in crop["restore_shape_zyx"])  # type: ignore[iteration-over-annotation]
    crop_probabilities = _resize_zyx(
        np.asarray(outputs["probabilities"], dtype=np.float32),
        restore_shape_zyx,
        order=1,
    ).astype(np.float32)
    crop_prediction = (crop_probabilities >= float(threshold)).astype(np.uint8)
    prompt_region_restore = _resize_zyx(
        np.asarray(prompt_region, dtype=np.float32),
        restore_shape_zyx,
        order=1,
    ).astype(np.float32)
    if apply_postprocess:
        crop_prediction = keep_prompt_component(
            crop_prediction,
            prompt_region_restore,
            mode=postprocess_mode,
            prompt_endpoints_xyz=effective_prompt_endpoints_xyz,
            prompt_mode=prompt_mode,
            tube_radius_xy=postprocess_tube_radius_xy,
            tube_z_thickness=postprocess_tube_z_thickness,
            box_margin_zyx=postprocess_box_margin_zyx,
            tiny_island_min_voxels=postprocess_tiny_island_min_voxels,
            tiny_island_min_fraction=postprocess_tiny_island_min_fraction,
        )
    full_prediction = restore_crop_to_full_shape(
        crop_mask=crop_prediction,
        full_shape=case.image.shape,
        crop_slices=crop["crop_slices"],  # type: ignore[arg-type]
    )
    full_probabilities = restore_crop_to_full_shape(
        crop_mask=crop_probabilities,
        full_shape=case.image.shape,
        crop_slices=crop["crop_slices"],  # type: ignore[arg-type]
    )
    return {
        "crop": crop,
        "crop_prediction": crop_prediction,
        "full_prediction": full_prediction,
        "full_probabilities": full_probabilities,
        "prompt_region": prompt_region,
        "prompt_region_restore": prompt_region_restore,
        "elapsed": float(outputs["elapsed"]),
    }


def _axis_border_area(shape_zyx: tuple[int, int, int], axis: int) -> int:
    other_dims = [int(dim) for index, dim in enumerate(shape_zyx) if index != axis]
    return int(np.prod(other_dims))


def _border_thresholds(
    axis: int,
    border_shape_zyx: tuple[int, int, int],
    ref_patch_zyx: tuple[int, int, int],
    base_abs_changed: int,
    base_min_changed: int,
) -> tuple[float, float]:
    border_area = float(_axis_border_area(border_shape_zyx, axis))
    ref_area = float(max(_axis_border_area(ref_patch_zyx, axis), 1))
    scale = border_area / ref_area
    return float(base_abs_changed) * scale, float(base_min_changed) * scale


def _crop_border_axes_from_mask(mask: np.ndarray, margin_voxels: int = 0) -> list[int]:
    mask_bool = np.asarray(mask) > 0
    if not mask_bool.any():
        return []
    axes: list[int] = []
    margin = max(int(margin_voxels), 0)
    for axis, dim in enumerate(mask_bool.shape):
        width = min(margin + 1, int(dim))
        front = np.take(mask_bool, indices=range(width), axis=axis)
        back = np.take(mask_bool, indices=range(max(int(dim) - width, 0), int(dim)), axis=axis)
        if bool(front.any()) or bool(back.any()):
            axes.append(axis)
    return axes


def _border_foreground_load(mask: np.ndarray, margin_voxels: int = 0) -> tuple[list[int], list[float]]:
    mask_bool = np.asarray(mask) > 0
    counts: list[int] = []
    fractions: list[float] = []
    margin = max(int(margin_voxels), 0)
    for axis, dim in enumerate(mask_bool.shape):
        width = min(margin + 1, int(dim))
        front = np.take(mask_bool, indices=range(width), axis=axis)
        back = np.take(mask_bool, indices=range(max(int(dim) - width, 0), int(dim)), axis=axis)
        count = int(front.sum()) + int(back.sum())
        area = int(front.size) + int(back.size)
        counts.append(count)
        fractions.append(float(count / max(area, 1)))
    return counts, fractions


def _border_change_axes(
    previous_full: np.ndarray,
    current_full: np.ndarray,
    crop_slices: tuple[slice, slice, slice],
    ref_patch_zyx: tuple[int, int, int],
    base_abs_changed: int,
    base_min_changed: int,
    base_rel_changed: float,
) -> list[int]:
    previous = np.asarray(previous_full) > 0
    current = np.asarray(current_full) > 0
    axes: list[int] = []
    crop_shape = tuple(int(slc.stop - slc.start) for slc in crop_slices)
    for axis in range(3):
        changed_total = 0
        foreground_total = 0
        for plane_index in (int(crop_slices[axis].start), int(crop_slices[axis].stop) - 1):
            plane_slices: list[slice | int] = [crop_slices[0], crop_slices[1], crop_slices[2]]
            plane_slices[axis] = plane_index
            plane_key = tuple(plane_slices)
            previous_plane = previous[plane_key]
            current_plane = current[plane_key]
            changed_total += int(np.logical_xor(previous_plane, current_plane).sum())
            foreground_total += int(np.logical_or(previous_plane, current_plane).sum())
        abs_thr, min_thr = _border_thresholds(
            axis=axis,
            border_shape_zyx=crop_shape,
            ref_patch_zyx=ref_patch_zyx,
            base_abs_changed=base_abs_changed,
            base_min_changed=base_min_changed,
        )
        relative_changed = float(changed_total / max(foreground_total, 1))
        if changed_total > abs_thr or (changed_total > min_thr and relative_changed > float(base_rel_changed)):
            axes.append(axis)
    return axes


def _grow_zoom(
    zoom_zyx: np.ndarray,
    axes: list[int],
    growth_zyx: tuple[float, float, float],
    max_zoom_zyx: tuple[float, float, float],
) -> np.ndarray:
    next_zoom = np.asarray(zoom_zyx, dtype=np.float32).copy()
    growth = np.asarray(growth_zyx, dtype=np.float32)
    max_zoom = np.asarray(max_zoom_zyx, dtype=np.float32)
    for axis in axes:
        next_zoom[axis] = min(float(next_zoom[axis] * growth[axis]), float(max_zoom[axis]))
    return next_zoom


def _autozoom_next_zoom(
    zoom_zyx: np.ndarray,
    trigger_axes: list[int],
    growth_zyx: tuple[float, float, float],
    max_zoom_zyx: tuple[float, float, float],
    candidate_zoom_zyx: np.ndarray,
    border_fractions_zyx: list[float],
    scale_mode: str = "fixed",
) -> np.ndarray:
    mode = str(scale_mode).strip().lower()
    if mode not in {"fixed", "adaptive"}:
        raise ValueError(f"Unsupported AutoZoom scale mode: {scale_mode!r}")
    if mode == "adaptive":
        next_zoom = np.asarray(zoom_zyx, dtype=np.float32).copy()
        growth = np.asarray(growth_zyx, dtype=np.float32)
        max_zoom = np.asarray(max_zoom_zyx, dtype=np.float32)
        candidate_zoom = np.asarray(candidate_zoom_zyx, dtype=np.float32)
        for axis in trigger_axes:
            border_fraction = float(border_fractions_zyx[axis])
            border_strength = min(max(border_fraction / 0.05, 0.0), 1.0)
            border_zoom = float(next_zoom[axis]) * (1.0 + (float(growth[axis]) - 1.0) * border_strength)
            needed_zoom = max(float(candidate_zoom[axis]), border_zoom)
            next_zoom[axis] = min(needed_zoom, float(max_zoom[axis]))
        return np.minimum(next_zoom, max_zoom).astype(np.float32)

    next_zoom = _grow_zoom(zoom_zyx, trigger_axes, growth_zyx, max_zoom_zyx)
    current_zoom = np.asarray(zoom_zyx, dtype=np.float32)
    growth = np.asarray(growth_zyx, dtype=np.float32)
    max_zoom = np.asarray(max_zoom_zyx, dtype=np.float32)
    candidate_zoom = np.asarray(candidate_zoom_zyx, dtype=np.float32)
    for axis in trigger_axes:
        if float(border_fractions_zyx[axis]) >= 0.05:
            next_zoom[axis] = max(float(next_zoom[axis]), float(current_zoom[axis] * growth[axis] * growth[axis]))
        next_zoom[axis] = max(float(next_zoom[axis]), float(candidate_zoom[axis]))
    return np.minimum(next_zoom, max_zoom).astype(np.float32)


def _mask_bbox_size_zyx(mask: np.ndarray) -> np.ndarray | None:
    points = np.argwhere(np.asarray(mask) > 0)
    if len(points) == 0:
        return None
    return (points.max(axis=0) - points.min(axis=0) + 1).astype(np.float32)


def _direct_candidate_zoom(
    candidate_mask: np.ndarray,
    crop_size_zyx: tuple[int, int, int],
    current_zoom_zyx: np.ndarray,
    max_zoom_zyx: tuple[float, float, float],
) -> np.ndarray:
    bbox_size = _mask_bbox_size_zyx(candidate_mask)
    current_zoom = np.asarray(current_zoom_zyx, dtype=np.float32)
    if bbox_size is None:
        return current_zoom.copy()
    patch_size = np.asarray(crop_size_zyx, dtype=np.float32)
    max_zoom = np.asarray(max_zoom_zyx, dtype=np.float32)
    requested_size = bbox_size + patch_size / 3.0
    target_zoom = np.maximum(current_zoom, requested_size / patch_size)
    return np.minimum(target_zoom, max_zoom).astype(np.float32)


def _make_refinement_boxes(
    diff_mask: np.ndarray,
    coarse_mask: np.ndarray,
    crop_size_zyx: tuple[int, int, int],
    margin_zyx: tuple[int, int, int],
    max_boxes: int,
) -> list[tuple[slice, slice, slice]]:
    labeled, component_count = ndimage.label(np.asarray(diff_mask) > 0)
    boxes: list[tuple[float, tuple[slice, slice, slice]]] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = set()
    for component_id in range(1, int(component_count) + 1):
        points = np.argwhere(labeled == component_id)
        if len(points) == 0:
            continue
        min_corner = points.min(axis=0) - np.asarray(margin_zyx, dtype=np.int32)
        max_corner = points.max(axis=0) + np.asarray(margin_zyx, dtype=np.int32) + 1
        center_zyx = (min_corner.astype(np.float32) + max_corner.astype(np.float32)) / 2.0
        crop_slices = compute_crop_slices(coarse_mask.shape, center_zyx, crop_size_zyx)
        key = tuple((int(slc.start), int(slc.stop)) for slc in crop_slices)
        if key in seen:
            continue
        seen.add(key)
        score = float(np.asarray(coarse_mask)[crop_slices].sum())
        boxes.append((score, crop_slices))
    boxes.sort(key=lambda row: row[0], reverse=True)
    return [crop_slices for _, crop_slices in boxes[: max(int(max_boxes), 0)]]


def _extract_crop_for_slices(
    image: np.ndarray,
    crop_slices: tuple[slice, slice, slice],
    crop_size_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, tuple[int, int, int]]:
    image_crop = np.asarray(image, dtype=np.float32)[crop_slices]
    restore_shape_zyx = tuple(int(dim) for dim in image_crop.shape)
    pad_width = []
    for dim, target_dim in zip(image_crop.shape, crop_size_zyx, strict=True):
        pad_width.append((0, max(int(target_dim) - int(dim), 0)))
    if any(after > 0 for _, after in pad_width):
        image_crop = np.pad(image_crop, pad_width, mode="constant", constant_values=0.0)
    return image_crop.astype(np.float32), restore_shape_zyx


def _run_refinement_boxes(
    model: torch.nn.Module,
    case,
    full_prediction: np.ndarray,
    boxes: list[tuple[slice, slice, slice]],
    crop_size_zyx: tuple[int, int, int],
    prompt_sigma: float,
    threshold: float,
    device: str,
    prompt_mode: str,
    apply_postprocess: bool,
    postprocess_mode: str,
    postprocess_tube_radius_xy: int,
    postprocess_tube_z_thickness: int,
    postprocess_box_margin_zyx: tuple[int, int, int],
    postprocess_tiny_island_min_voxels: int,
    postprocess_tiny_island_min_fraction: float,
) -> tuple[np.ndarray, float]:
    refined = np.asarray(full_prediction, dtype=np.uint8).copy()
    elapsed = 0.0
    for crop_slices in boxes:
        image_crop, restore_shape_zyx = _extract_crop_for_slices(case.image, crop_slices, crop_size_zyx)
        crop_start_xyz = np.asarray(
            [crop_slices[2].start, crop_slices[1].start, crop_slices[0].start],
            dtype=np.float32,
        )
        endpoints_xyz = np.asarray(case.recist_endpoints_xyz, dtype=np.float32) - crop_start_xyz
        normalization_stats = getattr(case, "normalization_stats", None)
        normalization_mode = (
            str(normalization_stats.get("mode", "dataset_stats"))
            if isinstance(normalization_stats, dict)
            else "legacy_crop_zscore"
        )
        inputs, prompt_region = build_input_channels(
            image_crop=image_crop,
            endpoints_xyz=endpoints_xyz,
            prompt_sigma=prompt_sigma,
            prompt_mode=prompt_mode,
            case_id=case.case_id,
            prompt_source=getattr(case, "prompt_source", "unknown"),
            normalization_stats=normalization_stats,
            normalization_mode=normalization_mode,
        )
        outputs = run_model_on_case_inputs(
            model=model,
            prepared={"inputs": inputs},
            device=device,
        )
        elapsed += float(outputs["elapsed"])
        crop_prediction = (np.asarray(outputs["probabilities"], dtype=np.float32) >= float(threshold)).astype(np.uint8)
        restore_slices = tuple(slice(0, int(dim)) for dim in restore_shape_zyx)
        if apply_postprocess:
            crop_prediction = keep_prompt_component(
                crop_prediction,
                prompt_region,
                mode=postprocess_mode,
                prompt_endpoints_xyz=endpoints_xyz,
                prompt_mode=prompt_mode,
                tube_radius_xy=postprocess_tube_radius_xy,
                tube_z_thickness=postprocess_tube_z_thickness,
                box_margin_zyx=postprocess_box_margin_zyx,
                tiny_island_min_voxels=postprocess_tiny_island_min_voxels,
                tiny_island_min_fraction=postprocess_tiny_island_min_fraction,
            )
        refined[crop_slices] = crop_prediction[restore_slices]
    return refined, elapsed


def _predict_case_autozoom(
    model: torch.nn.Module,
    case,
    crop_size_zyx: tuple[int, int, int],
    prompt_sigma: float,
    threshold: float,
    device: str,
    prompt_mode: str,
    apply_postprocess: bool,
    prompt_endpoints_xyz: np.ndarray | None,
    postprocess_mode: str,
    postprocess_tube_radius_xy: int,
    postprocess_tube_z_thickness: int,
    postprocess_box_margin_zyx: tuple[int, int, int],
    postprocess_tiny_island_min_voxels: int,
    postprocess_tiny_island_min_fraction: float,
    max_passes: int,
    min_passes: int,
    growth_zyx: tuple[float, float, float],
    max_zoom_zyx: tuple[float, float, float],
    scale_mode: str,
    base_abs_changed: int,
    base_min_changed: int,
    base_rel_changed: float,
    ref_patch_zyx: tuple[int, int, int],
    refine_enable: bool,
    refine_margin_zyx: tuple[int, int, int],
    refine_max_boxes: int,
    collect_progression: bool,
) -> tuple[np.ndarray, float, np.ndarray, dict[str, object]]:
    zoom_zyx = np.ones(3, dtype=np.float32)
    previous_full: np.ndarray | None = None
    first_full: np.ndarray | None = None
    best_result: dict[str, object] | None = None
    elapsed_total = 0.0
    trace_passes: list[dict[str, object]] = []
    pass_count = max(int(max_passes), 1)
    minimum_pass_count = min(max(int(min_passes), 1), pass_count)
    for pass_index in range(pass_count):
        result = _run_zoom_pass(
            model=model,
            case=case,
            crop_size_zyx=crop_size_zyx,
            zoom_zyx=zoom_zyx,
            prompt_sigma=prompt_sigma,
            threshold=threshold,
            device=device,
            prompt_mode=prompt_mode,
            apply_postprocess=apply_postprocess,
            prompt_endpoints_xyz=prompt_endpoints_xyz,
            postprocess_mode=postprocess_mode,
            postprocess_tube_radius_xy=postprocess_tube_radius_xy,
            postprocess_tube_z_thickness=postprocess_tube_z_thickness,
            postprocess_box_margin_zyx=postprocess_box_margin_zyx,
            postprocess_tiny_island_min_voxels=postprocess_tiny_island_min_voxels,
            postprocess_tiny_island_min_fraction=postprocess_tiny_island_min_fraction,
        )
        elapsed_total += float(result["elapsed"])
        current_full = np.asarray(result["full_prediction"], dtype=np.uint8)
        if first_full is None:
            first_full = current_full.copy()
        if previous_full is None:
            trigger_axes = _crop_border_axes_from_mask(np.asarray(result["crop_prediction"], dtype=np.uint8), margin_voxels=0)
            trigger_kind = "border_foreground"
        else:
            trigger_axes = _border_change_axes(
                previous_full=previous_full,
                current_full=current_full,
                crop_slices=result["crop"]["crop_slices"],  # type: ignore[index]
                ref_patch_zyx=ref_patch_zyx,
                base_abs_changed=base_abs_changed,
                base_min_changed=base_min_changed,
                base_rel_changed=base_rel_changed,
            )
            trigger_kind = "border_change"
        forced_min_pass = False
        if not trigger_axes and pass_index + 1 < minimum_pass_count:
            trigger_axes = [0, 1, 2]
            trigger_kind = f"{trigger_kind}_forced_min_pass"
            forced_min_pass = True
        crop_prediction = np.asarray(result["crop_prediction"], dtype=np.uint8)
        candidate_zoom = _direct_candidate_zoom(
            candidate_mask=crop_prediction,
            crop_size_zyx=crop_size_zyx,
            current_zoom_zyx=zoom_zyx,
            max_zoom_zyx=max_zoom_zyx,
        )
        border_counts, border_fractions = _border_foreground_load(crop_prediction, margin_voxels=0)
        crop_slices = result["crop"]["crop_slices"]  # type: ignore[index]
        trace_entry: dict[str, object] = {
            "pass_index": pass_index,
            "zoom_zyx": [float(value) for value in zoom_zyx.tolist()],
            "candidate_zoom_zyx": [float(value) for value in candidate_zoom.tolist()],
            "crop_size_zyx": list(_zoomed_crop_size(crop_size_zyx, zoom_zyx)),
            "crop_slices_zyx": [[int(slc.start or 0), int(slc.stop or 0)] for slc in crop_slices],
            "trigger_axes_zyx": [int(axis) for axis in trigger_axes],
            "trigger_kind": trigger_kind,
            "forced_min_pass": bool(forced_min_pass),
            "scale_mode": str(scale_mode),
            "border_foreground_voxels_zyx": [int(value) for value in border_counts],
            "border_foreground_fraction_zyx": [float(value) for value in border_fractions],
            "predicted_voxels": int(current_full.sum()),
        }
        if collect_progression:
            trace_entry["full_prediction"] = current_full.copy()
        trace_passes.append(trace_entry)
        best_result = result
        if not trigger_axes:
            break
        next_zoom = _autozoom_next_zoom(
            zoom_zyx=zoom_zyx,
            trigger_axes=trigger_axes,
            growth_zyx=growth_zyx,
            max_zoom_zyx=max_zoom_zyx,
            candidate_zoom_zyx=candidate_zoom,
            border_fractions_zyx=border_fractions,
            scale_mode=scale_mode,
        )
        if np.allclose(next_zoom, zoom_zyx):
            break
        previous_full = current_full
        zoom_zyx = next_zoom

    if best_result is None:
        raise RuntimeError("AutoZoom did not run any inference passes")
    final_prediction = np.asarray(best_result["full_prediction"], dtype=np.uint8)
    refinement_boxes: list[tuple[slice, slice, slice]] = []
    if refine_enable and first_full is not None:
        diff_mask = np.logical_xor(final_prediction > 0, first_full > 0)
        refinement_boxes = _make_refinement_boxes(
            diff_mask=diff_mask,
            coarse_mask=final_prediction,
            crop_size_zyx=crop_size_zyx,
            margin_zyx=refine_margin_zyx,
            max_boxes=refine_max_boxes,
        )
        refinement_box_ranges = [
            [[int(slc.start or 0), int(slc.stop or 0)] for slc in crop_slices]
            for crop_slices in refinement_boxes
        ]
        if refinement_boxes:
            final_prediction, refine_elapsed = _run_refinement_boxes(
                model=model,
                case=case,
                full_prediction=final_prediction,
                boxes=refinement_boxes,
                crop_size_zyx=crop_size_zyx,
                prompt_sigma=prompt_sigma,
                threshold=threshold,
                device=device,
                prompt_mode=prompt_mode,
                apply_postprocess=apply_postprocess,
                postprocess_mode=postprocess_mode,
                postprocess_tube_radius_xy=postprocess_tube_radius_xy,
                postprocess_tube_z_thickness=postprocess_tube_z_thickness,
                postprocess_box_margin_zyx=postprocess_box_margin_zyx,
                postprocess_tiny_island_min_voxels=postprocess_tiny_island_min_voxels,
                postprocess_tiny_island_min_fraction=postprocess_tiny_island_min_fraction,
            )
            elapsed_total += refine_elapsed
    else:
        refinement_box_ranges = []

    trace = {
        "autozoom_enabled": True,
        "passes": trace_passes,
        "refinement_box_count": len(refinement_boxes),
        "refinement_boxes_zyx": refinement_box_ranges,
    }
    return final_prediction, elapsed_total, best_result["prompt_region"], trace  # type: ignore[return-value]


def predict_case(
    model: torch.nn.Module,
    case,
    crop_size_zyx: tuple[int, int, int],
    prompt_sigma: float,
    threshold: float,
    device: str,
    prompt_mode: str = "full",
    apply_postprocess: bool = True,
    prompt_endpoints_xyz: np.ndarray | None = None,
    postprocess_mode: str = "prompt_component",
    postprocess_tube_radius_xy: int = 6,
    postprocess_tube_z_thickness: int = 1,
    postprocess_box_margin_zyx: tuple[int, int, int] = (1, 8, 8),
    postprocess_tiny_island_min_voxels: int = 0,
    postprocess_tiny_island_min_fraction: float = 0.0,
    autozoom_enable: bool = False,
    autozoom_max_passes: int = 3,
    autozoom_min_passes: int = 1,
    autozoom_growth_zyx: tuple[float, float, float] = (1.5, 1.25, 1.25),
    autozoom_max_zoom_zyx: tuple[float, float, float] = (4.0, 2.0, 2.0),
    autozoom_scale_mode: str = "fixed",
    autozoom_base_abs_changed: int = 1500,
    autozoom_base_min_changed: int = 100,
    autozoom_base_rel_changed: float = 0.2,
    autozoom_ref_patch_zyx: tuple[int, int, int] = (128, 160, 160),
    autozoom_refine_enable: bool = True,
    autozoom_refine_margin_zyx: tuple[int, int, int] = (10, 10, 10),
    autozoom_refine_max_boxes: int = 4,
    return_details: bool = False,
    return_progression: bool = False,
) -> tuple[np.ndarray, float, np.ndarray] | tuple[np.ndarray, float, np.ndarray, dict[str, object]]:
    if autozoom_enable:
        prediction, elapsed, prompt_region, trace = _predict_case_autozoom(
            model=model,
            case=case,
            crop_size_zyx=crop_size_zyx,
            prompt_sigma=prompt_sigma,
            threshold=threshold,
            device=device,
            prompt_mode=prompt_mode,
            apply_postprocess=apply_postprocess,
            prompt_endpoints_xyz=prompt_endpoints_xyz,
            postprocess_mode=postprocess_mode,
            postprocess_tube_radius_xy=postprocess_tube_radius_xy,
            postprocess_tube_z_thickness=postprocess_tube_z_thickness,
            postprocess_box_margin_zyx=postprocess_box_margin_zyx,
            postprocess_tiny_island_min_voxels=postprocess_tiny_island_min_voxels,
            postprocess_tiny_island_min_fraction=postprocess_tiny_island_min_fraction,
            max_passes=autozoom_max_passes,
            min_passes=autozoom_min_passes,
            growth_zyx=autozoom_growth_zyx,
            max_zoom_zyx=autozoom_max_zoom_zyx,
            scale_mode=autozoom_scale_mode,
            base_abs_changed=autozoom_base_abs_changed,
            base_min_changed=autozoom_base_min_changed,
            base_rel_changed=autozoom_base_rel_changed,
            ref_patch_zyx=autozoom_ref_patch_zyx,
            refine_enable=autozoom_refine_enable,
            refine_margin_zyx=autozoom_refine_margin_zyx,
            refine_max_boxes=autozoom_refine_max_boxes,
            collect_progression=bool(return_details and return_progression),
        )
        if return_details:
            return prediction, elapsed, prompt_region, trace
        return prediction, elapsed, prompt_region
    prepared = prepare_case_inference(
        case=case,
        crop_size_zyx=crop_size_zyx,
        prompt_sigma=prompt_sigma,
        prompt_mode=prompt_mode,
        prompt_endpoints_xyz=prompt_endpoints_xyz,
    )
    outputs = run_model_on_case_inputs(
        model=model,
        prepared=prepared,
        device=device,
    )
    crop_prediction, full_prediction = decode_prediction(
        probabilities=outputs["probabilities"],  # type: ignore[arg-type]
        prompt_region=prepared["prompt_region"],  # type: ignore[arg-type]
        crop_slices=prepared["crop"]["crop_slices"],  # type: ignore[index]
        restore_shape_zyx=prepared["crop"]["restore_shape_zyx"],  # type: ignore[index]
        full_shape=case.image.shape,
        threshold=threshold,
        apply_postprocess=apply_postprocess,
        postprocess_mode=postprocess_mode,
        prompt_endpoints_xyz=prepared["prompt_endpoints_xyz"],  # type: ignore[arg-type]
        prompt_mode=prompt_mode,
        postprocess_tube_radius_xy=postprocess_tube_radius_xy,
        postprocess_tube_z_thickness=postprocess_tube_z_thickness,
        postprocess_box_margin_zyx=postprocess_box_margin_zyx,
        postprocess_tiny_island_min_voxels=postprocess_tiny_island_min_voxels,
        postprocess_tiny_island_min_fraction=postprocess_tiny_island_min_fraction,
    )
    trace = {"autozoom_enabled": False, "passes": [], "refinement_box_count": 0, "refinement_boxes_zyx": []}
    if return_details:
        return full_prediction, float(outputs["elapsed"]), prepared["prompt_region"], trace  # type: ignore[index]
    return full_prediction, float(outputs["elapsed"]), prepared["prompt_region"]  # type: ignore[index]
