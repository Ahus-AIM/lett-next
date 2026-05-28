from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

from ...prompts import rasterize_line_mask, xyz_to_zyx
from .records import CaseRecord, LoadedCase
from .validation import _validate_recist_geometry


@dataclass(slots=True)
class MaskComponent:
    label_value: int
    component_id: int
    voxel_count: int
    mask: np.ndarray
    bbox_zyx: tuple[np.ndarray, np.ndarray]
    slice_areas: np.ndarray


def iter_labeled_components(mask_zyx: np.ndarray, *, min_voxels: int = 0) -> Iterator[MaskComponent]:
    mask = np.asarray(mask_zyx)
    for label_value in [int(value) for value in np.unique(mask).tolist() if int(value) > 0]:
        labeled, component_count = ndimage.label(mask == label_value)
        for component_id in range(1, int(component_count) + 1):
            component_mask = labeled == component_id
            voxel_count = int(component_mask.sum())
            if voxel_count <= 0 or voxel_count < int(min_voxels):
                continue
            points = np.argwhere(component_mask)
            slice_areas = component_mask.reshape(component_mask.shape[0], -1).sum(axis=1)
            yield MaskComponent(
                label_value=int(label_value),
                component_id=int(component_id),
                voxel_count=int(voxel_count),
                mask=component_mask.astype(np.uint8),
                bbox_zyx=(points.min(axis=0).astype(np.int64), (points.max(axis=0) + 1).astype(np.int64)),
                slice_areas=np.asarray(slice_areas),
            )


def _farthest_points_2d(points_yx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_yx, dtype=np.float32)
    if len(points) == 1:
        return points[0], points[0]
    candidates = points
    if len(points) >= 3:
        try:
            hull = ConvexHull(points)
        except QhullError:
            candidates = points
        else:
            candidates = points[hull.vertices]
    best_start = candidates[0]
    best_end = candidates[0]
    best_distance = -1.0
    chunk_size = 4096
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start : start + chunk_size]
        squared = np.sum((chunk[:, None, :] - candidates[None, :, :]) ** 2, axis=-1)
        chunk_index, candidate_index = np.unravel_index(int(np.argmax(squared)), squared.shape)
        distance = float(squared[chunk_index, candidate_index])
        if distance > best_distance:
            best_distance = distance
            best_start = chunk[chunk_index]
            best_end = candidates[candidate_index]
    return best_start, best_end


def _derive_endpoints_from_binary_mask(mask_zyx: np.ndarray) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask_zyx) > 0
    points = np.argwhere(mask)
    if len(points) == 0:
        raise ValueError("Cannot derive RECIST endpoints from an empty mask")
    if len(np.unique(points[:, 0])) == 1:
        slice_index = int(points[0, 0])
    else:
        slices, counts = np.unique(points[:, 0], return_counts=True)
        slice_index = int(slices[np.argmax(counts)])
    slice_points = points[points[:, 0] == slice_index][:, 1:3]
    if len(slice_points) == 1:
        point_yx = slice_points[0]
        endpoints_xyz = np.asarray(
            [
                [float(point_yx[1]), float(point_yx[0]), float(slice_index)],
                [float(point_yx[1]), float(point_yx[0]), float(slice_index)],
            ],
            dtype=np.float32,
        )
        return endpoints_xyz, slice_index
    start_yx, end_yx = _farthest_points_2d(slice_points)
    endpoints_xyz = np.asarray(
        [
            [float(start_yx[1]), float(start_yx[0]), float(slice_index)],
            [float(end_yx[1]), float(end_yx[0]), float(slice_index)],
        ],
        dtype=np.float32,
    )
    return endpoints_xyz, slice_index


def _select_target_instance_mask(mask_zyx: np.ndarray, min_voxels: int = 0) -> np.ndarray:
    components = list(iter_labeled_components(mask_zyx, min_voxels=min_voxels))
    if not components:
        raise ValueError("Cannot select a target instance from an empty mask")
    selected = max(components, key=lambda component: (component.voxel_count, int(component.slice_areas.max())))
    return selected.mask.astype(np.uint8)


def _labeled_component_masks(
    mask_zyx: np.ndarray,
    *,
    min_voxels: int = 0,
) -> list[dict[str, object]]:
    components: list[dict[str, object]] = [
        {
            "label_value": component.label_value,
            "component_id": component.component_id,
            "voxel_count": component.voxel_count,
            "mask": component.mask,
        }
        for component in iter_labeled_components(mask_zyx, min_voxels=min_voxels)
    ]
    components.sort(key=lambda item: (int(item["label_value"]), int(item["component_id"])))
    return components


def _target_mask_from_manifest_selection(
    raw_mask_zyx: np.ndarray,
    target_label_value: int,
    target_component_id: int | None = None,
) -> np.ndarray:
    label_mask = np.asarray(raw_mask_zyx) == int(target_label_value)
    if target_component_id is None:
        return label_mask.astype(np.uint8)
    labeled, component_count = ndimage.label(label_mask)
    component_id = int(target_component_id)
    if component_id < 1 or component_id > int(component_count):
        return np.zeros(label_mask.shape, dtype=np.uint8)
    return (labeled == component_id).astype(np.uint8)


def _recist_component_masks(recist_zyx: np.ndarray) -> list[tuple[str, np.ndarray]]:
    recist = np.asarray(recist_zyx)
    positive_values = [int(value) for value in np.unique(recist).tolist() if int(value) > 0]
    components: list[tuple[str, np.ndarray]] = []
    if not positive_values:
        return components

    if len(positive_values) == 1:
        labeled, component_count = ndimage.label(recist > 0)
        for component_id in range(1, int(component_count) + 1):
            suffix = "recist" if int(component_count) == 1 else f"recist_component{component_id}"
            components.append((suffix, labeled == component_id))
        return components

    for label_value in positive_values:
        labeled, component_count = ndimage.label(recist == label_value)
        for component_id in range(1, int(component_count) + 1):
            suffix = (
                f"recist{label_value}"
                if int(component_count) == 1
                else f"recist{label_value}_component{component_id}"
            )
            components.append((suffix, labeled == component_id))
    return components


def _select_target_label_from_recist_component(
    mask_zyx: np.ndarray | None,
    component_mask_zyx: np.ndarray,
) -> tuple[int | None, int | None, int | None]:
    if mask_zyx is None:
        return None, None, None
    mask = np.asarray(mask_zyx)
    recist_component = np.asarray(component_mask_zyx) > 0
    candidates: list[tuple[int, int, int, int]] = []
    for label_value in [int(value) for value in np.unique(mask).tolist() if int(value) > 0]:
        labeled, component_count = ndimage.label(mask == label_value)
        for component_id in range(1, int(component_count) + 1):
            target_component = labeled == component_id
            overlap = int((target_component & recist_component).sum())
            if overlap <= 0:
                continue
        candidates.append(
                (
                    int(label_value),
                    int(component_id),
                    int(overlap),
                    int(target_component.sum()),
                )
            )
    if not candidates:
        return None, None, None
    target_label, target_component_id, _, target_voxel_count = max(
        candidates,
        key=lambda item: (item[2], item[3]),
    )
    return int(target_label), int(target_component_id), int(target_voxel_count)


def _case_id_for_recist_component(case_id: str, suffix: str, component_count: int) -> str:
    if int(component_count) <= 1:
        return case_id
    return f"{case_id}__{suffix}"


def _scan_recist_component_records(
    *,
    npz_path: Path,
    split: str,
    source: str,
    spacing: np.ndarray,
    image_shape: tuple[int, int, int],
    recist_mask: np.ndarray,
    mask: np.ndarray | None,
) -> list[CaseRecord]:
    components = _recist_component_masks(recist_mask)
    records: list[CaseRecord] = []
    for suffix, component_mask in components:
        try:
            endpoints, slice_index = _derive_endpoints_from_binary_mask(component_mask)
        except ValueError:
            continue
        case_id = _case_id_for_recist_component(npz_path.stem, suffix, len(components))
        _validate_recist_geometry(
            np.asarray(endpoints, dtype=np.float32).reshape(2, 3),
            int(slice_index),
            image_shape,
            case_id,
        )
        target_label_value, target_component_id, target_voxel_count = _select_target_label_from_recist_component(
            mask,
            component_mask,
        )
        records.append(
            CaseRecord(
                case_id=case_id,
                image_path=str(npz_path),
                label_path=None,
                spacing_xyz=np.asarray(spacing, dtype=np.float32).reshape(-1).tolist()[:3],
                recist_endpoints_xyz=np.asarray(endpoints, dtype=np.float32).reshape(2, 3).tolist(),
                recist_slice_index=int(slice_index),
                split=split,
                source=source,
                prompt_source="recist" if len(components) == 1 else "recist_component",
                target_label_value=target_label_value,
                target_component_id=target_component_id,
                target_voxel_count=target_voxel_count,
            )
        )
    return records


def _scan_labeled_mask_component_records(
    *,
    npz_path: Path,
    split: str,
    source: str,
    spacing: np.ndarray,
    image_shape: tuple[int, int, int],
    mask: np.ndarray,
    min_voxels: int = 0,
) -> list[CaseRecord]:
    components = _labeled_component_masks(mask, min_voxels=min_voxels)
    records: list[CaseRecord] = []
    for component in components:
        component_mask = np.asarray(component["mask"], dtype=np.uint8)
        try:
            endpoints, slice_index = _derive_endpoints_from_binary_mask(component_mask)
        except ValueError:
            continue
        component_count = len(components)
        if component_count <= 1:
            case_id = npz_path.stem
        else:
            case_id = (
                f"{npz_path.stem}__label{int(component['label_value'])}"
                f"_component{int(component['component_id'])}"
            )
        _validate_recist_geometry(
            np.asarray(endpoints, dtype=np.float32).reshape(2, 3),
            int(slice_index),
            image_shape,
            case_id,
        )
        records.append(
            CaseRecord(
                case_id=case_id,
                image_path=str(npz_path),
                label_path=None,
                spacing_xyz=np.asarray(spacing, dtype=np.float32).reshape(-1).tolist()[:3],
                recist_endpoints_xyz=np.asarray(endpoints, dtype=np.float32).reshape(2, 3).tolist(),
                recist_slice_index=int(slice_index),
                split=split,
                source=source,
                prompt_source="derived_from_gt_component",
                target_label_value=int(component["label_value"]),
                target_component_id=int(component["component_id"]),
                target_voxel_count=int(component["voxel_count"]),
            )
        )
    return records


def _endpoint_sets_xyz(endpoints: np.ndarray) -> np.ndarray:
    endpoint_array = np.asarray(endpoints, dtype=np.float32)
    if endpoint_array.shape == (2, 3):
        return endpoint_array.reshape(1, 2, 3)
    if endpoint_array.ndim == 3 and endpoint_array.shape[1:] == (2, 3):
        return endpoint_array
    if endpoint_array.ndim == 3 and endpoint_array.shape[0] == 2 and endpoint_array.shape[2] == 3:
        return np.transpose(endpoint_array, (1, 0, 2))
    raise ValueError(f"RECIST endpoints must have shape (2, 3) or (N, 2, 3), got {endpoint_array.shape}")


def _slice_indices_for_endpoint_sets(
    endpoints_xyz: np.ndarray,
    slice_index_array: np.ndarray | None,
) -> list[int]:
    endpoint_sets = np.asarray(endpoints_xyz, dtype=np.float32)
    if slice_index_array is None:
        return [int(round(float(endpoint_set[:, 2].mean()))) for endpoint_set in endpoint_sets]
    slice_indices = np.asarray(slice_index_array)
    if slice_indices.ndim == 0:
        return [int(slice_indices.item()) for _ in range(endpoint_sets.shape[0])]
    flat_indices = slice_indices.reshape(-1)
    if len(flat_indices) != endpoint_sets.shape[0]:
        raise ValueError(
            f"RECIST slice indices must be scalar or length {endpoint_sets.shape[0]}, got {slice_indices.shape}"
        )
    return [int(value) for value in flat_indices.tolist()]


def _target_selection_from_endpoints(
    mask: np.ndarray | None,
    endpoints_xyz: np.ndarray,
    slice_index: int,
) -> tuple[int | None, int | None, int | None]:
    if mask is None:
        return None, None, None
    try:
        _, selection = select_recist_target_instance(
            mask,
            np.asarray(endpoints_xyz, dtype=np.float32),
            recist_slice_index=int(slice_index),
        )
    except ValueError:
        return None, None, None
    label_value = selection.get("selected_label_value")
    component_id = selection.get("selected_component_id")
    voxel_count = selection.get("selected_voxel_count")
    return (
        None if label_value is None else int(label_value),
        None if component_id is None else int(component_id),
        None if voxel_count is None else int(voxel_count),
    )


def _scan_endpoint_records(
    *,
    npz_path: Path,
    split: str,
    source: str,
    spacing: np.ndarray,
    image_shape: tuple[int, int, int],
    endpoints: np.ndarray,
    slice_index_array: np.ndarray | None,
    mask: np.ndarray | None,
    prompt_source: str,
) -> list[CaseRecord]:
    endpoint_sets = _endpoint_sets_xyz(endpoints)
    slice_indices = _slice_indices_for_endpoint_sets(endpoint_sets, slice_index_array)
    records: list[CaseRecord] = []
    for index, endpoint_set in enumerate(endpoint_sets):
        slice_index = int(slice_indices[index])
        case_id = npz_path.stem if len(endpoint_sets) == 1 else f"{npz_path.stem}__recist_endpoint{index + 1}"
        _validate_recist_geometry(
            np.asarray(endpoint_set, dtype=np.float32).reshape(2, 3),
            slice_index,
            image_shape,
            case_id,
        )
        target_label_value, target_component_id, target_voxel_count = _target_selection_from_endpoints(
            mask,
            endpoint_set,
            slice_index,
        )
        records.append(
            CaseRecord(
                case_id=case_id,
                image_path=str(npz_path),
                label_path=None,
                spacing_xyz=np.asarray(spacing, dtype=np.float32).reshape(-1).tolist()[:3],
                recist_endpoints_xyz=np.asarray(endpoint_set, dtype=np.float32).reshape(2, 3).tolist(),
                recist_slice_index=slice_index,
                split=split,
                source=source,
                prompt_source=prompt_source,
                target_label_value=target_label_value,
                target_component_id=target_component_id,
                target_voxel_count=target_voxel_count,
            )
        )
    return records


def _clip_index(value: float, upper: int) -> int:
    if upper <= 0:
        return 0
    return int(np.clip(round(float(value)), 0, upper - 1))


def _component_endpoint_distance(component_points_zyx: np.ndarray, endpoints_zyx: np.ndarray) -> float:
    component_points = np.asarray(component_points_zyx, dtype=np.float32)
    if len(component_points) == 0:
        return float("inf")
    distances: list[float] = []
    for endpoint_zyx in np.asarray(endpoints_zyx, dtype=np.float32):
        distances.append(float(np.linalg.norm(component_points - endpoint_zyx[None, :], axis=1).min()))
    return float(np.mean(distances)) if distances else float("inf")


def _candidate_score(candidate: dict[str, object]) -> tuple[float, ...]:
    return (
        float(int(bool(candidate["endpoint_label_hit"]))),
        float(int(bool(candidate["line_label_hit"]))),
        float(int(candidate["line_overlap"])),
        float(int(candidate["endpoint_hits"])),
        -float(candidate["endpoint_distance"]),
        -float(candidate["slice_distance"]),
        float(int(candidate["voxel_count"])),
    )


def select_recist_target_instance(
    mask_zyx: np.ndarray,
    endpoints_xyz: np.ndarray,
    recist_slice_index: int | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    mask = np.asarray(mask_zyx)
    positive_labels = [int(value) for value in np.unique(mask).tolist() if int(value) > 0]
    if not positive_labels:
        raise ValueError("Cannot select a RECIST target target instance from an empty mask")

    endpoints_xyz = np.asarray(endpoints_xyz, dtype=np.float32).reshape(2, 3)
    endpoints_zyx = np.asarray([xyz_to_zyx(endpoint) for endpoint in endpoints_xyz], dtype=np.float32)
    midpoint_zyx = endpoints_zyx.mean(axis=0)
    recist_slice = (
        int(recist_slice_index)
        if recist_slice_index is not None
        else int(round(float(midpoint_zyx[0])))
    )
    recist_slice = int(np.clip(recist_slice, 0, mask.shape[0] - 1))
    line_mask = rasterize_line_mask(mask.shape, endpoints_xyz, radius_xy=0)
    line_slice = line_mask[recist_slice] > 0

    endpoint_indices_zyx = np.asarray(
        [
            [
                _clip_index(endpoint_zyx[0], mask.shape[0]),
                _clip_index(endpoint_zyx[1], mask.shape[1]),
                _clip_index(endpoint_zyx[2], mask.shape[2]),
            ]
            for endpoint_zyx in endpoints_zyx
        ],
        dtype=np.int32,
    )
    labels_at_endpoints = {
        int(mask[int(index[0]), int(index[1]), int(index[2])])
        for index in endpoint_indices_zyx
        if int(mask[int(index[0]), int(index[1]), int(index[2])]) > 0
    }
    labels_on_line = {
        int(value)
        for value in np.unique(mask[recist_slice][line_slice]).tolist()
        if int(value) > 0
    }

    candidates: list[dict[str, object]] = []
    for component in iter_labeled_components(mask):
        component_mask = component.mask > 0
        component_points = np.argwhere(component_mask)
        component_slice_mask = component_mask[recist_slice]
        slice_points = np.argwhere(component_slice_mask)
        points_for_distance = slice_points.copy()
        if len(points_for_distance) > 0:
            points_for_distance = np.column_stack(
                [
                    np.full(len(points_for_distance), recist_slice, dtype=np.float32),
                    points_for_distance[:, 0].astype(np.float32),
                    points_for_distance[:, 1].astype(np.float32),
                ]
            )
        else:
            points_for_distance = component_points.astype(np.float32)
        endpoint_distance = _component_endpoint_distance(points_for_distance, endpoints_zyx)
        slice_distance = float(np.min(np.abs(component_points[:, 0] - recist_slice)))
        endpoint_hits = int(
            sum(
                int(component_mask[int(index[0]), int(index[1]), int(index[2])])
                for index in endpoint_indices_zyx
            )
        )
        candidates.append(
            {
                "label_value": int(component.label_value),
                "component_id": int(component.component_id),
                "mask": component.mask,
                "voxel_count": int(component.voxel_count),
                "line_overlap": int((component_slice_mask & line_slice).sum()),
                "endpoint_hits": int(endpoint_hits),
                "endpoint_distance": float(endpoint_distance),
                "slice_distance": float(slice_distance),
                "endpoint_label_hit": bool(component.label_value in labels_at_endpoints),
                "line_label_hit": bool(component.label_value in labels_on_line),
            }
        )

    if not candidates:
        fallback_mask = _select_target_instance_mask(mask_zyx)
        return fallback_mask, {
            "selection_mode": "largest_component_fallback",
            "used_fallback": True,
            "positive_label_count": int(len(positive_labels)),
            "component_candidate_count": 0,
            "labels_at_endpoints": sorted(int(value) for value in labels_at_endpoints),
            "labels_on_recist_line": sorted(int(value) for value in labels_on_line),
            "recist_slice_index": int(recist_slice),
            "selected_label_value": None,
            "selected_component_id": None,
            "selected_voxel_count": int(fallback_mask.sum()),
        }

    selected = max(candidates, key=_candidate_score)
    if bool(selected["endpoint_label_hit"]):
        selection_mode = "direct_endpoint_label_hit"
    elif bool(selected["line_label_hit"]) or int(selected["line_overlap"]) > 0 or int(selected["endpoint_hits"]) > 0:
        selection_mode = "recist_line_match"
    else:
        selection_mode = "nearest_component_fallback"

    selection = {
        "selection_mode": selection_mode,
        "used_fallback": bool(selection_mode == "nearest_component_fallback"),
        "positive_label_count": int(len(positive_labels)),
        "component_candidate_count": int(len(candidates)),
        "labels_at_endpoints": sorted(int(value) for value in labels_at_endpoints),
        "labels_on_recist_line": sorted(int(value) for value in labels_on_line),
        "recist_slice_index": int(recist_slice),
        "selected_label_value": int(selected["label_value"]),
        "selected_component_id": int(selected["component_id"]),
        "selected_voxel_count": int(selected["voxel_count"]),
        "selected_line_overlap": int(selected["line_overlap"]),
        "selected_endpoint_hits": int(selected["endpoint_hits"]),
        "selected_endpoint_distance": float(selected["endpoint_distance"]),
        "selected_slice_distance": float(selected["slice_distance"]),
        "selected_endpoint_label_hit": bool(selected["endpoint_label_hit"]),
        "selected_line_label_hit": bool(selected["line_label_hit"]),
    }
    return np.asarray(selected["mask"], dtype=np.uint8), selection


def get_recist_target_mask(case: LoadedCase) -> np.ndarray | None:
    if case.recist_target_mask is not None:
        return case.recist_target_mask
    return case.mask


def enumerate_lesion_instances(mask_zyx: np.ndarray) -> list[dict[str, object]]:
    instances: list[dict[str, object]] = []
    for component in iter_labeled_components(mask_zyx):
        points = np.argwhere(component.mask > 0)
        center_zyx = points.mean(axis=0).astype(np.float32)
        instances.append(
            {
                "label_value": component.label_value,
                "component_id": component.component_id,
                "voxel_count": component.voxel_count,
                "center_zyx": center_zyx,
                "slice_index": int(np.argmax(component.slice_areas)),
            }
        )
    instances.sort(key=lambda row: (int(row["voxel_count"]), -int(row["slice_index"])), reverse=True)
    return instances
