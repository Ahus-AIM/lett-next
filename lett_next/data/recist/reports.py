from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import MASK_KEYS
from .io import _find_first_key
from .recist_geometry import (
    _target_mask_from_manifest_selection,
    enumerate_lesion_instances,
    select_recist_target_instance,
)
from .records import CaseRecord


def _summarize_values(values: list[int | float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    array = np.asarray(values, dtype=np.float32)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
    }


def _dedupe_case_records(records: list[CaseRecord]) -> list[CaseRecord]:
    unique_records: list[CaseRecord] = []
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        key = (record.case_id, record.image_path, record.split)
        if key in seen:
            continue
        seen.add(key)
        unique_records.append(record)
    return unique_records


def build_recist_target_report(
    records: list[CaseRecord],
    target_spacing_xyz: tuple[float, float, float] | None = None,
    resampling_mode: str = "standard",
) -> dict[str, object]:
    unique_records = _dedupe_case_records(records)
    selection_mode_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    union_voxel_fractions: list[float] = []
    union_voxel_counts: list[int] = []
    selected_voxel_counts: list[int] = []
    component_counts: list[int] = []
    multi_component_count = 0
    direct_endpoint_count = 0
    fallback_count = 0
    case_rows: list[dict[str, object]] = []
    for record in unique_records:
        split_counts[record.split] = split_counts.get(record.split, 0) + 1
        summary = load_recist_target_summary(
            record,
            target_spacing_xyz=target_spacing_xyz,
            resampling_mode=resampling_mode,
        )
        if summary is None:
            case_rows.append(
                {
                    "case_id": record.case_id,
                    "split": record.split,
                    "selection_mode": "missing_mask",
                    "component_count": 0,
                    "all_positive_voxels": 0,
                    "selected_target_voxels": 0,
                    "selected_fraction_of_all_positive": 0.0,
                }
            )
            continue
        selection = summary["selection"]
        selection_mode = str(selection.get("selection_mode", "unknown"))
        selection_mode_counts[selection_mode] = selection_mode_counts.get(selection_mode, 0) + 1
        if selection_mode == "direct_endpoint_label_hit":
            direct_endpoint_count += 1
        if bool(selection.get("used_fallback", False)):
            fallback_count += 1
        component_count = int(summary["component_count"])
        if component_count > 1:
            multi_component_count += 1
        all_positive_voxels = int(summary["all_positive_voxels"])
        selected_target_voxels = int(summary["selected_target_voxels"])
        selected_fraction = (
            float(selected_target_voxels / max(all_positive_voxels, 1))
            if all_positive_voxels > 0
            else 0.0
        )
        union_voxel_counts.append(all_positive_voxels)
        selected_voxel_counts.append(selected_target_voxels)
        union_voxel_fractions.append(selected_fraction)
        component_counts.append(component_count)
        case_rows.append(
            {
                "case_id": record.case_id,
                "split": record.split,
                "selection_mode": selection_mode,
                "component_count": component_count,
                "all_positive_voxels": all_positive_voxels,
                "selected_target_voxels": selected_target_voxels,
                "selected_fraction_of_all_positive": selected_fraction,
                "selected_label_value": selection.get("selected_label_value"),
                "selected_component_id": selection.get("selected_component_id"),
                "selected_line_overlap": selection.get("selected_line_overlap"),
                "selected_endpoint_distance": selection.get("selected_endpoint_distance"),
            }
        )
    return {
        "case_count": len(unique_records),
        "split_counts": split_counts,
        "selection_mode_counts": selection_mode_counts,
        "multi_component_case_count": int(multi_component_count),
        "multi_component_case_fraction": float(multi_component_count / max(len(unique_records), 1)),
        "direct_endpoint_selection_count": int(direct_endpoint_count),
        "fallback_selection_count": int(fallback_count),
        "all_positive_voxel_summary": _summarize_values(union_voxel_counts),
        "selected_target_voxel_summary": _summarize_values(selected_voxel_counts),
        "selected_fraction_of_all_positive_summary": _summarize_values(union_voxel_fractions),
        "component_count_summary": _summarize_values(component_counts),
        "cases": case_rows,
    }


def load_recist_target_summary(
    record: CaseRecord,
    target_spacing_xyz: tuple[float, float, float] | None = None,
    resampling_mode: str = "standard",
) -> dict[str, object] | None:
    if target_spacing_xyz is None and record.target_voxel_count is not None and record.target_voxel_count <= 0:
        return None
    if target_spacing_xyz is None and record.target_voxel_count is not None and record.target_label_value is not None:
        target_voxels = int(record.target_voxel_count)
        return {
            "selection": {
                "selection_mode": "manifest_target_label",
                "used_fallback": False,
                "selected_label_value": int(record.target_label_value),
                "selected_component_id": record.target_component_id,
                "selected_voxel_count": target_voxels,
                "summary_source": "manifest_cached",
            },
            "component_count": 1,
            "all_positive_voxels": target_voxels,
            "selected_target_voxels": target_voxels,
        }
    if target_spacing_xyz is not None or record.label_path:
        from .case_loader import load_case

        case = load_case(record, target_spacing_xyz=target_spacing_xyz, resampling_mode=resampling_mode)
        if case.raw_mask is None or case.mask is None or case.recist_target_mask is None:
            return None
        selection = case.recist_target_selection or {}
        return {
            "selection": selection,
            "component_count": int(len(enumerate_lesion_instances(case.raw_mask))),
            "all_positive_voxels": int(case.mask.sum()),
            "selected_target_voxels": int(case.recist_target_mask.sum()),
        }

    npz_path = Path(record.image_path)
    with np.load(npz_path, allow_pickle=False) as data:
        raw_mask = _find_first_key(data, MASK_KEYS)
        if raw_mask is None:
            return None
        raw_mask = np.asarray(raw_mask)
    if not np.any(raw_mask > 0):
        return None
    if record.target_label_value is not None:
        target_mask = _target_mask_from_manifest_selection(
            raw_mask,
            int(record.target_label_value),
            record.target_component_id,
        )
        target_voxels = int(target_mask.sum())
        if target_voxels <= 0:
            return None
        return {
            "selection": {
                "selection_mode": "manifest_target_label",
                "used_fallback": False,
                "selected_label_value": int(record.target_label_value),
                "selected_component_id": record.target_component_id,
                "selected_voxel_count": target_voxels,
            },
            "component_count": int(len(enumerate_lesion_instances(raw_mask))),
            "all_positive_voxels": int((raw_mask > 0).sum()),
            "selected_target_voxels": target_voxels,
        }
    try:
        target_mask, selection = select_recist_target_instance(
            raw_mask,
            np.asarray(record.recist_endpoints_xyz, dtype=np.float32),
            recist_slice_index=int(record.recist_slice_index),
        )
    except ValueError:
        return None
    return {
        "selection": selection,
        "component_count": int(len(enumerate_lesion_instances(raw_mask))),
        "all_positive_voxels": int((raw_mask > 0).sum()),
        "selected_target_voxels": int(np.asarray(target_mask).sum()),
    }


def recist_target_voxel_count_for_record(
    record: CaseRecord,
    target_spacing_xyz: tuple[float, float, float] | None = None,
    resampling_mode: str = "standard",
) -> int | None:
    if target_spacing_xyz is None and record.target_voxel_count is not None:
        return int(record.target_voxel_count)
    summary = load_recist_target_summary(record, target_spacing_xyz=target_spacing_xyz, resampling_mode=resampling_mode)
    if summary is None:
        return None
    return int(summary["selected_target_voxels"])
