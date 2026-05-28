from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...pants_aux import PantsAuxClassMap, class_map_metadata
from .constants import PANTS_LESION_LABEL


def _find_first_key(values: dict[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key in values:
            return values[key]
    return None


def _find_first_present_key(keys_available: set[str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in keys_available:
            return key
    return None


def _import_nibabel() -> Any:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - dependency failure path
        raise ImportError("NIfTI support requires nibabel. Install the project dependencies again.") from exc
    return nib


def _is_nifti_path(path: Path) -> bool:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    return suffixes[-1:] == [".nii"] or suffixes[-2:] == [".nii", ".gz"]


def _load_nifti_zyx(path: Path) -> tuple[np.ndarray, np.ndarray]:
    nib = _import_nibabel()
    image = nib.load(str(path))
    array_xyz = np.asanyarray(image.dataobj)
    if array_xyz.ndim > 3 and all(int(dim) == 1 for dim in array_xyz.shape[3:]):
        array_xyz = array_xyz.reshape(array_xyz.shape[:3])
    if array_xyz.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume in {path}, got shape {array_xyz.shape}")
    spacing_xyz = np.asarray(image.header.get_zooms()[:3], dtype=np.float32)
    return np.asarray(array_xyz).transpose(2, 1, 0), spacing_xyz


def _read_nifti_spacing_xyz(path: Path) -> np.ndarray:
    nib = _import_nibabel()
    image = nib.load(str(path))
    return np.asarray(image.header.get_zooms()[:3], dtype=np.float32)


def _load_pants_label_mask_zyx(label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    label_zyx, spacing_xyz = _load_nifti_zyx(label_path)
    if label_path.name == "combined_labels.nii.gz" or label_path.name == "combined_labels.nii":
        label_zyx = label_zyx == PANTS_LESION_LABEL
    else:
        label_zyx = label_zyx > 0
    return label_zyx.astype(np.uint8), spacing_xyz


def _pants_label_case_dir(label_path: Path) -> Path:
    return label_path.parent.parent if label_path.parent.name == "segmentations" else label_path.parent


def _pants_segmentation_path(label_case_dir: Path, name: str) -> Path | None:
    segmentations_dir = label_case_dir / "segmentations"
    for suffix in (".nii.gz", ".nii"):
        candidate = segmentations_dir / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _load_pants_aux_target_zyx(
    *,
    label_path: Path,
    class_map: PantsAuxClassMap,
    image_shape_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    ignore_index = int(class_map.ignore_index)
    aux_target = np.full(image_shape_zyx, ignore_index, dtype=np.int16)
    aux_valid_mask = np.zeros(image_shape_zyx, dtype=bool)
    label_case_dir = _pants_label_case_dir(label_path)
    loaded_segmentations: list[str] = []
    missing_segmentations: list[str] = []
    occupied = np.zeros(image_shape_zyx, dtype=bool)
    all_foreground_available = True
    for class_id, segmentation in sorted(class_map.segmentations_by_class_id.items()):
        if int(class_id) == 1:
            continue
        segmentation_path = _pants_segmentation_path(label_case_dir, segmentation)
        if segmentation_path is None:
            missing_segmentations.append(segmentation)
            all_foreground_available = False
            continue
        mask_zyx, _ = _load_nifti_zyx(segmentation_path)
        mask = np.asarray(mask_zyx) > 0
        if tuple(int(dim) for dim in mask.shape) != image_shape_zyx:
            raise ValueError(
                f"PanTS aux segmentation shape {mask.shape} for {segmentation_path} "
                f"does not match image shape {image_shape_zyx}"
            )
        loaded_segmentations.append(segmentation)
        aux_target[mask] = np.int16(class_id)
        aux_valid_mask[mask] = True
        occupied |= mask
    tumor_segmentation = _pants_segmentation_path(label_case_dir, "pancreatic_lesion")
    if tumor_segmentation is not None:
        tumor_zyx, _ = _load_nifti_zyx(tumor_segmentation)
        tumor_mask = np.asarray(tumor_zyx) > 0
        if tuple(int(dim) for dim in tumor_mask.shape) != image_shape_zyx:
            raise ValueError(
                f"PanTS aux tumor segmentation shape {tumor_mask.shape} for {tumor_segmentation} "
                f"does not match image shape {image_shape_zyx}"
            )
        loaded_segmentations.append("pancreatic_lesion")
        aux_target[tumor_mask] = np.int16(1)
        aux_valid_mask[tumor_mask] = True
        occupied |= tumor_mask
    elif label_path.name in {"combined_labels.nii.gz", "combined_labels.nii"}:
        combined_zyx, _ = _load_nifti_zyx(label_path)
        tumor_mask = np.asarray(combined_zyx) == PANTS_LESION_LABEL
        aux_target[tumor_mask] = np.int16(1)
        aux_valid_mask[tumor_mask] = True
        occupied |= tumor_mask
    else:
        missing_segmentations.append("pancreatic_lesion")
        all_foreground_available = False
    if all_foreground_available:
        background = ~occupied
        aux_target[background] = np.int16(0)
        aux_valid_mask[background] = True
    metadata = class_map_metadata(class_map)
    metadata |= {
        "loaded_segmentations": loaded_segmentations,
        "missing_segmentations": missing_segmentations,
        "aux_background_is_valid": bool(all_foreground_available),
    }
    return aux_target, aux_valid_mask, metadata
