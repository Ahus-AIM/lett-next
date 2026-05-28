from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import _import_nibabel
from .records import CaseRecord, IntegrityReport, LoadedCase


def _integrity_error(case_id: str, reason: str) -> ValueError:
    return ValueError(f"Dataset integrity check failed for {case_id}: {reason}")


def _validate_spacing_xyz(spacing_xyz: np.ndarray, case_id: str) -> np.ndarray:
    spacing = np.asarray(spacing_xyz, dtype=np.float32).reshape(-1)
    if spacing.shape != (3,):
        raise _integrity_error(case_id, f"spacing_xyz must have length 3, got shape {spacing.shape}")
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise _integrity_error(case_id, f"spacing_xyz must be finite and positive, got {spacing.tolist()}")
    return spacing


def _validate_image_array(image: np.ndarray, case_id: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim > 3 and all(int(dim) == 1 for dim in array.shape[3:]):
        array = array.reshape(array.shape[:3])
    if array.ndim != 3:
        raise _integrity_error(case_id, f"image must be 3D, got shape {array.shape}")
    if any(int(dim) <= 0 for dim in array.shape):
        raise _integrity_error(case_id, f"image shape must be positive, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise _integrity_error(case_id, f"image dtype must be numeric, got {array.dtype}")
    if not np.isfinite(array).all():
        raise _integrity_error(case_id, "image contains NaN or Inf")
    return array


def _validate_mask_array(mask: np.ndarray | None, image_shape: tuple[int, int, int], case_id: str, require_nonempty: bool) -> None:
    if mask is None:
        if require_nonempty:
            raise _integrity_error(case_id, "mask is required but missing")
        return
    array = np.asarray(mask)
    if array.ndim != 3:
        raise _integrity_error(case_id, f"mask must be 3D, got shape {array.shape}")
    if tuple(int(dim) for dim in array.shape) != tuple(int(dim) for dim in image_shape):
        raise _integrity_error(case_id, f"mask shape {array.shape} does not match image shape {image_shape}")
    if not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_)):
        raise _integrity_error(case_id, f"mask dtype must be integer or bool, got {array.dtype}")
    if not np.isfinite(array).all():
        raise _integrity_error(case_id, "mask contains NaN or Inf")
    if require_nonempty and not np.any(array > 0):
        raise _integrity_error(case_id, "mask has no positive voxels")


def _validate_recist_geometry(
    endpoints_xyz: np.ndarray,
    recist_slice_index: int,
    image_shape_zyx: tuple[int, int, int],
    case_id: str,
) -> np.ndarray:
    endpoints = np.asarray(endpoints_xyz, dtype=np.float32)
    if endpoints.shape != (2, 3):
        raise _integrity_error(case_id, f"RECIST endpoints must have shape (2, 3), got {endpoints.shape}")
    if not np.isfinite(endpoints).all():
        raise _integrity_error(case_id, "RECIST endpoints contain NaN or Inf")
    max_xyz = np.asarray(
        [image_shape_zyx[2] - 1, image_shape_zyx[1] - 1, image_shape_zyx[0] - 1],
        dtype=np.float32,
    )
    if np.any(endpoints < 0) or np.any(endpoints > max_xyz[None, :]):
        raise _integrity_error(
            case_id,
            f"RECIST endpoints out of bounds for image shape {image_shape_zyx}: {endpoints.tolist()}",
        )
    if int(recist_slice_index) < 0 or int(recist_slice_index) >= int(image_shape_zyx[0]):
        raise _integrity_error(
            case_id,
            f"recist_slice_index {recist_slice_index} out of bounds for depth {image_shape_zyx[0]}",
        )
    endpoint_slice_delta = np.abs(endpoints[:, 2] - float(recist_slice_index))
    if not np.any(endpoint_slice_delta <= 0.5001):
        raise _integrity_error(
            case_id,
            f"recist_slice_index {recist_slice_index} is inconsistent with endpoint z values {endpoints[:, 2].tolist()}",
        )
    return endpoints


def _validate_nifti_pair_geometry(image_path: Path, label_path: Path, case_id: str) -> None:
    nib = _import_nibabel()
    image = nib.load(str(image_path))
    label = nib.load(str(label_path))
    if tuple(image.shape[:3]) != tuple(label.shape[:3]):
        raise _integrity_error(case_id, f"NIfTI label shape {label.shape[:3]} does not match image shape {image.shape[:3]}")
    image_spacing = np.asarray(image.header.get_zooms()[:3], dtype=np.float32)
    label_spacing = np.asarray(label.header.get_zooms()[:3], dtype=np.float32)
    if not np.allclose(image_spacing, label_spacing, rtol=1e-4, atol=1e-4):
        raise _integrity_error(case_id, f"NIfTI label spacing {label_spacing.tolist()} does not match image spacing {image_spacing.tolist()}")
    if not np.allclose(np.asarray(image.affine), np.asarray(label.affine), rtol=1e-4, atol=1e-4):
        raise _integrity_error(case_id, "NIfTI label affine does not match image affine")


def validate_loaded_case_integrity(
    case: LoadedCase,
    *,
    require_mask: bool = False,
    require_recist_target: bool = False,
) -> None:
    image = _validate_image_array(case.image, case.case_id)
    _validate_spacing_xyz(case.spacing_xyz, case.case_id)
    _validate_mask_array(case.raw_mask, tuple(int(dim) for dim in image.shape), case.case_id, require_mask)
    _validate_recist_geometry(
        case.recist_endpoints_xyz,
        case.recist_slice_index,
        tuple(int(dim) for dim in image.shape),
        case.case_id,
    )
    if require_recist_target and (case.recist_target_mask is None or not np.any(case.recist_target_mask > 0)):
        raise _integrity_error(case.case_id, "RECIST target selected target mask is missing or empty")


def validate_records_integrity(
    records: list[CaseRecord],
    *,
    target_spacing_xyz: tuple[float, float, float] | None = None,
    resampling_mode: str = "standard",
    require_mask: bool = False,
    require_recist_target: bool = False,
) -> IntegrityReport:
    failures: list[dict[str, str]] = []
    for record in records:
        if (
            target_spacing_xyz is None
            and require_recist_target
            and record.target_voxel_count is not None
            and int(record.target_voxel_count) > 0
            and record.target_label_value is not None
        ):
            continue
        try:
            from .case_loader import load_case

            load_case(
                record,
                target_spacing_xyz=target_spacing_xyz,
                resampling_mode=resampling_mode,
                validate_integrity=True,
                require_mask=require_mask,
                require_recist_target=require_recist_target,
            )
        except Exception as exc:
            failures.append({"case_id": record.case_id, "path": record.image_path, "reason": str(exc)})
    if failures:
        preview = "; ".join(
            f"{row['case_id']}: {row['reason']}" for row in failures[:5]
        )
        suffix = "" if len(failures) <= 5 else f"; ... {len(failures) - 5} more"
        raise ValueError(f"Dataset integrity validation failed for {len(failures)} case(s): {preview}{suffix}")
    return IntegrityReport(
        checked_case_count=int(len(records)),
        failed_case_count=0,
        failures=[],
    )
