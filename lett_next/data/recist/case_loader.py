from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...pants_aux import PantsAuxClassMap, class_map_metadata
from .constants import IMAGE_KEYS, LEGACY_RECIST_RESAMPLING_MODE, MASK_KEYS
from .io import _find_first_key, _is_nifti_path, _load_nifti_zyx, _load_pants_aux_target_zyx, _load_pants_label_mask_zyx
from .recist_geometry import _target_mask_from_manifest_selection, select_recist_target_instance
from .records import CaseRecord, LoadedCase
from .resampling import _resample_anatomy_aux_arrays, _resample_loaded_case, _resample_loaded_case_recist, _zoom_array
from .split_cache import load_split_case_cache
from .validation import _validate_nifti_pair_geometry, validate_loaded_case_integrity


@dataclass(slots=True)
class RawCaseArrays:
    image: np.ndarray
    raw_mask: np.ndarray | None
    spacing_xyz: np.ndarray
    anatomy_aux_target: np.ndarray | None = None
    anatomy_aux_valid_mask: np.ndarray | None = None
    anatomy_aux_metadata: dict[str, object] | None = None


@dataclass(slots=True)
class ResampledCaseArrays:
    image: np.ndarray
    raw_mask: np.ndarray | None
    spacing_xyz: np.ndarray
    recist_endpoints_xyz: np.ndarray
    recist_slice_index: int
    anatomy_aux_target: np.ndarray | None = None
    anatomy_aux_valid_mask: np.ndarray | None = None
    anatomy_aux_metadata: dict[str, object] | None = None


def maybe_load_cached_case(
    record: CaseRecord,
    *,
    resampling_mode: str = "standard",
    require_recist_target: bool = False,
    normalization_stats: dict[str, object] | None = None,
    target_spacing_xyz: tuple[float, float, float] | None = None,
    case_cache_dir: Path | str | None = None,
    pants_aux_class_map: PantsAuxClassMap | None = None,
) -> LoadedCase | None:
    if case_cache_dir is None or target_spacing_xyz is None:
        return None
    if resampling_mode != LEGACY_RECIST_RESAMPLING_MODE or not require_recist_target:
        return None
    return load_split_case_cache(
        record,
        cache_dir=Path(case_cache_dir),
        target_spacing_xyz=target_spacing_xyz,
        resampling_mode=resampling_mode,
        normalization_stats=normalization_stats,
        pants_aux_class_map=pants_aux_class_map,
    )


def load_raw_case_arrays(
    record: CaseRecord,
    *,
    load_mask: bool,
    pants_aux_class_map: PantsAuxClassMap | None = None,
) -> RawCaseArrays:
    should_load_mask = bool(load_mask)
    image_path = Path(record.image_path)
    aux_target = None
    aux_valid_mask = None
    aux_metadata = None
    if _is_nifti_path(image_path):
        if should_load_mask and record.label_path not in (None, ""):
            _validate_nifti_pair_geometry(image_path, Path(record.label_path), record.case_id)
        image_array, spacing_xyz = _load_nifti_zyx(image_path)
        image_array = np.asarray(image_array, dtype=np.float32)
        raw_mask = None
        if should_load_mask and record.label_path not in (None, ""):
            label_path = Path(record.label_path)
            if record.source == "pants":
                raw_mask, _ = _load_pants_label_mask_zyx(label_path)
                if pants_aux_class_map is not None:
                    aux_target, aux_valid_mask, aux_metadata = _load_pants_aux_target_zyx(
                        label_path=label_path,
                        class_map=pants_aux_class_map,
                        image_shape_zyx=tuple(int(dim) for dim in image_array.shape),
                    )
            else:
                label_array, _ = _load_nifti_zyx(label_path)
                raw_mask = np.asarray(label_array)
    else:
        npz_path = image_path
        with np.load(npz_path, allow_pickle=False) as data:
            image = _find_first_key(data, IMAGE_KEYS)
            if image is None:
                raise KeyError(f"Could not find image array in {npz_path}")
            raw_mask = None
            if should_load_mask:
                mask = _find_first_key(data, MASK_KEYS)
                raw_mask = None if mask is None else np.asarray(mask)
            image_array = np.asarray(image, dtype=np.float32)
            spacing_xyz = np.asarray(record.spacing_xyz, dtype=np.float32)
    return RawCaseArrays(
        image=image_array,
        raw_mask=raw_mask,
        spacing_xyz=np.asarray(spacing_xyz, dtype=np.float32),
        anatomy_aux_target=aux_target,
        anatomy_aux_valid_mask=aux_valid_mask,
        anatomy_aux_metadata=aux_metadata,
    )


def maybe_resample_case_arrays(
    raw: RawCaseArrays,
    record: CaseRecord,
    *,
    target_spacing_xyz: tuple[float, float, float] | None,
    resampling_mode: str,
) -> ResampledCaseArrays:
    recist_endpoints_xyz = np.asarray(record.recist_endpoints_xyz, dtype=np.float32)
    recist_slice_index = int(record.recist_slice_index)
    source_spacing_xyz = np.asarray(raw.spacing_xyz, dtype=np.float32)
    image_array = raw.image
    raw_mask = raw.raw_mask
    spacing_xyz = raw.spacing_xyz
    aux_target = raw.anatomy_aux_target
    aux_valid_mask = raw.anatomy_aux_valid_mask
    if target_spacing_xyz is not None:
        resampler = _resample_loaded_case_recist if resampling_mode == LEGACY_RECIST_RESAMPLING_MODE else _resample_loaded_case
        image_array, raw_mask, spacing_xyz, recist_endpoints_xyz, recist_slice_index = resampler(
            image=image_array,
            raw_mask=raw_mask,
            spacing_xyz=spacing_xyz,
            recist_endpoints_xyz=recist_endpoints_xyz,
            recist_slice_index=recist_slice_index,
            target_spacing_xyz=target_spacing_xyz,
        )
        if aux_target is not None and aux_valid_mask is not None:
            if resampling_mode != LEGACY_RECIST_RESAMPLING_MODE:
                zoom_zyx = tuple(float(v) for v in (source_spacing_xyz[::-1] / np.asarray(target_spacing_xyz, dtype=np.float32)[::-1]).tolist())
                aux_target = _zoom_array(np.asarray(aux_target, dtype=np.int16), zoom_zyx, order=0).astype(np.int16)
                aux_valid_mask = _zoom_array(np.asarray(aux_valid_mask, dtype=np.uint8), zoom_zyx, order=0) > 0
            else:
                aux_target, aux_valid_mask = _resample_anatomy_aux_arrays(
                    aux_target,
                    aux_valid_mask,
                    source_spacing_xyz=source_spacing_xyz,
                    target_spacing_xyz=target_spacing_xyz,
                )
    return ResampledCaseArrays(
        image=image_array,
        raw_mask=raw_mask,
        spacing_xyz=np.asarray(spacing_xyz, dtype=np.float32),
        recist_endpoints_xyz=recist_endpoints_xyz,
        recist_slice_index=recist_slice_index,
        anatomy_aux_target=aux_target,
        anatomy_aux_valid_mask=aux_valid_mask,
        anatomy_aux_metadata=raw.anatomy_aux_metadata,
    )


def ensure_aux_placeholder(
    arrays: ResampledCaseArrays,
    pants_aux_class_map: PantsAuxClassMap | None,
) -> ResampledCaseArrays:
    aux_target = arrays.anatomy_aux_target
    aux_valid_mask = arrays.anatomy_aux_valid_mask
    aux_metadata = arrays.anatomy_aux_metadata
    if pants_aux_class_map is not None and aux_target is None:
        aux_target = np.full(tuple(int(dim) for dim in arrays.image.shape), int(pants_aux_class_map.ignore_index), dtype=np.int16)
        aux_valid_mask = np.zeros(tuple(int(dim) for dim in arrays.image.shape), dtype=bool)
        aux_metadata = class_map_metadata(pants_aux_class_map) | {
            "loaded_segmentations": [],
            "missing_segmentations": [],
            "aux_background_is_valid": False,
        }
    return ResampledCaseArrays(
        image=arrays.image,
        raw_mask=arrays.raw_mask,
        spacing_xyz=arrays.spacing_xyz,
        recist_endpoints_xyz=arrays.recist_endpoints_xyz,
        recist_slice_index=arrays.recist_slice_index,
        anatomy_aux_target=aux_target,
        anatomy_aux_valid_mask=aux_valid_mask,
        anatomy_aux_metadata=aux_metadata,
    )


def select_case_target_mask(
    record: CaseRecord,
    arrays: ResampledCaseArrays,
) -> tuple[np.ndarray | None, dict[str, object] | None]:
    recist_target_mask = None
    recist_target_selection = None
    if arrays.raw_mask is not None:
        if record.target_label_value is not None:
            target_mask = _target_mask_from_manifest_selection(
                arrays.raw_mask,
                int(record.target_label_value),
                record.target_component_id,
            )
            if int(target_mask.sum()) > 0:
                recist_target_mask = target_mask
                recist_target_selection = {
                    "selection_mode": "manifest_target_label",
                    "used_fallback": False,
                    "selected_label_value": int(record.target_label_value),
                    "selected_component_id": record.target_component_id,
                    "selected_voxel_count": int(target_mask.sum()),
                }
        if recist_target_mask is None:
            try:
                recist_target_mask, recist_target_selection = select_recist_target_instance(
                    arrays.raw_mask,
                    arrays.recist_endpoints_xyz,
                    recist_slice_index=arrays.recist_slice_index,
                )
            except ValueError:
                recist_target_mask = None
                recist_target_selection = None
    return recist_target_mask, recist_target_selection


def build_loaded_case(
    record: CaseRecord,
    arrays: ResampledCaseArrays,
    *,
    recist_target_mask: np.ndarray | None,
    recist_target_selection: dict[str, object] | None,
    normalization_stats: dict[str, object] | None,
) -> LoadedCase:
    return LoadedCase(
        case_id=record.case_id,
        image=arrays.image,
        raw_mask=arrays.raw_mask,
        mask=None if arrays.raw_mask is None else (arrays.raw_mask > 0).astype(np.uint8),
        spacing_xyz=arrays.spacing_xyz,
        recist_endpoints_xyz=arrays.recist_endpoints_xyz,
        recist_slice_index=arrays.recist_slice_index,
        prompt_source=record.prompt_source,
        recist_target_mask=recist_target_mask,
        recist_target_selection=recist_target_selection,
        anatomy_aux_target=arrays.anatomy_aux_target,
        anatomy_aux_valid_mask=arrays.anatomy_aux_valid_mask,
        anatomy_aux_metadata=arrays.anatomy_aux_metadata,
        normalization_stats=normalization_stats,
    )


def validate_and_return(
    case: LoadedCase,
    *,
    validate_integrity: bool,
    require_mask: bool,
    require_recist_target: bool,
) -> LoadedCase:
    if validate_integrity:
        validate_loaded_case_integrity(
            case,
            require_mask=require_mask,
            require_recist_target=require_recist_target,
        )
    return case


def load_case(
    record: CaseRecord,
    target_spacing_xyz: tuple[float, float, float] | None = None,
    *,
    resampling_mode: str = "standard",
    validate_integrity: bool = True,
    require_mask: bool = False,
    require_recist_target: bool = False,
    normalization_stats: dict[str, object] | None = None,
    load_mask: bool = True,
    case_cache_dir: Path | str | None = None,
    pants_aux_class_map: PantsAuxClassMap | None = None,
) -> LoadedCase:
    cached_case = maybe_load_cached_case(
        record,
        resampling_mode=resampling_mode,
        require_recist_target=require_recist_target,
        normalization_stats=normalization_stats,
        target_spacing_xyz=target_spacing_xyz,
        case_cache_dir=case_cache_dir,
        pants_aux_class_map=pants_aux_class_map,
    )
    if cached_case is not None:
        return validate_and_return(
            cached_case,
            validate_integrity=validate_integrity,
            require_mask=require_mask,
            require_recist_target=require_recist_target,
        )

    raw = load_raw_case_arrays(
        record,
        load_mask=bool(load_mask or require_mask or require_recist_target),
        pants_aux_class_map=pants_aux_class_map,
    )
    resampled = maybe_resample_case_arrays(
        raw,
        record,
        target_spacing_xyz=target_spacing_xyz,
        resampling_mode=resampling_mode,
    )
    resampled = ensure_aux_placeholder(resampled, pants_aux_class_map)
    recist_target_mask, recist_target_selection = select_case_target_mask(record, resampled)
    case = build_loaded_case(
        record,
        resampled,
        recist_target_mask=recist_target_mask,
        recist_target_selection=recist_target_selection,
        normalization_stats=normalization_stats,
    )
    return validate_and_return(
        case,
        validate_integrity=validate_integrity,
        require_mask=require_mask,
        require_recist_target=require_recist_target,
    )
