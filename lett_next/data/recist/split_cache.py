from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ...pants_aux import PantsAuxClassMap, class_map_metadata
from .constants import (
    LEGACY_AUX_CASE_CACHE_VERSION,
    LEGACY_RECIST_RESAMPLING_MODE,
    LEGACY_SPLIT_SOURCE_CACHE_VERSION,
    LEGACY_SPLIT_TARGET_CACHE_VERSION,
)
from .io import _pants_label_case_dir, _pants_segmentation_path
from .recist_geometry import get_recist_target_mask
from .records import CaseRecord, LoadedCase


_SPLIT_SOURCE_CACHE_MAX_ITEMS = max(0, int(os.environ.get("LETT_NEXT_SPLIT_SOURCE_CACHE_MAX_ITEMS", "1")))
_SPLIT_SOURCE_CACHE: OrderedDict[str, dict[str, object]] = OrderedDict()
_SPLIT_TARGET_CASE_INDEX: dict[str, dict[str, tuple[Path, ...]]] = {}


def _file_fingerprint(path: str | None) -> dict[str, object] | None:
    if path in (None, ""):
        return None
    file_path = Path(str(path))
    try:
        stat = file_path.stat()
    except FileNotFoundError:
        return {"path": str(file_path), "missing": True}
    return {
        "path": str(file_path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _safe_cache_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def _pants_aux_segmentation_fingerprints(label_path: str | None, class_map: PantsAuxClassMap | None) -> list[dict[str, object]]:
    if label_path in (None, "") or class_map is None:
        return []
    label_case_dir = _pants_label_case_dir(Path(str(label_path)))
    rows: list[dict[str, object]] = []
    for class_id, segmentation in sorted(class_map.segmentations_by_class_id.items()):
        path = _pants_segmentation_path(label_case_dir, segmentation)
        rows.append(
            {
                "class_id": int(class_id),
                "segmentation": segmentation,
                "file": _file_fingerprint(None if path is None else str(path)),
            }
        )
    return rows


def split_case_source_cache_key(
    record: CaseRecord,
    *,
    target_spacing_xyz: tuple[float, float, float],
    resampling_mode: str = LEGACY_RECIST_RESAMPLING_MODE,
    pants_aux_class_map: PantsAuxClassMap | None = None,
) -> str:
    payload = {
        "version": LEGACY_SPLIT_SOURCE_CACHE_VERSION,
        "source": record.source,
        "image": _file_fingerprint(record.image_path),
        "label": _file_fingerprint(record.label_path),
        "spacing_xyz": [float(v) for v in record.spacing_xyz],
        "target_spacing_xyz": [float(v) for v in target_spacing_xyz],
        "resampling_mode": resampling_mode,
    }
    if pants_aux_class_map is not None:
        payload["aux"] = {
            "version": LEGACY_AUX_CASE_CACHE_VERSION,
            **class_map_metadata(pants_aux_class_map),
            "segmentation_files": _pants_aux_segmentation_fingerprints(record.label_path, pants_aux_class_map)
            if record.source == "pants"
            else [],
        }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def split_case_target_cache_key(
    record: CaseRecord,
    *,
    source_cache_key: str,
    target_spacing_xyz: tuple[float, float, float],
    resampling_mode: str = LEGACY_RECIST_RESAMPLING_MODE,
    pants_aux_class_map: PantsAuxClassMap | None = None,
) -> str:
    payload = {
        "version": LEGACY_SPLIT_TARGET_CACHE_VERSION,
        "source_cache_key": source_cache_key,
        "case_id": record.case_id,
        "source": record.source,
        "prompt_source": record.prompt_source,
        "target_spacing_xyz": [float(v) for v in target_spacing_xyz],
        "resampling_mode": resampling_mode,
        "recist_endpoints_xyz": [[float(v) for v in row] for row in record.recist_endpoints_xyz],
        "recist_slice_index": int(record.recist_slice_index),
        "target_label_value": record.target_label_value,
        "target_component_id": record.target_component_id,
    }
    if pants_aux_class_map is not None:
        payload["aux"] = {
            "version": LEGACY_AUX_CASE_CACHE_VERSION,
            **class_map_metadata(pants_aux_class_map),
        }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def split_case_source_cache_path(
    record: CaseRecord,
    *,
    cache_dir: Path,
    target_spacing_xyz: tuple[float, float, float],
    resampling_mode: str = LEGACY_RECIST_RESAMPLING_MODE,
    pants_aux_class_map: PantsAuxClassMap | None = None,
) -> Path:
    source_key = split_case_source_cache_key(
        record,
        target_spacing_xyz=target_spacing_xyz,
        resampling_mode=resampling_mode,
        pants_aux_class_map=pants_aux_class_map,
    )
    return cache_dir / "sources" / source_key[:2] / f"{source_key}.npz"


def split_case_target_cache_path(
    record: CaseRecord,
    *,
    cache_dir: Path,
    target_spacing_xyz: tuple[float, float, float],
    resampling_mode: str = LEGACY_RECIST_RESAMPLING_MODE,
    pants_aux_class_map: PantsAuxClassMap | None = None,
) -> Path:
    source_key = split_case_source_cache_key(
        record,
        target_spacing_xyz=target_spacing_xyz,
        resampling_mode=resampling_mode,
        pants_aux_class_map=pants_aux_class_map,
    )
    target_key = split_case_target_cache_key(
        record,
        source_cache_key=source_key,
        target_spacing_xyz=target_spacing_xyz,
        resampling_mode=resampling_mode,
        pants_aux_class_map=pants_aux_class_map,
    )
    return cache_dir / "targets" / target_key[:2] / f"{_safe_cache_name(record.case_id)}_{target_key[:16]}.npz"


def validate_anatomy_aux_cache_metadata(
    *,
    metadata: dict[str, object],
    class_map: PantsAuxClassMap,
    target_spacing_xyz: tuple[float, float, float],
) -> None:
    expected = class_map_metadata(class_map)
    checks = {
        "class_map_version": str(expected["class_map_version"]),
        "ignore_index": int(expected["ignore_index"]),
        "num_aux_classes": int(expected["num_aux_classes"]),
        "spacing_xyz": [float(value) for value in target_spacing_xyz],
    }
    for key, expected_value in checks.items():
        actual_value = metadata.get(key)
        if key in {"ignore_index", "num_aux_classes"}:
            actual_value = None if actual_value is None else int(actual_value)  # type: ignore[arg-type]
        elif key == "spacing_xyz" and actual_value is not None:
            actual_value = [float(value) for value in actual_value]  # type: ignore[union-attr]
        if actual_value != expected_value:
            raise ValueError(
                f"anatomy auxiliary cache metadata mismatch for {key}: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )


def _split_source_cache_is_usable(
    path: Path,
    *,
    expected_source_key: str,
    pants_aux_class_map: PantsAuxClassMap | None = None,
    target_spacing_xyz: tuple[float, float, float] | None = None,
) -> bool:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return False
        with np.load(path, allow_pickle=False) as data:
            required_keys = {"image", "spacing_xyz", "source_cache_key", "cache_version"}
            if not required_keys.issubset(set(data.files)):
                return False
            if str(np.asarray(data["source_cache_key"]).item()) != expected_source_key:
                return False
            if str(np.asarray(data["cache_version"]).item()) != LEGACY_SPLIT_SOURCE_CACHE_VERSION:
                return False
            if pants_aux_class_map is not None:
                aux_keys = {"task2_aux_target", "task2_aux_valid_mask", "aux_metadata"}
                if not aux_keys.issubset(set(data.files)):
                    return False
                if target_spacing_xyz is not None:
                    aux_metadata = json.loads(str(np.asarray(data["aux_metadata"]).item()))
                    if not isinstance(aux_metadata, dict):
                        return False
                    validate_anatomy_aux_cache_metadata(
                        metadata=aux_metadata,
                        class_map=pants_aux_class_map,
                        target_spacing_xyz=target_spacing_xyz,
                    )
        return True
    except (EOFError, OSError, KeyError, ValueError, json.JSONDecodeError, zipfile.BadZipFile):
        return False


def _split_target_cache_is_usable(
    path: Path,
    *,
    expected_source_key: str,
) -> bool:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return False
        with np.load(path, allow_pickle=False) as data:
            required_keys = {
                "task2_target_mask",
                "recist_endpoints_xyz",
                "recist_slice_index",
                "selected_label_value",
                "selected_voxel_count",
                "source_cache_key",
                "cache_version",
            }
            if not required_keys.issubset(set(data.files)):
                return False
            if str(np.asarray(data["source_cache_key"]).item()) != expected_source_key:
                return False
            if str(np.asarray(data["cache_version"]).item()) != LEGACY_SPLIT_TARGET_CACHE_VERSION:
                return False
        return True
    except (EOFError, OSError, KeyError, ValueError, zipfile.BadZipFile):
        return False


def _record_has_missing_raw_file(record: CaseRecord) -> bool:
    paths = [record.image_path]
    if record.label_path not in (None, ""):
        paths.append(str(record.label_path))
    return any(not Path(str(path)).exists() for path in paths)


def _target_cache_filename_matches_case_id(path: Path, case_id: str) -> bool:
    prefix = f"{_safe_cache_name(case_id)}_"
    if not path.name.endswith(".npz") or not path.name.startswith(prefix):
        return False
    suffix = path.name[len(prefix) : -4]
    return len(suffix) == 16 and all(char in "0123456789abcdef" for char in suffix)


def _split_target_case_index(target_root: Path) -> dict[str, tuple[Path, ...]]:
    index_key = str(target_root)
    cached = _SPLIT_TARGET_CASE_INDEX.get(index_key)
    if cached is not None:
        return cached
    grouped: dict[str, list[Path]] = {}
    if target_root.exists():
        for path in target_root.rglob("*.npz"):
            stem = path.stem
            separator = stem.rfind("_")
            if separator <= 0:
                continue
            suffix = stem[separator + 1 :]
            if len(suffix) != 16 or not all(char in "0123456789abcdef" for char in suffix):
                continue
            grouped.setdefault(stem[:separator], []).append(path)
    index = {case_id: tuple(sorted(paths)) for case_id, paths in grouped.items()}
    _SPLIT_TARGET_CASE_INDEX[index_key] = index
    return index


def _split_target_cache_record_source_key(path: Path, record: CaseRecord) -> str | None:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return None
        with np.load(path, allow_pickle=False) as data:
            required_keys = {
                "task2_target_mask",
                "recist_endpoints_xyz",
                "recist_slice_index",
                "selected_label_value",
                "selected_voxel_count",
                "source_cache_key",
                "cache_version",
            }
            if not required_keys.issubset(set(data.files)):
                return None
            if str(np.asarray(data["cache_version"]).item()) != LEGACY_SPLIT_TARGET_CACHE_VERSION:
                return None
            if int(np.asarray(data["task2_target_mask"], dtype=np.uint8).sum()) <= 0:
                return None
            if record.target_label_value is not None:
                if int(np.asarray(data["selected_label_value"]).item()) != int(record.target_label_value):
                    return None
            if record.target_component_id is not None and "selected_component_id" in data.files:
                if int(np.asarray(data["selected_component_id"]).item()) != int(record.target_component_id):
                    return None
            return str(np.asarray(data["source_cache_key"]).item())
    except (EOFError, OSError, KeyError, ValueError, zipfile.BadZipFile):
        return None


def _find_split_cache_paths_by_case_id(
    record: CaseRecord,
    *,
    cache_dir: Path,
) -> tuple[Path, Path, str] | None:
    target_root = cache_dir / "targets"
    if not target_root.exists():
        return None
    for target_path in _split_target_case_index(target_root).get(_safe_cache_name(record.case_id), ()):
        if not _target_cache_filename_matches_case_id(target_path, record.case_id):
            continue
        source_key = _split_target_cache_record_source_key(target_path, record)
        if source_key is None:
            continue
        source_path = cache_dir / "sources" / source_key[:2] / f"{source_key}.npz"
        if source_path.exists() and source_path.stat().st_size > 0:
            return source_path, target_path, source_key
    return None


def _write_npz_compressed_atomic(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp.npz")
    try:
        np.savez_compressed(tmp_path, **arrays)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _load_split_source_payload(
    path: Path,
    *,
    source_cache_key: str,
    pants_aux_class_map: PantsAuxClassMap | None,
    target_spacing_xyz: tuple[float, float, float],
) -> dict[str, object] | None:
    cached = _SPLIT_SOURCE_CACHE.get(str(path))
    if cached is not None and cached.get("source_cache_key") == source_cache_key:
        _SPLIT_SOURCE_CACHE.move_to_end(str(path))
        return cached
    if not _split_source_cache_is_usable(
        path,
        expected_source_key=source_cache_key,
        pants_aux_class_map=pants_aux_class_map,
        target_spacing_xyz=target_spacing_xyz,
    ):
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            payload: dict[str, object] = {
                "source_cache_key": source_cache_key,
                "image": np.asarray(data["image"], dtype=np.float32),
                "spacing_xyz": np.asarray(data["spacing_xyz"], dtype=np.float32),
                "aux_target": None,
                "aux_valid_mask": None,
                "aux_metadata": None,
            }
            if pants_aux_class_map is not None:
                aux_metadata = json.loads(str(np.asarray(data["aux_metadata"]).item()))
                if not isinstance(aux_metadata, dict):
                    raise ValueError(f"RECIST target split aux cache metadata must be a mapping: {path}")
                validate_anatomy_aux_cache_metadata(
                    metadata=aux_metadata,
                    class_map=pants_aux_class_map,
                    target_spacing_xyz=target_spacing_xyz,
                )
                payload["aux_target"] = np.asarray(data["task2_aux_target"], dtype=np.int16)
                payload["aux_valid_mask"] = np.asarray(data["task2_aux_valid_mask"], dtype=np.uint8) > 0
                payload["aux_metadata"] = aux_metadata
    except (EOFError, OSError, zipfile.BadZipFile):
        return None
    if _SPLIT_SOURCE_CACHE_MAX_ITEMS > 0:
        _SPLIT_SOURCE_CACHE[str(path)] = payload
        _SPLIT_SOURCE_CACHE.move_to_end(str(path))
        while len(_SPLIT_SOURCE_CACHE) > _SPLIT_SOURCE_CACHE_MAX_ITEMS:
            _SPLIT_SOURCE_CACHE.popitem(last=False)
    return payload


def load_split_case_cache(
    record: CaseRecord,
    *,
    cache_dir: Path,
    target_spacing_xyz: tuple[float, float, float],
    resampling_mode: str = LEGACY_RECIST_RESAMPLING_MODE,
    normalization_stats: dict[str, object] | None = None,
    pants_aux_class_map: PantsAuxClassMap | None = None,
) -> LoadedCase | None:
    source_key = split_case_source_cache_key(
        record,
        target_spacing_xyz=target_spacing_xyz,
        resampling_mode=resampling_mode,
        pants_aux_class_map=pants_aux_class_map,
    )
    source_path = split_case_source_cache_path(
        record,
        cache_dir=cache_dir,
        target_spacing_xyz=target_spacing_xyz,
        resampling_mode=resampling_mode,
        pants_aux_class_map=pants_aux_class_map,
    )
    target_path = split_case_target_cache_path(
        record,
        cache_dir=cache_dir,
        target_spacing_xyz=target_spacing_xyz,
        resampling_mode=resampling_mode,
        pants_aux_class_map=pants_aux_class_map,
    )
    fallback_used = False
    if not _split_target_cache_is_usable(target_path, expected_source_key=source_key):
        fallback_paths = _find_split_cache_paths_by_case_id(record, cache_dir=cache_dir)
        if fallback_paths is None:
            return None
        source_path, target_path, source_key = fallback_paths
        fallback_used = True
    source_payload = _load_split_source_payload(
        source_path,
        source_cache_key=source_key,
        pants_aux_class_map=pants_aux_class_map,
        target_spacing_xyz=target_spacing_xyz,
    )
    if source_payload is None:
        return None
    try:
        with np.load(target_path, allow_pickle=False) as data:
            target_mask = np.asarray(data["task2_target_mask"], dtype=np.uint8)
            endpoints_xyz = np.asarray(data["recist_endpoints_xyz"], dtype=np.float32)
            recist_slice_index = int(np.asarray(data["recist_slice_index"]).item())
            selected_label_value = int(np.asarray(data["selected_label_value"]).item())
            if "selected_component_id" in data:
                selected_component_id = int(np.asarray(data["selected_component_id"]).item())
            else:
                selected_component_id = record.target_component_id
            selected_voxel_count = int(np.asarray(data["selected_voxel_count"]).item())
    except (EOFError, OSError, zipfile.BadZipFile):
        return None
    if int(target_mask.sum()) <= 0:
        return None
    return LoadedCase(
        case_id=record.case_id,
        image=np.asarray(source_payload["image"], dtype=np.float32),
        raw_mask=target_mask,
        mask=target_mask,
        spacing_xyz=np.asarray(source_payload["spacing_xyz"], dtype=np.float32),
        recist_endpoints_xyz=endpoints_xyz,
        recist_slice_index=recist_slice_index,
        prompt_source=record.prompt_source,
        recist_target_mask=target_mask,
        recist_target_selection={
            "selection_mode": "resampled_split_case_cache",
            "used_fallback": fallback_used,
            "selected_label_value": selected_label_value,
            "selected_component_id": selected_component_id,
            "selected_voxel_count": selected_voxel_count,
            "cache_path": str(target_path),
            "source_cache_path": str(source_path),
        },
        anatomy_aux_target=source_payload["aux_target"],  # type: ignore[arg-type]
        anatomy_aux_valid_mask=source_payload["aux_valid_mask"],  # type: ignore[arg-type]
        anatomy_aux_metadata=source_payload["aux_metadata"],  # type: ignore[arg-type]
        normalization_stats=normalization_stats,
    )


def write_split_case_cache_from_loaded_case(
    record: CaseRecord,
    case: LoadedCase,
    *,
    cache_dir: Path,
    target_spacing_xyz: tuple[float, float, float],
    resampling_mode: str = LEGACY_RECIST_RESAMPLING_MODE,
    overwrite: bool = False,
    pants_aux_class_map: PantsAuxClassMap | None = None,
) -> dict[str, object]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_key = split_case_source_cache_key(
        record,
        target_spacing_xyz=target_spacing_xyz,
        resampling_mode=resampling_mode,
        pants_aux_class_map=pants_aux_class_map,
    )
    source_path = split_case_source_cache_path(
        record,
        cache_dir=cache_dir,
        target_spacing_xyz=target_spacing_xyz,
        resampling_mode=resampling_mode,
        pants_aux_class_map=pants_aux_class_map,
    )
    target_path = split_case_target_cache_path(
        record,
        cache_dir=cache_dir,
        target_spacing_xyz=target_spacing_xyz,
        resampling_mode=resampling_mode,
        pants_aux_class_map=pants_aux_class_map,
    )
    target_mask = get_recist_target_mask(case)
    if target_mask is None or int(np.asarray(target_mask).sum()) <= 0:
        return {"case_id": record.case_id, "path": str(target_path), "status": "empty_target"}
    source_written = False
    source_exists = _split_source_cache_is_usable(
        source_path,
        expected_source_key=source_key,
        pants_aux_class_map=pants_aux_class_map,
        target_spacing_xyz=target_spacing_xyz,
    )
    if overwrite or not source_exists:
        source_payload: dict[str, object] = {
            "cache_version": np.asarray(LEGACY_SPLIT_SOURCE_CACHE_VERSION, dtype=np.str_),
            "source_cache_key": np.asarray(source_key, dtype=np.str_),
            "image": np.asarray(case.image, dtype=np.float32),
            "spacing_xyz": np.asarray(case.spacing_xyz, dtype=np.float32),
        }
        if pants_aux_class_map is not None:
            if case.anatomy_aux_target is None or case.anatomy_aux_valid_mask is None or case.anatomy_aux_metadata is None:
                raise ValueError(f"RECIST target split aux cache requested but aux target was not built for {record.case_id}")
            aux_metadata = dict(case.anatomy_aux_metadata)
            aux_metadata |= {
                "spacing_xyz": [float(value) for value in target_spacing_xyz],
            }
            validate_anatomy_aux_cache_metadata(
                metadata=aux_metadata,
                class_map=pants_aux_class_map,
                target_spacing_xyz=target_spacing_xyz,
            )
            source_payload |= {
                "task2_aux_target": np.asarray(case.anatomy_aux_target, dtype=np.int16),
                "task2_aux_valid_mask": np.asarray(case.anatomy_aux_valid_mask, dtype=np.uint8),
                "aux_metadata": np.asarray(json.dumps(aux_metadata, sort_keys=True), dtype=np.str_),
            }
        _write_npz_compressed_atomic(source_path, **source_payload)
        if not _split_source_cache_is_usable(
            source_path,
            expected_source_key=source_key,
            pants_aux_class_map=pants_aux_class_map,
            target_spacing_xyz=target_spacing_xyz,
        ):
            raise ValueError(f"RECIST target split source cache write produced an unreadable file: {source_path}")
        source_written = True
    target_exists = _split_target_cache_is_usable(target_path, expected_source_key=source_key)
    if target_exists and not overwrite:
        return {
            "case_id": record.case_id,
            "path": str(target_path),
            "source_path": str(source_path),
            "status": "exists",
            "cache_format": "split",
        }
    selection = case.recist_target_selection or {}
    _write_npz_compressed_atomic(
        target_path,
        cache_version=np.asarray(LEGACY_SPLIT_TARGET_CACHE_VERSION, dtype=np.str_),
        source_cache_key=np.asarray(source_key, dtype=np.str_),
        task2_target_mask=(np.asarray(target_mask) > 0).astype(np.uint8),
        recist_endpoints_xyz=np.asarray(case.recist_endpoints_xyz, dtype=np.float32),
        recist_slice_index=np.asarray(case.recist_slice_index, dtype=np.int32),
        selected_label_value=np.asarray(
            int(selection.get("selected_label_value", record.target_label_value or 1)),
            dtype=np.int32,
        ),
        selected_component_id=np.asarray(
            int(selection.get("selected_component_id", record.target_component_id or 0)),
            dtype=np.int32,
        ),
        selected_voxel_count=np.asarray(int(np.asarray(target_mask).sum()), dtype=np.int64),
    )
    if not _split_target_cache_is_usable(target_path, expected_source_key=source_key):
        raise ValueError(f"RECIST target split target cache write produced an unreadable file: {target_path}")
    bytes_written = int(target_path.stat().st_size)
    if source_written:
        bytes_written += int(source_path.stat().st_size)
    return {
        "case_id": record.case_id,
        "path": str(target_path),
        "source_path": str(source_path),
        "status": "written",
        "cache_format": "split",
        "source_written": bool(source_written),
        "image_shape": [int(v) for v in case.image.shape],
        "target_voxels": int(np.asarray(target_mask).sum()),
        "bytes": bytes_written,
    }


def write_split_case_cache(
    record: CaseRecord,
    *,
    cache_dir: Path,
    target_spacing_xyz: tuple[float, float, float],
    resampling_mode: str = LEGACY_RECIST_RESAMPLING_MODE,
    overwrite: bool = False,
    pants_aux_class_map: PantsAuxClassMap | None = None,
) -> dict[str, object]:
    source_key = split_case_source_cache_key(
        record,
        target_spacing_xyz=target_spacing_xyz,
        resampling_mode=resampling_mode,
        pants_aux_class_map=pants_aux_class_map,
    )
    source_path = split_case_source_cache_path(
        record,
        cache_dir=cache_dir,
        target_spacing_xyz=target_spacing_xyz,
        resampling_mode=resampling_mode,
        pants_aux_class_map=pants_aux_class_map,
    )
    target_path = split_case_target_cache_path(
        record,
        cache_dir=cache_dir,
        target_spacing_xyz=target_spacing_xyz,
        resampling_mode=resampling_mode,
        pants_aux_class_map=pants_aux_class_map,
    )
    if (
        not overwrite
        and _split_source_cache_is_usable(
            source_path,
            expected_source_key=source_key,
            pants_aux_class_map=pants_aux_class_map,
            target_spacing_xyz=target_spacing_xyz,
        )
        and _split_target_cache_is_usable(target_path, expected_source_key=source_key)
    ):
        return {
            "case_id": record.case_id,
            "path": str(target_path),
            "source_path": str(source_path),
            "status": "exists",
            "cache_format": "split",
        }
    from .case_loader import load_case

    case = load_case(
        record,
        target_spacing_xyz=target_spacing_xyz,
        resampling_mode=resampling_mode,
        validate_integrity=True,
        require_mask=False,
        require_recist_target=True,
        load_mask=True,
        case_cache_dir=None,
        pants_aux_class_map=pants_aux_class_map,
    )
    return write_split_case_cache_from_loaded_case(
        record,
        case,
        cache_dir=cache_dir,
        target_spacing_xyz=target_spacing_xyz,
        resampling_mode=resampling_mode,
        overwrite=overwrite,
        pants_aux_class_map=pants_aux_class_map,
    )
