from __future__ import annotations

from .constants import (
    IMAGE_KEYS,
    MASK_KEYS,
    SPACING_KEYS,
    LEGACY_AUX_CASE_CACHE_VERSION,
    LEGACY_RECIST_RESAMPLING_MODE,
    LEGACY_SPLIT_SOURCE_CACHE_VERSION,
    LEGACY_SPLIT_TARGET_CACHE_VERSION,
)
from .records import CaseRecord, IntegrityReport, LoadedCase, NormalizationStats
from .io import _find_first_present_key
from .manifest import load_manifest
from .case_loader import load_case
from .dataset import RecistTrainingConfig, RecistTrainingDataset
from .normalization import compute_ct_normalization_stats, normalize_image
from .reports import build_recist_target_report, load_recist_target_summary, recist_target_voxel_count_for_record
from .recist_geometry import get_recist_target_mask, select_recist_target_instance
from .resampling import _resample_loaded_case_recist
from .crops import compute_crop_slices, extract_prompt_crop, restore_crop_to_full_shape
from .input_channels import build_input_channels
from .split_cache import (
    load_split_case_cache,
    split_case_source_cache_key,
    split_case_source_cache_path,
    split_case_target_cache_path,
    write_split_case_cache,
)
from .validation import validate_records_integrity

__all__ = [
    "CaseRecord",
    "IMAGE_KEYS",
    "IntegrityReport",
    "LoadedCase",
    "MASK_KEYS",
    "NormalizationStats",
    "RecistTrainingConfig",
    "RecistTrainingDataset",
    "SPACING_KEYS",
    "LEGACY_AUX_CASE_CACHE_VERSION",
    "LEGACY_RECIST_RESAMPLING_MODE",
    "LEGACY_SPLIT_SOURCE_CACHE_VERSION",
    "LEGACY_SPLIT_TARGET_CACHE_VERSION",
    "_find_first_present_key",
    "_resample_loaded_case_recist",
    "build_recist_target_report",
    "build_input_channels",
    "compute_crop_slices",
    "compute_ct_normalization_stats",
    "extract_prompt_crop",
    "get_recist_target_mask",
    "load_case",
    "load_manifest",
    "load_recist_target_summary",
    "load_split_case_cache",
    "normalize_image",
    "recist_target_voxel_count_for_record",
    "restore_crop_to_full_shape",
    "select_recist_target_instance",
    "split_case_source_cache_key",
    "split_case_source_cache_path",
    "split_case_target_cache_path",
    "validate_records_integrity",
    "write_split_case_cache",
]
