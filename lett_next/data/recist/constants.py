from __future__ import annotations

IMAGE_KEYS = ("image", "ct", "volume", "img", "imgs", "arr_0")
MASK_KEYS = ("mask", "label", "seg", "lesion_mask", "labels", "gts")
SPACING_KEYS = ("spacing_xyz", "spacing", "voxel_spacing")
ENDPOINT_KEYS = (
    "recist_endpoints_xyz",
    "recist_endpoints",
    "diameter_endpoints_xyz",
    "endpoints_xyz",
)
SLICE_INDEX_KEYS = ("recist_slice_index", "slice_index", "diameter_slice_index")
PANTS_LESION_LABEL = 28
PANTS_PROMPT_SOURCE = "derived_from_pants_pancreatic_lesion"

# Existing split caches were written with these names; changing them would force a full cache rebuild.
LEGACY_RECIST_RESAMPLING_MODE = "task2_anisotropic"
LEGACY_AUX_CASE_CACHE_VERSION = "task2_resampled_case_aux_v1"
LEGACY_SPLIT_SOURCE_CACHE_VERSION = "task2_resampled_split_source_v1"
LEGACY_SPLIT_TARGET_CACHE_VERSION = "task2_resampled_split_target_v1"
