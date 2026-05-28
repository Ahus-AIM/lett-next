from __future__ import annotations

from pathlib import Path


MODEL_NAME = "mednextv2_f32"
DESCRIPTION = "LETT-NeXt MedNeXt-v2 f32 submitted LETT-NeXt reproducibility recipe"

DEEP_SUPERVISION = True
CROP_SIZE_ZYX = (72, 160, 160)
TARGET_SPACING_XYZ = (1.0, 1.0, 2.4)
PROMPT_SIGMA = 2.0
PRIMARY_PROMPT_MODE = "full"

NORMALIZATION_MODE = "dataset_stats"
EXCLUDE_EMPTY_TRAIN_MASKS = True
TRAIN_CROP_MODE = "mixed_recist_prompt"
TRAIN_CROP_JITTER_ZYX = (4, 8, 8)
TRAIN_BBOX_FIT_MARGIN_ZYX = (2, 4, 4)
MIXED_PROMPT_BASE_FRACTION = 0.60
MIXED_PROMPT_MULTIFOV_FRACTION = 0.30
MIXED_RECIST_BBOX_FIT_TAIL_FRACTION = 0.10
MIXED_RECIST_MULTIFOV_SCALES_ZYX = (
    (1.0, 1.0, 1.0),
    (1.15, 1.4, 1.4),
    (1.35, 1.8, 1.8),
    (1.35, 2.4, 2.4),
)
ENSURE_RECIST_ENDPOINTS_INSIDE_CROP = True
RECIST_ENDPOINT_MARGIN_MM = 15.0
TRAIN_INTENSITY_SHIFT = 0.03
TRAIN_INTENSITY_SCALE = 0.05
TRAIN_NOISE_STD = 0.01

DICE_WEIGHT = 1.0
BCE_WEIGHT = 2.0
PANTS_AUX_LOSS_WEIGHT = 0.25
NEGATIVE_LOSS_WEIGHT = 0.25
PANTS_AUX_CLASS_MAP_RELATIVE_PATH = Path("configs/pants/class_map_v1.json")

POSTPROCESS_KWARGS = {
    "postprocess_mode": "prompt_component",
    "postprocess_tube_radius_xy": 6,
    "postprocess_tube_z_thickness": 1,
    "postprocess_box_margin_zyx": (1, 8, 8),
    "postprocess_tiny_island_min_voxels": 0,
    "postprocess_tiny_island_min_fraction": 0.0,
}

VALIDATION_AUTOZOOM_KWARGS = {
    "autozoom_enable": True,
    "autozoom_max_passes": 3,
    "autozoom_growth_zyx": (1.5, 1.25, 1.25),
    "autozoom_max_zoom_zyx": (4.0, 2.0, 2.0),
    "autozoom_base_abs_changed": 1500,
    "autozoom_base_min_changed": 100,
    "autozoom_base_rel_changed": 0.2,
    "autozoom_ref_patch_zyx": (128, 160, 160),
    "autozoom_refine_enable": True,
    "autozoom_refine_margin_zyx": (10, 10, 10),
    "autozoom_refine_max_boxes": 4,
}

DEFAULT_THRESHOLDS = (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75)

SUBMISSION_THRESHOLD = 0.35
SUBMISSION_AUTOZOOM_KWARGS = {
    "autozoom_enable": True,
    "autozoom_max_passes": 2,
    "autozoom_min_passes": 1,
    "autozoom_growth_zyx": (1.25, 1.15, 1.15),
    "autozoom_max_zoom_zyx": (4.0, 2.0, 2.0),
    "autozoom_scale_mode": "adaptive",
    "autozoom_refine_enable": False,
    "autozoom_refine_max_boxes": 4,
}
