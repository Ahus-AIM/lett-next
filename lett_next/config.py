from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from . import recipe


def _strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(value):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return value[:index].rstrip()
    return value.rstrip()


def _parse_simple_yaml(config_path: Path) -> dict[str, object]:
    data: dict[str, object] = {}
    for line_number, raw_line in enumerate(config_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = _strip_inline_comment(raw_line)
        if not line.strip():
            continue
        if line[:1].isspace():
            raise ValueError(f"Nested YAML is not supported in {config_path}:{line_number}")
        if ":" not in line:
            raise ValueError(f"Expected 'key: value' in {config_path}:{line_number}")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Expected a non-empty key in {config_path}:{line_number}")
        data[key] = _parse_scalar_value(raw_value.strip())
    return data


def _parse_scalar_value(value: str) -> object:
    if value == "":
        return None
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if value[:1] in {"'", '"'} or value[:1] in {"[", "{", "("}:
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _lookup(mapping: dict[str, object], key: str, default: object = None) -> object | None:
    return mapping.get(key, default)


def _parse_optional_int(value: object | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _parse_float_tuple_3(value: object | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if value in (None, ""):
        return default
    parts = [float(part) for part in value] if not isinstance(value, str) else [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 floats, got {value!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _resolve_path(repo_root: Path, value: object | None, default: str | Path | None = None) -> Path | None:
    raw_value = value if value not in (None, "") else default
    if raw_value in (None, ""):
        return None
    path = Path(str(raw_value))
    return path if path.is_absolute() else repo_root / path


@dataclass(slots=True)
class ExperimentConfig:
    repo_root: Path
    config_path: Path | None
    manifest_path: Path
    runs_dir: Path
    epochs: int
    eval_every: int
    batch_size: int
    learning_rate: float
    num_workers: int
    max_train_cases: int | None
    max_val_cases: int | None
    seed: int
    threshold: float
    train_log_every_steps: int
    train_device: str
    eval_device: str
    train_run_suffix: str
    train_cuda_visible_devices: str
    train_nproc_per_node: int
    train_data_dir: Path | None
    val_data_dir: Path | None
    pants_data_dir: Path | None
    pants_enable: bool
    combine_manifest_path: Path | None
    case_cache_dir: Path | None
    target_audit_path: Path | None
    normalization_stats_path: Path | None
    integrity_mode: str
    integrity_report_path: Path | None
    resume_run_dir: Path | None
    resume_checkpoint_path: Path | None

    model_name: str = recipe.MODEL_NAME
    description: str = recipe.DESCRIPTION
    crop_size_zyx: tuple[int, int, int] = recipe.CROP_SIZE_ZYX
    target_spacing_xyz: tuple[float, float, float] = recipe.TARGET_SPACING_XYZ
    prompt_sigma: float = recipe.PROMPT_SIGMA
    deep_supervision: bool = recipe.DEEP_SUPERVISION
    primary_prompt_mode: str = recipe.PRIMARY_PROMPT_MODE
    normalization_mode: str = recipe.NORMALIZATION_MODE
    normalization_stats: dict[str, object] | None = None
    exclude_empty_train_masks: bool = recipe.EXCLUDE_EMPTY_TRAIN_MASKS
    train_crop_mode: str = recipe.TRAIN_CROP_MODE
    train_crop_jitter_zyx: tuple[int, int, int] = recipe.TRAIN_CROP_JITTER_ZYX
    train_bbox_fit_margin_zyx: tuple[int, int, int] = recipe.TRAIN_BBOX_FIT_MARGIN_ZYX
    mixed_prompt_base_fraction: float = recipe.MIXED_PROMPT_BASE_FRACTION
    mixed_prompt_multifov_fraction: float = recipe.MIXED_PROMPT_MULTIFOV_FRACTION
    mixed_recist_bbox_fit_tail_fraction: float = recipe.MIXED_RECIST_BBOX_FIT_TAIL_FRACTION
    mixed_recist_multifov_scales_zyx: tuple[tuple[float, float, float], ...] = recipe.MIXED_RECIST_MULTIFOV_SCALES_ZYX
    ensure_recist_endpoints_inside_crop: bool = recipe.ENSURE_RECIST_ENDPOINTS_INSIDE_CROP
    recist_endpoint_margin_mm: float = recipe.RECIST_ENDPOINT_MARGIN_MM
    train_intensity_shift: float = recipe.TRAIN_INTENSITY_SHIFT
    train_intensity_scale: float = recipe.TRAIN_INTENSITY_SCALE
    train_noise_std: float = recipe.TRAIN_NOISE_STD
    train_recist_aug_probability: float = 0.0
    train_recist_aug_shift_xy: int = 0
    train_recist_aug_endpoint_jitter_xy: int = 0
    train_recist_aug_slice_jitter: int = 0
    train_recist_aug_length_scale_min: float = 1.0
    train_recist_aug_length_scale_max: float = 1.0
    train_line_dropout_probability: float = 0.0
    hard_negative_fraction: float = 0.0
    anatomy_aux_enable: bool = True
    anatomy_aux_class_map_path: Path | None = None
    anatomy_aux_loss_weight: float = recipe.PANTS_AUX_LOSS_WEIGHT
    negative_loss_weight: float = recipe.NEGATIVE_LOSS_WEIGHT

    @classmethod
    def from_source(cls, repo_root: Path, config_path: Path) -> "ExperimentConfig":
        return cls.from_yaml(repo_root, config_path)

    @classmethod
    def from_yaml(cls, repo_root: Path, config_path: Path) -> "ExperimentConfig":
        return cls._from_mapping(repo_root, _parse_simple_yaml(config_path), config_path=config_path)

    @classmethod
    def _from_mapping(
        cls,
        repo_root: Path,
        values: dict[str, object],
        config_path: Path | None = None,
    ) -> "ExperimentConfig":
        train_device = str(_lookup(values, "train_device", "cuda" if torch.cuda.is_available() else "cpu"))
        eval_device = str(_lookup(values, "eval_device", train_device))
        return cls(
            repo_root=repo_root,
            config_path=config_path,
            manifest_path=_resolve_path(
                repo_root,
                _lookup(values, "manifest_path"),
                default="data/prepared/manifest.json",
            )
            or (repo_root / "data/prepared/manifest.json"),
            runs_dir=_resolve_path(repo_root, _lookup(values, "runs_dir"), default="runs") or (repo_root / "runs"),
            epochs=int(_lookup(values, "epochs", 25)),
            eval_every=int(_lookup(values, "eval_every", 3)),
            batch_size=int(_lookup(values, "batch_size", 6)),
            learning_rate=float(_lookup(values, "learning_rate", 1e-3)),
            num_workers=int(_lookup(values, "num_workers", 8)),
            max_train_cases=_parse_optional_int(_lookup(values, "max_train_cases")),
            max_val_cases=_parse_optional_int(_lookup(values, "max_val_cases")),
            seed=int(_lookup(values, "seed", 43)),
            threshold=float(_lookup(values, "threshold", 0.5)),
            train_log_every_steps=int(_lookup(values, "train_log_every_steps", 50)),
            train_device=train_device,
            eval_device=eval_device,
            train_run_suffix=str(
                _lookup(values, "train_run_suffix", "lett-next-mednext-f32-xy1p0-z2p4-z72xy160-25ep-b6-w8-3gpu")
            ),
            train_cuda_visible_devices=str(_lookup(values, "train_cuda_visible_devices", "0,1,2")),
            train_nproc_per_node=int(_lookup(values, "train_nproc_per_node", 3)),
            train_data_dir=_resolve_path(repo_root, _lookup(values, "train_data_dir")),
            val_data_dir=_resolve_path(repo_root, _lookup(values, "val_data_dir")),
            pants_data_dir=_resolve_path(repo_root, _lookup(values, "pants_data_dir")),
            pants_enable=bool(_lookup(values, "pants_enable", True)),
            combine_manifest_path=_resolve_path(repo_root, _lookup(values, "combine_manifest_path")),
            case_cache_dir=_resolve_path(repo_root, _lookup(values, "case_cache_dir")),
            target_audit_path=_resolve_path(repo_root, _lookup(values, "target_audit_path")),
            normalization_stats_path=_resolve_path(repo_root, _lookup(values, "normalization_stats_path")),
            integrity_mode=str(_lookup(values, "integrity_mode", "skip")),
            integrity_report_path=_resolve_path(repo_root, _lookup(values, "integrity_report_path")),
            resume_run_dir=_resolve_path(repo_root, _lookup(values, "resume_run_dir")),
            resume_checkpoint_path=_resolve_path(repo_root, _lookup(values, "resume_checkpoint_path")),
            target_spacing_xyz=_parse_float_tuple_3(_lookup(values, "target_spacing_xyz"), recipe.TARGET_SPACING_XYZ),
            anatomy_aux_class_map_path=_resolve_path(
                repo_root,
                _lookup(values, "anatomy_aux_class_map_path"),
                default=recipe.PANTS_AUX_CLASS_MAP_RELATIVE_PATH,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in data.items()}

    @property
    def is_resume(self) -> bool:
        return self.resume_run_dir is not None or self.resume_checkpoint_path is not None
