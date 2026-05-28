from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from ...pants_aux import PantsAuxClassMap
from .case_loader import load_case
from .crops import extract_multifov_recist_prompt_crop, extract_prompt_crop, extract_recist_bbox_fit_crop, prompt_crop_needs_bbox_fit_tail
from .input_channels import build_input_channels, perturb_recist_endpoints_for_training
from .normalization import NormalizationStats, _coerce_normalization_stats
from .recist_geometry import get_recist_target_mask
from .records import CaseRecord, LoadedCase
from .constants import LEGACY_RECIST_RESAMPLING_MODE


@dataclass(slots=True)
class RecistTrainingConfig:
    crop_size_zyx: tuple[int, int, int]
    prompt_sigma: float
    prompt_mode: str = "full"
    crop_mode: str = "mixed_recist_prompt"
    crop_jitter_zyx: tuple[int, int, int] = (0, 0, 0)
    bbox_fit_margin_zyx: tuple[int, int, int] = (2, 4, 4)
    mixed_prompt_base_fraction: float = 0.60
    mixed_prompt_multifov_fraction: float = 0.30
    mixed_recist_bbox_fit_tail_fraction: float = 0.10
    mixed_recist_multifov_scales_zyx: tuple[tuple[float, float, float], ...] = (
        (1.0, 1.0, 1.0),
        (1.25, 1.25, 1.25),
        (1.5, 1.5, 1.5),
        (1.5, 2.0, 2.0),
    )
    ensure_recist_endpoints_inside_crop: bool = False
    recist_endpoint_margin_mm: float = 0.0
    recist_aug_probability: float = 0.0
    recist_aug_shift_xy: int = 0
    recist_aug_endpoint_jitter_xy: int = 0
    recist_aug_slice_jitter: int = 0
    recist_aug_length_scale_min: float = 1.0
    recist_aug_length_scale_max: float = 1.0
    line_dropout_probability: float = 0.0
    intensity_shift: float = 0.0
    intensity_scale: float = 0.0
    noise_std: float = 0.0
    target_spacing_xyz: tuple[float, float, float] | None = None
    normalization_stats: NormalizationStats | dict[str, object] | None = None
    normalization_mode: str = "legacy_crop_zscore"
    pants_aux_class_map: PantsAuxClassMap | None = None
    hard_negative_fraction: float = 0.0
    negative_loss_weight: float = 0.25


def _pad_aux_to_shape(
    aux_target: np.ndarray,
    aux_valid_mask: np.ndarray,
    target_shape_zyx: tuple[int, int, int],
    *,
    ignore_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    pad_width = []
    needs_padding = False
    for dim, target_dim in zip(aux_target.shape, target_shape_zyx, strict=True):
        pad_after = max(int(target_dim) - int(dim), 0)
        pad_width.append((0, pad_after))
        needs_padding = needs_padding or pad_after > 0
    if needs_padding:
        aux_target = np.pad(aux_target, pad_width, mode="constant", constant_values=int(ignore_index))
        aux_valid_mask = np.pad(aux_valid_mask, pad_width, mode="constant", constant_values=False)
    slices = tuple(slice(0, int(dim)) for dim in target_shape_zyx)
    return np.asarray(aux_target[slices], dtype=np.int16), np.asarray(aux_valid_mask[slices], dtype=bool)


def _resize_aux_nearest_exact(
    aux_target: np.ndarray,
    aux_valid_mask: np.ndarray,
    target_shape_zyx: tuple[int, int, int],
    *,
    ignore_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    source_shape = np.asarray(aux_target.shape, dtype=np.float32)
    target_shape = np.asarray(target_shape_zyx, dtype=np.int64)
    zoom = target_shape.astype(np.float32) / np.maximum(source_shape, 1.0)
    if np.allclose(zoom, np.ones(3, dtype=np.float32), atol=1e-6):
        return _pad_aux_to_shape(aux_target, aux_valid_mask, target_shape_zyx, ignore_index=ignore_index)
    resized_target = ndimage.zoom(np.asarray(aux_target, dtype=np.int16), zoom=tuple(float(v) for v in zoom), order=0)
    resized_valid = ndimage.zoom(np.asarray(aux_valid_mask, dtype=np.uint8), zoom=tuple(float(v) for v in zoom), order=0) > 0
    return _pad_aux_to_shape(resized_target, resized_valid, target_shape_zyx, ignore_index=ignore_index)


def _aux_crop_for_training_crop(
    case: LoadedCase,
    crop: dict[str, object],
    crop_size_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    if case.anatomy_aux_target is None or case.anatomy_aux_valid_mask is None or case.anatomy_aux_metadata is None:
        return None
    ignore_index = int(case.anatomy_aux_metadata.get("ignore_index", -1))
    aux_target = np.asarray(case.anatomy_aux_target, dtype=np.int16)
    aux_valid = np.asarray(case.anatomy_aux_valid_mask, dtype=bool)
    crop_slices = crop["crop_slices"]  # type: ignore[assignment]
    if "source_fov_scale_zyx" in crop:
        source_target = aux_target[crop_slices]  # type: ignore[index]
        source_valid = aux_valid[crop_slices]  # type: ignore[index]
        source_size = tuple(int(v) for v in crop["source_crop_size_zyx"])  # type: ignore[union-attr]
        source_target, source_valid = _pad_aux_to_shape(
            source_target,
            source_valid,
            source_size,
            ignore_index=ignore_index,
        )
        return _resize_aux_nearest_exact(source_target, source_valid, crop_size_zyx, ignore_index=ignore_index)
    if bool(crop.get("bbox_fit_rescaled", False)):
        source_slices_rows = crop.get("bbox_fit_source_slices_zyx")
        if not isinstance(source_slices_rows, list) or len(source_slices_rows) != 2:
            raise ValueError("bbox-fit aux crop is missing source slices")
        source_slices = tuple(
            slice(int(start), int(stop))
            for start, stop in zip(source_slices_rows[0], source_slices_rows[1], strict=True)
        )
        source_target = aux_target[source_slices]
        source_valid = aux_valid[source_slices]
        scale_zyx = tuple(float(v) for v in crop["bbox_fit_scale_zyx"])  # type: ignore[union-attr]
        scaled_target = ndimage.zoom(source_target, zoom=scale_zyx, order=0).astype(np.int16)
        scaled_valid = ndimage.zoom(source_valid.astype(np.uint8), zoom=scale_zyx, order=0) > 0
        target_crop = scaled_target[crop_slices]  # type: ignore[index]
        valid_crop = scaled_valid[crop_slices]  # type: ignore[index]
        return _pad_aux_to_shape(target_crop, valid_crop, crop_size_zyx, ignore_index=ignore_index)
    target_crop = aux_target[crop_slices]  # type: ignore[index]
    valid_crop = aux_valid[crop_slices]  # type: ignore[index]
    return _pad_aux_to_shape(target_crop, valid_crop, crop_size_zyx, ignore_index=ignore_index)


class RecistTrainingDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        records: list[CaseRecord],
        crop_size_zyx: tuple[int, int, int] | RecistTrainingConfig,
        prompt_sigma: float | None = None,
        prompt_mode: str = "full",
        crop_mode: str = "prompt_centered",
        crop_jitter_zyx: tuple[int, int, int] = (0, 0, 0),
        bbox_fit_margin_zyx: tuple[int, int, int] = (2, 4, 4),
        mixed_prompt_base_fraction: float = 0.60,
        mixed_prompt_multifov_fraction: float = 0.30,
        mixed_recist_bbox_fit_tail_fraction: float = 0.10,
        mixed_recist_multifov_scales_zyx: tuple[tuple[float, float, float], ...] = (
            (1.0, 1.0, 1.0),
            (1.25, 1.25, 1.25),
            (1.5, 1.5, 1.5),
            (1.5, 2.0, 2.0),
        ),
        ensure_recist_endpoints_inside_crop: bool = False,
        recist_endpoint_margin_mm: float = 0.0,
        recist_aug_probability: float = 0.0,
        recist_aug_shift_xy: int = 0,
        recist_aug_endpoint_jitter_xy: int = 0,
        recist_aug_slice_jitter: int = 0,
        recist_aug_length_scale_min: float = 1.0,
        recist_aug_length_scale_max: float = 1.0,
        line_dropout_probability: float = 0.0,
        intensity_shift: float = 0.0,
        intensity_scale: float = 0.0,
        noise_std: float = 0.0,
        target_spacing_xyz: tuple[float, float, float] | None = None,
        case_cache_dir: Path | str | None = None,
        normalization_stats: NormalizationStats | dict[str, object] | None = None,
        normalization_mode: str = "legacy_crop_zscore",
        pants_aux_class_map: PantsAuxClassMap | None = None,
        hard_negative_fraction: float = 0.0,
        negative_loss_weight: float = 0.25,
    ) -> None:
        if isinstance(crop_size_zyx, RecistTrainingConfig):
            training_config = crop_size_zyx
            crop_size_zyx = training_config.crop_size_zyx
            prompt_sigma = training_config.prompt_sigma
            prompt_mode = training_config.prompt_mode
            crop_mode = training_config.crop_mode
            crop_jitter_zyx = training_config.crop_jitter_zyx
            bbox_fit_margin_zyx = training_config.bbox_fit_margin_zyx
            mixed_prompt_base_fraction = training_config.mixed_prompt_base_fraction
            mixed_prompt_multifov_fraction = training_config.mixed_prompt_multifov_fraction
            mixed_recist_bbox_fit_tail_fraction = training_config.mixed_recist_bbox_fit_tail_fraction
            mixed_recist_multifov_scales_zyx = training_config.mixed_recist_multifov_scales_zyx
            ensure_recist_endpoints_inside_crop = training_config.ensure_recist_endpoints_inside_crop
            recist_endpoint_margin_mm = training_config.recist_endpoint_margin_mm
            recist_aug_probability = training_config.recist_aug_probability
            recist_aug_shift_xy = training_config.recist_aug_shift_xy
            recist_aug_endpoint_jitter_xy = training_config.recist_aug_endpoint_jitter_xy
            recist_aug_slice_jitter = training_config.recist_aug_slice_jitter
            recist_aug_length_scale_min = training_config.recist_aug_length_scale_min
            recist_aug_length_scale_max = training_config.recist_aug_length_scale_max
            line_dropout_probability = training_config.line_dropout_probability
            intensity_shift = training_config.intensity_shift
            intensity_scale = training_config.intensity_scale
            noise_std = training_config.noise_std
            target_spacing_xyz = training_config.target_spacing_xyz
            normalization_stats = training_config.normalization_stats
            normalization_mode = training_config.normalization_mode
            pants_aux_class_map = training_config.pants_aux_class_map
            hard_negative_fraction = training_config.hard_negative_fraction
            negative_loss_weight = training_config.negative_loss_weight
        if prompt_sigma is None:
            raise ValueError("prompt_sigma must be provided when not using RecistTrainingConfig")
        self.records = records
        self.crop_size_zyx = crop_size_zyx
        self.prompt_sigma = prompt_sigma
        self.prompt_mode = prompt_mode
        self.crop_mode = crop_mode
        self.crop_jitter_zyx = crop_jitter_zyx
        self.bbox_fit_margin_zyx = bbox_fit_margin_zyx
        self.mixed_prompt_base_fraction = float(max(mixed_prompt_base_fraction, 0.0))
        self.mixed_prompt_multifov_fraction = float(max(mixed_prompt_multifov_fraction, 0.0))
        self.mixed_recist_bbox_fit_tail_fraction = float(max(mixed_recist_bbox_fit_tail_fraction, 0.0))
        self.mixed_recist_multifov_scales_zyx = tuple(
            (float(scale[0]), float(scale[1]), float(scale[2])) for scale in mixed_recist_multifov_scales_zyx
        )
        if not self.mixed_recist_multifov_scales_zyx:
            raise ValueError("mixed_recist_multifov_scales_zyx must contain at least one scale")
        self.ensure_recist_endpoints_inside_crop = bool(ensure_recist_endpoints_inside_crop)
        self.recist_endpoint_margin_mm = float(max(recist_endpoint_margin_mm, 0.0))
        self.recist_aug_probability = recist_aug_probability
        self.recist_aug_shift_xy = recist_aug_shift_xy
        self.recist_aug_endpoint_jitter_xy = recist_aug_endpoint_jitter_xy
        self.recist_aug_slice_jitter = recist_aug_slice_jitter
        self.recist_aug_length_scale_min = recist_aug_length_scale_min
        self.recist_aug_length_scale_max = recist_aug_length_scale_max
        self.line_dropout_probability = float(np.clip(line_dropout_probability, 0.0, 1.0))
        self.intensity_shift = intensity_shift
        self.intensity_scale = intensity_scale
        self.noise_std = noise_std
        self.target_spacing_xyz = target_spacing_xyz
        self.case_cache_dir = None if case_cache_dir is None else Path(case_cache_dir)
        self.normalization_stats = None if normalization_stats is None else _coerce_normalization_stats(normalization_stats).to_dict()  # type: ignore[union-attr]
        self.normalization_mode = normalization_mode
        self.pants_aux_class_map = pants_aux_class_map
        self.hard_negative_fraction = float(np.clip(hard_negative_fraction, 0.0, 1.0))
        self.negative_loss_weight = float(max(negative_loss_weight, 0.0))

    def __len__(self) -> int:
        return len(self.records)

    def _sample_mixed_recist_crop(self, case: LoadedCase, target_mask: np.ndarray) -> dict[str, object]:
        prompt_crop = extract_prompt_crop(
            image=case.image,
            mask=target_mask,
            endpoints_xyz=case.recist_endpoints_xyz,
            crop_size_zyx=self.crop_size_zyx,
            jitter_zyx=self.crop_jitter_zyx,
        )
        prompt_crop["crop_kind"] = "mixed_prompt_base"
        needs_tail = prompt_crop_needs_bbox_fit_tail(
            full_mask=target_mask,
            prompt_crop=prompt_crop,
            crop_size_zyx=self.crop_size_zyx,
            spacing_xyz=case.spacing_xyz,
            endpoint_margin_mm=self.recist_endpoint_margin_mm,
            require_endpoint_margin=self.ensure_recist_endpoints_inside_crop,
        )
        base_fraction = self.mixed_prompt_base_fraction
        multifov_fraction = self.mixed_prompt_multifov_fraction
        tail_fraction = self.mixed_recist_bbox_fit_tail_fraction
        total = max(base_fraction + multifov_fraction + tail_fraction, 1e-6)
        draw = float(np.random.random()) * total
        if needs_tail and draw >= base_fraction + multifov_fraction and draw < base_fraction + multifov_fraction + tail_fraction:
            crop = extract_recist_bbox_fit_crop(
                image=case.image,
                mask=target_mask,
                endpoints_xyz=case.recist_endpoints_xyz,
                crop_size_zyx=self.crop_size_zyx,
                margin_zyx=self.bbox_fit_margin_zyx,
                jitter_zyx=self.crop_jitter_zyx,
            )
            crop["crop_kind"] = "mixed_recist_bbox_fit_tail"
            crop["mixed_prompt_tail_eligible"] = True
            return crop
        if draw >= base_fraction and draw < base_fraction + multifov_fraction:
            scale_index = int(np.random.randint(0, len(self.mixed_recist_multifov_scales_zyx)))
            scale_zyx = self.mixed_recist_multifov_scales_zyx[scale_index]
            crop = extract_multifov_recist_prompt_crop(
                image=case.image,
                mask=target_mask,
                endpoints_xyz=case.recist_endpoints_xyz,
                crop_size_zyx=self.crop_size_zyx,
                source_fov_scale_zyx=scale_zyx,
                jitter_zyx=self.crop_jitter_zyx,
                ensure_endpoints_inside=self.ensure_recist_endpoints_inside_crop,
                endpoint_margin_mm=self.recist_endpoint_margin_mm,
                spacing_xyz=case.spacing_xyz,
            )
            crop["crop_kind"] = "mixed_prompt_multifov"
            crop["mixed_prompt_tail_eligible"] = bool(needs_tail)
            return crop
        prompt_crop["mixed_prompt_tail_eligible"] = bool(needs_tail)
        return prompt_crop

    def _sample_training_crop(self, case: LoadedCase, target_mask: np.ndarray) -> dict[str, object]:
        if self.crop_mode in {"mixed_recist", "mixed_recist_prompt"}:
            return self._sample_mixed_recist_crop(case, target_mask)
        if self.crop_mode == "prompt_centered":
            crop = extract_prompt_crop(
                image=case.image,
                mask=target_mask,
                endpoints_xyz=case.recist_endpoints_xyz,
                crop_size_zyx=self.crop_size_zyx,
                jitter_zyx=self.crop_jitter_zyx,
            )
            crop["crop_kind"] = "prompt_centered"
            return crop
        if self.crop_mode == "bbox_fit":
            crop = extract_recist_bbox_fit_crop(
                image=case.image,
                mask=target_mask,
                endpoints_xyz=case.recist_endpoints_xyz,
                crop_size_zyx=self.crop_size_zyx,
                margin_zyx=self.bbox_fit_margin_zyx,
                jitter_zyx=self.crop_jitter_zyx,
            )
            crop["crop_kind"] = "bbox_fit"
            return crop
        raise ValueError(f"Unsupported crop_mode: {self.crop_mode}")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        case = load_case(
            record,
            target_spacing_xyz=self.target_spacing_xyz,
            resampling_mode=LEGACY_RECIST_RESAMPLING_MODE,
            require_mask=False,
            require_recist_target=True,
            normalization_stats=self.normalization_stats,
            case_cache_dir=self.case_cache_dir,
            pants_aux_class_map=self.pants_aux_class_map,
        )
        target_mask = get_recist_target_mask(case)
        if target_mask is None:
            raise ValueError(f"Training case {record.case_id} is missing a mask")
        crop = self._sample_training_crop(case, target_mask)
        hard_negative = self.hard_negative_fraction > 0.0 and float(np.random.random()) < self.hard_negative_fraction
        if hard_negative:
            crop["mask"] = np.zeros_like(np.asarray(crop["mask"], dtype=np.uint8), dtype=np.uint8)
            crop["crop_kind"] = f"{crop.get('crop_kind', self.crop_mode)}_hard_negative"
        prompt_endpoints_xyz = perturb_recist_endpoints_for_training(
            endpoints_xyz=crop["endpoints_xyz"],  # type: ignore[arg-type]
            shape_zyx=np.asarray(crop["image"]).shape,  # type: ignore[arg-type]
            probability=self.recist_aug_probability,
            shift_xy=self.recist_aug_shift_xy,
            endpoint_jitter_xy=self.recist_aug_endpoint_jitter_xy,
            slice_jitter=self.recist_aug_slice_jitter,
            length_scale_min=self.recist_aug_length_scale_min,
            length_scale_max=self.recist_aug_length_scale_max,
        )
        prompt_mode = self.prompt_mode
        if hard_negative:
            prompt_mode = "wrong_prompt"
        if self.line_dropout_probability > 0.0 and float(np.random.random()) < self.line_dropout_probability:
            prompt_mode = "endpoint_only"
        inputs, _ = build_input_channels(
            image_crop=crop["image"],  # type: ignore[arg-type]
            endpoints_xyz=prompt_endpoints_xyz,
            prompt_sigma=self.prompt_sigma,
            case_id=case.case_id,
            prompt_source=case.prompt_source,
            prompt_mode=prompt_mode,
            intensity_shift=self.intensity_shift,
            intensity_scale=self.intensity_scale,
            noise_std=self.noise_std,
            normalization_stats=case.normalization_stats,
            normalization_mode=self.normalization_mode,
        )
        target = np.asarray(crop["mask"], dtype=np.float32)[None, ...]
        sample: dict[str, torch.Tensor | str] = {
            "image": torch.from_numpy(inputs),
            "mask": torch.from_numpy(target),
            "case_id": record.case_id,
            "crop_kind": str(crop.get("crop_kind", self.crop_mode)),
            "bbox_fit_scale_zyx": torch.tensor(crop.get("bbox_fit_scale_zyx", (1.0, 1.0, 1.0)), dtype=torch.float32),
            "bbox_fit_rescaled": torch.tensor(float(crop.get("bbox_fit_rescaled", False)), dtype=torch.float32),
            "bbox_fit_target_coverage": torch.tensor(float(crop.get("bbox_fit_target_coverage", 1.0)), dtype=torch.float32),
            "source_fov_scale_zyx": torch.tensor(crop.get("source_fov_scale_zyx", (1.0, 1.0, 1.0)), dtype=torch.float32),
            "mixed_prompt_tail_eligible": torch.tensor(float(crop.get("mixed_prompt_tail_eligible", False)), dtype=torch.float32),
            "main_loss_weight": torch.tensor(self.negative_loss_weight if hard_negative else 1.0, dtype=torch.float32),
            "hard_negative": torch.tensor(float(hard_negative), dtype=torch.float32),
        }
        aux_crop = _aux_crop_for_training_crop(case, crop, self.crop_size_zyx)
        if aux_crop is not None:
            aux_target, aux_valid_mask = aux_crop
            sample["aux_target"] = torch.from_numpy(aux_target.astype(np.int64, copy=False))
            sample["aux_valid_mask"] = torch.from_numpy(aux_valid_mask.astype(np.bool_, copy=False))
        return sample
