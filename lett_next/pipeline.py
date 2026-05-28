from __future__ import annotations

import fcntl
import json
import os
import random
import resource
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .artifacts import (
    create_run_context,
    ensure_layout,
    get_git_commit,
    tee_stdout,
    write_json,
)
from . import recipe
from .config import ExperimentConfig
from .data import (
    LEGACY_RECIST_RESAMPLING_MODE,
    RecistTrainingConfig,
    RecistTrainingDataset,
    build_recist_target_report,
    compute_ct_normalization_stats,
    load_manifest,
    load_case,
    recist_target_voxel_count_for_record,
    validate_records_integrity,
)
from .evaluation import evaluate_cases
from .losses import make_training_loss
from .models import _final_logits, build_model
from .pants_aux import PantsAuxClassMap, load_pants_aux_class_map
from .submission_validation import run_checkpoint_submission_validation


@dataclass(frozen=True, slots=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    enabled: bool

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0


def get_distributed_context(environ: dict[str, str] | None = None) -> DistributedContext:
    env = os.environ if environ is None else environ
    world_size = int(env.get("WORLD_SIZE", "1") or "1")
    rank = int(env.get("RANK", "0") or "0")
    local_rank = int(env.get("LOCAL_RANK", "0") or "0")
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"RANK must be in [0, WORLD_SIZE), got rank={rank} world_size={world_size}")
    if local_rank < 0:
        raise ValueError(f"LOCAL_RANK must be >= 0, got {local_rank}")
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        enabled=world_size > 1,
    )


def _initialize_distributed_context(config: ExperimentConfig) -> DistributedContext:
    context = get_distributed_context()
    if not context.enabled:
        return context
    if torch.cuda.is_available():
        torch.cuda.set_device(context.local_rank)
        config.train_device = f"cuda:{context.local_rank}"
        if str(config.eval_device).startswith("cuda"):
            config.eval_device = f"cuda:{context.local_rank}"
        backend = "nccl"
    else:
        config.train_device = "cpu"
        backend = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return context


def _distributed_barrier(context: DistributedContext) -> None:
    if context.enabled and dist.is_available() and dist.is_initialized():
        dist.barrier()


def _cleanup_distributed_context(context: DistributedContext) -> None:
    if context.enabled and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _broadcast_string(value: str | None, context: DistributedContext) -> str | None:
    if not context.enabled:
        return value
    payload: list[str | None] = [value if context.is_rank_zero else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _distributed_batch_size(config: ExperimentConfig, context: DistributedContext) -> int:
    batch_size = int(config.batch_size)
    if not context.enabled:
        return batch_size
    if batch_size % int(context.world_size) != 0:
        raise ValueError(
            f"DDP requires global batch_size to be divisible by WORLD_SIZE: "
            f"batch_size={batch_size} world_size={context.world_size}"
        )
    return max(batch_size // int(context.world_size), 1)


def _distributed_data_parallel_kwargs(config: ExperimentConfig, context: DistributedContext) -> dict[str, object]:
    if not context.enabled:
        return {}
    kwargs: dict[str, object] = {}
    if str(config.train_device).startswith("cuda"):
        kwargs |= {"device_ids": [context.local_rank], "output_device": context.local_rank}
    if config.anatomy_aux_enable:
        kwargs["find_unused_parameters"] = True
    return kwargs


def _set_distributed_sampler_epoch(loader: DataLoader, epoch: int) -> None:
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(int(epoch))


def _unwrap_parallel_model(model: torch.nn.Module) -> torch.nn.Module:
    while hasattr(model, "module") and isinstance(getattr(model, "module"), torch.nn.Module):
        model = getattr(model, "module")
    return model


def _strip_module_prefix(model_state: dict[str, object]) -> dict[str, object]:
    if not model_state:
        return dict(model_state)
    if not all(str(key).startswith("module.") for key in model_state):
        return dict(model_state)
    return {str(key).removeprefix("module."): value for key, value in model_state.items()}


def _load_model_state_dict(model: torch.nn.Module, model_state: object) -> None:
    if not isinstance(model_state, dict):
        raise ValueError("Checkpoint model_state must be a state-dict mapping")
    target_model = _unwrap_parallel_model(model)
    try:
        target_model.load_state_dict(model_state)
    except RuntimeError:
        target_model.load_state_dict(_strip_module_prefix(model_state))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _limit_records(records, limit: int | None):
    if limit is None:
        return records
    return records[:limit]


def _recist_resampling_mode(config: ExperimentConfig) -> str:
    return LEGACY_RECIST_RESAMPLING_MODE


def _normalization_stats(config: ExperimentConfig) -> dict[str, object] | None:
    return config.normalization_stats


def _normalization_mode(config: ExperimentConfig) -> str:
    if str(config.normalization_mode) in {"dataset_stats", "fixed_hu"} and config.normalization_stats is None:
        return "legacy_crop_zscore"
    return str(config.normalization_mode)


def _anatomy_aux_class_map(config: ExperimentConfig) -> PantsAuxClassMap | None:
    if not config.anatomy_aux_enable:
        return None
    if config.anatomy_aux_class_map_path is None:
        raise ValueError("anatomy_aux_class_map_path must be set when anatomy_aux_enable=true")
    return load_pants_aux_class_map(config.anatomy_aux_class_map_path)


def _load_case_for_config(record, config: ExperimentConfig, *, require_mask: bool = False, require_recist_target: bool = False):
    return load_case(
        record,
        target_spacing_xyz=config.target_spacing_xyz,
        resampling_mode=_recist_resampling_mode(config),
        require_mask=require_mask,
        require_recist_target=require_recist_target,
        normalization_stats=_normalization_stats(config),
        case_cache_dir=config.case_cache_dir,
        pants_aux_class_map=_anatomy_aux_class_map(config),
    )


def _threshold_name(threshold: float) -> str:
    return f"{threshold:.3f}".rstrip("0").rstrip(".")


def _threshold_label(threshold: float) -> str:
    return _threshold_name(threshold).replace(".", "p")


def _mean_numeric_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {
        str(key): float(np.mean([float(row[str(key)]) for row in rows]))
        for key in keys
    }


def _learning_rate_for_epoch(config: ExperimentConfig, epoch: int) -> float:
    return float(config.learning_rate)


def _set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)


def _latest_deep_supervision_losses(criterion: torch.nn.Module) -> dict[str, float]:
    payload = getattr(criterion, "latest_component_losses", None)
    return dict(payload) if isinstance(payload, dict) else {}


def _slice_batch_outputs(outputs, index: int):
    if isinstance(outputs, torch.Tensor):
        return outputs[index : index + 1]
    if isinstance(outputs, tuple):
        return tuple(_slice_batch_outputs(output, index) for output in outputs)
    if isinstance(outputs, list):
        return [_slice_batch_outputs(output, index) for output in outputs]
    raise TypeError(f"Unsupported output type for sample-weighted loss: {type(outputs)!r}")


def _mean_hard_dice_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> float:
    probabilities = torch.sigmoid(logits.detach())
    hard_predictions = (probabilities >= float(threshold)).float()
    targets = targets.detach().float()
    intersection = torch.sum(hard_predictions * targets, dim=(1, 2, 3, 4))
    denominator = torch.sum(hard_predictions + targets, dim=(1, 2, 3, 4))
    dice = torch.where(
        denominator > 0,
        (2.0 * intersection) / denominator.clamp_min(1.0),
        torch.ones_like(denominator),
    )
    return float(dice.mean().item())


def _recist_target_voxel_count(
    record,
    target_spacing_xyz: tuple[float, float, float] | None = None,
    resampling_mode: str = "standard",
) -> int | None:
    return recist_target_voxel_count_for_record(
        record,
        target_spacing_xyz=target_spacing_xyz,
        resampling_mode=resampling_mode,
    )


def _filter_train_records(
    records,
    *,
    exclude_empty: bool,
    target_spacing_xyz: tuple[float, float, float] | None = None,
    verify_resampled_empty: bool = True,
) -> tuple[list[object], dict[str, object]]:
    kept = []
    empty_rows: list[dict[str, object]] = []
    nonempty_count = 0
    missing_count = 0
    for record in records:
        manifest_voxel_count = getattr(record, "target_voxel_count", None)
        verify_after_resample = bool(verify_resampled_empty) and target_spacing_xyz is not None
        if manifest_voxel_count is not None and not verify_after_resample:
            voxel_count = int(record.target_voxel_count)
        else:
            voxel_count = _recist_target_voxel_count(
                record,
                target_spacing_xyz=target_spacing_xyz,
                resampling_mode=LEGACY_RECIST_RESAMPLING_MODE,
            )
        is_empty = voxel_count in (None, 0)
        if is_empty:
            if voxel_count is None:
                missing_count += 1
            empty_rows.append(
                {
                    "case_id": record.case_id,
                    "split": record.split,
                    "selected_target_voxels": 0 if voxel_count is None else int(voxel_count),
                    "reason": "missing_target_mask" if voxel_count is None else "empty_target_mask",
                }
            )
            if exclude_empty:
                continue
        else:
            nonempty_count += 1
        kept.append(record)
    audit = {
        "input_train_case_count": int(len(records)),
        "kept_train_case_count": int(len(kept)),
        "nonempty_target_case_count": int(nonempty_count),
        "empty_or_missing_target_case_count": int(len(empty_rows)),
        "missing_target_case_count": int(missing_count),
        "excluded_empty_train_masks": bool(exclude_empty),
        "verified_resampled_empty": bool(verify_resampled_empty),
        "empty_or_missing_cases": empty_rows,
    }
    return kept, audit


def _record_paths(record) -> list[Path]:
    paths = [Path(str(record.image_path))]
    if getattr(record, "label_path", None) not in (None, ""):
        paths.append(Path(str(record.label_path)))
    return paths


def _filter_available_records(records) -> tuple[list[object], dict[str, object]]:
    kept = []
    skipped_by_source: dict[str, int] = {}
    for record in records:
        if all(path.exists() for path in _record_paths(record)):
            kept.append(record)
            continue
        source = str(getattr(record, "source", "unknown"))
        skipped_by_source[source] = skipped_by_source.get(source, 0) + 1
    return kept, {
        "input_records": int(len(records)),
        "available_records": int(len(kept)),
        "skipped_missing_records": int(len(records) - len(kept)),
        "skipped_missing_by_source": skipped_by_source,
    }


def _make_dataloaders(
    config: ExperimentConfig,
    manifest_records,
    distributed_context: DistributedContext | None = None,
):
    context = distributed_context or get_distributed_context({})
    manifest_records, availability_audit = _filter_available_records(manifest_records)
    if context.is_rank_zero:
        print("availability=" + json.dumps(availability_audit, sort_keys=True), flush=True)
    train_records = [record for record in manifest_records if record.split == "train"]
    val_records = [record for record in manifest_records if record.split == "val"]
    train_target_audit: dict[str, object] = {}
    verify_resampled_empty = config.case_cache_dir is None
    train_records, train_target_audit = _filter_train_records(
        train_records,
        exclude_empty=config.exclude_empty_train_masks,
        target_spacing_xyz=config.target_spacing_xyz,
        verify_resampled_empty=verify_resampled_empty,
    )
    train_records = _limit_records(train_records, config.max_train_cases)
    val_records = _limit_records(val_records, config.max_val_cases)
    if not train_records:
        raise ValueError(
            "No local training records are available. Download or rsync at least one training case "
            "matching data/prepared/train_manifest.json."
        )
    if not val_records:
        val_records = train_records[: min(2, len(train_records))]
    integrity_report = _integrity_report_for_records(config, list(train_records) + list(val_records))
    normalization_mode = str(config.normalization_mode)
    if normalization_mode == "dataset_stats":
        config.normalization_stats = _normalization_stats_for_records(config, train_records)
    elif normalization_mode == "fixed_hu":
        if config.normalization_stats is None:
            config.normalization_stats = {
                "mode": "fixed_hu",
                "clip_low": -1000.0,
                "clip_high": 1000.0,
                "mean": 0.0,
                "std": 500.0,
                "source_case_count": 0,
                "source_voxel_count": 0,
            }
    elif normalization_mode in {"legacy_crop_zscore", "crop_zscore", "legacy"}:
        config.normalization_stats = None
    else:
        raise ValueError(f"Unsupported normalization_mode: {normalization_mode}")
    pants_aux_class_map = _anatomy_aux_class_map(config)
    training_config = RecistTrainingConfig(
        crop_size_zyx=config.crop_size_zyx,
        prompt_sigma=config.prompt_sigma,
        prompt_mode=config.primary_prompt_mode,
        crop_mode=config.train_crop_mode,
        crop_jitter_zyx=config.train_crop_jitter_zyx,
        bbox_fit_margin_zyx=config.train_bbox_fit_margin_zyx,
        mixed_prompt_base_fraction=config.mixed_prompt_base_fraction,
        mixed_prompt_multifov_fraction=config.mixed_prompt_multifov_fraction,
        mixed_recist_bbox_fit_tail_fraction=config.mixed_recist_bbox_fit_tail_fraction,
        mixed_recist_multifov_scales_zyx=config.mixed_recist_multifov_scales_zyx,
        ensure_recist_endpoints_inside_crop=config.ensure_recist_endpoints_inside_crop,
        recist_endpoint_margin_mm=config.recist_endpoint_margin_mm,
        recist_aug_probability=config.train_recist_aug_probability,
        recist_aug_shift_xy=config.train_recist_aug_shift_xy,
        recist_aug_endpoint_jitter_xy=config.train_recist_aug_endpoint_jitter_xy,
        recist_aug_slice_jitter=config.train_recist_aug_slice_jitter,
        recist_aug_length_scale_min=config.train_recist_aug_length_scale_min,
        recist_aug_length_scale_max=config.train_recist_aug_length_scale_max,
        line_dropout_probability=config.train_line_dropout_probability,
        intensity_shift=config.train_intensity_shift,
        intensity_scale=config.train_intensity_scale,
        noise_std=config.train_noise_std,
        target_spacing_xyz=config.target_spacing_xyz,
        normalization_stats=config.normalization_stats,
        normalization_mode=_normalization_mode(config),
        pants_aux_class_map=pants_aux_class_map,
        hard_negative_fraction=config.hard_negative_fraction,
        negative_loss_weight=config.negative_loss_weight,
    )
    dataset = RecistTrainingDataset(
        records=train_records,
        crop_size_zyx=training_config,
        case_cache_dir=config.case_cache_dir,
    )
    sampler = None
    shuffle = True
    if context.enabled:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=config.seed,
        )
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=_distributed_batch_size(config, context),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
    )
    return loader, train_records, val_records, train_target_audit, integrity_report


_RESUME_IMMUTABLE_FIELDS = (
    "model_name",
    "deep_supervision",
    "exclude_empty_train_masks",
    "manifest_path",
    "crop_size_zyx",
    "target_spacing_xyz",
    "normalization_mode",
    "primary_prompt_mode",
    "train_crop_mode",
    "train_bbox_fit_margin_zyx",
    "mixed_prompt_base_fraction",
    "mixed_prompt_multifov_fraction",
    "mixed_recist_bbox_fit_tail_fraction",
    "mixed_recist_multifov_scales_zyx",
    "ensure_recist_endpoints_inside_crop",
    "recist_endpoint_margin_mm",
    "anatomy_aux_enable",
    "anatomy_aux_class_map_path",
    "anatomy_aux_loss_weight",
    "hard_negative_fraction",
    "negative_loss_weight",
    "batch_size",
)


def _read_json(path: Path, default: dict[str, object] | None = None) -> dict[str, object]:
    if not path.exists():
        return {} if default is None else dict(default)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _read_or_compute_json_cache(
    path: Path,
    compute_payload,
    *,
    label: str | None = None,
    heartbeat_seconds: float = 30.0,
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    cache_label = str(label or path.name)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        wait_started = time.monotonic()
        last_wait_report = wait_started - float(heartbeat_seconds)
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                now = time.monotonic()
                if now - last_wait_report >= float(heartbeat_seconds):
                    print(
                        json.dumps(
                            {
                                "event": "json_cache_waiting_for_lock",
                                "label": cache_label,
                                "path": str(path),
                                "lock_path": str(lock_path),
                                "elapsed_seconds": round(float(now - wait_started), 1),
                            }
                        ),
                        flush=True,
                    )
                    last_wait_report = now
                time.sleep(min(max(float(heartbeat_seconds), 1.0), 5.0))
        if path.exists():
            print(
                json.dumps(
                    {
                        "event": "json_cache_hit",
                        "label": cache_label,
                        "path": str(path),
                    }
                ),
                flush=True,
            )
            return _read_json(path)
        print(
            json.dumps(
                {
                    "event": "json_cache_compute_start",
                    "label": cache_label,
                    "path": str(path),
                }
            ),
            flush=True,
        )
        payload = compute_payload()
        if not isinstance(payload, dict):
            raise ValueError(f"Expected cache payload for {path} to be a JSON object")
        write_json(path, payload)
        print(
            json.dumps(
                {
                    "event": "json_cache_compute_done",
                    "label": cache_label,
                    "path": str(path),
                }
            ),
            flush=True,
        )
        return payload


def _validate_records_for_config(
    config: ExperimentConfig,
    records,
) -> dict[str, object]:
    report = validate_records_integrity(
        list(records),
        target_spacing_xyz=config.target_spacing_xyz,
        resampling_mode=_recist_resampling_mode(config),
        require_mask=True,
        require_recist_target=True,
    )
    return report.to_dict()


def _integrity_report_for_records(
    config: ExperimentConfig,
    records,
) -> dict[str, object]:
    integrity_mode = str(config.integrity_mode)
    if integrity_mode in {"skip", "disabled", "none"}:
        return {
            "checked_case_count": 0,
            "failed_case_count": 0,
            "failures": [],
            "skipped": True,
            "integrity_mode": integrity_mode,
        }
    if config.integrity_report_path is not None:
        return _read_or_compute_json_cache(
            config.integrity_report_path,
            lambda: _validate_records_for_config(config, records),
            label="integrity_report",
        )
    if integrity_mode == "fail_fast":
        return _validate_records_for_config(config, records)
    raise ValueError(f"Unsupported integrity_mode: {integrity_mode}")


def _target_audit_for_records(
    config: ExperimentConfig,
    train_records,
    val_records,
    train_filter_audit: dict[str, object],
) -> dict[str, object]:
    def compute_payload() -> dict[str, object]:
        return {
            "train_filter": train_filter_audit,
            "train": build_recist_target_report(
                train_records,
                target_spacing_xyz=config.target_spacing_xyz,
                resampling_mode=_recist_resampling_mode(config),
            ),
            "val": build_recist_target_report(
                val_records,
                target_spacing_xyz=config.target_spacing_xyz,
                resampling_mode=_recist_resampling_mode(config),
            ),
        }

    if config.target_audit_path is None:
        return compute_payload()
    return _read_or_compute_json_cache(
        config.target_audit_path,
        compute_payload,
        label="target_audit",
    )


def _normalization_stats_for_records(
    config: ExperimentConfig,
    train_records,
) -> dict[str, object]:
    def compute_payload() -> dict[str, object]:
        return compute_ct_normalization_stats(
            list(train_records),
            target_spacing_xyz=config.target_spacing_xyz,
            resampling_mode=_recist_resampling_mode(config),
            seed=config.seed,
            use_mask_values=False,
            progress_label="normalization_stats",
        ).to_dict()

    if config.normalization_stats_path is not None:
        return _read_or_compute_json_cache(
            config.normalization_stats_path,
            compute_payload,
            label="normalization_stats",
        )
    return compute_payload()


def _normalize_config_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_normalize_config_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_config_value(item) for item in value]
    return value


def _resolve_resume_run_dir(config: ExperimentConfig) -> Path | None:
    if config.resume_run_dir is not None:
        return config.resume_run_dir
    if config.resume_checkpoint_path is None:
        return None
    checkpoint_path = config.resume_checkpoint_path
    if checkpoint_path.parent.name == "checkpoints" and checkpoint_path.parent.parent != checkpoint_path.parent:
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def _resolve_resume_checkpoint_path(config: ExperimentConfig) -> Path:
    if config.resume_checkpoint_path is not None:
        return config.resume_checkpoint_path
    resume_run_dir = _resolve_resume_run_dir(config)
    if resume_run_dir is None:
        raise ValueError("resume_run_dir or resume_checkpoint_path must be set to resume a run")
    return resume_run_dir / "checkpoints" / "last.pt"


def _load_resume_checkpoint(config: ExperimentConfig) -> dict[str, object]:
    checkpoint_path = _resolve_resume_checkpoint_path(config)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"Failed to load resume checkpoint: {checkpoint_path}: {error}") from error
    if not isinstance(state, dict):
        raise ValueError(f"Resume checkpoint has an unexpected payload: {checkpoint_path}")
    state["checkpoint_path"] = str(checkpoint_path)
    return state


def _validate_resume_config(config: ExperimentConfig, checkpoint_state: dict[str, object]) -> dict[str, object]:
    snapshot = checkpoint_state.get("config_snapshot") or checkpoint_state.get("config")
    if not isinstance(snapshot, dict):
        raise ValueError("Resume checkpoint is missing a config snapshot")
    current = config.to_dict()
    mismatches: list[str] = []
    for field in _RESUME_IMMUTABLE_FIELDS:
        current_value = _normalize_config_value(current.get(field))
        saved_value = _normalize_config_value(snapshot.get(field))
        if current_value != saved_value:
            mismatches.append(f"{field}: saved={saved_value!r} current={current_value!r}")
    if mismatches:
        raise ValueError("Resume config mismatch for immutable fields: " + "; ".join(mismatches))
    return snapshot


def _resume_progress_value(config: ExperimentConfig, checkpoint_state: dict[str, object]) -> int:
    key = "epoch"
    raw_value = checkpoint_state.get(key)
    if raw_value in (None, ""):
        raise ValueError(f"Resume checkpoint is missing completed {key}")
    return int(raw_value)


class _TrainingLogger:
    def log_history_row(self, history_row: dict[str, float]) -> None:
        return

    def log_summary(self, summary: dict[str, object]) -> None:
        return

    def finish(self) -> None:
        return


def _restore_training_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_state: dict[str, object],
) -> None:
    _load_model_state_dict(model, checkpoint_state["model_state"])
    optimizer_state = checkpoint_state.get("optimizer_state")
    if optimizer_state is None:
        checkpoint_path = checkpoint_state.get("checkpoint_path", "<unknown>")
        raise ValueError(f"Resume checkpoint is missing optimizer_state: {checkpoint_path}")
    optimizer.load_state_dict(optimizer_state)


def _atomic_torch_save(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: ExperimentConfig,
    *,
    epoch: int | None = None,
    step: int | None = None,
    best_score: float | None = None,
    history: list[dict[str, float]] | None = None,
    extra_state: dict[str, object] | None = None,
) -> None:
    unwrapped_model = _unwrap_parallel_model(model)
    payload: dict[str, object] = {
        "model_state": unwrapped_model.state_dict(),
        "model_name": config.model_name,
        "config": config.to_dict(),
        "config_snapshot": config.to_dict(),
        "best_score": best_score,
        "history": [] if history is None else history,
        "extra_state": {} if extra_state is None else extra_state,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if step is not None:
        payload["step"] = int(step)
    _atomic_torch_save(path, payload)


def _load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
    state = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has an unexpected payload: {path}")
    _load_model_state_dict(model, state["model_state"])
    if optimizer is not None:
        optimizer_state = state.get("optimizer_state")
        if optimizer_state is None:
            raise ValueError(f"Checkpoint is missing optimizer_state: {path}")
        optimizer.load_state_dict(optimizer_state)
    return state


def _peak_memory_gb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(usage) / (1024.0**2)


def _make_loss(config: ExperimentConfig, device: str):
    return make_training_loss(deep_supervision=config.deep_supervision, aux_class_map=_anatomy_aux_class_map(config))


def _write_run_description(
    run,
    config: ExperimentConfig,
    git_commit: str,
    selected_case_ids: list[str] | None = None,
    train_case_ids: list[str] | None = None,
    val_case_ids: list[str] | None = None,
) -> None:
    lines = [
        f"description: {config.description}",
        f"model: {config.model_name}",
        f"deep_supervision: {config.deep_supervision}",
        f"commit: {git_commit}",
        f"manifest_path: {config.manifest_path}",
        "loss: weighted_dice_bce_aux",
        f"dice_weight: {recipe.DICE_WEIGHT}",
        f"bce_weight: {recipe.BCE_WEIGHT}",
        f"epochs: {config.epochs}",
        f"batch_size: {config.batch_size}",
        f"learning_rate: {config.learning_rate}",
        f"eval_every: {config.eval_every}",
        f"crop_size_zyx: {config.crop_size_zyx}",
        f"primary_prompt_mode: {config.primary_prompt_mode}",
        f"exclude_empty_train_masks: {config.exclude_empty_train_masks}",
        f"seed: {config.seed}",
        f"postprocess: {recipe.POSTPROCESS_KWARGS}",
        f"validation_autozoom: {recipe.VALIDATION_AUTOZOOM_KWARGS}",
        f"train_crop_mode: {config.train_crop_mode}",
        f"train_bbox_fit_margin_zyx: {config.train_bbox_fit_margin_zyx}",
        f"mixed_prompt_base_fraction: {config.mixed_prompt_base_fraction}",
        f"mixed_prompt_multifov_fraction: {config.mixed_prompt_multifov_fraction}",
        f"mixed_recist_bbox_fit_tail_fraction: {config.mixed_recist_bbox_fit_tail_fraction}",
        f"mixed_recist_multifov_scales_zyx: {config.mixed_recist_multifov_scales_zyx}",
        f"ensure_recist_endpoints_inside_crop: {config.ensure_recist_endpoints_inside_crop}",
        f"recist_endpoint_margin_mm: {config.recist_endpoint_margin_mm}",
        f"train_crop_jitter_zyx: {config.train_crop_jitter_zyx}",
        f"train_intensity_shift: {config.train_intensity_shift}",
        f"train_intensity_scale: {config.train_intensity_scale}",
        f"train_noise_std: {config.train_noise_std}",
        f"normalization_mode: {config.normalization_mode}",
        f"normalization_stats_path: {config.normalization_stats_path}",
        f"case_cache_dir: {config.case_cache_dir}",
        f"anatomy_aux_enable: {config.anatomy_aux_enable}",
        f"anatomy_aux_class_map_path: {config.anatomy_aux_class_map_path}",
        f"anatomy_aux_loss_weight: {config.anatomy_aux_loss_weight}",
        f"hard_negative_fraction: {config.hard_negative_fraction}",
        f"negative_loss_weight: {config.negative_loss_weight}",
        f"integrity_report_path: {config.integrity_report_path}",
        f"target_audit_path: {config.target_audit_path}",
        f"target_spacing_xyz: {config.target_spacing_xyz}",
    ]
    if selected_case_ids:
        lines.append("selected_cases:")
        lines.extend(f"- {case_id}" for case_id in selected_case_ids)
    if train_case_ids:
        lines.append("train_cases:")
        lines.extend(f"- {case_id}" for case_id in train_case_ids)
    if val_case_ids:
        lines.append("val_cases:")
        lines.extend(f"- {case_id}" for case_id in val_case_ids)
    run.description_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _run_standard_experiment(
    config: ExperimentConfig,
    manifest_records,
    run,
    git_commit: str,
    started: float,
    resume_state: dict[str, object] | None = None,
    distributed_context: DistributedContext | None = None,
) -> dict[str, object]:
    context = distributed_context or get_distributed_context({})
    history: list[dict[str, float]] = []
    train_loader, train_records, val_records, train_target_audit, integrity_report = _make_dataloaders(
        config,
        manifest_records,
        distributed_context=context,
    )
    last_case_rows: list[dict[str, object]] = []
    if context.is_rank_zero:
        _write_run_description(
            run,
            config,
            git_commit,
            train_case_ids=[record.case_id for record in train_records],
            val_case_ids=[record.case_id for record in val_records],
        )
    target_audit = {}
    if context.is_rank_zero:
        target_audit = _target_audit_for_records(
            config,
            train_records,
            val_records,
            train_target_audit,
        )
        write_json(run.run_dir / "target_audit.json", target_audit)
        if config.normalization_stats is not None:
            write_json(run.run_dir / "normalization_stats.json", config.normalization_stats)
        if integrity_report:
            write_json(run.run_dir / "integrity_report.json", integrity_report)
    sample_batch = next(iter(train_loader))
    in_channels = int(sample_batch["image"].shape[1])
    anatomy_aux_map = _anatomy_aux_class_map(config)
    model = build_model(
        config.model_name,
        in_channels=in_channels,
        deep_supervision=config.deep_supervision,
        aux_output_classes=None if anatomy_aux_map is None else anatomy_aux_map.num_classes,
    ).to(config.train_device)
    criterion = _make_loss(config, config.train_device)
    if context.enabled:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            **_distributed_data_parallel_kwargs(config, context),
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_score = None
    start_epoch = 0
    if resume_state is not None:
        _restore_training_state(model, optimizer, resume_state)
        start_epoch = _resume_progress_value(config, resume_state)
        best_score = None if resume_state.get("best_score") in (None, "") else float(resume_state["best_score"])
        history_payload = resume_state.get("history", [])
        if not isinstance(history_payload, list):
            raise ValueError("Resume checkpoint history must be a list")
        history = [dict(row) for row in history_payload]
        if start_epoch >= config.epochs:
            raise ValueError(
                f"Resume checkpoint already reached epoch {start_epoch}, which meets or exceeds target epochs={config.epochs}"
            )
        print(f"resuming run_id={run.run_id} from {resume_state['checkpoint_path']} at epoch={start_epoch}")
    training_logger = _TrainingLogger()
    try:
        for epoch in range(start_epoch + 1, config.epochs + 1):
            _set_distributed_sampler_epoch(train_loader, epoch)
            epoch_learning_rate = _learning_rate_for_epoch(config, epoch)
            _set_optimizer_learning_rate(optimizer, epoch_learning_rate)
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            epoch_started = time.perf_counter()
            batches_per_epoch = len(train_loader)
            deep_supervision_loss_rows: list[dict[str, float]] = []
            train_crop_dsc_rows: list[float] = []
            bbox_fit_rescaled_rows: list[float] = []
            bbox_fit_coverage_rows: list[float] = []
            bbox_fit_min_scale_rows: list[float] = []
            hard_negative_rows: list[float] = []
            for batch in train_loader:
                images = batch["image"].to(config.train_device)
                masks = batch["mask"].to(config.train_device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                if config.anatomy_aux_enable:
                    if "aux_target" not in batch or "aux_valid_mask" not in batch:
                        raise ValueError("anatomy auxiliary training is enabled but batch is missing aux_target/aux_valid_mask")
                    loss = criterion(
                        logits,
                        masks,
                        batch["aux_target"].to(config.train_device),
                        batch["aux_valid_mask"].to(config.train_device),
                        batch.get("main_loss_weight", torch.ones(masks.shape[0])).to(config.train_device),
                    )
                else:
                    loss = criterion(logits, masks)
                deep_supervision_loss_rows.append(_latest_deep_supervision_losses(criterion))
                train_crop_dsc_rows.append(
                    _mean_hard_dice_from_logits(
                        logits=_final_logits(logits),
                        targets=masks,
                        threshold=config.threshold,
                    )
                )
                if "bbox_fit_rescaled" in batch:
                    bbox_fit_rescaled_rows.append(float(batch["bbox_fit_rescaled"].float().mean().item()))
                if "bbox_fit_target_coverage" in batch:
                    bbox_fit_coverage_rows.append(float(batch["bbox_fit_target_coverage"].float().mean().item()))
                if "bbox_fit_scale_zyx" in batch:
                    bbox_fit_min_scale_rows.append(float(batch["bbox_fit_scale_zyx"].float().amin(dim=1).mean().item()))
                if "hard_negative" in batch:
                    hard_negative_rows.append(float(batch["hard_negative"].float().mean().item()))
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                batch_count += 1
                log_every = max(int(config.train_log_every_steps), 0)
                if log_every and (batch_count % log_every == 0 or batch_count == batches_per_epoch):
                    elapsed = time.perf_counter() - epoch_started
                    steps_per_second = float(batch_count / max(elapsed, 1e-9))
                    remaining_steps = max(int(batches_per_epoch - batch_count), 0)
                    progress_row: dict[str, float | str] = {
                        "event": "train_step",
                        "epoch": float(epoch),
                        "batch_step": float(batch_count),
                        "batches_per_epoch": float(batches_per_epoch),
                        "global_step": float((epoch - 1) * batches_per_epoch + batch_count),
                        "batch_loss": float(loss.item()),
                        "train_loss_running": float(epoch_loss / max(batch_count, 1)),
                        "learning_rate": float(epoch_learning_rate),
                        "elapsed_seconds": float(elapsed),
                        "steps_per_second": steps_per_second,
                        "eta_epoch_seconds": float(remaining_steps / max(steps_per_second, 1e-9)),
                    }
                    if train_crop_dsc_rows:
                        progress_row["train_crop_dsc_running"] = float(np.mean(train_crop_dsc_rows))
                    if bbox_fit_rescaled_rows:
                        progress_row["bbox_fit_rescaled_fraction_running"] = float(np.mean(bbox_fit_rescaled_rows))
                    if bbox_fit_coverage_rows:
                        progress_row["bbox_fit_target_coverage_running"] = float(np.mean(bbox_fit_coverage_rows))
                    if bbox_fit_min_scale_rows:
                        progress_row["bbox_fit_min_scale_running"] = float(np.mean(bbox_fit_min_scale_rows))
                    if hard_negative_rows:
                        progress_row["hard_negative_fraction_running"] = float(np.mean(hard_negative_rows))
                    print(json.dumps(progress_row), flush=True)
            epoch_loss = epoch_loss / max(batch_count, 1)
            history_row = {
                "epoch": float(epoch),
                "train_loss": epoch_loss,
                "learning_rate": float(epoch_learning_rate),
            }
            if deep_supervision_loss_rows:
                history_row |= _mean_numeric_dict(deep_supervision_loss_rows)
            if train_crop_dsc_rows:
                history_row["train_crop_dsc"] = float(np.mean(train_crop_dsc_rows))
            if bbox_fit_rescaled_rows:
                history_row["bbox_fit_rescaled_fraction"] = float(np.mean(bbox_fit_rescaled_rows))
            if bbox_fit_coverage_rows:
                history_row["bbox_fit_target_coverage"] = float(np.mean(bbox_fit_coverage_rows))
            if bbox_fit_min_scale_rows:
                history_row["bbox_fit_min_scale"] = float(np.mean(bbox_fit_min_scale_rows))
            if hard_negative_rows:
                history_row["hard_negative_fraction"] = float(np.mean(hard_negative_rows))
            should_eval = epoch % max(config.eval_every, 1) == 0 or epoch == config.epochs
            best_updated = False
            if should_eval and context.is_rank_zero:
                eval_model = _unwrap_parallel_model(model).to(config.eval_device)
                val_metrics, val_case_rows = evaluate_cases(
                    model=eval_model,
                    records=val_records,
                    config=config,
                    device=config.eval_device,
                    threshold=config.threshold,
                    aux_class_map=anatomy_aux_map,
                )
                last_case_rows = val_case_rows
                eval_model.to(config.train_device)
                history_row |= {
                    "val_dsc": val_metrics["val_dsc"],
                    "val_nsd": val_metrics["val_nsd"],
                    "val_score": val_metrics["val_score"],
                    "val_cpu_seconds_per_case": val_metrics["cpu_seconds_per_case"],
                }
                if best_score is None or val_metrics["val_score"] > best_score:
                    best_score = float(val_metrics["val_score"])
                    best_updated = True
            _distributed_barrier(context)
            if context.is_rank_zero:
                history.append(history_row)
                print(json.dumps(history_row))
                training_logger.log_history_row(history_row)
            checkpoint_extra = {}
            if should_eval and context.is_rank_zero:
                if best_updated:
                    _save_checkpoint(
                        run.checkpoints_dir / "best.pt",
                        model,
                        optimizer,
                        config,
                        epoch=epoch,
                        best_score=best_score,
                        history=history,
                        extra_state=checkpoint_extra,
                    )
            if context.is_rank_zero:
                _save_checkpoint(
                    run.checkpoints_dir / "last.pt",
                    model,
                    optimizer,
                    config,
                    epoch=epoch,
                    best_score=best_score,
                    history=history,
                    extra_state=checkpoint_extra,
                )

        if not context.is_rank_zero:
            return {
                "rank": context.rank,
                "world_size": context.world_size,
                "status": "worker_complete",
            }
        best_checkpoint_path = run.checkpoints_dir / "best.pt"
        final_row = history[-1] if history else {}
        if "val_score" not in final_row:
            raise RuntimeError("Training finished without a validation row; cannot write run summary")
        serializable_case_rows = [
            {key: value for key, value in row.items() if key != "thumbnail"}
            for row in last_case_rows
        ]
        component_summary = {
            "component_val_score": round(float(final_row["val_score"]), 6),
            "component_val_dsc": round(float(final_row["val_dsc"]), 6),
            "component_val_nsd": round(float(final_row["val_nsd"]), 6),
            "component_cpu_seconds_per_case": round(float(final_row.get("val_cpu_seconds_per_case", 0.0)), 6),
            "component_selected_threshold": round(float(config.threshold), 6),
        }
        submission_validation = run_checkpoint_submission_validation(
            repo_root=config.repo_root,
            checkpoint_path=best_checkpoint_path,
            outputs_dir=run.run_dir / "submission_validation",
            inputs_dir=config.val_data_dir,
        )
        total_seconds = time.perf_counter() - started
        summary: dict[str, object] = {
            "timestamp": run.timestamp,
            "run_id": run.run_id,
            "commit": git_commit,
            "model_name": config.model_name,
            "val_score": round(float(submission_validation["val_score"]), 6),
            "val_dsc": round(float(submission_validation["val_dsc"]), 6),
            "val_nsd": round(float(submission_validation["val_nsd"]), 6),
            "cpu_seconds_per_case": round(float(submission_validation.get("mean_total_seconds", 0.0)), 6),
            "memory_gb": round(_peak_memory_gb(), 6),
            "total_seconds": round(total_seconds, 6),
            "total_flops": 0.0,
            "description": config.description,
            "run_dir": str(run.run_dir),
            "best_checkpoint": str(best_checkpoint_path),
            "validation_path": "submission",
            "submission_validation_dir": str(run.run_dir / "submission_validation"),
            **component_summary,
        }
        summary["selected_threshold"] = round(float(recipe.SUBMISSION_THRESHOLD), 6)
        metrics_payload = {
            "summary": summary,
            "history": history,
            "case_metrics": serializable_case_rows,
            "submission_validation": submission_validation,
        }
        metrics_payload["target_audit"] = target_audit
        write_json(run.metrics_path, metrics_payload)
        training_logger.log_summary(summary)
        print(json.dumps(summary, indent=2))
        return summary
    finally:
        training_logger.finish()


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    distributed_context = _initialize_distributed_context(config)
    try:
        if distributed_context.is_rank_zero:
            ensure_layout(config.repo_root)
            config.runs_dir.mkdir(parents=True, exist_ok=True)
        _distributed_barrier(distributed_context)
        set_seed(config.seed)
        if distributed_context.is_rank_zero and not config.manifest_path.exists():
            raise FileNotFoundError(
                f"Training manifest not found: {config.manifest_path}. "
                "Run prepare.py or provide a committed manifest before training."
            )
        _distributed_barrier(distributed_context)
        manifest_records = load_manifest(config.manifest_path)
        resume_run_dir = _resolve_resume_run_dir(config)
        resume_state = _load_resume_checkpoint(config) if config.is_resume else None
        if resume_state is not None:
            _validate_resume_config(config, resume_state)
        if distributed_context.is_rank_zero:
            run = create_run_context(
                config.repo_root,
                config.model_name,
                runs_dir=config.runs_dir,
                resume_run_dir=resume_run_dir,
                run_suffix=config.train_run_suffix,
            )
            run_dir = run.run_dir
        else:
            run_dir = None
        broadcast_run_dir = _broadcast_string(None if run_dir is None else str(run_dir), distributed_context)
        if distributed_context.is_rank_zero:
            assert run.run_dir == Path(str(broadcast_run_dir))
        else:
            run = create_run_context(
                config.repo_root,
                config.model_name,
                runs_dir=config.runs_dir,
                resume_run_dir=Path(str(broadcast_run_dir)),
                run_suffix=config.train_run_suffix,
            )
        if distributed_context.is_rank_zero:
            write_json(run.config_path, config.to_dict())
        git_commit = get_git_commit(config.repo_root)
        if distributed_context.is_rank_zero:
            _write_run_description(run, config, git_commit)
        started = time.perf_counter()
        log_context = tee_stdout(run.log_path) if distributed_context.is_rank_zero else nullcontext()
        with log_context:
            if distributed_context.is_rank_zero:
                print(f"run_id={run.run_id}")
                print(f"model={config.model_name}")
                if distributed_context.enabled:
                    print(
                        "ddp="
                        + json.dumps(
                            {
                                "rank": distributed_context.rank,
                                "local_rank": distributed_context.local_rank,
                                "world_size": distributed_context.world_size,
                                "train_device": config.train_device,
                            }
                        )
                    )
            return _run_standard_experiment(
                config,
                manifest_records,
                run,
                git_commit,
                started,
                resume_state=resume_state,
                distributed_context=distributed_context,
            )
    finally:
        _cleanup_distributed_context(distributed_context)


def write_crash_summary(
    config: ExperimentConfig,
    error: Exception,
) -> dict[str, object]:
    distributed_context = get_distributed_context()
    if not distributed_context.is_rank_zero:
        return {
            "rank": distributed_context.rank,
            "world_size": distributed_context.world_size,
            "status": "crash_worker",
            "error": repr(error),
        }
    ensure_layout(config.repo_root)
    config.runs_dir.mkdir(parents=True, exist_ok=True)
    run = create_run_context(
        config.repo_root,
        config.model_name,
        runs_dir=config.runs_dir,
        resume_run_dir=_resolve_resume_run_dir(config) if config.is_resume else None,
        run_suffix=config.train_run_suffix,
    )
    summary = {
        "timestamp": run.timestamp,
        "run_id": run.run_id,
        "commit": get_git_commit(config.repo_root),
        "model_name": config.model_name,
        "val_score": "",
        "val_dsc": "",
        "val_nsd": "",
        "cpu_seconds_per_case": "",
        "memory_gb": round(_peak_memory_gb(), 6),
        "total_seconds": "",
        "total_flops": 0.0,
        "status": "crash",
        "description": f"{config.description}: {error}",
        "run_dir": str(run.run_dir),
        "best_checkpoint": "",
        "error": repr(error),
    }
    write_json(run.config_path, config.to_dict())
    write_json(run.metrics_path, {"summary": summary, "error": repr(error)})
    return summary
