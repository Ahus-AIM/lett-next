from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lett_next.config import ExperimentConfig
from lett_next.data import (
    LEGACY_RECIST_RESAMPLING_MODE,
    load_manifest,
    write_split_case_cache,
)
from lett_next.pipeline import (
    _filter_train_records,
    _limit_records,
    _normalization_stats_for_records,
    _integrity_report_for_records,
    _target_audit_for_records,
    _anatomy_aux_class_map,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="LETT-NeXt config whose shared caches should be built")
    parser.add_argument("--case-cache", action="store_true", help="Precompute resampled per-case training cache")
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers for per-case cache precompute")
    parser.add_argument("--limit", type=int, default=None, help="Optional case limit for smoke testing")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing per-case cache files")
    return parser.parse_args()


def _write_case_cache_worker(
    payload: tuple[object, str, tuple[float, float, float], str, bool, str | None],
) -> dict[str, object]:
    record, cache_dir, target_spacing_xyz, resampling_mode, overwrite, class_map_path = payload
    class_map = None
    if class_map_path is not None:
        from lett_next.pants_aux import load_pants_aux_class_map

        class_map = load_pants_aux_class_map(Path(class_map_path))
    return write_split_case_cache(
        record,  # type: ignore[arg-type]
        cache_dir=Path(cache_dir),
        target_spacing_xyz=target_spacing_xyz,
        resampling_mode=resampling_mode,
        overwrite=overwrite,
        pants_aux_class_map=class_map,
    )


def _precompute_case_cache(
    *,
    records: list[object],
    cache_dir: Path,
    target_spacing_xyz: tuple[float, float, float],
    workers: int,
    limit: int | None,
    overwrite: bool,
    class_map_path: Path | None,
) -> dict[str, object]:
    selected_records = records if limit is None else records[: int(limit)]
    cache_dir.mkdir(parents=True, exist_ok=True)
    totals: dict[str, int] = {"written": 0, "exists": 0, "empty_target": 0, "error": 0}
    bytes_written = 0
    worker_payloads = [
        (
            record,
            str(cache_dir),
            target_spacing_xyz,
            LEGACY_RECIST_RESAMPLING_MODE,
            overwrite,
            None if class_map_path is None else str(class_map_path),
        )
        for record in selected_records
    ]
    print(
        json.dumps(
            {
                "event": "case_cache_start",
                "cache_dir": str(cache_dir),
                "case_count": len(worker_payloads),
                "workers": int(workers),
                "overwrite": bool(overwrite),
                "aux_class_map_path": None if class_map_path is None else str(class_map_path),
                "cache_format": "split",
            }
        ),
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = [executor.submit(_write_case_cache_worker, payload) for payload in worker_payloads]
        for completed, future in enumerate(as_completed(futures), start=1):
            try:
                row = future.result()
            except Exception as error:
                row = {"status": "error", "error": repr(error)}
            status = str(row.get("status", "error"))
            totals[status] = totals.get(status, 0) + 1
            bytes_written += int(row.get("bytes", 0) or 0)
            if completed == 1 or completed % 25 == 0 or completed == len(futures):
                print(
                    json.dumps(
                        {
                            "event": "case_cache_progress",
                            "completed": completed,
                            "total": len(futures),
                            "latest_status": status,
                            "totals": totals,
                            "written_gb": round(bytes_written / (1024.0**3), 3),
                        }
                    ),
                    flush=True,
                )
    return {
        "cache_dir": str(cache_dir),
        "case_count": len(worker_payloads),
        "totals": totals,
        "written_gb": round(bytes_written / (1024.0**3), 3),
        "cache_format": "split",
    }


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config = ExperimentConfig.from_source(repo_root, args.config)
    aux_class_map = _anatomy_aux_class_map(config)
    manifest_records = load_manifest(config.manifest_path)
    train_records = [record for record in manifest_records if record.split == "train"]
    val_records = [record for record in manifest_records if record.split == "val"]
    train_records, train_filter_audit = _filter_train_records(
        train_records,
        exclude_empty=config.exclude_empty_train_masks,
        target_spacing_xyz=config.target_spacing_xyz,
        verify_resampled_empty=False,
    )
    train_records = _limit_records(train_records, config.max_train_cases)
    val_records = _limit_records(val_records, config.max_val_cases)
    if not val_records:
        val_records = train_records[: min(2, len(train_records))]

    json_train_records = train_records
    json_val_records = val_records
    if args.limit is not None:
        json_train_records = train_records[: int(args.limit)]
        json_val_records = val_records[: min(len(val_records), max(1, int(args.limit)))]

    print(
        json.dumps(
            {
                "event": "cache_precompute_start",
                "config": str(args.config),
                "manifest_path": str(config.manifest_path),
                "train_cases": len(train_records),
                "val_cases": len(val_records),
                "json_train_cases": len(json_train_records),
                "json_val_cases": len(json_val_records),
                "limit": args.limit,
                "normalization_stats_path": (
                    None if config.normalization_stats_path is None else str(config.normalization_stats_path)
                ),
                "integrity_report_path": (
                    None if config.integrity_report_path is None else str(config.integrity_report_path)
                ),
                "target_audit_path": (
                    None if config.target_audit_path is None else str(config.target_audit_path)
                ),
            }
        ),
        flush=True,
    )
    case_cache_summary: dict[str, object] | None = None
    if args.case_cache:
        if config.target_spacing_xyz is None:
            raise ValueError("target_spacing_xyz must be set for case caching")
        if config.case_cache_dir is None:
            raise ValueError("case_cache_dir must be set for --case-cache")
        case_cache_records = list(train_records) + list(val_records)
        case_cache_summary = _precompute_case_cache(
            records=case_cache_records,
            cache_dir=config.case_cache_dir,
            target_spacing_xyz=config.target_spacing_xyz,
            workers=args.workers,
            limit=args.limit,
            overwrite=args.overwrite,
            class_map_path=None if aux_class_map is None else config.anatomy_aux_class_map_path,
        )
    integrity_report = _integrity_report_for_records(config, list(json_train_records) + list(json_val_records))
    normalization_stats = _normalization_stats_for_records(config, json_train_records)
    target_audit = _target_audit_for_records(
        config,
        json_train_records,
        json_val_records,
        train_filter_audit,
    )
    print(
        json.dumps(
            {
                "event": "cache_precompute_done",
                "config": str(args.config),
                "integrity_checked_cases": int(integrity_report.get("checked_case_count", 0)),
                "normalization_source_cases": int(normalization_stats.get("source_case_count", 0)),
                "normalization_source_voxels": int(normalization_stats.get("source_voxel_count", 0)),
                "target_audit_sections": sorted(target_audit.keys()),
                "case_cache": case_cache_summary,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
