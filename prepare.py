from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "configs/lett_next.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Preparation config path")
    parser.add_argument(
        "--all-records",
        action="store_true",
        help="Attempt every manifest record, including records whose raw files are not present locally",
    )
    return parser.parse_args()


def _resolve_config_path(path: Path | None = None) -> Path:
    path = CONFIG_PATH if path is None else path
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _record_paths(record: object) -> list[Path]:
    paths = [Path(str(getattr(record, "image_path")))]
    label_path = getattr(record, "label_path", None)
    if label_path not in (None, ""):
        paths.append(Path(str(label_path)))
    return paths


def _filter_available_records(records: list[object]) -> tuple[list[object], dict[str, object]]:
    available: list[object] = []
    skipped_by_source: dict[str, int] = {}
    for record in records:
        missing_paths = [str(path) for path in _record_paths(record) if not path.exists()]
        if not missing_paths:
            available.append(record)
            continue
        source = str(getattr(record, "source", "unknown"))
        skipped_by_source[source] = skipped_by_source.get(source, 0) + 1
    return available, {
        "input_records": len(records),
        "available_records": len(available),
        "skipped_missing_records": len(records) - len(available),
        "skipped_missing_by_source": skipped_by_source,
    }


def main() -> None:
    from scripts.precompute_case_caches import _precompute_case_cache
    from lett_next.config import ExperimentConfig
    from lett_next.data import load_manifest
    from lett_next.pipeline import _filter_train_records, _limit_records, _anatomy_aux_class_map

    args = _parse_args()
    config_path = _resolve_config_path(args.config)
    _require(config_path)
    config = ExperimentConfig.from_source(REPO_ROOT, config_path)
    _require(config.manifest_path)
    if config.case_cache_dir is None:
        raise ValueError("configs/lett_next.yaml must define case_cache_dir")
    if config.target_spacing_xyz is None:
        raise ValueError("configs/lett_next.yaml must define target_spacing_xyz")

    manifest_records = load_manifest(config.manifest_path)
    manifest_train_records = [record for record in manifest_records if record.split == "train"]
    manifest_val_records = [record for record in manifest_records if record.split == "val"]
    train_records, train_filter_audit = _filter_train_records(
        manifest_train_records,
        exclude_empty=config.exclude_empty_train_masks,
        target_spacing_xyz=config.target_spacing_xyz,
        verify_resampled_empty=False,
    )
    train_records = _limit_records(train_records, config.max_train_cases)
    val_records = _limit_records(manifest_val_records, config.max_val_cases)
    aux_class_map = _anatomy_aux_class_map(config)

    cache_records = list(train_records) + list(val_records)
    availability_summary: dict[str, object] | None = None
    if not args.all_records:
        cache_records, availability_summary = _filter_available_records(cache_records)
        if not cache_records:
            raise ValueError(
                "No manifest records have all required raw files locally. "
                "Download a small FLARE/PanTS subset or pass --all-records to see raw loading failures."
            )

    cache_summary = _precompute_case_cache(
        records=cache_records,
        cache_dir=config.case_cache_dir,
        target_spacing_xyz=config.target_spacing_xyz,
        workers=8,
        limit=None,
        overwrite=False,
        class_map_path=None if aux_class_map is None else config.anatomy_aux_class_map_path,
    )
    print(
        json.dumps(
            {
                "config": _display_path(config_path),
                "manifest_path": _display_path(config.manifest_path),
                "train_records": len(train_records),
                "val_records": len(val_records),
                "available_records": len(cache_records),
                "availability": availability_summary,
                "train_filter_audit": train_filter_audit,
                "cache_dir": _display_path(config.case_cache_dir),
                "case_cache": cache_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
