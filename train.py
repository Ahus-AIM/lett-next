from __future__ import annotations

import json
import os
import subprocess
import sys
import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "configs/lett_next.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lett_next.config import ExperimentConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Training config path")
    return parser.parse_args()


def _resolve_config_path(path: Path | None = None) -> Path:
    path = CONFIG_PATH if path is None else path
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _run_training_worker() -> None:
    from lett_next.pipeline import run_experiment, write_crash_summary

    config = ExperimentConfig.from_source(REPO_ROOT, _resolve_config_path(Path(os.environ["LETT_NEXT_CONFIG_PATH"])))
    try:
        summary = run_experiment(config)
    except Exception as error:
        crash_summary = write_crash_summary(config, error)
        print(json.dumps(crash_summary, indent=2))
        raise
    print(json.dumps(summary, indent=2))


def _launcher_command(config: ExperimentConfig) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={config.train_nproc_per_node}",
        "train.py",
    ]


def main() -> None:
    if os.environ.get("LETT_NEXT_TRAIN_WORKER") == "1":
        _run_training_worker()
        return

    args = _parse_args()
    config_path = _resolve_config_path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    env = dict(os.environ)
    config = ExperimentConfig.from_source(REPO_ROOT, config_path)
    env["LETT_NEXT_TRAIN_WORKER"] = "1"
    env["LETT_NEXT_CONFIG_PATH"] = str(config_path)
    env.setdefault("CUDA_VISIBLE_DEVICES", config.train_cuda_visible_devices)

    command = _launcher_command(config)
    print(
        json.dumps(
            {
                "config": _display_path(config_path),
                "run_suffix": config.train_run_suffix,
                "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
                "nproc_per_node": config.train_nproc_per_node,
                "command": command,
            },
            indent=2,
        ),
        flush=True,
    )
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
