from __future__ import annotations

import json
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

@dataclass(slots=True)
class RunContext:
    timestamp: str
    run_id: str
    run_dir: Path
    description_path: Path
    checkpoints_dir: Path
    log_path: Path
    metrics_path: Path
    config_path: Path


class TeeStream:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_stdout(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("a", encoding="utf-8") as handle:
        tee = TeeStream(original_stdout, handle)
        sys.stdout = tee
        sys.stderr = tee
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def ensure_layout(repo_root: Path) -> None:
    (repo_root / "runs").mkdir(parents=True, exist_ok=True)
    (repo_root / "data" / "prepared").mkdir(parents=True, exist_ok=True)


def get_git_commit(repo_root: Path) -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return "unknown"
    return output.strip() or "unknown"


def _sanitize_run_id_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return sanitized.strip("._-")


def _timestamp_for_run_id(run_id: str) -> str:
    prefix = run_id.split("_", 1)[0]
    if re.fullmatch(r"\d{8}T\d{6}Z", prefix):
        return prefix
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def create_run_context(
    repo_root: Path,
    model_name: str,
    runs_dir: Path | None = None,
    resume_run_dir: Path | None = None,
    run_suffix: str = "",
) -> RunContext:
    runs_root = runs_dir or (repo_root / "runs")
    if resume_run_dir is not None:
        run_dir = resume_run_dir
        run_id = run_dir.name
        timestamp = _timestamp_for_run_id(run_id)
        checkpoints_dir = run_dir / "checkpoints"
        for path in (run_dir, checkpoints_dir):
            path.mkdir(parents=True, exist_ok=True)
        return RunContext(
            timestamp=timestamp,
            run_id=run_id,
            run_dir=run_dir,
            description_path=run_dir / "description.txt",
            checkpoints_dir=checkpoints_dir,
            log_path=run_dir / "train.log",
            metrics_path=run_dir / "metrics.json",
            config_path=run_dir / "config.json",
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_suffix = _sanitize_run_id_component(run_suffix)
    run_id_base = f"{timestamp}_{model_name}"
    if run_suffix:
        run_id_base = f"{run_id_base}_{run_suffix}"
    run_id = run_id_base
    run_dir = runs_root / run_id
    collision_index = 2
    while run_dir.exists():
        run_id = f"{run_id_base}_{collision_index}"
        run_dir = runs_root / run_id
        collision_index += 1
    checkpoints_dir = run_dir / "checkpoints"
    for path in (checkpoints_dir,):
        path.mkdir(parents=True, exist_ok=True)
    return RunContext(
        timestamp=timestamp,
        run_id=run_id,
        run_dir=run_dir,
        description_path=run_dir / "description.txt",
        checkpoints_dir=checkpoints_dir,
        log_path=run_dir / "train.log",
        metrics_path=run_dir / "metrics.json",
        config_path=run_dir / "config.json",
    )


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
