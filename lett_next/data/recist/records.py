from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(slots=True)
class CaseRecord:
    case_id: str
    image_path: str
    label_path: str | None
    spacing_xyz: list[float]
    recist_endpoints_xyz: list[list[float]]
    recist_slice_index: int
    split: str
    source: str
    prompt_source: str
    target_label_value: int | None = None
    target_component_id: int | None = None
    target_voxel_count: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CaseRecord":
        prompt_source = data.get("prompt_source")
        if prompt_source in (None, ""):
            prompt_source = "recist" if "recist_endpoints_xyz" in data else "unknown"
        return cls(
            case_id=str(data["case_id"]),
            image_path=str(data["image_path"]),
            label_path=None if data.get("label_path") in (None, "") else str(data["label_path"]),
            spacing_xyz=[float(value) for value in data["spacing_xyz"]],
            recist_endpoints_xyz=[
                [float(coord) for coord in point]
                for point in data["recist_endpoints_xyz"]
            ],
            recist_slice_index=int(data["recist_slice_index"]),
            split=str(data["split"]),
            source=str(data["source"]),
            prompt_source=str(prompt_source),
            target_label_value=(
                None
                if data.get("target_label_value") in (None, "")
                else int(data["target_label_value"])
            ),
            target_component_id=(
                None
                if data.get("target_component_id") in (None, "")
                else int(data["target_component_id"])
            ),
            target_voxel_count=(
                None
                if data.get("target_voxel_count") in (None, "")
                else int(data["target_voxel_count"])
            ),
        )


@dataclass(slots=True)
class LoadedCase:
    case_id: str
    image: np.ndarray
    raw_mask: np.ndarray | None
    mask: np.ndarray | None
    spacing_xyz: np.ndarray
    recist_endpoints_xyz: np.ndarray
    recist_slice_index: int
    prompt_source: str
    recist_target_mask: np.ndarray | None
    recist_target_selection: dict[str, object] | None
    anatomy_aux_target: np.ndarray | None = None
    anatomy_aux_valid_mask: np.ndarray | None = None
    anatomy_aux_metadata: dict[str, object] | None = None
    normalization_stats: dict[str, object] | None = None


@dataclass(slots=True)
class IntegrityReport:
    checked_case_count: int
    failed_case_count: int
    failures: list[dict[str, str]]

    def to_dict(self) -> dict[str, object]:
        return {
            "checked_case_count": int(self.checked_case_count),
            "failed_case_count": int(self.failed_case_count),
            "failures": self.failures,
        }


@dataclass(slots=True)
class NormalizationStats:
    mode: str
    clip_low: float
    clip_high: float
    mean: float
    std: float
    source_case_count: int
    source_voxel_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "clip_low": float(self.clip_low),
            "clip_high": float(self.clip_high),
            "mean": float(self.mean),
            "std": float(self.std),
            "source_case_count": int(self.source_case_count),
            "source_voxel_count": int(self.source_voxel_count),
        }
