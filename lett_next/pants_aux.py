from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PantsAuxClass:
    class_id: int
    name: str
    segmentation: str | None


@dataclass(frozen=True, slots=True)
class PantsAuxClassMap:
    version: str
    ignore_index: int
    classes: tuple[PantsAuxClass, ...]

    @property
    def num_classes(self) -> int:
        return max((item.class_id for item in self.classes), default=-1) + 1

    @property
    def class_names(self) -> dict[int, str]:
        return {item.class_id: item.name for item in self.classes}

    @property
    def segmentations_by_class_id(self) -> dict[int, str]:
        return {
            item.class_id: item.segmentation
            for item in self.classes
            if item.segmentation not in (None, "")
        }


def load_pants_aux_class_map(path: Path) -> PantsAuxClassMap:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in PanTS aux class map: {path}")
    version = str(payload.get("version") or "")
    if not version:
        raise ValueError(f"PanTS aux class map is missing version: {path}")
    ignore_index = int(payload.get("ignore_index", -1))
    raw_classes = payload.get("classes")
    if not isinstance(raw_classes, dict):
        raise ValueError(f"PanTS aux class map must contain a classes mapping: {path}")
    classes: list[PantsAuxClass] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for raw_id, raw_value in raw_classes.items():
        class_id = int(raw_id)
        if class_id < 0:
            raise ValueError(f"PanTS aux class IDs must be non-negative, got {class_id}")
        if class_id in seen_ids:
            raise ValueError(f"Duplicate PanTS aux class ID {class_id} in {path}")
        seen_ids.add(class_id)
        if isinstance(raw_value, str):
            name = raw_value
            segmentation = None if class_id == 0 else raw_value
        elif isinstance(raw_value, dict):
            name = str(raw_value.get("name") or "")
            segmentation_value = raw_value.get("segmentation")
            segmentation = None if segmentation_value in (None, "") else str(segmentation_value)
        else:
            raise ValueError(f"Unsupported PanTS aux class entry for ID {class_id}: {raw_value!r}")
        if not name:
            raise ValueError(f"PanTS aux class ID {class_id} is missing a name")
        if name in seen_names:
            raise ValueError(f"Duplicate PanTS aux class name {name!r} in {path}")
        seen_names.add(name)
        classes.append(PantsAuxClass(class_id=class_id, name=name, segmentation=segmentation))
    classes.sort(key=lambda item: item.class_id)
    expected_ids = list(range(max(seen_ids) + 1 if seen_ids else 0))
    if [item.class_id for item in classes] != expected_ids:
        raise ValueError(f"PanTS aux class IDs must be contiguous from 0 in {path}")
    if not classes or classes[0].name != "background":
        raise ValueError("PanTS aux class ID 0 must be background")
    return PantsAuxClassMap(version=version, ignore_index=ignore_index, classes=tuple(classes))


def class_map_metadata(class_map: PantsAuxClassMap) -> dict[str, Any]:
    return {
        "class_map_version": class_map.version,
        "ignore_index": int(class_map.ignore_index),
        "num_aux_classes": int(class_map.num_classes),
        "class_names": {str(key): value for key, value in class_map.class_names.items()},
    }
