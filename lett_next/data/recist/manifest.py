from __future__ import annotations

import json
from pathlib import Path

from .records import CaseRecord


def load_manifest(manifest_path: Path) -> list[CaseRecord]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent.parent.parent if manifest_path.parent.name == "prepared" else manifest_path.parent
    records = []
    for record in payload["records"]:
        resolved = dict(record)
        for key in ("image_path", "label_path"):
            value = resolved.get(key)
            if isinstance(value, str) and value and not Path(value).is_absolute():
                resolved[key] = str(root / value)
        records.append(CaseRecord.from_dict(resolved))
    return records
