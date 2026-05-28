from __future__ import annotations

from importlib import import_module

import numpy as np

_surface_dice = import_module("eval.SurfaceDice")


def compute_case_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    spacing_xyz: np.ndarray,
) -> dict[str, float]:
    gt = np.asarray(ground_truth, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt={gt.shape}, pred={pred.shape}")
    if gt.sum() == 0 and pred.sum() == 0:
        return {"dsc": 1.0, "nsd": 1.0}
    if gt.sum() == 0 or pred.sum() == 0:
        return {"dsc": 0.0, "nsd": 0.0}
    dsc = float(_surface_dice.compute_dice_coefficient(gt, pred))
    if dsc < 0.2:
        return {"dsc": dsc, "nsd": 0.0}
    spacing_zyx = tuple(float(value) for value in np.asarray(spacing_xyz, dtype=np.float32)[::-1])
    surface_distances = _surface_dice.compute_surface_distances(gt, pred, spacing_zyx)
    nsd = float(_surface_dice.compute_surface_dice_at_tolerance(surface_distances, 2.0))
    return {"dsc": dsc, "nsd": nsd}


def compute_labeled_case_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    spacing_xyz: np.ndarray,
    label_values: list[int] | tuple[int, ...] | np.ndarray | None = None,
) -> dict[str, object]:
    gt = np.asarray(ground_truth)
    pred = np.asarray(prediction)
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt={gt.shape}, pred={pred.shape}")
    if label_values is None:
        labels = sorted(
            {
                int(value)
                for value in np.unique(gt).tolist() + np.unique(pred).tolist()
                if int(value) > 0
            }
        )
    else:
        labels = sorted({int(value) for value in np.asarray(label_values).reshape(-1).tolist() if int(value) > 0})
    if not labels:
        return {"dsc": 1.0, "nsd": 1.0, "lesion_count": 0, "lesions": []}

    lesion_rows: list[dict[str, float | int]] = []
    for label_value in labels:
        metrics = compute_case_metrics(gt == label_value, pred == label_value, spacing_xyz)
        lesion_rows.append(
            {
                "label_value": int(label_value),
                "dsc": float(metrics["dsc"]),
                "nsd": float(metrics["nsd"]),
                "gt_voxels": int((gt == label_value).sum()),
                "pred_voxels": int((pred == label_value).sum()),
            }
        )
    return {
        "dsc": float(np.mean([row["dsc"] for row in lesion_rows])),
        "nsd": float(np.mean([row["nsd"] for row in lesion_rows])),
        "lesion_count": int(len(lesion_rows)),
        "lesions": lesion_rows,
    }


def compute_binary_stats(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    gt = np.asarray(ground_truth, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt={gt.shape}, pred={pred.shape}")
    tp = int(np.logical_and(gt, pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())
    gt_count = int(gt.sum())
    pred_count = int(pred.sum())
    denominator = gt_count + pred_count
    if denominator == 0:
        dsc = 1.0
    else:
        dsc = float(2.0 * tp / denominator)
    if gt_count == 0:
        recall = 1.0 if pred_count == 0 else 0.0
    else:
        recall = float(tp / gt_count)
    if pred_count == 0:
        precision = 1.0 if gt_count == 0 else 0.0
    else:
        precision = float(tp / pred_count)
    pred_positive_fraction = float(pred_count / max(pred.size, 1))
    return {
        "dsc": dsc,
        "recall": recall,
        "precision": precision,
        "pred_positive_fraction": pred_positive_fraction,
        "gt_positive_voxels": float(gt_count),
        "pred_positive_voxels": float(pred_count),
        "tp_voxels": float(tp),
        "fp_voxels": float(fp),
        "fn_voxels": float(fn),
    }


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"val_dsc": 0.0, "val_nsd": 0.0, "val_score": 0.0}
    dsc = float(np.mean([row["dsc"] for row in rows]))
    nsd = float(np.mean([row["nsd"] for row in rows]))
    return {
        "val_dsc": dsc,
        "val_nsd": nsd,
        "val_score": 0.5 * (dsc + nsd),
    }
