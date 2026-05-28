#!/usr/bin/env python3
"""Render a polished LETT-NeXt RECIST-to-mask visual abstract.

Pipeline shown in the figure:

    CT slice + RECIST line -> RECIST-guided local 3D crop + prompt channels
    -> LETT-NeXt -> volumetric lesion mask mapped back to the full CT volume

This version intentionally avoids freehand-looking pseudo-3D objects.  It uses
fixed panel coordinates, aligned headers, a controlled axial-slice stack for the
local 3D crop, and a deliberately simplified model placeholder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy import ndimage

DEFAULT_INPUTS_DIR = Path("data/validation_npz/validation_public_npz")
DEFAULT_PREDICTIONS_DIR = Path("results/submission_validation_public")
DEFAULT_OUTPUT_DIR = Path("figures")

COLORS = {
    "ink": "#0f172a",
    "muted": "#475569",
    "light_text": "#64748b",
    "panel": "#ffffff",
    "panel_fill": "#f8fafc",
    "panel_edge": "#cbd5e1",
    "cyan": "#22d3ee",
    "cyan_dark": "#0891b2",
    "cyan_light": "#cffafe",
    "blue": "#2563eb",
    "blue_dark": "#1e3a8a",
    "blue_mid": "#60a5fa",
    "blue_light": "#dbeafe",
    "green": "#3b8c46",
    "green_light": "#e9f6e6",
    "orange": "#f97316",
    "orange_dark": "#c2410c",
    "orange_light": "#ffedd5",
    "orange_pale": "#fff7ed",
}


# -----------------------------------------------------------------------------
# Data loading and image utilities
# -----------------------------------------------------------------------------


def _load_case(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["imgs"]), np.asarray(data["gts"]), np.asarray(data["recist"])


def _load_prediction(path: Path, expected_shape: tuple[int, int, int]) -> np.ndarray:
    # NIfTI is commonly stored x-y-z. The challenge arrays here are z-y-x.
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("nibabel is required to load .nii.gz predictions. Install it with `pip install nibabel`, or use --demo for layout testing.") from exc
    prediction = np.asarray(nib.load(str(path)).dataobj).transpose(2, 1, 0)
    if tuple(prediction.shape) != expected_shape:
        raise ValueError(f"Prediction shape {prediction.shape} does not match {expected_shape}: {path}")
    return prediction


def _normalize(image: np.ndarray) -> np.ndarray:
    """Robustly normalize one CT slice/crop for display."""
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image, dtype=np.float32)

    image = np.where(finite, image, 0.0)
    low, high = np.percentile(image[finite], [1.0, 99.5])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def _rgba(mask: np.ndarray, color: tuple[float, float, float], alpha: float) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    rgba = np.zeros(mask.shape + (4,), dtype=np.float32)
    rgba[..., :3] = np.asarray(color, dtype=np.float32)
    rgba[..., 3] = mask.astype(np.float32) * alpha
    return rgba


def _display_slice(recist_mask: np.ndarray, target_mask: np.ndarray, prediction_mask: np.ndarray) -> int:
    recist_area = recist_mask.reshape(recist_mask.shape[0], -1).sum(axis=1)
    if recist_area.max() > 0:
        return int(np.argmax(recist_area))

    union = target_mask | prediction_mask
    union_area = union.reshape(union.shape[0], -1).sum(axis=1)
    return int(np.argmax(union_area)) if union_area.max() > 0 else int(union.shape[0] // 2)


def _crop_slices(mask: np.ndarray, *, margin: int, min_size: int) -> tuple[slice, slice]:
    points = np.argwhere(mask)
    height, width = mask.shape
    if len(points) == 0:
        center_y, center_x = height // 2, width // 2
        size = min_size
    else:
        y0, x0 = points.min(axis=0)
        y1, x1 = points.max(axis=0) + 1
        center_y = int(round((int(y0) + int(y1)) / 2.0))
        center_x = int(round((int(x0) + int(x1)) / 2.0))
        size = max(int(y1 - y0), int(x1 - x0), min_size) + 2 * margin

    size = min(int(size), int(max(height, width)))
    half = int(np.ceil(size / 2.0))

    y_start = max(0, center_y - half)
    y_stop = min(height, y_start + size)
    y_start = max(0, y_stop - size)

    x_start = max(0, center_x - half)
    x_stop = min(width, x_start + size)
    x_start = max(0, x_stop - size)

    return slice(int(y_start), int(y_stop)), slice(int(x_start), int(x_stop))


def _farthest_endpoints(mask: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return RECIST endpoints as ((y0, x0), (y1, x1)) from a rasterized line mask."""
    points = np.argwhere(mask > 0)
    if len(points) < 2:
        return None
    distances = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
    first, second = np.unravel_index(int(np.argmax(distances)), distances.shape)
    return tuple(float(v) for v in points[first]), tuple(float(v) for v in points[second])


def _endpoint_heatmap(line_mask: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    endpoints = _farthest_endpoints(line_mask)
    heatmap = np.zeros_like(line_mask, dtype=np.float32)
    if endpoints is None:
        return heatmap

    for y, x in endpoints:
        heatmap[int(round(y)), int(round(x))] = 1.0
    heatmap = ndimage.gaussian_filter(heatmap, sigma=float(sigma))
    maximum = float(heatmap.max())
    return heatmap / maximum if maximum > 0 else heatmap


def _safe_z(volume: np.ndarray, z: int) -> int:
    return int(np.clip(z, 0, volume.shape[0] - 1))


def _make_demo_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a synthetic case for layout testing via --demo."""
    zdim, height, width = 56, 512, 512
    yy, xx = np.mgrid[:height, :width]
    zz = np.arange(zdim)[:, None, None]

    body = (((yy - 265) / 190) ** 2 + ((xx - 256) / 225) ** 2) < 1
    liver = (((yy - 210) / 85) ** 2 + ((xx - 185) / 120) ** 2) < 1
    spleen = (((yy - 230) / 70) ** 2 + ((xx - 360) / 60) ** 2) < 1
    spine = (((yy - 380) / 46) ** 2 + ((xx - 255) / 58) ** 2) < 1

    image = np.full((zdim, height, width), -900.0, dtype=np.float32)
    image[:, body] = 35.0
    image[:, liver] = 75.0
    image[:, spleen] = 95.0
    image[:, spine] = 300.0
    image += np.random.default_rng(4).normal(0, 12, image.shape).astype(np.float32)

    lesion = (((zz - 29) / 5.5) ** 2 + ((yy - 196) / 22) ** 2 + ((xx - 150) / 28) ** 2) < 1
    gt = np.zeros_like(image, dtype=np.uint8)
    prediction = np.zeros_like(image, dtype=np.uint8)
    recist = np.zeros_like(image, dtype=np.uint8)
    gt[lesion] = 2
    prediction[ndimage.binary_dilation(lesion, iterations=1)] = 2

    # Rasterized RECIST line on the central lesion slice.
    z = 29
    y0, x0, y1, x1 = 216, 126, 174, 174
    n = 90
    ys = np.linspace(y0, y1, n).round().astype(int)
    xs = np.linspace(x0, x1, n).round().astype(int)
    recist[z, ys, xs] = 2
    recist[z] = ndimage.binary_dilation(recist[z] == 2, iterations=1).astype(np.uint8) * 2
    return image, gt, recist, prediction


# -----------------------------------------------------------------------------
# Figure drawing primitives
# -----------------------------------------------------------------------------


def _fig_square_height(fig: plt.Figure, width_frac: float) -> float:
    return float(width_frac * fig.get_figwidth() / fig.get_figheight())


def _soft_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str = "white",
    edgecolor: str = COLORS["panel_edge"],
    linewidth: float = 1.0,
    rounding: float = 0.018,
    alpha: float = 1.0,
    zorder: float = 1.0,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={rounding}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    axis.add_patch(patch)
    return patch


def _draw_panel(
    canvas: plt.Axes,
    rect: tuple[float, float, float, float],
    *,
    number: int,
    title: str,
    subtitle: str,
) -> None:
    x, y, w, h = rect
    _soft_box(canvas, (x, y), w, h, facecolor=COLORS["panel"], edgecolor=COLORS["panel_edge"], linewidth=1.0, rounding=0.018)

    header_y = y + h - 0.070
    badge = Circle((x + 0.027, header_y + 0.032), 0.018, facecolor=COLORS["ink"], edgecolor="none", zorder=5)
    canvas.add_patch(badge)
    canvas.text(x + 0.027, header_y + 0.032, str(number), ha="center", va="center", fontsize=10.5, weight="bold", color="white", zorder=6)
    canvas.text(x + 0.057, header_y + 0.044, title, ha="left", va="center", fontsize=14.4, weight="bold", color=COLORS["ink"], zorder=6)
    canvas.text(x + 0.057, header_y - 0.015, subtitle, ha="left", va="center", fontsize=12.2, color=COLORS["muted"], linespacing=0.92, zorder=6)


def _arrow_between(canvas: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str) -> None:
    canvas.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.0,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=8,
        )
    )


def _draw_ct(axis: plt.Axes, image: np.ndarray) -> None:
    axis.imshow(_normalize(image), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor("black")
    for spine in axis.spines.values():
        spine.set_visible(False)


def _draw_recist_line(axis: plt.Axes, recist: np.ndarray) -> None:
    endpoints = _farthest_endpoints(recist)
    if endpoints is None:
        line = ndimage.binary_dilation(recist > 0, structure=np.ones((3, 3), dtype=bool), iterations=1)
        axis.imshow(_rgba(line, (1.0, 0.45, 0.0), 0.95))
        return

    (y0, x0), (y1, x1) = endpoints
    stroke = [pe.Stroke(linewidth=5.5, foreground="black", alpha=0.35), pe.Normal()]
    axis.plot([x0, x1], [y0, y1], color=COLORS["orange"], linewidth=2.9, solid_capstyle="round", path_effects=stroke, zorder=7)
    for y, x in ((y0, x0), (y1, x1)):
        axis.add_patch(
            Circle(
                (x, y),
                radius=5.0,
                facecolor=COLORS["orange"],
                edgecolor="white",
                linewidth=1.1,
                zorder=9,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="black", alpha=0.25)],
            )
        )


def _draw_input_image(axis: plt.Axes, image: np.ndarray, recist: np.ndarray, crop_y: slice, crop_x: slice) -> None:
    _draw_ct(axis, image)
    _draw_recist_line(axis, recist)
    axis.add_patch(
        Rectangle(
            (crop_x.start, crop_y.start),
            crop_x.stop - crop_x.start,
            crop_y.stop - crop_y.start,
            fill=False,
            edgecolor=COLORS["orange"],
            linewidth=2.0,
            linestyle=(0, (5, 2.5)),
            zorder=8,
        )
    )


def _draw_output_image(axis: plt.Axes, image: np.ndarray, prediction: np.ndarray) -> None:
    _draw_ct(axis, image)
    axis.imshow(_rgba(prediction > 0, (1.0, 0.36, 0.12), 0.58), interpolation="nearest")
    if np.asarray(prediction).any():
        axis.contour(prediction > 0, levels=[0.5], colors=[COLORS["orange"]], linewidths=1.8)


def _draw_cyan_channel(axis: plt.Axes, array: np.ndarray, *, dilate: int = 0) -> None:
    values = np.asarray(array, dtype=np.float32)
    if values.max() > 0:
        values = values / float(values.max())
    if dilate > 0:
        values = ndimage.grey_dilation(values, size=(dilate, dilate))
    rgb = np.zeros(values.shape + (3,), dtype=np.float32)
    rgb[..., 0] = 1.00 * values
    rgb[..., 1] = 0.45 * values
    axis.imshow(rgb, vmin=0.0, vmax=1.0, interpolation="nearest")
    axis.set_facecolor("black")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


# -----------------------------------------------------------------------------
# Center-panel components
# -----------------------------------------------------------------------------


def _add_slice_stack(
    fig: plt.Figure,
    canvas: plt.Axes,
    *,
    image: np.ndarray,
    z: int,
    crop_y: slice,
    crop_x: slice,
    line_crop: np.ndarray,
    pos: tuple[float, float, float, float],
) -> None:
    """Show the local 3D crop as a controlled axial-slice stack."""
    x, y, w, h = pos
    dx, dy = 0.012, 0.018

    # Draw two ghost slices behind the actual middle slice. This reads as 3D
    # without distorting the CT crop into an unnatural perspective transform.
    for i, alpha in [(2, 0.34), (1, 0.48)]:
        canvas.add_patch(
            Rectangle(
                (x + i * dx, y + i * dy),
                w,
                h,
                facecolor=COLORS["orange_light"],
                edgecolor=COLORS["orange_dark"],
                linewidth=0.9,
                alpha=alpha,
                zorder=3,
            )
        )

    front_ax = fig.add_axes([x, y, w, h], zorder=7)
    front_ax.imshow(_normalize(image[_safe_z(image, z), crop_y, crop_x]), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    _draw_recist_line(front_ax, line_crop)
    front_ax.set_xticks([])
    front_ax.set_yticks([])
    for spine in front_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["orange"])
        spine.set_linewidth(1.5)

    canvas.text(x + w / 2 + dx, y + h + 0.058, "RECIST-guided\nlocal 3D crop", ha="center", va="bottom", fontsize=10.8, weight="bold", color=COLORS["orange_dark"], linespacing=0.95, zorder=10)


def _add_channel_stack(
    fig: plt.Figure,
    canvas: plt.Axes,
    *,
    ct_crop: np.ndarray,
    line_crop: np.ndarray,
    endpoint_crop: np.ndarray,
    rect: tuple[float, float, float, float],
) -> None:
    x, y, w, h = rect
    _soft_box(canvas, (x, y), w, h, facecolor=COLORS["orange_pale"], edgecolor=COLORS["orange"], linewidth=1.0, rounding=0.014, zorder=4)
    canvas.text(x + w / 2, y + h - 0.032, "dense prompt\nchannels", ha="center", va="top", fontsize=10.2, weight="bold", color=COLORS["orange_dark"], linespacing=0.90, zorder=7)

    thumb_w = w * 0.36
    thumb_h = _fig_square_height(fig, thumb_w)
    thumb_x = x + 0.013
    label_x = thumb_x + thumb_w + 0.012
    y_positions = [y + h - 0.220, y + h - 0.385, y + h - 0.550]
    labels = ["CT\ncrop", "RECIST\nline", "Endpoint\nheatmap"]

    ax = fig.add_axes([thumb_x, y_positions[0], thumb_w, thumb_h], zorder=8)
    ax.imshow(_normalize(ct_crop), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#64748b")
        spine.set_linewidth(0.9)
    canvas.text(label_x, y_positions[0] + thumb_h / 2, labels[0], ha="left", va="center", fontsize=8.7, color=COLORS["ink"], linespacing=0.9, zorder=9)

    ax = fig.add_axes([thumb_x, y_positions[1], thumb_w, thumb_h], zorder=8)
    _draw_cyan_channel(ax, line_crop.astype(float), dilate=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["orange_dark"])
        spine.set_linewidth(0.9)
    canvas.text(label_x, y_positions[1] + thumb_h / 2, labels[1], ha="left", va="center", fontsize=8.7, color=COLORS["ink"], linespacing=0.9, zorder=9)

    ax = fig.add_axes([thumb_x, y_positions[2], thumb_w, thumb_h], zorder=8)
    _draw_cyan_channel(ax, endpoint_crop, dilate=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["orange_dark"])
        spine.set_linewidth(0.9)
    canvas.text(label_x, y_positions[2] + thumb_h / 2, labels[2], ha="left", va="center", fontsize=8.7, color=COLORS["ink"], linespacing=0.9, zorder=9)


def _draw_model_placeholder(axis: plt.Axes) -> None:
    """Draw a compact MedNeXt-style U-Net placeholder, not the exact model."""
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    _soft_box(axis, (0.035, 0.045), 0.93, 0.91, facecolor=COLORS["orange_pale"], edgecolor=COLORS["orange"], linewidth=1.15, rounding=0.045, zorder=0)
    axis.text(0.50, 0.900, "LETT-NeXt", ha="center", va="center", fontsize=13.0, weight="bold", color=COLORS["orange_dark"], zorder=5)

    def mini_box(x: float, y: float, w: float, h: float, text: str, *, face: str, edge: str, fontsize: float = 6.7) -> tuple[float, float]:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.010,rounding_size=0.018",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.8,
            zorder=3,
        )
        axis.add_patch(patch)
        axis.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight="bold", color=COLORS["ink"], linespacing=0.92, zorder=4)
        return x + w / 2, y + h / 2

    def mini_arrow(start: tuple[float, float], end: tuple[float, float], *, color: str = COLORS["ink"], rad: float = 0.0) -> None:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=7.5,
                linewidth=0.85,
                color=color,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=3,
                shrinkB=3,
                zorder=2,
            )
        )

    enc_x, dec_x = 0.155, 0.635
    box_w, box_h = 0.210, 0.082
    enc_y = [0.705, 0.540, 0.375]
    dec_y = [0.705, 0.540, 0.375]
    enc_centers = [
        mini_box(enc_x, y, box_w, box_h, label, face=COLORS["orange_light"], edge=COLORS["orange"])
        for y, label in zip(enc_y, ["Enc 0", "Enc 1", "Enc 2"])
    ]
    bottleneck = mini_box(0.395, 0.230, box_w, box_h, "Bottleneck", face="#fed7aa", edge=COLORS["orange"], fontsize=6.2)
    dec_centers = [
        mini_box(dec_x, y, box_w, box_h, label, face=COLORS["orange_light"], edge=COLORS["orange_dark"])
        for y, label in zip(dec_y, ["Dec 0", "Dec 1", "Dec 2"])
    ]

    for upper, lower in zip(enc_centers, enc_centers[1:]):
        mini_arrow((upper[0], upper[1] - box_h / 2), (lower[0], lower[1] + box_h / 2))
    mini_arrow((enc_centers[-1][0] + box_w / 2, enc_centers[-1][1]), (bottleneck[0] - box_w / 2, bottleneck[1]))
    mini_arrow((bottleneck[0] + box_w / 2, bottleneck[1]), (dec_centers[-1][0] - box_w / 2, dec_centers[-1][1]))
    for lower, upper in zip(reversed(dec_centers[1:]), reversed(dec_centers[:-1])):
        mini_arrow((lower[0], lower[1] + box_h / 2), (upper[0], upper[1] - box_h / 2))

    for enc, dec in zip(enc_centers, dec_centers):
        plus = Circle((0.535, enc[1]), 0.022, facecolor="white", edgecolor=COLORS["ink"], linewidth=0.75, zorder=4)
        axis.add_patch(plus)
        axis.text(0.535, enc[1] - 0.001, "+", ha="center", va="center", fontsize=6.5, weight="bold", color=COLORS["ink"], zorder=5)
        mini_arrow((enc[0] + box_w / 2, enc[1]), (0.512, enc[1]), color=COLORS["orange_dark"])
        mini_arrow((0.558, enc[1]), (dec[0] - box_w / 2, dec[1]), color=COLORS["orange_dark"])

    axis.text(
        0.50,
        0.125,
        "compact placeholder\nnot exact architecture",
        ha="center",
        va="center",
        fontsize=7.4,
        color=COLORS["muted"],
        linespacing=0.90,
        zorder=5,
    )


def _draw_center_panel(
    fig: plt.Figure,
    canvas: plt.Axes,
    *,
    image: np.ndarray,
    z: int,
    crop_y: slice,
    crop_x: slice,
    ct_crop: np.ndarray,
    line_crop: np.ndarray,
    endpoint_crop: np.ndarray,
) -> None:
    # Fixed center-panel layout.
    stack_pos = (0.300, 0.330, 0.108, _fig_square_height(fig, 0.108))
    channels_rect = (0.430, 0.200, 0.134, 0.585)
    model_pos = (0.598, 0.265, 0.168, 0.455)

    _add_slice_stack(fig, canvas, image=image, z=z, crop_y=crop_y, crop_x=crop_x, line_crop=line_crop, pos=stack_pos)

    # Crop-to-channel arrow.
    _arrow_between(canvas, (0.418, 0.505), (0.428, 0.505), color=COLORS["orange_dark"])
    _add_channel_stack(fig, canvas, ct_crop=ct_crop, line_crop=line_crop, endpoint_crop=endpoint_crop, rect=channels_rect)

    # Channel-to-model arrow.
    _arrow_between(canvas, (0.565, 0.505), (0.590, 0.505), color=COLORS["orange"])
    model_axis = fig.add_axes(model_pos, zorder=6)
    _draw_model_placeholder(model_axis)


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------


def render(args: argparse.Namespace) -> dict[str, str]:
    if args.demo:
        image, gt, recist, prediction = _make_demo_case()
        input_path = Path("<demo>")
        prediction_path = Path("<demo>")
    else:
        input_path = args.inputs_dir / f"{args.case_id}.npz"
        prediction_path = args.predictions_dir / f"{args.case_id}.nii.gz"
        image, gt, recist = _load_case(input_path)
        prediction = _load_prediction(prediction_path, tuple(int(dim) for dim in image.shape))

    label = int(args.label_value)
    recist_mask = recist == label
    gt_mask = gt == label
    prediction_mask = prediction == label

    z = _display_slice(recist_mask, gt_mask, prediction_mask)
    crop_y, crop_x = _crop_slices(recist_mask[z] | gt_mask[z] | prediction_mask[z], margin=args.crop_margin, min_size=args.crop_min_size)

    image_slice = image[z]
    line_slice = recist_mask[z]
    prediction_slice = prediction_mask[z]
    ct_crop = image_slice[crop_y, crop_x]
    line_crop = line_slice[crop_y, crop_x]
    endpoint_crop = _endpoint_heatmap(line_crop, sigma=args.endpoint_sigma)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(15.8, 5.55), dpi=args.dpi, facecolor="white")
    canvas = fig.add_axes([0, 0, 1, 1], zorder=0)
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis("off")

    input_panel = (0.035, 0.130, 0.205, 0.785)
    center_panel = (0.270, 0.130, 0.505, 0.785)
    output_panel = (0.805, 0.130, 0.170, 0.785)

    _draw_panel(canvas, input_panel, number=1, title="Input", subtitle="Real CT lesion\n+ RECIST")
    _draw_panel(canvas, center_panel, number=2, title="LETT-NeXt inference", subtitle="Prompt channels\n+ compact schematic")
    _draw_panel(canvas, output_panel, number=3, title="Output", subtitle="Volumetric\nlesion mask")

    # Arrows between panels are drawn on the global canvas, so they remain aligned.
    _arrow_between(canvas, (0.248, 0.505), (0.263, 0.505), color=COLORS["ink"])
    _arrow_between(canvas, (0.785, 0.505), (0.798, 0.505), color=COLORS["orange"])

    # Input CT slice.
    input_img_w = 0.172
    input_img_h = _fig_square_height(fig, input_img_w)
    input_img_x = input_panel[0] + (input_panel[2] - input_img_w) / 2
    input_img_y = 0.298
    input_axis = fig.add_axes([input_img_x, input_img_y, input_img_w, input_img_h], zorder=5)
    _draw_input_image(input_axis, image_slice, line_slice, crop_y, crop_x)
    canvas.text(input_panel[0] + input_panel[2] / 2, 0.225, "RECIST diameter + endpoint dots", ha="center", va="center", fontsize=10.5, color=COLORS["muted"], zorder=9)

    # Center panel: local 3D crop, prompt channels, model placeholder.
    _draw_center_panel(
        fig,
        canvas,
        image=image,
        z=z,
        crop_y=crop_y,
        crop_x=crop_x,
        ct_crop=ct_crop,
        line_crop=line_crop,
        endpoint_crop=endpoint_crop,
    )

    # Output CT slice.
    output_img_w = 0.138
    output_img_h = _fig_square_height(fig, output_img_w)
    output_img_x = output_panel[0] + (output_panel[2] - output_img_w) / 2
    output_img_y = 0.320
    output_axis = fig.add_axes([output_img_x, output_img_y, output_img_w, output_img_h], zorder=5)
    _draw_output_image(output_axis, image_slice, prediction_slice)
    canvas.text(output_panel[0] + output_panel[2] / 2, 0.240, "mapped back to\nfull CT volume", ha="center", va="center", fontsize=10.2, color=COLORS["orange_dark"], linespacing=0.95, zorder=9)
    _arrow_between(canvas, (output_panel[0] + output_panel[2] / 2, 0.282), (output_panel[0] + output_panel[2] / 2, 0.315), color=COLORS["orange"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "lett_next_visual_abstract"
    if args.demo:
        stem += "_demo"
    png_path = args.output_dir / f"{stem}.png"
    pdf_path = args.output_dir / f"{stem}.pdf"
    svg_path = args.output_dir / f"{stem}.svg"

    # Do not use bbox_inches="tight" here. Fixed panel geometry is intentional,
    # and tight cropping can make publication-panel alignment look inconsistent.
    fig.savefig(png_path, pad_inches=0.02)
    fig.savefig(pdf_path, pad_inches=0.02)
    fig.savefig(svg_path, pad_inches=0.02)
    plt.close(fig)

    metadata_path = args.output_dir / f"{stem}.json"
    metadata_path.write_text(
        json.dumps(
            {
                "case_id": args.case_id if not args.demo else "demo",
                "label_value": label,
                "slice_index": int(z),
                "input_path": str(input_path),
                "prediction_path": str(prediction_path),
                "crop_y": [int(crop_y.start), int(crop_y.stop)],
                "crop_x": [int(crop_x.start), int(crop_x.stop)],
                "outputs": {"png": str(png_path), "pdf": str(pdf_path), "svg": str(svg_path)},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return {"png": str(png_path), "pdf": str(pdf_path), "svg": str(svg_path), "metadata": str(metadata_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir", type=Path, default=DEFAULT_INPUTS_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-id", default="CT_Lesion_FLARE23Ts_0003")
    parser.add_argument("--label-value", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--crop-margin", type=int, default=48)
    parser.add_argument("--crop-min-size", type=int, default=150)
    parser.add_argument("--endpoint-sigma", type=float, default=5.0)
    parser.add_argument("--demo", action="store_true", help="Render a synthetic case to check layout without local CT files.")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(render(parse_args()), indent=2))


if __name__ == "__main__":
    main()
