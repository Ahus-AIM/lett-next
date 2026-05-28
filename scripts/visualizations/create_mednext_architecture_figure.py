"""
Reproduce a publication-style architecture figure for MedNextv2-f32.

The script creates an editable Matplotlib diagram and exports it as PNG, SVG, and PDF.
It is intentionally written with small helper functions so the layout, colors,
channel counts, and block labels can be changed easily.

Run:
    python reproduce_mednextv2unet3d_figure.py

Outputs:
    figures/mednext_architecture.png
    figures/mednext_architecture.svg
    figures/mednext_architecture.pdf
    figures/mednext_blocks.png
    figures/mednext_blocks.svg
    figures/mednext_blocks.pdf
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D


# -----------------------------------------------------------------------------
# Global style
# -----------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.linewidth": 0.8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "svg.fonttype": "none",  # keep text editable in SVG
    }
)

# Soft, paper-friendly colors.
COLORS = {
    "panel_edge": "#5f6368",
    "text": "#111111",
    "encoder_fill": "#eaf3ff",
    "encoder_edge": "#2f6db3",
    "decoder_fill": "#e9f6e6",
    "decoder_edge": "#3b8c46",
    "neutral_fill": "#ffffff",
    "neutral_edge": "#444444",
    "conv_fill": "#edf5ff",
    "conv_green_fill": "#edf8eb",
    "norm_fill": "#eef5ff",
    "gelu_fill": "#fff2cc",
    "grn_fill": "#efe3f8",
    "residual_fill": "#f6f6f6",
    "arrow": "#111111",
}


# -----------------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------------
def add_round_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str = "white",
    ec: str = "black",
    lw: float = 1.0,
    ls: str = "-",
    fontsize: float = 8,
    weight: str | None = None,
    color: str = COLORS["text"],
    rounding: float = 0.075,
    zorder: int = 3,
):
    """Draw a rounded box with centered text."""
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.025,rounding_size={rounding}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        zorder=zorder + 1,
        linespacing=1.1,
    )
    return patch


def add_panel_frame(ax, x: float, y: float, w: float, h: float, label: str | None, title: str | None):
    """Draw a large panel frame with a black letter tag and title."""
    add_round_box(
        ax,
        x,
        y,
        w,
        h,
        "",
        fc="white",
        ec=COLORS["panel_edge"],
        lw=1.0,
        rounding=0.035,
        zorder=0,
    )
    title_x = x + 0.62
    if label:
        tag_w, tag_h = 0.34, 0.34
        add_round_box(
            ax,
            x + 0.14,
            y + h - 0.45,
            tag_w,
            tag_h,
            label,
            fc="black",
            ec="black",
            fontsize=12,
            weight="bold",
            color="white",
            rounding=0.04,
            zorder=5,
        )
    else:
        title_x = x + 0.32
    if title:
        ax.text(
            title_x,
            y + h - 0.27,
            title,
            ha="left",
            va="center",
            fontsize=17,
            fontweight="bold",
            color=COLORS["text"],
            zorder=5,
        )


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    lw: float = 1.1,
    mutation_scale: float = 10,
    color: str = COLORS["arrow"],
    connectionstyle: str = "arc3,rad=0.0",
    shrinkA: float = 0,
    shrinkB: float = 0,
    zorder: int = 2,
):
    """Draw a single arrow from start to end."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=shrinkA,
        shrinkB=shrinkB,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def add_polyline_arrow(
    ax,
    points: Sequence[tuple[float, float]],
    *,
    lw: float = 1.1,
    color: str = COLORS["arrow"],
    mutation_scale: float = 10,
    zorder: int = 2,
):
    """Draw a segmented connector with an arrow head on the final segment."""
    if len(points) < 2:
        raise ValueError("A polyline arrow needs at least two points.")
    if len(points) > 2:
        xs, ys = zip(*points[:-1])
        ax.add_line(Line2D(xs, ys, linewidth=lw, color=color, zorder=zorder))
    add_arrow(
        ax,
        points[-2],
        points[-1],
        lw=lw,
        color=color,
        mutation_scale=mutation_scale,
        zorder=zorder,
    )


def add_plus(ax, x: float, y: float, r: float = 0.16):
    circ = Circle((x, y), r, facecolor="white", edgecolor="black", linewidth=1.2, zorder=4)
    ax.add_patch(circ)
    ax.text(x, y, "+", ha="center", va="center", fontsize=10, fontweight="bold", zorder=5)
    return circ


def add_junction(ax, x: float, y: float, r: float = 0.03):
    dot = Circle((x, y), r, facecolor="black", edgecolor="black", linewidth=0.5, zorder=4)
    ax.add_patch(dot)
    return dot


def add_text(ax, x: float, y: float, text: str, *, size: float = 8, weight: str | None = None, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, fontweight=weight, color=COLORS["text"])


def load_example_input_slice(example_path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a small real CT crop and RECIST prompt overlay for the input icon."""
    if example_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        example_path = repo_root / "data/validation_npz/validation_public_npz/CT_Lesion_FLARE23Ts_0003.npz"
    example_path = Path(example_path)
    if not example_path.exists():
        return None

    with np.load(example_path, allow_pickle=False) as data:
        image = data["imgs"]
        recist = data.get("recist")
        if recist is None:
            z = image.shape[0] // 2
            prompt = np.zeros_like(image[z], dtype=bool)
        else:
            recist_sums = recist.reshape(recist.shape[0], -1).sum(axis=1)
            z = int(np.argmax(recist_sums)) if np.any(recist_sums) else image.shape[0] // 2
            prompt = recist[z] > 0
        slice_image = image[z].astype(float)

    coords = np.argwhere(prompt)
    if coords.size:
        cy, cx = coords.mean(axis=0)
    else:
        cy, cx = np.array(slice_image.shape) / 2

    crop_size = 150
    half = crop_size // 2
    y0 = int(np.clip(round(cy) - half, 0, max(slice_image.shape[0] - crop_size, 0)))
    x0 = int(np.clip(round(cx) - half, 0, max(slice_image.shape[1] - crop_size, 0)))
    y1 = min(y0 + crop_size, slice_image.shape[0])
    x1 = min(x0 + crop_size, slice_image.shape[1])

    crop = slice_image[y0:y1, x0:x1]
    prompt_crop = prompt[y0:y1, x0:x1]
    lo, hi = np.percentile(crop, [1, 99])
    crop = np.clip((crop - lo) / max(hi - lo, 1e-6), 0, 1)
    return crop, prompt_crop


def draw_input_volume_icon(ax, x: float, y: float, w: float = 0.75, h: float = 0.75):
    """Draw a compact input volume icon using a real validation CT crop when present."""
    example = load_example_input_slice()

    # Stacked slices behind the front image.
    for i in range(6):
        dx, dy = 0.035 * i, 0.028 * i
        rect = Rectangle(
            (x + dx, y + dy),
            w,
            h,
            facecolor=str(0.08 + 0.03 * i),
            edgecolor="black",
            linewidth=0.45,
            zorder=2 + i,
        )
        ax.add_patch(rect)

    if example is None:
        cx, cy = x + 0.39, y + 0.39
        ax.add_patch(Circle((cx, cy), 0.24, facecolor="#3b3b3b", edgecolor="#d9d9d9", linewidth=0.7, zorder=10))
        ax.add_patch(Circle((cx - 0.05, cy + 0.02), 0.05, facecolor="#bfbfbf", edgecolor="none", zorder=11))
        ax.add_patch(Circle((cx + 0.06, cy + 0.01), 0.05, facecolor="#bfbfbf", edgecolor="none", zorder=11))
        ax.add_patch(Circle((cx, cy - 0.08), 0.035, facecolor="#bfbfbf", edgecolor="none", zorder=11))
        return

    crop, prompt = example
    extent = (x + 0.175, x + 0.175 + w, y + 0.140, y + 0.140 + h)
    ax.imshow(crop, cmap="gray", extent=extent, origin="lower", vmin=0, vmax=1, zorder=14)
    overlay = np.ma.masked_where(~prompt, prompt)
    ax.imshow(overlay, cmap="cool", alpha=0.90, extent=extent, origin="lower", interpolation="nearest", zorder=15)
    ax.add_patch(
        Rectangle(
            (extent[0], extent[2]),
            w,
            h,
            facecolor="none",
            edgecolor="black",
            linewidth=0.55,
            zorder=16,
        )
    )



# -----------------------------------------------------------------------------
# Panel A: U-Net architecture
# -----------------------------------------------------------------------------
def draw_architecture_panel(ax):
    """Draw the high-level U-Net ladder.

    The decoder stream is deliberately routed directly into each decoder stage.
    The plus symbols annotate the lateral additive skip inputs only, which keeps
    the visual interpretation simple: "upsampled decoder feature + encoder skip"
    becomes the input to the following decoder block.
    """
    add_panel_frame(ax, 0.15, 5.15, 15.7, 5.15, None, None)

    # Input and stem
    draw_input_volume_icon(ax, 0.38, 8.28, 0.88, 0.88)
    add_text(ax, 0.88, 8.05, "Input\nvolume", size=9, weight="bold")
    add_arrow(ax, (1.48, 8.85), (1.72, 8.85), lw=1.2)
    add_round_box(ax, 1.77, 8.58, 0.72, 0.54, "1×1\nConv", fc="white", ec="#333333", fontsize=8.5)

    # Encoder coordinates
    stage_x, stage_w, stage_h = 3.05, 1.92, 0.56
    stage_centers = [9.90, 8.85, 7.80, 6.75, 5.70]
    down_h = 0.40
    stage_labels = [
        ("Stage 0", "C=32, depth=2"),
        ("Stage 1", "C=64, depth=2"),
        ("Stage 2", "C=128, depth=4"),
        ("Stage 3", "C=256, depth=4"),
        ("Stage 4", "C=512, depth=2"),
    ]

    # Stem route into Stage 0
    add_polyline_arrow(ax, [(2.49, 8.85), (2.72, 8.85), (2.72, 9.90), (3.05, 9.90)], lw=1.2)

    # Encoder stages + down blocks
    for i, (cy, (name, info)) in enumerate(zip(stage_centers, stage_labels)):
        add_round_box(
            ax,
            stage_x,
            cy - stage_h / 2,
            stage_w,
            stage_h,
            f"{name}\n{info}",
            fc=COLORS["encoder_fill"],
            ec=COLORS["encoder_edge"],
            lw=1.2,
            fontsize=9,
            weight="bold",
            color="#073b84",
        )
        if i < len(stage_centers) - 1:
            next_cy = stage_centers[i + 1]
            down_cy = (cy + next_cy) / 2
            add_arrow(ax, (stage_x + stage_w / 2, cy - stage_h / 2), (stage_x + stage_w / 2, down_cy + 0.24), lw=1.0)
            add_round_box(
                ax,
                stage_x + 0.18,
                down_cy - 0.24,
                stage_w - 0.36,
                0.48,
                "DownBlock3D,\nstride 2",
                fc="white",
                ec="#444444",
                lw=0.9,
                ls="--",
                fontsize=7.8,
            )
            add_arrow(ax, (stage_x + stage_w / 2, down_cy - down_h / 2), (stage_x + stage_w / 2, next_cy + stage_h / 2), lw=1.0)

    # Decoder coordinates
    dec_x, dec_w, dec_h = 10.35, 1.78, 0.56
    dec_center_x = dec_x + dec_w / 2
    plus_x = dec_x - 0.42
    decoder_labels = [
        ("Dec0", "C=32, depth=2"),
        ("Dec1", "C=64, depth=2"),
        ("Dec2", "C=128, depth=4"),
        ("Dec3", "C=256, depth=4"),
    ]
    decoder_centers = stage_centers[:4]  # y positions matching Stage 0-3

    # Decoder stage boxes first, so arrows can terminate cleanly at their edges.
    for cy, (name, info) in zip(decoder_centers, decoder_labels):
        add_round_box(
            ax,
            dec_x,
            cy - dec_h / 2,
            dec_w,
            dec_h,
            f"{name}\n{info}",
            fc=COLORS["decoder_fill"],
            ec=COLORS["decoder_edge"],
            lw=1.2,
            fontsize=9,
            weight="bold",
            color="#153b1d",
        )

    # Lateral additive skip inputs. These are drawn separately from the decoder stream.
    for cy in decoder_centers:
        add_arrow(ax, (stage_x + stage_w, cy), (plus_x - 0.17, cy), lw=1.15, mutation_scale=8)
        add_plus(ax, plus_x, cy, r=0.17)
        add_arrow(ax, (plus_x + 0.17, cy), (dec_x, cy), lw=1.15, mutation_scale=8)

    # Bottleneck upsample: Stage 4 -> UpBlock -> directly into Dec3.
    bottom_up_x, bottom_up_y, bottom_up_w, bottom_up_h = 8.65, 5.54, 1.60, 0.36
    add_arrow(ax, (stage_x + stage_w, stage_centers[4]), (bottom_up_x, stage_centers[4]), lw=1.1, mutation_scale=8)
    add_round_box(
        ax,
        bottom_up_x,
        bottom_up_y,
        bottom_up_w,
        bottom_up_h,
        "UpBlock3D ×2",
        fc="white",
        ec=COLORS["decoder_edge"],
        lw=1.0,
        ls="--",
        fontsize=8,
    )
    add_polyline_arrow(
        ax,
        [
            (bottom_up_x + bottom_up_w, bottom_up_y + bottom_up_h / 2),
            (dec_center_x, bottom_up_y + bottom_up_h / 2),
            (dec_center_x, stage_centers[3] - dec_h / 2),
        ],
        lw=1.05,
        mutation_scale=8,
    )

    # Up blocks between decoder levels: Dec3->Dec2, Dec2->Dec1, Dec1->Dec0.
    # These connectors now enter the next decoder box directly, rather than the '+' skip node.
    for lower_y, upper_y in zip(
        [stage_centers[3], stage_centers[2], stage_centers[1]],
        [stage_centers[2], stage_centers[1], stage_centers[0]],
    ):
        up_cy = (lower_y + upper_y) / 2
        up_h = 0.36
        up_w = dec_w - 0.14
        up_x = dec_x + 0.07
        add_arrow(ax, (dec_center_x, lower_y + dec_h / 2), (dec_center_x, up_cy - up_h / 2), lw=1.0, mutation_scale=8)
        add_round_box(
            ax,
            up_x,
            up_cy - up_h / 2,
            up_w,
            up_h,
            "UpBlock3D ×2",
            fc="white",
            ec=COLORS["decoder_edge"],
            lw=1.0,
            ls="--",
            fontsize=8,
        )
        add_arrow(ax, (dec_center_x, up_cy + up_h / 2), (dec_center_x, upper_y - dec_h / 2), lw=1.0, mutation_scale=8)

    # Final head
    add_arrow(ax, (dec_x + dec_w, stage_centers[0]), (12.50, stage_centers[0]), lw=1.1)
    add_round_box(ax, 12.58, stage_centers[0] - 0.28, 0.80, 0.56, "1×1\nConv", fc="white", ec="#333333", fontsize=8.5)
    add_arrow(ax, (13.38, stage_centers[0]), (13.92, stage_centers[0]), lw=1.1)
    ax.text(14.02, stage_centers[0], "Segmentation\nlogits", ha="left", va="center", fontsize=9.5, fontweight="bold")

    # Legend
    legend_x, legend_y, legend_w, legend_h = 12.82, 5.78, 2.35, 1.88
    add_round_box(
        ax,
        legend_x,
        legend_y,
        legend_w,
        legend_h,
        "",
        fc="white",
        ec="#666666",
        lw=0.9,
        ls=(0, (2, 2)),
        rounding=0.06,
        zorder=1,
    )
    y0 = legend_y + legend_h - 0.35
    add_round_box(ax, legend_x + 0.22, y0 - 0.13, 0.40, 0.26, "", fc=COLORS["encoder_fill"], ec=COLORS["encoder_edge"], lw=1)
    ax.text(legend_x + 0.80, y0, "Encoder (stages)", ha="left", va="center", fontsize=8)
    y1 = y0 - 0.42
    add_round_box(ax, legend_x + 0.22, y1 - 0.13, 0.40, 0.26, "", fc=COLORS["decoder_fill"], ec=COLORS["decoder_edge"], lw=1)
    ax.text(legend_x + 0.80, y1, "Decoder (stages)", ha="left", va="center", fontsize=8)
    y2 = y1 - 0.42
    add_round_box(ax, legend_x + 0.22, y2 - 0.13, 0.40, 0.26, "", fc="white", ec="#444444", lw=0.9, ls="--")
    ax.text(legend_x + 0.80, y2, "Down/Up blocks", ha="left", va="center", fontsize=8)
    y3 = y2 - 0.45
    add_plus(ax, legend_x + 0.42, y3, r=0.14)
    ax.text(legend_x + 0.80, y3, "Additive skip\ninput", ha="left", va="center", fontsize=8)


# -----------------------------------------------------------------------------
# Panel B: core blocks
# -----------------------------------------------------------------------------
def draw_core_block_panel(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    blocks: Sequence[tuple[str, str, str]],
    *,
    edge_color: str,
    residual_label: str | None,
    note: str | None = None,
    title_fontsize: float = 8.2,
    note_fontsize: float = 6.5,
    block_fontsize: float = 6.4,
    endpoint_fontsize: float = 6.8,
    branch_fontsize: float = 6.0,
):
    """Draw one of the compact core-block diagrams."""
    add_round_box(ax, x, y, w, h, "", fc="#fbfdff", ec=edge_color, lw=0.8, rounding=0.05, zorder=1)
    ax.text(x + 0.15, y + h - 0.22, title, ha="left", va="center", fontsize=title_fontsize, fontweight="bold", color=edge_color)
    if note:
        ax.text(x + 0.15, y + h - 0.48, note, ha="left", va="center", fontsize=note_fontsize, color="#444444")

    mid_y = y + 1.12
    block_h = 0.68
    plus_x = x + w - 0.70
    output_anchor_x = x + w - 0.20
    input_anchor_x = x + 0.30
    first_block_x = x + 0.78
    if w > 8:
        plus_x = x + w - 1.12
        output_anchor_x = x + w - 0.70
        input_anchor_x = x + 0.42
        first_block_x = x + 1.05

    ax.text(input_anchor_x, mid_y + 0.28, "Input", ha="center", va="center", fontsize=endpoint_fontsize)

    # One clean input arrow feeds the main path.
    # A junction dot marks where the same input also branches to the residual path.
    # This avoids the visually confusing "two arrowheads before the first block" effect.
    split_x = first_block_x - 0.16
    add_arrow(ax, (input_anchor_x, mid_y), (first_block_x - 0.06, mid_y), lw=1.0, mutation_scale=8)
    add_junction(ax, split_x, mid_y, r=0.028)

    # Compact, readable block widths. Detailed dimensions are kept in the code/caption,
    # not inside every box.
    n = len(blocks)
    box_gap = 0.10
    usable = plus_x - first_block_x - 0.18
    widths = []
    for label, _, _ in blocks:
        if "Depthwise" in label:
            widths.append(0.72)
        elif "Compress" in label or "Expand" in label:
            widths.append(0.72)
        elif "Instance" in label:
            widths.append(0.66)
        else:
            widths.append(0.52)
    max_scale = 1.85 if w > 8 else 1.0
    scale = min(max_scale, (usable - box_gap * (n - 1)) / sum(widths))
    widths = [ww * scale for ww in widths]

    cur_x = first_block_x
    previous_right = cur_x
    for i, ((label, fc, ec), bw) in enumerate(zip(blocks, widths)):
        if i > 0:
            add_arrow(ax, (previous_right, mid_y), (cur_x, mid_y), lw=0.9, mutation_scale=7)
        add_round_box(
            ax,
            cur_x,
            mid_y - block_h / 2,
            bw,
            block_h,
            label,
            fc=fc,
            ec=ec,
            lw=0.8,
            fontsize=block_fontsize,
            rounding=0.035,
        )
        previous_right = cur_x + bw
        cur_x = previous_right + box_gap

    add_arrow(ax, (previous_right, mid_y), (plus_x - 0.14, mid_y), lw=0.9, mutation_scale=7)
    add_plus(ax, plus_x, mid_y, r=0.13)
    add_arrow(ax, (plus_x + 0.13, mid_y), (output_anchor_x - 0.08, mid_y), lw=1.0, mutation_scale=8)
    ax.text(output_anchor_x, mid_y + 0.28, "Output", ha="center", va="center", fontsize=endpoint_fontsize)

    # Residual branch.
    res_y = y + 0.42
    if residual_label is None:
        # Identity residual: input routes below all internal operations.
        ax.add_line(Line2D([split_x, split_x], [mid_y, res_y], linewidth=1.0, color=COLORS["arrow"], zorder=2))
        ax.add_line(Line2D([split_x, plus_x], [res_y, res_y], linewidth=1.0, color=COLORS["arrow"], zorder=2))
        add_arrow(ax, (plus_x, res_y), (plus_x, mid_y - 0.13), lw=1.0, mutation_scale=7)
    else:
        branch_w, branch_h = 1.22, 0.40
        if "Upsample" in residual_label:
            branch_w = 1.75
        branch_x = x + w / 2 - branch_w / 2
        ax.add_line(Line2D([split_x, split_x], [mid_y, res_y], linewidth=1.0, color=COLORS["arrow"], zorder=2))
        add_arrow(ax, (split_x, res_y), (branch_x, res_y), lw=1.0, mutation_scale=7)
        add_round_box(
            ax,
            branch_x,
            res_y - branch_h / 2,
            branch_w,
            branch_h,
            residual_label,
            fc=COLORS["residual_fill"],
            ec="#555555",
            lw=0.8,
            fontsize=branch_fontsize,
            rounding=0.035,
        )
        ax.add_line(Line2D([branch_x + branch_w, plus_x], [res_y, res_y], linewidth=1.0, color=COLORS["arrow"], zorder=2))
        add_arrow(ax, (plus_x, res_y), (plus_x, mid_y - 0.13), lw=1.0, mutation_scale=7)

def core_block_definitions() -> tuple[
    list[tuple[str, str, str]],
    list[tuple[str, str, str]],
    list[tuple[str, str, str]],
]:
    """Return shared block operation labels and colors."""
    base_blocks = [
        ("Depthwise\nConv3D", COLORS["conv_fill"], COLORS["encoder_edge"]),
        ("Instance\nNorm3D", COLORS["norm_fill"], COLORS["encoder_edge"]),
        ("Expand\n1×1", COLORS["conv_fill"], COLORS["encoder_edge"]),
        ("GELU", COLORS["gelu_fill"], "#d59b2d"),
        ("GRN3d", COLORS["grn_fill"], "#8a61b0"),
        ("Compress\n1×1", COLORS["conv_fill"], COLORS["encoder_edge"]),
    ]
    down_blocks = [
        ("Depthwise\nConv3D", COLORS["conv_fill"], COLORS["encoder_edge"]),
        ("Instance\nNorm3D", COLORS["norm_fill"], COLORS["encoder_edge"]),
        ("Expand\n1×1", COLORS["conv_fill"], COLORS["encoder_edge"]),
        ("GELU", COLORS["gelu_fill"], "#d59b2d"),
        ("GRN3d", COLORS["grn_fill"], "#8a61b0"),
        ("Compress\n1×1", COLORS["conv_fill"], COLORS["encoder_edge"]),
    ]
    up_blocks = [
        ("Depthwise\nConvT3D", COLORS["conv_green_fill"], COLORS["decoder_edge"]),
        ("Instance\nNorm3D", COLORS["conv_green_fill"], COLORS["decoder_edge"]),
        ("Expand\n1×1", COLORS["conv_green_fill"], COLORS["decoder_edge"]),
        ("GELU", COLORS["gelu_fill"], "#d59b2d"),
        ("GRN3d", COLORS["grn_fill"], "#8a61b0"),
        ("Compress\n1×1", COLORS["conv_green_fill"], COLORS["decoder_edge"]),
    ]
    return base_blocks, down_blocks, up_blocks


def draw_core_blocks_panel(ax):
    add_panel_frame(ax, 0.15, 0.20, 15.7, 4.75, None, None)

    # Compact labels work better in the figure. The exact kernel, group, and channel
    # dimensions are better placed in the methods text or caption.
    base_blocks, down_blocks, up_blocks = core_block_definitions()

    panel_y, panel_h = 1.55, 2.66
    draw_core_block_panel(
        ax,
        0.35,
        panel_y,
        5.05,
        panel_h,
        "1) MedNeXtV2Block3D",
        base_blocks,
        edge_color=COLORS["encoder_edge"],
        residual_label=None,
        note="k=3, groups=C; expansion ratio=2",
    )
    draw_core_block_panel(
        ax,
        5.55,
        panel_y,
        5.05,
        panel_h,
        "2) DownBlock3D",
        down_blocks,
        edge_color=COLORS["encoder_edge"],
        residual_label="1×1 Conv\nstride 2",
        note="main and residual branches downsample ×2",
    )
    draw_core_block_panel(
        ax,
        10.75,
        panel_y,
        4.95,
        panel_h,
        "3) UpBlock3D",
        up_blocks,
        edge_color=COLORS["decoder_edge"],
        residual_label="Upsample ×2\n+ 1×1 Conv",
        note="ConvT3D in main branch upsamples ×2",
    )

    # GRN inset/callout
    callout_x, callout_y, callout_w, callout_h = 5.55, 0.43, 4.95, 0.88
    add_round_box(
        ax,
        callout_x,
        callout_y,
        callout_w,
        callout_h,
        "",
        fc="white",
        ec="#777777",
        lw=0.8,
        ls=(0, (2, 2)),
        rounding=0.045,
        zorder=1,
    )
    add_round_box(
        ax,
        callout_x + 0.12,
        callout_y + 0.24,
        0.58,
        0.42,
        "GRN3d",
        fc=COLORS["grn_fill"],
        ec="#8a61b0",
        lw=0.8,
        fontsize=7,
        rounding=0.035,
    )
    ax.text(
        callout_x + 0.82,
        callout_y + 0.67,
        "Global Response Normalization",
        ha="left",
        va="center",
        fontsize=7.2,
        fontweight="bold",
    )
    ax.text(
        callout_x + 0.82,
        callout_y + 0.48,
        r"$g_c=\|X_c\|_2$ over $(D,H,W)$;  $n_c=g_c/(\mathrm{mean}_j\,g_j+\epsilon)$",
        ha="left",
        va="center",
        fontsize=6.4,
    )
    ax.text(
        callout_x + 0.82,
        callout_y + 0.28,
        r"$\mathrm{GRN}(X)=\gamma\odot(X\odot n)+\beta+X$",
        ha="left",
        va="center",
        fontsize=6.8,
    )


def draw_core_blocks_panel_stacked(ax):
    """Draw a less panoramic block-detail figure with larger labels."""
    add_panel_frame(ax, 0.15, 1.05, 11.25, 8.35, None, None)
    base_blocks, down_blocks, up_blocks = core_block_definitions()

    rows = [
        (
            7.02,
            "1) MedNeXtV2Block3D",
            base_blocks,
            COLORS["encoder_edge"],
            None,
            None,
        ),
        (
            4.42,
            "2) DownBlock3D",
            down_blocks,
            COLORS["encoder_edge"],
            "1×1 Conv\nstride 2",
            None,
        ),
        (
            1.82,
            "3) UpBlock3D",
            up_blocks,
            COLORS["decoder_edge"],
            "Upsample ×2\n+ 1×1 Conv",
            None,
        ),
    ]
    for y, title, blocks, edge_color, residual_label, note in rows:
        draw_core_block_panel(
            ax,
            0.45,
            y,
            10.65,
            2.10,
            title,
            blocks,
            edge_color=edge_color,
            residual_label=residual_label,
            note=note,
            title_fontsize=11,
            note_fontsize=8.7,
            block_fontsize=8.6,
            endpoint_fontsize=9.0,
            branch_fontsize=8.0,
        )

# -----------------------------------------------------------------------------
# Public render functions
# -----------------------------------------------------------------------------
def save_figure(
    fig,
    output_stem: str | Path,
    *,
    save_png: bool = True,
    save_svg: bool = True,
    save_pdf: bool = True,
):
    """Save the current figure in the requested publication-friendly formats."""
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    if save_png:
        fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    if save_svg:
        fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    if save_pdf:
        fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def render_architecture_figure(
    output_stem: str | Path = "mednext_architecture",
    *,
    save_png: bool = True,
    save_svg: bool = True,
    save_pdf: bool = True,
    show: bool = False,
):
    """Render the high-level architecture figure.

    Parameters
    ----------
    output_stem:
        Path without extension. `.png`, `.svg`, and/or `.pdf` will be appended.
    save_png, save_svg, save_pdf:
        Toggle export formats.
    show:
        If True, display the figure in an interactive session.
    """
    fig, ax = plt.subplots(figsize=(13.2, 6.2), constrained_layout=False)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 16)
    ax.set_ylim(5.0, 11.0)
    ax.axis("off")

    draw_architecture_panel(ax)

    save_figure(fig, output_stem, save_png=save_png, save_svg=save_svg, save_pdf=save_pdf)
    if show:
        plt.show()
    plt.close(fig)


def render_blocks_figure(
    output_stem: str | Path = "mednext_blocks",
    *,
    save_png: bool = True,
    save_svg: bool = True,
    save_pdf: bool = True,
    show: bool = False,
):
    """Render the MedNextv2-f32 core block details as a separate figure."""
    fig, ax = plt.subplots(figsize=(8.2, 7.25), constrained_layout=False)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 11.6)
    ax.set_ylim(1.0, 10.0)
    ax.axis("off")

    draw_core_blocks_panel_stacked(ax)

    save_figure(fig, output_stem, save_png=save_png, save_svg=save_svg, save_pdf=save_pdf)
    if show:
        plt.show()
    plt.close(fig)


def render_mednextv2_figures(output_dir: str | Path):
    """Render both paper figures into the given directory."""
    output_dir = Path(output_dir)
    render_architecture_figure(output_dir / "mednext_architecture")
    render_blocks_figure(output_dir / "mednext_blocks")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    render_mednextv2_figures(repo_root / "figures")
