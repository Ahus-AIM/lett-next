from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PromptChannels:
    line: np.ndarray
    endpoints: np.ndarray
    endpoints_xyz: np.ndarray


def xyz_to_zyx(point_xyz: np.ndarray) -> np.ndarray:
    return np.asarray(point_xyz, dtype=np.float32)[::-1]


def zyx_to_xyz(point_zyx: np.ndarray) -> np.ndarray:
    return np.asarray(point_zyx, dtype=np.float32)[::-1]


def _clip_zyx(shape_zyx: tuple[int, int, int], point_zyx: np.ndarray) -> np.ndarray:
    point_zyx = np.asarray(point_zyx, dtype=np.float32)
    upper = np.asarray(shape_zyx, dtype=np.float32) - 1.0
    return np.clip(point_zyx, 0.0, upper)


def rasterize_line_mask(
    shape_zyx: tuple[int, int, int],
    endpoints_xyz: np.ndarray,
    radius_xy: int = 2,
    z_thickness: int = 0,
) -> np.ndarray:
    endpoints_xyz = np.asarray(endpoints_xyz, dtype=np.float32)
    start_zyx = _clip_zyx(shape_zyx, xyz_to_zyx(endpoints_xyz[0]))
    end_zyx = _clip_zyx(shape_zyx, xyz_to_zyx(endpoints_xyz[1]))

    # Force to a single RECIST slice unless you explicitly want a 3D tube
    z = int(round((start_zyx[0] + end_zyx[0]) / 2.0))
    start_yx = start_zyx[1:]
    end_yx = end_zyx[1:]

    distance = float(np.linalg.norm(end_yx - start_yx))
    steps = max(int(np.ceil(distance * 2.0)), 1)

    line_mask = np.zeros(shape_zyx, dtype=np.float32)

    yy, xx = np.ogrid[-radius_xy : radius_xy + 1, -radius_xy : radius_xy + 1]
    disk = (yy**2 + xx**2) <= radius_xy**2

    for alpha in np.linspace(0.0, 1.0, steps + 1, dtype=np.float32):
        point_yx = start_yx + alpha * (end_yx - start_yx)
        cy, cx = np.round(point_yx).astype(int)

        for zz in range(
            max(0, z - z_thickness), min(shape_zyx[0], z + z_thickness + 1)
        ):
            y0 = max(cy - radius_xy, 0)
            y1 = min(cy + radius_xy + 1, shape_zyx[1])
            x0 = max(cx - radius_xy, 0)
            x1 = min(cx + radius_xy + 1, shape_zyx[2])

            ky0 = y0 - (cy - radius_xy)
            ky1 = ky0 + (y1 - y0)
            kx0 = x0 - (cx - radius_xy)
            kx1 = kx0 + (x1 - x0)

            line_mask[zz, y0:y1, x0:x1] = np.maximum(
                line_mask[zz, y0:y1, x0:x1],
                disk[ky0:ky1, kx0:kx1].astype(np.float32),
            )

    return line_mask


def build_endpoint_gaussian_channel(
    shape_zyx: tuple[int, int, int],
    endpoints_xyz: np.ndarray,
    sigma: float,
) -> np.ndarray:
    grid_z, grid_y, grid_x = np.indices(shape_zyx, dtype=np.float32)
    channel = np.zeros(shape_zyx, dtype=np.float32)
    sigma_sq = max(float(sigma) ** 2, 1e-6)
    for endpoint_xyz in np.asarray(endpoints_xyz, dtype=np.float32):
        center_zyx = _clip_zyx(shape_zyx, xyz_to_zyx(endpoint_xyz))
        squared_distance = (
            (grid_z - center_zyx[0]) ** 2
            + (grid_y - center_zyx[1]) ** 2
            + (grid_x - center_zyx[2]) ** 2
        )
        channel = np.maximum(channel, np.exp(-squared_distance / (2.0 * sigma_sq)))
    return channel.astype(np.float32)


def recist_single_slice_delta(endpoints_xyz: np.ndarray) -> float:
    endpoints_xyz = np.asarray(endpoints_xyz, dtype=np.float32)
    start_zyx = xyz_to_zyx(endpoints_xyz[0])
    end_zyx = xyz_to_zyx(endpoints_xyz[1])
    return abs(float(start_zyx[0]) - float(end_zyx[0]))


def make_wrong_prompt(
    shape_zyx: tuple[int, int, int],
    endpoints_xyz: np.ndarray,
) -> np.ndarray:
    endpoints_xyz = np.asarray(endpoints_xyz, dtype=np.float32)
    offset_xyz = np.asarray(
        [
            max(shape_zyx[2] // 4, 2),
            max(shape_zyx[1] // 5, 2),
            max(shape_zyx[0] // 6, 1),
        ],
        dtype=np.float32,
    )
    shifted = endpoints_xyz + offset_xyz
    max_xyz = np.asarray(
        [shape_zyx[2] - 1, shape_zyx[1] - 1, shape_zyx[0] - 1], dtype=np.float32
    )
    return np.clip(shifted, 0.0, max_xyz)


def build_prompt_channels(
    shape_zyx: tuple[int, int, int],
    endpoints_xyz: np.ndarray,
    sigma: float,
    prompt_mode: str = "full",
    case_id: str | None = None,
    prompt_source: str | None = None,
) -> PromptChannels:
    endpoints_xyz = np.asarray(endpoints_xyz, dtype=np.float32)
    effective_endpoints = endpoints_xyz.copy()

    if prompt_mode == "wrong_prompt":
        effective_endpoints = make_wrong_prompt(shape_zyx, endpoints_xyz)

    z_delta = recist_single_slice_delta(effective_endpoints)
    if z_delta > 0.5:
        print(
            "[prompt warning] multi-slice RECIST-like endpoints "
            f"case={case_id or 'unknown'} "
            f"source={prompt_source or 'unknown'} "
            f"z_delta={z_delta:.3f} "
            f"start_zyx={xyz_to_zyx(effective_endpoints[0]).tolist()} "
            f"end_zyx={xyz_to_zyx(effective_endpoints[1]).tolist()}"
        )

    line = rasterize_line_mask(shape_zyx, effective_endpoints)
    endpoints = build_endpoint_gaussian_channel(
        shape_zyx, effective_endpoints, sigma=sigma
    )

    if prompt_mode == "no_prompt":
        line.fill(0.0)
        endpoints.fill(0.0)

    elif prompt_mode == "line_only":
        endpoints.fill(0.0)

    elif prompt_mode == "endpoint_only":
        line.fill(0.0)

    return PromptChannels(
        line=line,
        endpoints=endpoints,
        endpoints_xyz=effective_endpoints,
    )
