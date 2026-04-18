#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .config import PlotContext
from .fields import get_var_field, nearest_index
from .io import LoadedSnapshot


def axis_edges_from_centers(vec: np.ndarray) -> np.ndarray:
    if len(vec) == 1:
        return np.array([vec[0] - 0.5, vec[0] + 0.5], dtype=float)

    edges = np.empty(len(vec) + 1, dtype=float)
    edges[1:-1] = 0.5 * (vec[:-1] + vec[1:])
    edges[0] = vec[0] - 0.5 * (vec[1] - vec[0])
    edges[-1] = vec[-1] + 0.5 * (vec[-1] - vec[-2])
    return edges


def box_aspect_from_axes(x_vec: np.ndarray, y_vec: np.ndarray) -> float:
    dx = abs(x_vec[-1] - x_vec[0]) if len(x_vec) > 1 else 1.0
    dy = abs(y_vec[-1] - y_vec[0]) if len(y_vec) > 1 else 1.0

    if dx == 0.0:
        dx = 1.0
    if dy == 0.0:
        dy = 1.0

    return dy / dx


def figsize_from_box_aspect(box_aspect: float, base_width: float = 8.0) -> tuple[float, float]:
    height = base_width * box_aspect
    height = max(2.5, min(height, 10.0))
    return base_width, height


def contour_levels_from_field(field: np.ndarray, nlevels: Optional[int]) -> Optional[np.ndarray]:
    if nlevels is None or nlevels <= 0:
        return None

    finite = field[np.isfinite(field)]
    if finite.size == 0:
        return None

    fmin = np.min(finite)
    fmax = np.max(finite)

    if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin:
        return None

    return np.linspace(fmin, fmax, nlevels)


def default_output_path(snapshot_path: Path, var: int) -> Path:
    return snapshot_path.with_name(f"{snapshot_path.stem}_v{var}.png")


def build_axes_3d(ctx: PlotContext,
                  shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]:
    nz, ny, nx = shape
    ng = ctx.meta.ng if (ctx.meta.include_ghosts_default and not ctx.args.trim_ghosts) else 0

    if ctx.args.cell_axis:
        x_vec = np.arange(-ng, ctx.meta.nx + ng, dtype=float) if nx == ctx.meta.nx + 2 * ng else np.arange(nx, dtype=float)
        y_vec = np.arange(-ng, ctx.meta.ny + ng, dtype=float) if ny == ctx.meta.ny + 2 * ng else np.arange(ny, dtype=float)
        z_vec = np.arange(-ng, ctx.meta.nz + ng, dtype=float) if nz == ctx.meta.nz + 2 * ng else np.arange(nz, dtype=float)
        return x_vec, y_vec, z_vec, "i", "j", "k"

    if ctx.meta.x_min is not None and ctx.meta.x_max is not None:
        if nx == ctx.meta.nx + 2 * ng and ng > 0:
            x0 = ctx.meta.x_min - ng * ctx.meta.dx
            x1 = ctx.meta.x_max + ng * ctx.meta.dx
            x_vec = np.linspace(x0, x1, nx) if nx > 1 else np.array([0.5 * (x0 + x1)])
        else:
            x_vec = np.linspace(ctx.meta.x_min, ctx.meta.x_max, nx) if nx > 1 else np.array([0.5 * (ctx.meta.x_min + ctx.meta.x_max)])
        xlabel = "x"
    else:
        x_vec = np.arange(nx, dtype=float)
        xlabel = "i"

    dy = ctx.meta.dy if ctx.meta.dy is not None else 1.0
    if ctx.meta.y_min is not None and ctx.meta.y_max is not None:
        if ny == ctx.meta.ny + 2 * ng and ng > 0:
            y0 = ctx.meta.y_min - ng * dy
            y1 = ctx.meta.y_max + ng * dy
            y_vec = np.linspace(y0, y1, ny) if ny > 1 else np.array([0.5 * (y0 + y1)])
        else:
            y_vec = np.linspace(ctx.meta.y_min, ctx.meta.y_max, ny) if ny > 1 else np.array([0.5 * (ctx.meta.y_min + ctx.meta.y_max)])
        ylabel = "y"
    else:
        y_vec = np.arange(ny, dtype=float)
        ylabel = "j"

    dz = ctx.meta.dz if ctx.meta.dz is not None else 1.0
    if ctx.meta.z_min is not None and ctx.meta.z_max is not None:
        if nz == ctx.meta.nz + 2 * ng and ng > 0:
            z0 = ctx.meta.z_min - ng * dz
            z1 = ctx.meta.z_max + ng * dz
            z_vec = np.linspace(z0, z1, nz) if nz > 1 else np.array([0.5 * (z0 + z1)])
        else:
            z_vec = np.linspace(ctx.meta.z_min, ctx.meta.z_max, nz) if nz > 1 else np.array([0.5 * (ctx.meta.z_min + ctx.meta.z_max)])
        zlabel = "z"
    else:
        z_vec = np.arange(nz, dtype=float)
        zlabel = "k"

    return x_vec, y_vec, z_vec, xlabel, ylabel, zlabel


def build_title_3d(ctx: PlotContext,
                   snapshot: LoadedSnapshot,
                   var: int,
                   plane: str,
                   fixed_lab: str,
                   fixed_val: float) -> str:
    if ctx.args.title:
        return ctx.args.title
    if snapshot.time is not None:
        return f"step {snapshot.step}  t={snapshot.time:.6g}  v{var}  {plane}@{fixed_lab}={fixed_val:.6g}"
    if snapshot.step >= 0:
        return f"step {snapshot.step}  v{var}  {plane}@{fixed_lab}={fixed_val:.6g}"
    return f"v{var}  {plane}@{fixed_lab}={fixed_val:.6g}"


def plot_snapshot_3d(ctx: PlotContext,
                     snapshot: LoadedSnapshot,
                     save: bool = False,
                     show: bool = True) -> Optional[Path]:
    field = get_var_field(snapshot, ctx.args.var)
    if field.ndim != 3:
        raise ValueError(f"plot_snapshot_3d expected a 3D field, got shape {field.shape}")

    x_vec, y_vec, z_vec, xlab, ylab, zlab = build_axes_3d(ctx, field.shape)

    plane = ctx.args.plane if ctx.args.plane is not None else "xy"

    if ctx.args.plane_value is not None:
        if plane == "xy":
            fixed = nearest_index(z_vec, ctx.args.plane_value)
        elif plane == "xz":
            fixed = nearest_index(y_vec, ctx.args.plane_value)
        else:
            fixed = nearest_index(x_vec, ctx.args.plane_value)
    else:
        if ctx.args.plane_index is not None:
            fixed = ctx.args.plane_index
        else:
            if plane == "xy":
                fixed = field.shape[0] // 2
            elif plane == "xz":
                fixed = field.shape[1] // 2
            else:
                fixed = field.shape[2] // 2

    if plane == "xy":
        fixed = max(0, min(field.shape[0] - 1, fixed))
        Z2 = field[fixed, :, :]
        ax_u, ax_v = x_vec, y_vec
        lab_u, lab_v = xlab, ylab
        fixed_lab = zlab
        fixed_val = z_vec[fixed]
    elif plane == "xz":
        fixed = max(0, min(field.shape[1] - 1, fixed))
        Z2 = field[:, fixed, :]
        ax_u, ax_v = x_vec, z_vec
        lab_u, lab_v = xlab, zlab
        fixed_lab = ylab
        fixed_val = y_vec[fixed]
    else:
        fixed = max(0, min(field.shape[2] - 1, fixed))
        Z2 = field[:, :, fixed]
        ax_u, ax_v = y_vec, z_vec
        lab_u, lab_v = ylab, zlab
        fixed_lab = xlab
        fixed_val = x_vec[fixed]

    if ctx.args.slice_axis is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        in_plane = set(plane)
        slice_axis = ctx.args.slice_axis

        if slice_axis not in in_plane:
            raise ValueError(f"slice-axis {slice_axis} is not in plane {plane}")

        other_axis = (in_plane - {slice_axis}).pop()
        axis_map = {"x": x_vec, "y": y_vec, "z": z_vec}

        if ctx.args.slice_value is not None:
            other_idx = nearest_index(axis_map[other_axis], ctx.args.slice_value)
        else:
            if ctx.args.slice_index is not None:
                other_idx = ctx.args.slice_index
            else:
                other_idx = len(axis_map[other_axis]) // 2

        other_idx = max(0, min(len(axis_map[other_axis]) - 1, other_idx))

        if plane == "xy":
            if slice_axis == "x":
                line = Z2[other_idx, :]
                ax.plot(x_vec, line)
                ax.set_xlabel(xlab)
                other_lab = ylab
                other_val = y_vec[other_idx]
            else:
                line = Z2[:, other_idx]
                ax.plot(y_vec, line)
                ax.set_xlabel(ylab)
                other_lab = xlab
                other_val = x_vec[other_idx]

        elif plane == "xz":
            if slice_axis == "x":
                line = Z2[other_idx, :]
                ax.plot(x_vec, line)
                ax.set_xlabel(xlab)
                other_lab = zlab
                other_val = z_vec[other_idx]
            else:
                line = Z2[:, other_idx]
                ax.plot(z_vec, line)
                ax.set_xlabel(zlab)
                other_lab = xlab
                other_val = x_vec[other_idx]

        else:
            if slice_axis == "y":
                line = Z2[other_idx, :]
                ax.plot(y_vec, line)
                ax.set_xlabel(ylab)
                other_lab = zlab
                other_val = z_vec[other_idx]
            else:
                line = Z2[:, other_idx]
                ax.plot(z_vec, line)
                ax.set_xlabel(zlab)
                other_lab = ylab
                other_val = y_vec[other_idx]

        ax.set_ylabel(f"v{ctx.args.var}")
        if ctx.args.title:
            ax.set_title(ctx.args.title)
        else:
            ax.set_title(
                f"v{ctx.args.var} slice in {plane} at "
                f"{fixed_lab}={fixed_val:.6g}, {other_lab}={other_val:.6g}"
            )

    else:
        box_aspect = box_aspect_from_axes(ax_u, ax_v)
        fig, ax = plt.subplots(figsize=figsize_from_box_aspect(box_aspect))

        u_edges = axis_edges_from_centers(ax_u)
        v_edges = axis_edges_from_centers(ax_v)

        pcm = ax.pcolormesh(u_edges, v_edges, Z2, shading="auto")
        fig.colorbar(pcm, ax=ax, label=f"v{ctx.args.var}")

        levels = contour_levels_from_field(Z2, ctx.args.contours)
        if levels is not None:
            ax.contour(ax_u, ax_v, Z2, levels=levels, colors="k", linewidths=0.5)

        ax.set_xlabel(lab_u)
        ax.set_ylabel(lab_v)
        ax.set_xlim(u_edges[0], u_edges[-1])
        ax.set_ylim(v_edges[0], v_edges[-1])
        ax.set_box_aspect(box_aspect)
        ax.set_title(build_title_3d(ctx, snapshot, ctx.args.var, plane, fixed_lab, fixed_val))

    plt.tight_layout()

    outpath: Optional[Path] = None
    if save:
        outpath = ctx.args.output if ctx.args.output is not None else default_output_path(
            snapshot.path, ctx.args.var
        )
        fig.savefig(outpath, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return outpath