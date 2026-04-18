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


def build_axes_2d(ctx: PlotContext, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, str, str]:
    ny, nx = shape
    ng = ctx.meta.ng if (ctx.meta.include_ghosts_default and not ctx.args.trim_ghosts) else 0

    if ctx.args.cell_axis:
        x_vec = np.arange(-ng, ctx.meta.nx + ng, dtype=float) if nx == ctx.meta.nx + 2 * ng else np.arange(nx, dtype=float)
        y_vec = np.arange(-ng, ctx.meta.ny + ng, dtype=float) if ny == ctx.meta.ny + 2 * ng else np.arange(ny, dtype=float)
        return x_vec, y_vec, "i", "j"

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

    if ctx.meta.y_min is not None and ctx.meta.y_max is not None:
        dy = ctx.meta.dy if ctx.meta.dy is not None else 1.0
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

    return x_vec, y_vec, xlabel, ylabel


def build_title_2d(ctx: PlotContext, snapshot: LoadedSnapshot, var: int) -> str:
    if ctx.args.title:
        return ctx.args.title
    if snapshot.time is not None:
        return f"step {snapshot.step}  t={snapshot.time:.6g}  v{var}"
    if snapshot.step >= 0:
        return f"step {snapshot.step}  v{var}"
    return f"v{var}"


def plot_snapshot_2d(ctx: PlotContext,
                     snapshot: LoadedSnapshot,
                     save: bool = False,
                     show: bool = True) -> Optional[Path]:
    field = get_var_field(snapshot, ctx.args.var)
    if field.ndim != 2:
        raise ValueError(f"plot_snapshot_2d expected a 2D field, got shape {field.shape}")

    x_vec, y_vec, xlab, ylab = build_axes_2d(ctx, field.shape)

    if ctx.args.slice_axis is not None:
        fig, ax = plt.subplots(figsize=(8, 4))

        if ctx.args.slice_axis == "z":
            raise ValueError("2D data has no z axis.")

        if ctx.args.slice_axis == "x":
            if ctx.args.slice_value is not None:
                j0 = nearest_index(y_vec, ctx.args.slice_value)
            else:
                j0 = ctx.args.slice_index if ctx.args.slice_index is not None else (field.shape[0] // 2)
            j0 = max(0, min(field.shape[0] - 1, j0))

            ax.plot(x_vec, field[j0, :])
            ax.set_xlabel(xlab)
            ax.set_ylabel(f"v{ctx.args.var}")
            if ctx.args.title:
                ax.set_title(ctx.args.title)
            else:
                ax.set_title(f"v{ctx.args.var} slice at {ylab}={y_vec[j0]:.6g}")

        else:
            if ctx.args.slice_value is not None:
                i0 = nearest_index(x_vec, ctx.args.slice_value)
            else:
                i0 = ctx.args.slice_index if ctx.args.slice_index is not None else (field.shape[1] // 2)
            i0 = max(0, min(field.shape[1] - 1, i0))

            ax.plot(y_vec, field[:, i0])
            ax.set_xlabel(ylab)
            ax.set_ylabel(f"v{ctx.args.var}")
            if ctx.args.title:
                ax.set_title(ctx.args.title)
            else:
                ax.set_title(f"v{ctx.args.var} slice at {xlab}={x_vec[i0]:.6g}")

    else:
        box_aspect = box_aspect_from_axes(x_vec, y_vec)
        fig, ax = plt.subplots(figsize=figsize_from_box_aspect(box_aspect))

        x_edges = axis_edges_from_centers(x_vec)
        y_edges = axis_edges_from_centers(y_vec)

        pcm = ax.pcolormesh(x_edges, y_edges, field, shading="auto")
        fig.colorbar(pcm, ax=ax, label=f"v{ctx.args.var}")

        levels = contour_levels_from_field(field, ctx.args.contours)
        if levels is not None:
            ax.contour(x_vec, y_vec, field, levels=levels, colors="k", linewidths=0.5)

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])
        ax.set_box_aspect(box_aspect)
        ax.set_title(build_title_2d(ctx, snapshot, ctx.args.var))

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