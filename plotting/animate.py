#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from .config import PlotContext
from .fields import get_var_field, nearest_index
from .io import LoadedSnapshot


def default_animation_output_path(snapshot_dir: Path, var: int) -> Path:
    return snapshot_dir / f"animation_v{var}.mp4"


def axis_edges_from_centers(vec: np.ndarray) -> np.ndarray:
    if len(vec) == 1:
        return np.array([vec[0] - 0.5, vec[0] + 0.5], dtype=float)

    edges = np.empty(len(vec) + 1, dtype=float)
    edges[1:-1] = 0.5 * (vec[:-1] + vec[1:])
    edges[0] = vec[0] - 0.5 * (vec[1] - vec[0])
    edges[-1] = vec[-1] + 0.5 * (vec[-1] - vec[-2])
    return edges


def build_axis_1d(ctx: PlotContext, n: int) -> tuple[np.ndarray, str]:
    nx = ctx.meta.nx
    ng = ctx.meta.ng
    dx = ctx.meta.dx
    x_min = ctx.meta.x_min
    x_max = ctx.meta.x_max

    if ctx.args.cell_axis:
        if n == nx:
            return np.arange(nx, dtype=float), "i"
        if n == nx + 2 * ng:
            return np.arange(-ng, nx + ng, dtype=float), "i"
        return np.arange(n, dtype=float), "i"

    if x_min is None or x_max is None:
        return np.arange(n, dtype=float), "i"

    if n == nx:
        if n == 1:
            return np.array([0.5 * (x_min + x_max)], dtype=float), "x"
        return np.linspace(x_min, x_max, n), "x"

    if n == nx + 2 * ng:
        x0 = x_min - ng * dx
        x1 = x_max + ng * dx
        if n == 1:
            return np.array([0.5 * (x0 + x1)], dtype=float), "x"
        return np.linspace(x0, x1, n), "x"

    if n == 1:
        return np.array([0.5 * (x_min + x_max)], dtype=float), "x"
    return np.linspace(x_min, x_max, n), "x"


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


def build_axes_3d(ctx: PlotContext, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]:
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


def extract_3d_plane(ctx: PlotContext, field: np.ndarray):
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

    return plane, Z2, ax_u, ax_v, lab_u, lab_v, fixed_lab, fixed_val


def animate_snapshots(ctx: PlotContext,
                      snapshots: list[LoadedSnapshot],
                      save: bool = False,
                      show: bool = True) -> Optional[Path]:
    if not snapshots:
        raise ValueError("No snapshots were loaded for animation.")

    dim = ctx.meta.dimension
    var = ctx.args.var
    interval_ms = 1000.0 / max(ctx.args.fps, 1)

    if dim == 1:
        first = get_var_field(snapshots[0], var)
        if first.ndim != 1:
            raise ValueError(f"Expected 1D field, got shape {first.shape}")

        x_vec, xlab = build_axis_1d(ctx, len(first))

        ymin = min(np.nanmin(get_var_field(s, var)*0.9) for s in snapshots)
        ymax = max(np.nanmax(get_var_field(s, var)*1.1 ) for s in snapshots)

        fig, ax = plt.subplots(figsize=(8, 4))
        (line,) = ax.plot(x_vec, first)
        ax.set_xlabel(xlab)
        ax.set_ylabel(f"v{var}")
        ax.set_ylim(ymin, ymax)

        def update(frame_idx: int):
            snap = snapshots[frame_idx]
            field = get_var_field(snap, var)
            line.set_ydata(field)
            if snap.time is not None:
                ax.set_title(f"step {snap.step}  t={snap.time:.6g}  v{var}")
            else:
                ax.set_title(f"step {snap.step}  v{var}")
            return (line,)

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(snapshots),
            interval=interval_ms,
            blit=False,
        )

    elif dim == 2:
        first = get_var_field(snapshots[0], var)
        if first.ndim != 2:
            raise ValueError(f"Expected 2D field, got shape {first.shape}")

        x_vec, y_vec, xlab, ylab = build_axes_2d(ctx, first.shape)
        x_edges = axis_edges_from_centers(x_vec)
        y_edges = axis_edges_from_centers(y_vec)

        vmin = min(np.nanmin(get_var_field(s, var)) for s in snapshots)
        vmax = max(np.nanmax(get_var_field(s, var)) for s in snapshots)

        fig, ax = plt.subplots(figsize=(8, 6))
        pcm = ax.pcolormesh(x_edges, y_edges, first, shading="auto", vmin=vmin, vmax=vmax)
        fig.colorbar(pcm, ax=ax, label=f"v{var}")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

        def update(frame_idx: int):
            snap = snapshots[frame_idx]
            field = get_var_field(snap, var)
            pcm.set_array(field.ravel())
            if snap.time is not None:
                ax.set_title(f"step {snap.step}  t={snap.time:.6g}  v{var}")
            else:
                ax.set_title(f"step {snap.step}  v{var}")
            return (pcm,)

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(snapshots),
            interval=interval_ms,
            blit=False,
        )

    elif dim == 3:
        first = get_var_field(snapshots[0], var)
        if first.ndim != 3:
            raise ValueError(f"Expected 3D field, got shape {first.shape}")

        plane, first2d, ax_u, ax_v, lab_u, lab_v, fixed_lab, fixed_val = extract_3d_plane(ctx, first)
        u_edges = axis_edges_from_centers(ax_u)
        v_edges = axis_edges_from_centers(ax_v)

        planes = [extract_3d_plane(ctx, get_var_field(s, var))[1] for s in snapshots]
        vmin = min(np.nanmin(p) for p in planes)
        vmax = max(np.nanmax(p) for p in planes)

        fig, ax = plt.subplots(figsize=(8, 6))
        pcm = ax.pcolormesh(u_edges, v_edges, first2d, shading="auto", vmin=vmin, vmax=vmax)
        fig.colorbar(pcm, ax=ax, label=f"v{var}")
        ax.set_xlabel(lab_u)
        ax.set_ylabel(lab_v)

        def update(frame_idx: int):
            snap = snapshots[frame_idx]
            plane_name, field2d, _, _, _, _, fixed_lab_now, fixed_val_now = extract_3d_plane(
                ctx, get_var_field(snap, var)
            )
            pcm.set_array(field2d.ravel())
            if snap.time is not None:
                ax.set_title(
                    f"step {snap.step}  t={snap.time:.6g}  v{var}  "
                    f"{plane_name}@{fixed_lab_now}={fixed_val_now:.6g}"
                )
            else:
                ax.set_title(
                    f"step {snap.step}  v{var}  "
                    f"{plane_name}@{fixed_lab_now}={fixed_val_now:.6g}"
                )
            return (pcm,)

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(snapshots),
            interval=interval_ms,
            blit=False,
        )

    else:
        raise ValueError(f"Unsupported dimension {dim}")

    plt.tight_layout()

    outpath: Optional[Path] = None
    if save:
        outpath = ctx.args.output if ctx.args.output is not None else default_animation_output_path(
            ctx.snapshot_dir, var
        )
        ani.save(outpath, fps=ctx.args.fps)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return outpath