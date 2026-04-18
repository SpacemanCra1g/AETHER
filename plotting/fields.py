#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import PlotContext
from .io import LoadedSnapshot


@dataclass(frozen=True)
class Field1D:
    values: np.ndarray
    x: np.ndarray
    xlabel: str
    ylabel: str


def nearest_index(vec: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(vec - value)))


def get_var_field(snapshot: LoadedSnapshot, var: int) -> np.ndarray:
    """
    Extract one variable from the loaded snapshot.

    Internal snapshot storage convention:
        data.shape == (numvar, nz, ny, nx)

    Returns:
        dim=1 -> shape (nx,)
        dim=2 -> shape (ny, nx)
        dim=3 -> shape (nz, ny, nx)
    """
    data = snapshot.data

    if data.ndim != 4:
        raise ValueError(
            f"Expected snapshot.data to have rank 4 as (numvar,nz,ny,nx), got shape {data.shape}"
        )

    if var < 0 or var >= data.shape[0]:
        raise IndexError(f"Variable index {var} out of bounds for numvar={data.shape[0]}")

    field = np.asarray(data[var])

    nz, ny, nx = field.shape

    if nz == 1 and ny == 1:
        return field[0, 0, :]
    if nz == 1:
        return field[0, :, :]
    return field


def build_axis_1d(ctx: PlotContext, n: int) -> tuple[np.ndarray, str]:
    """
    Build the 1D plotting axis from metadata.

    If ghost cells are present in the loaded field, extend the coordinate
    axis beyond the physical interior domain using dx and ng.
    """
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


def prepare_field_1d(ctx: PlotContext, snapshot: LoadedSnapshot, var: int) -> Field1D:
    field = get_var_field(snapshot, var)

    if field.ndim != 1:
        raise ValueError(
            f"prepare_field_1d expected a 1D field, got shape {field.shape}. "
            "Use the future 2D/3D plotting path for higher-dimensional data."
        )

    x, xlabel = build_axis_1d(ctx, field.shape[0])

    return Field1D(
        values=np.asarray(field),
        x=x,
        xlabel=xlabel,
        ylabel=f"v{var}",
    )