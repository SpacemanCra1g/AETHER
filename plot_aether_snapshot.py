#!/usr/bin/env python3
import argparse
import os
import re
import sys
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Header / I/O
# ============================================================

def parse_header(lines):
    info = {
        "dim": None,
        "numvar": None,
        "step": None,
        "nx": None,
        "ny": None,
        "nz": None,
        "ng": None,
        "quad": None,
        "gamma": None,
        "t": None,
        "dt": None,
        "cfl": None,
    }

    patterns = {
        "dim": r"\bdim\s*=\s*(\d+)",
        "numvar": r"\bnumvar\s*=\s*(\d+)",
        "step": r"\bstep\s*=\s*(-?\d+)",
        "nx": r"\bnx\s*=\s*(-?\d+)",
        "ny": r"\bny\s*=\s*(-?\d+)",
        "nz": r"\bnz\s*=\s*(-?\d+)",
        "ng": r"\bng\s*=\s*(-?\d+)",
        "quad": r"\bquad\s*=\s*(-?\d+)",
        "gamma": r"\bgamma\s*=\s*([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)",
        "t": r"\bt\s*=\s*([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)",
        "dt": r"\bdt\s*=\s*([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)",
        "cfl": r"\bcfl\s*=\s*([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)",
    }

    for ln in lines:
        if not ln.lstrip().startswith("#"):
            break
        s = ln.strip()
        for key, pat in patterns.items():
            m = re.search(pat, s)
            if m:
                try:
                    if key in ("gamma", "t", "dt", "cfl"):
                        info[key] = float(m.group(1))
                    else:
                        info[key] = int(m.group(1))
                except Exception:
                    pass

    return info


def load_snapshot_table(path: str) -> Tuple[np.ndarray, dict]:
    with open(path, "r") as f:
        lines = f.readlines()
    hdr = parse_header(lines)
    data = np.genfromtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    return data, hdr


def load_coords_table(path: str) -> Optional[np.ndarray]:
    try:
        coords = np.genfromtxt(path, comments="#")
        if coords.ndim == 1:
            coords = coords[None, :]
        return coords
    except Exception:
        return None


# ============================================================
# Grid inference
# ============================================================

def infer_grid_from_indices_1d(i: np.ndarray):
    return int(np.min(i)), int(np.max(i))


def infer_grid_from_indices_2d(ij: np.ndarray):
    return (
        int(np.min(ij[:, 0])), int(np.max(ij[:, 0])),
        int(np.min(ij[:, 1])), int(np.max(ij[:, 1])),
    )


def infer_grid_from_indices_3d(ijk: np.ndarray):
    return (
        int(np.min(ijk[:, 0])), int(np.max(ijk[:, 0])),
        int(np.min(ijk[:, 1])), int(np.max(ijk[:, 1])),
        int(np.min(ijk[:, 2])), int(np.max(ijk[:, 2])),
    )


# ============================================================
# Dense field assembly
# ============================================================

def assemble_field_1d(table: np.ndarray, var_col: int,
                      trim_ghosts: bool, nx_hdr: Optional[int]):
    i = table[:, 0].astype(int)
    vals = table[:, var_col]

    i_min, i_max = infer_grid_from_indices_1d(i)
    n = i_max - i_min + 1

    y = np.full((n,), np.nan, dtype=float)
    for ii, v in zip(i, vals):
        y[ii - i_min] = v

    if trim_ghosts and nx_hdr is not None:
        s0 = max(0 - i_min, 0)
        s1 = min((nx_hdr - 1) - i_min, y.shape[0] - 1)
        y = y[s0:s1 + 1]

    return y


def assemble_field_2d(table: np.ndarray, var_col: int,
                      trim_ghosts: bool,
                      nx_hdr: Optional[int], ny_hdr: Optional[int]):
    ij = table[:, :2].astype(int)
    vals = table[:, var_col]

    i_min, i_max, j_min, j_max = infer_grid_from_indices_2d(ij)
    nx = i_max - i_min + 1
    ny = j_max - j_min + 1

    Z = np.full((ny, nx), np.nan, dtype=float)
    for (ii, jj), v in zip(ij, vals):
        Z[jj - j_min, ii - i_min] = v

    if trim_ghosts and nx_hdr is not None and ny_hdr is not None:
        si0 = max(0 - i_min, 0)
        sj0 = max(0 - j_min, 0)
        si1 = min((nx_hdr - 1) - i_min, Z.shape[1] - 1)
        sj1 = min((ny_hdr - 1) - j_min, Z.shape[0] - 1)
        Z = Z[sj0:sj1 + 1, si0:si1 + 1]

    return Z


def assemble_field_3d(table: np.ndarray, var_col: int,
                      trim_ghosts: bool,
                      nx_hdr: Optional[int], ny_hdr: Optional[int], nz_hdr: Optional[int]):
    ijk = table[:, :3].astype(int)
    vals = table[:, var_col]

    i_min, i_max, j_min, j_max, k_min, k_max = infer_grid_from_indices_3d(ijk)
    nx = i_max - i_min + 1
    ny = j_max - j_min + 1
    nz = k_max - k_min + 1

    V = np.full((nz, ny, nx), np.nan, dtype=float)
    for (ii, jj, kk), v in zip(ijk, vals):
        V[kk - k_min, jj - j_min, ii - i_min] = v

    if trim_ghosts and nx_hdr is not None and ny_hdr is not None and nz_hdr is not None:
        si0 = max(0 - i_min, 0)
        sj0 = max(0 - j_min, 0)
        sk0 = max(0 - k_min, 0)
        si1 = min((nx_hdr - 1) - i_min, V.shape[2] - 1)
        sj1 = min((ny_hdr - 1) - j_min, V.shape[1] - 1)
        sk1 = min((nz_hdr - 1) - k_min, V.shape[0] - 1)
        V = V[sk0:sk1 + 1, sj0:sj1 + 1, si0:si1 + 1]

    return V


# ============================================================
# Coordinate helpers
# ============================================================

def nearest_index(vec: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(vec - value)))


def dense_axis_from_coords_1d(coords_1d: Optional[np.ndarray],
                              trim_ghosts: bool,
                              nx_hdr: Optional[int]) -> Optional[np.ndarray]:
    if coords_1d is None or coords_1d.shape[1] < 2:
        return None

    i = coords_1d[:, 0].astype(int)
    x = coords_1d[:, 1].astype(float)

    i_min, i_max = infer_grid_from_indices_1d(i)
    nx = i_max - i_min + 1

    out = np.full((nx,), np.nan, dtype=float)
    for ii, xv in zip(i, x):
        out[ii - i_min] = xv

    if trim_ghosts and nx_hdr is not None:
        s0 = max(0 - i_min, 0)
        out = out[s0:s0 + nx_hdr]

    if np.all(np.isfinite(out)):
        return out
    return None


def dense_axes_from_coords_2d(coords_2d: Optional[np.ndarray],
                              trim_ghosts: bool,
                              nx_hdr: Optional[int], ny_hdr: Optional[int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if coords_2d is None or coords_2d.shape[1] < 4:
        return None, None

    ij = coords_2d[:, :2].astype(int)
    xs = coords_2d[:, 2].astype(float)
    ys = coords_2d[:, 3].astype(float)

    i_min, i_max, j_min, j_max = infer_grid_from_indices_2d(ij)
    nx = i_max - i_min + 1
    ny = j_max - j_min + 1

    x_vec = np.full((nx,), np.nan, dtype=float)
    y_vec = np.full((ny,), np.nan, dtype=float)

    for (ii, jj), xv, yv in zip(ij, xs, ys):
        if np.isnan(x_vec[ii - i_min]):
            x_vec[ii - i_min] = xv
        if np.isnan(y_vec[jj - j_min]):
            y_vec[jj - j_min] = yv

    if trim_ghosts and nx_hdr is not None and ny_hdr is not None:
        sx = max(0 - i_min, 0)
        sy = max(0 - j_min, 0)
        x_vec = x_vec[sx:sx + nx_hdr]
        y_vec = y_vec[sy:sy + ny_hdr]

    if np.all(np.isfinite(x_vec)) and np.all(np.isfinite(y_vec)):
        return x_vec, y_vec
    return None, None


def dense_axes_from_coords_3d(coords_3d: Optional[np.ndarray],
                              trim_ghosts: bool,
                              nx_hdr: Optional[int], ny_hdr: Optional[int], nz_hdr: Optional[int]
                              ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if coords_3d is None or coords_3d.shape[1] < 6:
        return None, None, None

    ijk = coords_3d[:, :3].astype(int)
    xs = coords_3d[:, 3].astype(float)
    ys = coords_3d[:, 4].astype(float)
    zs = coords_3d[:, 5].astype(float)

    i_min, i_max, j_min, j_max, k_min, k_max = infer_grid_from_indices_3d(ijk)
    nx = i_max - i_min + 1
    ny = j_max - j_min + 1
    nz = k_max - k_min + 1

    x_vec = np.full((nx,), np.nan, dtype=float)
    y_vec = np.full((ny,), np.nan, dtype=float)
    z_vec = np.full((nz,), np.nan, dtype=float)

    for (ii, jj, kk), xv, yv, zv in zip(ijk, xs, ys, zs):
        if np.isnan(x_vec[ii - i_min]):
            x_vec[ii - i_min] = xv
        if np.isnan(y_vec[jj - j_min]):
            y_vec[jj - j_min] = yv
        if np.isnan(z_vec[kk - k_min]):
            z_vec[kk - k_min] = zv

    if trim_ghosts and nx_hdr is not None and ny_hdr is not None and nz_hdr is not None:
        sx = max(0 - i_min, 0)
        sy = max(0 - j_min, 0)
        sz = max(0 - k_min, 0)
        x_vec = x_vec[sx:sx + nx_hdr]
        y_vec = y_vec[sy:sy + ny_hdr]
        z_vec = z_vec[sz:sz + nz_hdr]

    if np.all(np.isfinite(x_vec)) and np.all(np.isfinite(y_vec)) and np.all(np.isfinite(z_vec)):
        return x_vec, y_vec, z_vec
    return None, None, None


def build_axis_1d(n: int, coord_axis: Optional[np.ndarray], rng: Optional[Tuple[float, float]]):
    if coord_axis is not None and len(coord_axis) == n:
        return coord_axis, "x"

    if rng is not None:
        if n == 1:
            return np.array([0.5 * (rng[0] + rng[1])]), "x"
        return np.linspace(rng[0], rng[1], n), "x"

    return np.arange(n, dtype=float), "i"


def build_axes_2d(shape: Tuple[int, int],
                  x_coords: Optional[np.ndarray], y_coords: Optional[np.ndarray],
                  xr: Optional[Tuple[float, float]], yr: Optional[Tuple[float, float]]):
    ny, nx = shape

    if x_coords is not None and len(x_coords) == nx:
        x_vec = x_coords
        xlab = "x"
    elif xr is not None:
        x_vec = np.linspace(xr[0], xr[1], nx) if nx > 1 else np.array([0.5 * (xr[0] + xr[1])])
        xlab = "x"
    else:
        x_vec = np.arange(nx, dtype=float)
        xlab = "i"

    if y_coords is not None and len(y_coords) == ny:
        y_vec = y_coords
        ylab = "y"
    elif yr is not None:
        y_vec = np.linspace(yr[0], yr[1], ny) if ny > 1 else np.array([0.5 * (yr[0] + yr[1])])
        ylab = "y"
    else:
        y_vec = np.arange(ny, dtype=float)
        ylab = "j"

    return x_vec, y_vec, xlab, ylab


def build_axes_3d(shape: Tuple[int, int, int],
                  x_coords: Optional[np.ndarray], y_coords: Optional[np.ndarray], z_coords: Optional[np.ndarray],
                  xr: Optional[Tuple[float, float]], yr: Optional[Tuple[float, float]], zr: Optional[Tuple[float, float]]):
    nz, ny, nx = shape

    if x_coords is not None and len(x_coords) == nx:
        x_vec = x_coords
        xlab = "x"
    elif xr is not None:
        x_vec = np.linspace(xr[0], xr[1], nx) if nx > 1 else np.array([0.5 * (xr[0] + xr[1])])
        xlab = "x"
    else:
        x_vec = np.arange(nx, dtype=float)
        xlab = "i"

    if y_coords is not None and len(y_coords) == ny:
        y_vec = y_coords
        ylab = "y"
    elif yr is not None:
        y_vec = np.linspace(yr[0], yr[1], ny) if ny > 1 else np.array([0.5 * (yr[0] + yr[1])])
        ylab = "y"
    else:
        y_vec = np.arange(ny, dtype=float)
        ylab = "j"

    if z_coords is not None and len(z_coords) == nz:
        z_vec = z_coords
        zlab = "z"
    elif zr is not None:
        z_vec = np.linspace(zr[0], zr[1], nz) if nz > 1 else np.array([0.5 * (zr[0] + zr[1])])
        zlab = "z"
    else:
        z_vec = np.arange(nz, dtype=float)
        zlab = "k"

    return x_vec, y_vec, z_vec, xlab, ylab, zlab


# ============================================================
# Plot geometry helpers
# ============================================================

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


def figsize_from_box_aspect(box_aspect: float, base_width: float = 8.0) -> Tuple[float, float]:
    height = base_width * box_aspect
    height = max(2.5, min(height, 10.0))
    return base_width, height


def contour_levels_from_field(field: np.ndarray, nlevels: int) -> Optional[np.ndarray]:
    if nlevels is None or nlevels <= 0:
        return None

    finite = field[np.isfinite(field)]
    if finite.size == 0:
        return None

    fmin = np.min(finite)
    fmax = np.max(finite)

    if not np.isfinite(fmin) or not np.isfinite(fmax):
        return None
    if fmax <= fmin:
        return None

    return np.linspace(fmin, fmax, nlevels)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Plot AETHER plaintext snapshot.")
    ap.add_argument("snapshot", help="Path to snapshot .txt file")
    ap.add_argument("--var", type=int, default=0, help="Variable index to plot (vN). Default 0")
    ap.add_argument("--trim-ghosts", action="store_true", help="Trim ghost cells using header nx,ny,nz")
    ap.add_argument("--coords", type=str, default=None,
                    help="Optional coordinates file: 1D(i x), 2D(i j x y), 3D(i j k x y z)")
    ap.add_argument("--output", type=str, default=None, help="Optional output image path")
    ap.add_argument("--title", type=str, default=None, help="Optional title")
    ap.add_argument("--contours", type=int, default=None,
                    help="Overlay contour lines with N levels")

    ap.add_argument("--slice-axis", choices=["x", "y", "z"], default=None,
                    help="1D slice axis for 2D/3D data")
    ap.add_argument("--slice-index", type=int, default=None,
                    help="Index of orthogonal coordinate for line slices")
    ap.add_argument("--slice-value", type=float, default=None,
                    help="Physical value of orthogonal coordinate for line slices")

    ap.add_argument("--xrange", type=float, nargs=2, default=None, metavar=("XMIN", "XMAX"))
    ap.add_argument("--yrange", type=float, nargs=2, default=None, metavar=("YMIN", "YMAX"))
    ap.add_argument("--zrange", type=float, nargs=2, default=None, metavar=("ZMIN", "ZMAX"))

    ap.add_argument("--plane", choices=["xy", "xz", "yz"], default=None,
                    help="3D plane to plot")
    ap.add_argument("--plane-index", type=int, default=None,
                    help="3D plane orthogonal index")
    ap.add_argument("--plane-value", type=float, default=None,
                    help="3D plane orthogonal physical value")

    args = ap.parse_args()

    table, hdr = load_snapshot_table(args.snapshot)

    dim = hdr.get("dim")
    if dim is None:
        if table.shape[1] >= 4:
            dim = 3
        elif table.shape[1] >= 3:
            dim = 2
        else:
            dim = 1

    if dim == 1:
        var_col = 1 + args.var
        nvar = table.shape[1] - 1
    elif dim == 2:
        var_col = 2 + args.var
        nvar = table.shape[1] - 2
    elif dim == 3:
        var_col = 3 + args.var
        nvar = table.shape[1] - 3
    else:
        print(f"[error] Unsupported dim={dim}", file=sys.stderr)
        sys.exit(2)

    if args.var < 0 or args.var >= nvar:
        raise IndexError(f"Requested var index {args.var} exceeds available variables ({nvar}).")

    nx_hdr = hdr.get("nx")
    ny_hdr = hdr.get("ny")
    nz_hdr = hdr.get("nz")

    coords = load_coords_table(args.coords) if args.coords else None
    xr = tuple(args.xrange) if args.xrange is not None else None
    yr = tuple(args.yrange) if args.yrange is not None else None
    zr = tuple(args.zrange) if args.zrange is not None else None

    # --------------------------------------------------------
    # 1D
    # --------------------------------------------------------
    if dim == 1:
        y = assemble_field_1d(table, var_col, args.trim_ghosts, nx_hdr)
        x_coords = dense_axis_from_coords_1d(coords, args.trim_ghosts, nx_hdr)
        x_vec, xlab = build_axis_1d(len(y), x_coords, xr)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x_vec, y)
        ax.set_xlabel(xlab)
        ax.set_ylabel(f"v{args.var}")

        if args.title:
            ax.set_title(args.title)
        elif hdr.get("step") is not None and hdr.get("t") is not None:
            ax.set_title(f"step {hdr['step']}  t={hdr['t']:.6g}  v{args.var}")

    # --------------------------------------------------------
    # 2D
    # --------------------------------------------------------
    elif dim == 2:
        Z = assemble_field_2d(table, var_col, args.trim_ghosts, nx_hdr, ny_hdr)
        x_coords, y_coords = dense_axes_from_coords_2d(coords, args.trim_ghosts, nx_hdr, ny_hdr)
        x_vec, y_vec, xlab, ylab = build_axes_2d(Z.shape, x_coords, y_coords, xr, yr)

        if args.slice_axis is not None:
            fig, ax = plt.subplots(figsize=(8, 4))

            if args.slice_axis == "z":
                print("[error] 2D data has no z axis.", file=sys.stderr)
                sys.exit(2)

            if args.slice_axis == "x":
                if args.slice_value is not None:
                    j0 = nearest_index(y_vec, args.slice_value)
                else:
                    j0 = args.slice_index if args.slice_index is not None else (Z.shape[0] // 2)
                j0 = max(0, min(Z.shape[0] - 1, j0))

                ax.plot(x_vec, Z[j0, :])
                ax.set_xlabel(xlab)
                ax.set_ylabel(f"v{args.var}")
                if args.title:
                    ax.set_title(args.title)
                else:
                    ax.set_title(f"v{args.var} slice at {ylab}={y_vec[j0]:.6g}")

            else:
                if args.slice_value is not None:
                    i0 = nearest_index(x_vec, args.slice_value)
                else:
                    i0 = args.slice_index if args.slice_index is not None else (Z.shape[1] // 2)
                i0 = max(0, min(Z.shape[1] - 1, i0))

                ax.plot(y_vec, Z[:, i0])
                ax.set_xlabel(ylab)
                ax.set_ylabel(f"v{args.var}")
                if args.title:
                    ax.set_title(args.title)
                else:
                    ax.set_title(f"v{args.var} slice at {xlab}={x_vec[i0]:.6g}")

        else:
            box_aspect = box_aspect_from_axes(x_vec, y_vec)
            fig, ax = plt.subplots(figsize=figsize_from_box_aspect(box_aspect))

            x_edges = axis_edges_from_centers(x_vec)
            y_edges = axis_edges_from_centers(y_vec)

            pcm = ax.pcolormesh(x_edges, y_edges, Z, shading="auto")
            fig.colorbar(pcm, ax=ax, label=f"v{args.var}")

            levels = contour_levels_from_field(Z, args.contours)
            if levels is not None:
                ax.contour(x_vec, y_vec, Z, levels=levels, colors="k", linewidths=0.5)

            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.set_xlim(x_edges[0], x_edges[-1])
            ax.set_ylim(y_edges[0], y_edges[-1])
            ax.set_box_aspect(box_aspect)

            if args.title:
                ax.set_title(args.title)
            elif hdr.get("step") is not None and hdr.get("t") is not None:
                ax.set_title(f"step {hdr['step']}  t={hdr['t']:.6g}  v{args.var}")

    # --------------------------------------------------------
    # 3D
    # --------------------------------------------------------
    else:
        V = assemble_field_3d(table, var_col, args.trim_ghosts, nx_hdr, ny_hdr, nz_hdr)
        x_coords, y_coords, z_coords = dense_axes_from_coords_3d(coords, args.trim_ghosts, nx_hdr, ny_hdr, nz_hdr)
        x_vec, y_vec, z_vec, xlab, ylab, zlab = build_axes_3d(V.shape, x_coords, y_coords, z_coords, xr, yr, zr)

        plane = args.plane if args.plane is not None else "xy"

        if args.plane_value is not None:
            if plane == "xy":
                fixed = nearest_index(z_vec, args.plane_value)
            elif plane == "xz":
                fixed = nearest_index(y_vec, args.plane_value)
            else:
                fixed = nearest_index(x_vec, args.plane_value)
        else:
            if args.plane_index is not None:
                fixed = args.plane_index
            else:
                if plane == "xy":
                    fixed = V.shape[0] // 2
                elif plane == "xz":
                    fixed = V.shape[1] // 2
                else:
                    fixed = V.shape[2] // 2

        if plane == "xy":
            fixed = max(0, min(V.shape[0] - 1, fixed))
            Z2 = V[fixed, :, :]
            ax_u, ax_v = x_vec, y_vec
            lab_u, lab_v = xlab, ylab
            fixed_lab = zlab
            fixed_val = z_vec[fixed]
        elif plane == "xz":
            fixed = max(0, min(V.shape[1] - 1, fixed))
            Z2 = V[:, fixed, :]
            ax_u, ax_v = x_vec, z_vec
            lab_u, lab_v = xlab, zlab
            fixed_lab = ylab
            fixed_val = y_vec[fixed]
        else:
            fixed = max(0, min(V.shape[2] - 1, fixed))
            Z2 = V[:, :, fixed]
            ax_u, ax_v = y_vec, z_vec
            lab_u, lab_v = ylab, zlab
            fixed_lab = xlab
            fixed_val = x_vec[fixed]

        if args.slice_axis is not None:
            fig, ax = plt.subplots(figsize=(8, 4))
            in_plane = set(plane)
            slice_axis = args.slice_axis

            if slice_axis not in in_plane:
                print(f"[error] slice-axis {slice_axis} is not in plane {plane}", file=sys.stderr)
                sys.exit(2)

            other_axis = (in_plane - {slice_axis}).pop()

            axis_map = {"x": x_vec, "y": y_vec, "z": z_vec}

            if args.slice_value is not None:
                other_idx = nearest_index(axis_map[other_axis], args.slice_value)
            else:
                if args.slice_index is not None:
                    other_idx = args.slice_index
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

            ax.set_ylabel(f"v{args.var}")
            if args.title:
                ax.set_title(args.title)
            else:
                ax.set_title(f"v{args.var} slice in {plane} at {fixed_lab}={fixed_val:.6g}, {other_lab}={other_val:.6g}")

        else:
            box_aspect = box_aspect_from_axes(ax_u, ax_v)
            fig, ax = plt.subplots(figsize=figsize_from_box_aspect(box_aspect))

            u_edges = axis_edges_from_centers(ax_u)
            v_edges = axis_edges_from_centers(ax_v)

            pcm = ax.pcolormesh(u_edges, v_edges, Z2, shading="auto")
            fig.colorbar(pcm, ax=ax, label=f"v{args.var}")

            levels = contour_levels_from_field(Z2, args.contours)
            if levels is not None:
                ax.contour(ax_u, ax_v, Z2, levels=levels, colors="k", linewidths=0.5)

            ax.set_xlabel(lab_u)
            ax.set_ylabel(lab_v)
            ax.set_xlim(u_edges[0], u_edges[-1])
            ax.set_ylim(v_edges[0], v_edges[-1])
            ax.set_box_aspect(box_aspect)

            if args.title:
                ax.set_title(args.title)
            elif hdr.get("step") is not None and hdr.get("t") is not None:
                ax.set_title(f"step {hdr['step']}  t={hdr['t']:.6g}  v{args.var}  {plane}@{fixed_lab}={fixed_val:.6g}")
            else:
                ax.set_title(f"v{args.var}  {plane}@{fixed_lab}={fixed_val:.6g}")

    outpath = args.output if args.output else os.path.splitext(args.snapshot)[0] + f"_v{args.var}.png"
    plt.tight_layout()
    # plt.savefig(outpath, dpi=150)
    plt.show()
    print(f"[ok] Wrote {outpath}")


if __name__ == "__main__":
    main()