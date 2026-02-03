#!/usr/bin/env python3
import argparse
import os
import re
import sys
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -------------------- I/O + header parsing --------------------

def parse_header(lines):
    info = {
        'dim': None, 'numvar': None, 'step': None,
        'nx': None, 'ny': None, 'nz': None, 'ng': None,
        'quad': None, 'gamma': None, 't': None, 'dt': None, 'cfl': None
    }
    patterns = {
        'dim': r'dim\s*=\s*(\d+)',
        'numvar': r'numvar\s*=\s*(\d+)',
        'step': r'step\s*=\s*(-?\d+)',
        'nx': r'\bnx\s*=\s*(-?\d+)',
        'ny': r'\bny\s*=\s*(-?\d+)',
        'nz': r'\bnz\s*=\s*(-?\d+)',
        'ng': r'\bng\s*=\s*(-?\d+)',
        'quad': r'\bquad\s*=\s*(-?\d+)',
        'gamma': r'\bgamma\s*=\s*([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)',
        't': r'\bt\s*=\s*([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)',
        'dt': r'\bdt\s*=\s*([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)',
        'cfl': r'\bcfl\s*=\s*([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)',
    }
    for ln in lines:
        if not ln.lstrip().startswith('#'):
            break
        s = ln.strip()
        for key, pat in patterns.items():
            m = re.search(pat, s)
            if m:
                try:
                    if key in ('gamma', 't', 'dt', 'cfl'):
                        info[key] = float(m.group(1))
                    else:
                        info[key] = int(m.group(1))
                except Exception:
                    pass
    return info


def load_snapshot_table(path: str) -> Tuple[np.ndarray, dict]:
    with open(path, 'r') as f:
        lines = f.readlines()
    hdr = parse_header(lines)
    data = np.genfromtxt(path, comments='#')
    if data.ndim == 1:
        data = data[None, :]
    return data, hdr


def load_coords_table(path: str) -> Optional[np.ndarray]:
    try:
        coords = np.genfromtxt(path, comments='#')
        if coords.ndim == 1:
            coords = coords[None, :]
        return coords
    except Exception:
        return None


# -------------------- index-grid inference --------------------

def infer_grid_from_indices_1d(i: np.ndarray):
    i_min = int(np.min(i))
    i_max = int(np.max(i))
    return i_min, i_max


def infer_grid_from_indices_2d(ij: np.ndarray):
    i_min = int(np.min(ij[:, 0]))
    i_max = int(np.max(ij[:, 0]))
    j_min = int(np.min(ij[:, 1]))
    j_max = int(np.max(ij[:, 1]))
    return i_min, i_max, j_min, j_max


def infer_grid_from_indices_3d(ijk: np.ndarray):
    i_min = int(np.min(ijk[:, 0]))
    i_max = int(np.max(ijk[:, 0]))
    j_min = int(np.min(ijk[:, 1]))
    j_max = int(np.max(ijk[:, 1]))
    k_min = int(np.min(ijk[:, 2]))
    k_max = int(np.max(ijk[:, 2]))
    return i_min, i_max, j_min, j_max, k_min, k_max


# -------------------- field assembly (dense arrays) --------------------

def assemble_field_1d(table: np.ndarray, var_col: int,
                      trim_ghosts: bool, ng: Optional[int],
                      nx_hdr: Optional[int]):
    if table.shape[1] < 2:
        raise ValueError('Expected at least columns i and one variable')
    i = table[:, 0].astype(int)
    vals = table[:, var_col]

    i_min, i_max = infer_grid_from_indices_1d(i)
    nx_tot = (i_max - i_min + 1)

    y = np.full((nx_tot,), np.nan, dtype=float)
    for ii, v in zip(i, vals):
        y[ii - i_min] = v

    # Trim ghosts: keep i in [0, nx_hdr-1]
    if trim_ghosts and (ng is not None) and (nx_hdr is not None):
        i0, i1 = 0, nx_hdr - 1
        si0 = max(i0 - i_min, 0)
        si1 = min(i1 - i_min, y.shape[0] - 1)
        y = y[si0:si1 + 1]
        i_min = 0

    return y, (i_min, i_max)


def assemble_field_2d(table: np.ndarray, var_col: int,
                      trim_ghosts: bool, ng: Optional[int],
                      nx_hdr: Optional[int], ny_hdr: Optional[int]):
    if table.shape[1] < 3:
        raise ValueError('Expected at least columns i, j, and one variable')
    ij = table[:, :2].astype(int)
    vals = table[:, var_col]
    i_min, i_max, j_min, j_max = infer_grid_from_indices_2d(ij)
    nx_tot = (i_max - i_min + 1)
    ny_tot = (j_max - j_min + 1)

    Z = np.full((ny_tot, nx_tot), np.nan, dtype=float)
    for (ii, jj), v in zip(ij, vals):
        Z[jj - j_min, ii - i_min] = v

    if trim_ghosts and (ng is not None) and (nx_hdr is not None) and (ny_hdr is not None):
        i0, j0 = 0, 0
        i1, j1 = nx_hdr - 1, ny_hdr - 1
        si0, sj0 = max(i0 - i_min, 0), max(j0 - j_min, 0)
        si1, sj1 = min(i1 - i_min, Z.shape[1] - 1), min(j1 - j_min, Z.shape[0] - 1)
        Z = Z[sj0:sj1 + 1, si0:si1 + 1]
        i_min, j_min = 0, 0

    return Z, (i_min, i_max, j_min, j_max)


def assemble_field_3d(table: np.ndarray, var_col: int,
                      trim_ghosts: bool, ng: Optional[int],
                      nx_hdr: Optional[int], ny_hdr: Optional[int], nz_hdr: Optional[int]):
    if table.shape[1] < 4:
        raise ValueError('Expected at least columns i, j, k, and one variable')
    ijk = table[:, :3].astype(int)
    vals = table[:, var_col]

    i_min, i_max, j_min, j_max, k_min, k_max = infer_grid_from_indices_3d(ijk)
    nx_tot = (i_max - i_min + 1)
    ny_tot = (j_max - j_min + 1)
    nz_tot = (k_max - k_min + 1)

    # We'll store as V[k, j, i] (z, y, x) for easy slicing
    V = np.full((nz_tot, ny_tot, nx_tot), np.nan, dtype=float)
    for (ii, jj, kk), v in zip(ijk, vals):
        V[kk - k_min, jj - j_min, ii - i_min] = v

    if trim_ghosts and (ng is not None) and (nx_hdr is not None) and (ny_hdr is not None) and (nz_hdr is not None):
        # keep i,j,k in [0..nx_hdr-1], [0..ny_hdr-1], [0..nz_hdr-1]
        i0, j0, k0 = 0, 0, 0
        i1, j1, k1 = nx_hdr - 1, ny_hdr - 1, nz_hdr - 1

        si0, sj0, sk0 = max(i0 - i_min, 0), max(j0 - j_min, 0), max(k0 - k_min, 0)
        si1 = min(i1 - i_min, V.shape[2] - 1)
        sj1 = min(j1 - j_min, V.shape[1] - 1)
        sk1 = min(k1 - k_min, V.shape[0] - 1)

        V = V[sk0:sk1 + 1, sj0:sj1 + 1, si0:si1 + 1]
        i_min, j_min, k_min = 0, 0, 0

    return V, (i_min, i_max, j_min, j_max, k_min, k_max)


# -------------------- physical axes helpers --------------------

def nearest_index(vec: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(vec - value)))


def build_physical_axis_1d(n: int,
                           coords_1d: Optional[np.ndarray],
                           trim_ghosts: bool,
                           nx_hdr: Optional[int],
                           xr: Optional[Tuple[float, float]]):
    """
    Return x axis of length n in physical coordinates if possible.
    Priority: coords file -> xrange -> indices.
    """
    if coords_1d is not None and coords_1d.shape[1] >= 2:
        i = coords_1d[:, 0].astype(int)
        x = coords_1d[:, 1].astype(float)

        i_min, i_max = infer_grid_from_indices_1d(i)
        nx_tot = i_max - i_min + 1
        x_dense = np.full((nx_tot,), np.nan, dtype=float)
        for ii, xv in zip(i, x):
            x_dense[ii - i_min] = xv

        if trim_ghosts and (nx_hdr is not None):
            si0 = max(0 - i_min, 0)
            si1 = min((nx_hdr - 1) - i_min, x_dense.shape[0] - 1)
            x_dense = x_dense[si0:si1 + 1]

        if np.all(np.isfinite(x_dense)) and len(x_dense) == n:
            return x_dense, 'x'

    if xr is not None:
        xmin, xmax = xr
        if n > 1:
            return np.linspace(xmin, xmax, n), 'x'
        else:
            return np.array([0.5 * (xmin + xmax)]), 'x'

    return np.arange(n, dtype=float), 'i'


def build_physical_axes_2d(Z: np.ndarray,
                           coords_2d: Optional[np.ndarray],
                           trim_ghosts: bool,
                           nx_hdr: Optional[int], ny_hdr: Optional[int],
                           xr: Optional[Tuple[float, float]],
                           yr: Optional[Tuple[float, float]]):
    """
    Build (x_vec, y_vec, extent, (xlab, ylab)) for imshow and slicing.

    Priority:
      - coords file (i j x y): dense vectors for x and y if possible (structured grid assumption)
      - xrange/yrange: uniform vectors
      - else: index vectors
    """
    ny, nx = Z.shape

    if coords_2d is not None and coords_2d.shape[1] >= 4:
        ij = coords_2d[:, :2].astype(int)
        xs = coords_2d[:, 2].astype(float)
        ys = coords_2d[:, 3].astype(float)

        i_min, i_max, j_min, j_max = infer_grid_from_indices_2d(ij)
        nx_tot = i_max - i_min + 1
        ny_tot = j_max - j_min + 1

        x_vec = np.full((nx_tot,), np.nan, dtype=float)
        y_vec = np.full((ny_tot,), np.nan, dtype=float)

        for (ii, jj), xv, yv in zip(ij, xs, ys):
            if np.isnan(x_vec[ii - i_min]):
                x_vec[ii - i_min] = xv
            if np.isnan(y_vec[jj - j_min]):
                y_vec[jj - j_min] = yv

        if trim_ghosts and (nx_hdr is not None) and (ny_hdr is not None):
            si0 = max(0 - i_min, 0)
            sj0 = max(0 - j_min, 0)
            x_vec = x_vec[si0:si0 + nx_hdr]
            y_vec = y_vec[sj0:sj0 + ny_hdr]
        else:
            x_vec = x_vec[:nx]
            y_vec = y_vec[:ny]

        if (len(x_vec) == nx) and (len(y_vec) == ny) and np.all(np.isfinite(x_vec)) and np.all(np.isfinite(y_vec)):
            extent = (float(x_vec.min()), float(x_vec.max()), float(y_vec.min()), float(y_vec.max()))
            return x_vec, y_vec, extent, ('x', 'y')

    if xr is not None:
        x_vec = np.linspace(xr[0], xr[1], nx) if nx > 1 else np.array([0.5 * (xr[0] + xr[1])])
        xlab = 'x'
    else:
        x_vec = np.arange(nx, dtype=float)
        xlab = 'i'

    if yr is not None:
        y_vec = np.linspace(yr[0], yr[1], ny) if ny > 1 else np.array([0.5 * (yr[0] + yr[1])])
        ylab = 'y'
    else:
        y_vec = np.arange(ny, dtype=float)
        ylab = 'j'

    extent = (float(x_vec.min()), float(x_vec.max()), float(y_vec.min()), float(y_vec.max())) if (xr is not None or yr is not None) else None
    return x_vec, y_vec, extent, (xlab, ylab)


def build_physical_axes_3d(V: np.ndarray,
                           coords_3d: Optional[np.ndarray],
                           trim_ghosts: bool,
                           nx_hdr: Optional[int], ny_hdr: Optional[int], nz_hdr: Optional[int],
                           xr: Optional[Tuple[float, float]],
                           yr: Optional[Tuple[float, float]],
                           zr: Optional[Tuple[float, float]]):
    """
    Build (x_vec, y_vec, z_vec, (xlab, ylab, zlab)).

    coords_3d format expected: (i j k x y z).
    Assumes structured grid: x depends on i, y on j, z on k.
    """
    nz, ny, nx = V.shape

    if coords_3d is not None and coords_3d.shape[1] >= 6:
        ijk = coords_3d[:, :3].astype(int)
        xs = coords_3d[:, 3].astype(float)
        ys = coords_3d[:, 4].astype(float)
        zs = coords_3d[:, 5].astype(float)

        i_min, i_max, j_min, j_max, k_min, k_max = infer_grid_from_indices_3d(ijk)
        nx_tot = i_max - i_min + 1
        ny_tot = j_max - j_min + 1
        nz_tot = k_max - k_min + 1

        x_vec = np.full((nx_tot,), np.nan, dtype=float)
        y_vec = np.full((ny_tot,), np.nan, dtype=float)
        z_vec = np.full((nz_tot,), np.nan, dtype=float)

        for (ii, jj, kk), xv, yv, zv in zip(ijk, xs, ys, zs):
            if np.isnan(x_vec[ii - i_min]):
                x_vec[ii - i_min] = xv
            if np.isnan(y_vec[jj - j_min]):
                y_vec[jj - j_min] = yv
            if np.isnan(z_vec[kk - k_min]):
                z_vec[kk - k_min] = zv

        if trim_ghosts and (nx_hdr is not None) and (ny_hdr is not None) and (nz_hdr is not None):
            si0 = max(0 - i_min, 0)
            sj0 = max(0 - j_min, 0)
            sk0 = max(0 - k_min, 0)
            x_vec = x_vec[si0:si0 + nx_hdr]
            y_vec = y_vec[sj0:sj0 + ny_hdr]
            z_vec = z_vec[sk0:sk0 + nz_hdr]
        else:
            x_vec = x_vec[:nx]
            y_vec = y_vec[:ny]
            z_vec = z_vec[:nz]

        if (len(x_vec) == nx) and (len(y_vec) == ny) and (len(z_vec) == nz) and \
           np.all(np.isfinite(x_vec)) and np.all(np.isfinite(y_vec)) and np.all(np.isfinite(z_vec)):
            return x_vec, y_vec, z_vec, ('x', 'y', 'z')

    # fallback to ranges or indices
    if xr is not None:
        x_vec = np.linspace(xr[0], xr[1], nx) if nx > 1 else np.array([0.5 * (xr[0] + xr[1])])
        xlab = 'x'
    else:
        x_vec = np.arange(nx, dtype=float)
        xlab = 'i'

    if yr is not None:
        y_vec = np.linspace(yr[0], yr[1], ny) if ny > 1 else np.array([0.5 * (yr[0] + yr[1])])
        ylab = 'y'
    else:
        y_vec = np.arange(ny, dtype=float)
        ylab = 'j'

    if zr is not None:
        z_vec = np.linspace(zr[0], zr[1], nz) if nz > 1 else np.array([0.5 * (zr[0] + zr[1])])
        zlab = 'z'
    else:
        z_vec = np.arange(nz, dtype=float)
        zlab = 'k'

    return x_vec, y_vec, z_vec, (xlab, ylab, zlab)


def aspect_from_extent(extent):
    # extent = (xmin, xmax, ymin, ymax)
    if extent is None:
        return 'auto'
    dx = extent[1] - extent[0]
    dy = extent[3] - extent[2]
    return (dx / dy) if dy != 0 else 1.0


# -------------------- main plotting --------------------

def main():
    ap = argparse.ArgumentParser(description='Plot AETHER plaintext snapshot (1D/2D/3D with slicing).')
    ap.add_argument('snapshot', help='Path to snapshot .txt file')
    ap.add_argument('--var', type=int, default=0, help='Variable index to plot (vN). Default 0')
    ap.add_argument('--trim-ghosts', action='store_true', help='Trim ghost cells using header nx,ny,nz,ng if present')
    ap.add_argument('--coords', type=str, default=None,
                    help='Optional coordinates file: 1D (i x), 2D (i j x y), 3D (i j k x y z)')
    ap.add_argument('--output', type=str, default=None,
                    help='Optional output image path (.png). Defaults next to snapshot (but script uses plt.show())')
    ap.add_argument('--title', type=str, default=None, help='Optional plot title')

    # 2D slicing (legacy)
    ap.add_argument('--slice-axis', choices=['x', 'y'], default=None,
                    help='For 2D snapshots: plot a 1D slice along x or y instead of a 2D image.')
    ap.add_argument('--slice-index', type=int, default=None,
                    help='2D: index of orthogonal direction for slicing (j for x-slice, i for y-slice). '
                         'If omitted, uses the midpoint.')
    ap.add_argument('--slice-value', type=float, default=None,
                    help='2D: physical coordinate for orthogonal slicing (y for x-slice, x for y-slice). '
                         'Nearest slice is chosen. Requires --coords or --xrange/--yrange.')

    # Physical axis ranges (fallback when coords not available)
    ap.add_argument('--xrange', type=float, nargs=2, default=None, metavar=('XMIN', 'XMAX'),
                    help='Physical x-range when coords are not available (e.g. --xrange -1 1).')
    ap.add_argument('--yrange', type=float, nargs=2, default=None, metavar=('YMIN', 'YMAX'),
                    help='Physical y-range when coords are not available (e.g. --yrange -1 1).')
    ap.add_argument('--zrange', type=float, nargs=2, default=None, metavar=('ZMIN', 'ZMAX'),
                    help='Physical z-range when coords are not available (e.g. --zrange -1 1).')

    # 3D plane + line slicing
    ap.add_argument('--plane', choices=['xy', 'xz', 'yz'], default=None,
                    help='For 3D snapshots: choose a 2D slice plane (xy at fixed z, xz at fixed y, yz at fixed x). '
                         'Default is xy.')
    ap.add_argument('--plane-index', type=int, default=None,
                    help='3D: index of the orthogonal coordinate for the plane (k for xy, j for xz, i for yz). '
                         'If omitted, uses midpoint.')
    ap.add_argument('--plane-value', type=float, default=None,
                    help='3D: physical value of the orthogonal coordinate for the plane (z for xy, y for xz, x for yz). '
                         'Nearest slice is chosen. Requires coords or corresponding range.')

    ap.add_argument('--line-axis', choices=['x', 'y', 'z'], default=None,
                    help='For 3D plane slices: optionally extract a 1D line in the chosen plane along this axis.')
    ap.add_argument('--line-index', type=int, default=None,
                    help='3D: index for the remaining in-plane orthogonal coordinate when plotting a line. '
                         'Example: plane=xy, line-axis=x -> line-index selects j (y-index).')
    ap.add_argument('--line-value', type=float, default=None,
                    help='3D: physical value for the remaining in-plane orthogonal coordinate when plotting a line. '
                         'Requires coords or corresponding range.')

    args = ap.parse_args()

    table, hdr = load_snapshot_table(args.snapshot)
    dim = hdr.get('dim', None)
    if dim is None:
        # Heuristic based on minimum index columns
        if table.shape[1] >= 4:
            dim = 3
        elif table.shape[1] >= 3:
            dim = 2
        else:
            dim = 1

    ncols = table.shape[1]
    if dim == 1:
        var_col = 1 + args.var
        nvar = ncols - 1
    elif dim == 2:
        var_col = 2 + args.var
        nvar = ncols - 2
    elif dim == 3:
        var_col = 3 + args.var
        nvar = ncols - 3
    else:
        print(f"[error] dim={dim} not supported by this script.", file=sys.stderr)
        sys.exit(2)

    if args.var < 0 or args.var >= nvar:
        raise IndexError(f'Requested var index {args.var} exceeds available variables ({nvar}).')

    ng = hdr.get('ng')
    nx_hdr, ny_hdr, nz_hdr = hdr.get('nx'), hdr.get('ny'), hdr.get('nz')

    coords = None
    if args.coords:
        coords = load_coords_table(args.coords)
        if coords is None:
            print('[warn] Could not load coords file; proceeding without physical coordinates.', file=sys.stderr)

    xr = tuple(args.xrange) if args.xrange is not None else None
    yr = tuple(args.yrange) if args.yrange is not None else None
    zr = tuple(args.zrange) if args.zrange is not None else None

    # ---------------- 1D ----------------
    if dim == 1:
        y, _ = assemble_field_1d(table, var_col, args.trim_ghosts, ng, nx_hdr)
        x_coords, x_label = build_physical_axis_1d(
            n=len(y),
            coords_1d=coords,
            trim_ghosts=args.trim_ghosts,
            nx_hdr=nx_hdr,
            xr=xr
        )

        plt.figure()
        plt.plot(x_coords, y)
        plt.xlabel(x_label)
        plt.ylabel(f'v{args.var}')

        if args.title:
            plt.title(args.title)
        elif hdr.get('step') is not None and hdr.get('t') is not None:
            plt.title(f"step {hdr['step']}  t={hdr['t']:.6g}  v{args.var}")

    # ---------------- 2D ----------------
    elif dim == 2:
        Z, _ = assemble_field_2d(table, var_col, args.trim_ghosts, ng, nx_hdr, ny_hdr)

        x_vec, y_vec, extent, (xlab, ylab) = build_physical_axes_2d(
            Z=Z,
            coords_2d=coords,
            trim_ghosts=args.trim_ghosts,
            nx_hdr=nx_hdr,
            ny_hdr=ny_hdr,
            xr=xr,
            yr=yr
        )

        plt.figure()

        # Optional 1D slice of 2D
        if args.slice_axis is not None:
            if args.slice_value is not None:
                if (args.slice_axis == 'x') and (coords is None) and (yr is None):
                    print('[error] 2D: --slice-value with --slice-axis x requires --coords or --yrange.', file=sys.stderr)
                    sys.exit(2)
                if (args.slice_axis == 'y') and (coords is None) and (xr is None):
                    print('[error] 2D: --slice-value with --slice-axis y requires --coords or --xrange.', file=sys.stderr)
                    sys.exit(2)

            if args.slice_axis == 'x':
                # plot vs x at fixed j (fixed y)
                if args.slice_value is not None:
                    j0 = nearest_index(y_vec, args.slice_value)
                else:
                    j0 = args.slice_index if args.slice_index is not None else (Z.shape[0] // 2)
                j0 = max(0, min(Z.shape[0] - 1, j0))

                line = Z[j0, :]
                plt.plot(x_vec, line)
                plt.xlabel(xlab)
                plt.ylabel(f'v{args.var}')
                y_val = float(y_vec[j0])
                plt.title(args.title if args.title else f"v{args.var} slice at {ylab}={y_val:.6g}")

            else:  # slice_axis == 'y'
                # plot vs y at fixed i (fixed x)
                if args.slice_value is not None:
                    i0 = nearest_index(x_vec, args.slice_value)
                else:
                    i0 = args.slice_index if args.slice_index is not None else (Z.shape[1] // 2)
                i0 = max(0, min(Z.shape[1] - 1, i0))

                line = Z[:, i0]
                plt.plot(y_vec, line)
                plt.xlabel(ylab)
                plt.ylabel(f'v{args.var}')
                x_val = float(x_vec[i0])
                plt.title(args.title if args.title else f"v{args.var} slice at {xlab}={x_val:.6g}")

        else:
            im = plt.imshow(Z, origin='lower',
                            extent=extent if extent is not None else None,
                            aspect=aspect_from_extent(extent) if extent is not None else 'auto')
            plt.colorbar(im, label=f'v{args.var}')
            plt.xlabel(xlab)
            plt.ylabel(ylab)

            if args.title:
                plt.title(args.title)
            elif hdr.get('step') is not None and hdr.get('t') is not None:
                plt.title(f"step {hdr['step']}  t={hdr['t']:.6g}  v{args.var}")

    # ---------------- 3D ----------------
    elif dim == 3:
        V, _ = assemble_field_3d(table, var_col, args.trim_ghosts, ng, nx_hdr, ny_hdr, nz_hdr)
        # V[k, j, i]
        z_vec, y_vec, x_vec = None, None, None

        x_vec, y_vec, z_vec, (xlab, ylab, zlab) = build_physical_axes_3d(
            V=V,
            coords_3d=coords,
            trim_ghosts=args.trim_ghosts,
            nx_hdr=nx_hdr,
            ny_hdr=ny_hdr,
            nz_hdr=nz_hdr,
            xr=xr,
            yr=yr,
            zr=zr
        )

        # Choose plane
        plane = args.plane if args.plane is not None else 'xy'

        # Determine fixed index for plane
        # plane=xy -> fixed k (z)
        # plane=xz -> fixed j (y)
        # plane=yz -> fixed i (x)
        if args.plane_value is not None:
            if plane == 'xy':
                if (coords is None) and (zr is None):
                    print('[error] 3D: --plane-value with plane=xy requires --coords or --zrange.', file=sys.stderr)
                    sys.exit(2)
                fixed = nearest_index(z_vec, args.plane_value)
            elif plane == 'xz':
                if (coords is None) and (yr is None):
                    print('[error] 3D: --plane-value with plane=xz requires --coords or --yrange.', file=sys.stderr)
                    sys.exit(2)
                fixed = nearest_index(y_vec, args.plane_value)
            else:  # yz
                if (coords is None) and (xr is None):
                    print('[error] 3D: --plane-value with plane=yz requires --coords or --xrange.', file=sys.stderr)
                    sys.exit(2)
                fixed = nearest_index(x_vec, args.plane_value)
        else:
            if args.plane_index is not None:
                fixed = args.plane_index
            else:
                if plane == 'xy':
                    fixed = V.shape[0] // 2
                elif plane == 'xz':
                    fixed = V.shape[1] // 2
                else:
                    fixed = V.shape[2] // 2

        # Clamp fixed index
        if plane == 'xy':
            fixed = max(0, min(V.shape[0] - 1, fixed))
        elif plane == 'xz':
            fixed = max(0, min(V.shape[1] - 1, fixed))
        else:
            fixed = max(0, min(V.shape[2] - 1, fixed))

        # Extract plane data as a 2D array Z2 in (row, col) = (y, x) ordering for imshow
        # and define the in-plane axes.
        if plane == 'xy':
            # Z2[j, i] at fixed k
            Z2 = V[fixed, :, :]
            ax_u, ax_v = x_vec, y_vec      # u=x (cols), v=y (rows)
            lab_u, lab_v = xlab, ylab
            fixed_lab = zlab
            fixed_val = float(z_vec[fixed])
            extent = (float(ax_u.min()), float(ax_u.max()), float(ax_v.min()), float(ax_v.max()))
        elif plane == 'xz':
            # Z2[k, i] at fixed j  -> rows=z, cols=x
            Z2 = V[:, fixed, :]
            ax_u, ax_v = x_vec, z_vec
            lab_u, lab_v = xlab, zlab
            fixed_lab = ylab
            fixed_val = float(y_vec[fixed])
            extent = (float(ax_u.min()), float(ax_u.max()), float(ax_v.min()), float(ax_v.max()))
        else:  # yz
            # Z2[k, j] at fixed i -> rows=z, cols=y
            Z2 = V[:, :, fixed]
            ax_u, ax_v = y_vec, z_vec
            lab_u, lab_v = ylab, zlab
            fixed_lab = xlab
            fixed_val = float(x_vec[fixed])
            extent = (float(ax_u.min()), float(ax_u.max()), float(ax_v.min()), float(ax_v.max()))

        plt.figure()

        # Optional: line in plane
        if args.line_axis is not None:
            line_axis = args.line_axis

            # Validate that requested line axis lies in the plane
            in_plane_axes = set(plane)  # 'xy' => {'x','y'}
            if line_axis not in in_plane_axes:
                print(f"[error] 3D: --line-axis {line_axis} is not in plane {plane}.", file=sys.stderr)
                print("        Valid choices: plane=xy -> line-axis x or y; plane=xz -> x or z; plane=yz -> y or z.", file=sys.stderr)
                sys.exit(2)

            # The other in-plane axis (to fix)
            other_axis = (in_plane_axes - {line_axis}).pop()

            # Map axis vectors
            vec = {'x': x_vec, 'y': y_vec, 'z': z_vec}
            lab = {'x': xlab, 'y': ylab, 'z': zlab}

            # Choose index for "other_axis" within this plane
            # If line_value provided: interpret as physical value of other_axis
            if args.line_value is not None:
                # require coords or relevant range for other_axis
                need = {'x': xr, 'y': yr, 'z': zr}[other_axis]
                if (coords is None) and (need is None):
                    print(f"[error] 3D: --line-value requires --coords or corresponding range for axis '{other_axis}'.", file=sys.stderr)
                    sys.exit(2)
                other_idx = nearest_index(vec[other_axis], args.line_value)
            else:
                other_idx = args.line_index
                if other_idx is None:
                    # midpoint
                    if other_axis == 'x':
                        other_idx = V.shape[2] // 2
                    elif other_axis == 'y':
                        other_idx = V.shape[1] // 2
                    else:
                        other_idx = V.shape[0] // 2

            # Clamp other_idx based on that axis' length
            other_idx = max(0, min(len(vec[other_axis]) - 1, other_idx))

            # Now extract the line, depending on plane + line_axis
            if plane == 'xy':
                # Z2 is (y, x)
                if line_axis == 'x':
                    # x-line at fixed y (j=other_idx)
                    y_idx = other_idx
                    line = Z2[y_idx, :]
                    plt.plot(x_vec, line)
                    plt.xlabel(xlab)
                    plt.ylabel(f'v{args.var}')
                    other_val = float(y_vec[y_idx])
                    title = f"v{args.var} line in {plane} at {fixed_lab}={fixed_val:.6g}, {ylab}={other_val:.6g}"
                else:  # line_axis == 'y'
                    x_idx = other_idx
                    line = Z2[:, x_idx]
                    plt.plot(y_vec, line)
                    plt.xlabel(ylab)
                    plt.ylabel(f'v{args.var}')
                    other_val = float(x_vec[x_idx])
                    title = f"v{args.var} line in {plane} at {fixed_lab}={fixed_val:.6g}, {xlab}={other_val:.6g}"

            elif plane == 'xz':
                # Z2 is (z, x)
                if line_axis == 'x':
                    z_idx = other_idx
                    line = Z2[z_idx, :]
                    plt.plot(x_vec, line)
                    plt.xlabel(xlab)
                    plt.ylabel(f'v{args.var}')
                    other_val = float(z_vec[z_idx])
                    title = f"v{args.var} line in {plane} at {fixed_lab}={fixed_val:.6g}, {zlab}={other_val:.6g}"
                else:  # line_axis == 'z'
                    x_idx = other_idx
                    line = Z2[:, x_idx]
                    plt.plot(z_vec, line)
                    plt.xlabel(zlab)
                    plt.ylabel(f'v{args.var}')
                    other_val = float(x_vec[x_idx])
                    title = f"v{args.var} line in {plane} at {fixed_lab}={fixed_val:.6g}, {xlab}={other_val:.6g}"

            else:  # plane == 'yz'
                # Z2 is (z, y)
                if line_axis == 'y':
                    z_idx = other_idx
                    line = Z2[z_idx, :]
                    plt.plot(y_vec, line)
                    plt.xlabel(ylab)
                    plt.ylabel(f'v{args.var}')
                    other_val = float(z_vec[z_idx])
                    title = f"v{args.var} line in {plane} at {fixed_lab}={fixed_val:.6g}, {zlab}={other_val:.6g}"
                else:  # line_axis == 'z'
                    y_idx = other_idx
                    line = Z2[:, y_idx]
                    plt.plot(z_vec, line)
                    plt.xlabel(zlab)
                    plt.ylabel(f'v{args.var}')
                    other_val = float(y_vec[y_idx])
                    title = f"v{args.var} line in {plane} at {fixed_lab}={fixed_val:.6g}, {ylab}={other_val:.6g}"

            if args.title:
                plt.title(args.title)
            else:
                plt.title(title)

        else:
            # 2D plane image
            im = plt.imshow(Z2, origin='lower',
                            extent=extent,
                            aspect=aspect_from_extent(extent))

            plt.colorbar(im, label=f'v{args.var}')
            plt.xlabel(lab_u)
            plt.ylabel(lab_v)

            if args.title:
                plt.title(args.title)
            elif hdr.get('step') is not None and hdr.get('t') is not None:
                plt.title(f"step {hdr['step']}  t={hdr['t']:.6g}  v{args.var}  {plane}@{fixed_lab}={fixed_val:.6g}")
            else:
                plt.title(f"v{args.var}  {plane}@{fixed_lab}={fixed_val:.6g}")

    # --------------- finalize ---------------
    outpath = args.output if args.output else os.path.splitext(args.snapshot)[0] + f"_v{args.var}.png"
    plt.tight_layout()
    # plt.savefig(outpath, dpi=150)
    plt.show()
    print(f"[ok] Wrote {outpath}")


if __name__ == '__main__':
    main()

