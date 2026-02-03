#!/usr/bin/env python3
import argparse
import os
import re
import sys
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


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
        si1, sj1 = min(i1 - i_min, Z.shape[1]-1), min(j1 - j_min, Z.shape[0]-1)
        Z = Z[sj0: sj1+1, si0: si1+1]
        i_min, j_min = 0, 0
    return Z, (i_min, i_max, j_min, j_max)


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

    # 1) From coords file: attempt to make dense x(i) and y(j)
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

        # Match trimming used by assemble_field_2d
        if trim_ghosts and (nx_hdr is not None) and (ny_hdr is not None):
            # assemble_field_2d trims to i,j in [0..nx_hdr-1],[0..ny_hdr-1]
            # but coords may have i_min/j_min offsets due to ghosts, so slice accordingly
            si0 = max(0 - i_min, 0)
            sj0 = max(0 - j_min, 0)
            x_vec = x_vec[si0: si0 + nx_hdr]
            y_vec = y_vec[sj0: sj0 + ny_hdr]
        else:
            # Otherwise just truncate to Z shape
            x_vec = x_vec[:nx]
            y_vec = y_vec[:ny]

        if (len(x_vec) == nx) and (len(y_vec) == ny) and np.all(np.isfinite(x_vec)) and np.all(np.isfinite(y_vec)):
            extent = (float(x_vec.min()), float(x_vec.max()), float(y_vec.min()), float(y_vec.max()))
            return x_vec, y_vec, extent, ('x', 'y')

    # 2) From ranges / fallback
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


def nearest_index(vec: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(vec - value)))


def main():
    ap = argparse.ArgumentParser(description='Plot AETHER plaintext snapshot (1D/2D).')
    ap.add_argument('snapshot', help='Path to snapshot .txt file')
    ap.add_argument('--var', type=int, default=0, help='Variable index to plot (vN). Default 0')
    ap.add_argument('--trim-ghosts', action='store_true', help='Trim ghost cells using header nx,ny,ng if present')
    ap.add_argument('--coords', type=str, default=None, help='Optional coordinates file: 1D (i x) or 2D (i j x y)')
    ap.add_argument('--output', type=str, default=None, help='Optional output image path (.png). Defaults next to snapshot')
    ap.add_argument('--title', type=str, default=None, help='Optional plot title')

    # New: 2D slicing options
    ap.add_argument('--slice-axis', choices=['x', 'y'], default=None,
                    help='For 2D snapshots: plot a 1D slice along x or y instead of a 2D image.')
    ap.add_argument('--slice-index', type=int, default=None,
                    help='Index of the orthogonal direction for slicing (j for x-slice, i for y-slice). '
                         'If omitted, uses the midpoint.')
    ap.add_argument('--slice-value', type=float, default=None,
                    help='Physical coordinate of the orthogonal direction for slicing (y for x-slice, x for y-slice). '
                         'Nearest slice is chosen. Requires --coords or --xrange/--yrange.')

    # New: physical axis ranges (fallback when coords not available)
    ap.add_argument('--xrange', type=float, nargs=2, default=None, metavar=('XMIN', 'XMAX'),
                    help='Physical x-range to use when coords are not available (e.g. --xrange -1 1).')
    ap.add_argument('--yrange', type=float, nargs=2, default=None, metavar=('YMIN', 'YMAX'),
                    help='Physical y-range to use when coords are not available (e.g. --yrange -1 1).')

    args = ap.parse_args()

    table, hdr = load_snapshot_table(args.snapshot)
    dim = hdr.get('dim', None)

    # Detect dimension: prefer header; fall back to number of index columns by dim=None
    if dim is None:
        dim = 2 if table.shape[1] >= 3 else 1

    ncols = table.shape[1]
    if dim == 1:
        var_col = 1 + args.var
        if var_col >= ncols:
            raise IndexError(f'Requested var index {args.var} exceeds available columns ({ncols-1} variables)')
    else:
        var_col = 2 + args.var
        if var_col >= ncols:
            raise IndexError(f'Requested var index {args.var} exceeds available columns ({ncols-2} variables)')

    ng, nx_hdr, ny_hdr = hdr.get('ng'), hdr.get('nx'), hdr.get('ny')

    # Optional coordinates
    coords = None
    if args.coords:
        coords = load_coords_table(args.coords)
        if coords is None:
            print('[warn] Could not load coords file; proceeding without physical coordinates.', file=sys.stderr)

    xr = tuple(args.xrange) if args.xrange is not None else None
    yr = tuple(args.yrange) if args.yrange is not None else None

    plt.figure()

    if dim == 1:
        y, _ = assemble_field_1d(table, var_col, args.trim_ghosts, ng, nx_hdr)
        x_coords, x_label = build_physical_axis_1d(
            n=len(y),
            coords_1d=coords,
            trim_ghosts=args.trim_ghosts,
            nx_hdr=nx_hdr,
            xr=xr
        )

        plt.plot(x_coords, y)
        plt.xlabel(x_label)
        plt.ylabel(f'v{args.var}')

        if args.title:
            plt.title(args.title)
        elif hdr.get('step') is not None and hdr.get('t') is not None:
            plt.title(f"step {hdr['step']}  t={hdr['t']:.6g}  v{args.var}")

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

        # Slice requested -> 1D plot
        if args.slice_axis is not None:
            # slice_value requires some notion of physical coordinates
            if args.slice_value is not None:
                if (args.slice_axis == 'x') and (coords is None) and (yr is None):
                    print('[error] --slice-value with --slice-axis x requires --coords or --yrange.', file=sys.stderr)
                    sys.exit(2)
                if (args.slice_axis == 'y') and (coords is None) and (xr is None):
                    print('[error] --slice-value with --slice-axis y requires --coords or --xrange.', file=sys.stderr)
                    sys.exit(2)

            if args.slice_axis == 'x':
                # plot vs x at fixed j (fixed y)
                if args.slice_value is not None:
                    j0 = nearest_index(y_vec, args.slice_value)
                else:
                    j0 = args.slice-index if args.slice_index is not None else (Z.shape[0] // 2)
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
            # Default 2D image
            if extent is not None:
                dx, dy = extent[1] - extent[0], extent[3] - extent[2]
                aspect_ratio = dx / dy if dy != 0 else 1.0
            else:
                ny, nx = Z.shape
                aspect_ratio = nx / ny if ny != 0 else 1.0

            im = plt.imshow(
                Z, origin='lower',
                extent=extent if extent is not None else None,
                aspect=aspect_ratio
            )

            plt.colorbar(im, label=f'v{args.var}')
            plt.xlabel(xlab)
            plt.ylabel(ylab)

            if args.title:
                plt.title(args.title)
            elif hdr.get('step') is not None and hdr.get('t') is not None:
                plt.title(f"step {hdr['step']}  t={hdr['t']:.6g}  v{args.var}")

    else:
        print(f"[error] dim={dim} not supported by this script (only 1D/2D).", file=sys.stderr)
        sys.exit(2)

    outpath = args.output if args.output else os.path.splitext(args.snapshot)[0] + f"_v{args.var}.png"
    plt.tight_layout()
    # plt.savefig(outpath, dpi=150)
    plt.show()
    print(f"[ok] Wrote {outpath}")


if __name__ == '__main__':
    main()
