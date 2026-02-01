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


def main():
    ap = argparse.ArgumentParser(description='Plot AETHER plaintext snapshot (1D/2D).')
    ap.add_argument('snapshot', help='Path to snapshot .txt file')
    ap.add_argument('--var', type=int, default=0, help='Variable index to plot (vN). Default 0')
    ap.add_argument('--trim-ghosts', action='store_true', help='Trim ghost cells using header nx,ny,ng if present')
    ap.add_argument('--coords', type=str, default=None, help='Optional coordinates file: 1D (i x) or 2D (i j x y)')
    ap.add_argument('--output', type=str, default=None, help='Optional output image path (.png). Defaults next to snapshot')
    ap.add_argument('--title', type=str, default=None, help='Optional plot title')
    args = ap.parse_args()

    table, hdr = load_snapshot_table(args.snapshot)
    dim = hdr.get('dim', None)

    # Detect dimension: prefer header; fall back to number of index columns by dim=None
    if dim is None:
        # Heuristic: if table has >=3 cols, assume first two are i,j (2D). If only >=2, assume 1D.
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
    x_coords = None
    y_coords = None
    extent = None

    coords = None
    if args.coords:
        coords = load_coords_table(args.coords)
        if coords is None:
            print('[warn] Could not load coords file; proceeding without physical coordinates.', file=sys.stderr)

    plt.figure()

    if dim == 1:
        # Assemble 1D
        y, bounds = assemble_field_1d(table, var_col, args.trim_ghosts, ng, nx_hdr)

        # Build x-axis
        if coords is not None and coords.shape[1] >= 2:
            # Expect (i x) at least; if user includes more cols, take col 1 as x
            i = coords[:, 0].astype(int)
            x = coords[:, 1].astype(float)

            # Map by i index into a dense vector
            i_min, i_max = infer_grid_from_indices_1d(i)
            nx_tot = i_max - i_min + 1
            x_dense = np.full((nx_tot,), np.nan, dtype=float)
            for ii, xv in zip(i, x):
                x_dense[ii - i_min] = xv

            if args.trim_ghosts and (nx_hdr is not None):
                # Trim to [0, nx_hdr-1]
                si0 = max(0 - i_min, 0)
                si1 = min((nx_hdr - 1) - i_min, x_dense.shape[0] - 1)
                x_dense = x_dense[si0:si1 + 1]

            x_coords = x_dense
        else:
            # Fall back to integer i locations
            x_coords = np.arange(len(y), dtype=float)

        plt.plot(x_coords, y)
        plt.xlabel('x' if (coords is not None and coords.shape[1] >= 2) else 'i')
        plt.ylabel(f'v{args.var}')

        if args.title:
            plt.title(args.title)
        elif hdr.get('step') is not None and hdr.get('t') is not None:
            plt.title(f"step {hdr['step']}  t={hdr['t']:.6g}  v{args.var}")

    elif dim == 2:
        # Assemble 2D
        Z, bounds = assemble_field_2d(table, var_col, args.trim_ghosts, ng, nx_hdr, ny_hdr)

        if coords is not None and coords.shape[1] >= 4:
            x_min, x_max = coords[:, 2].min(), coords[:, 2].max()
            y_min, y_max = coords[:, 3].min(), coords[:, 3].max()
            extent = (x_min, x_max, y_min, y_max)
        elif args.coords:
            print('[warn] Coords file format for 2D should be (i j x y); proceeding without extent.', file=sys.stderr)

        # Compute proportional aspect
        if extent is not None:
            dx, dy = extent[1] - extent[0], extent[3] - extent[2]
            aspect_ratio = dx / dy if dy != 0 else 1.0
        else:
            ny, nx = Z.shape
            aspect_ratio = nx / ny if ny != 0 else 1.0

        im = plt.imshow(Z, origin='lower',
                        extent=extent if extent is not None else None,
                        aspect=aspect_ratio)

        plt.colorbar(im, label=f'v{args.var}')
        plt.xlabel('i' if extent is None else 'x')
        plt.ylabel('j' if extent is None else 'y')

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
