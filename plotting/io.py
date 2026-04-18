#!/usr/bin/env python3
from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .config import PlotContext


# ============================================================
# Binary snapshot header
# ============================================================

AETHER_MAGIC = 0x4145544845523031  # "AETHER01"
SNAPSHOT_HEADER_STRUCT = struct.Struct("<QII d Q")
SNAPSHOT_HEADER_SIZE = SNAPSHOT_HEADER_STRUCT.size


@dataclass(frozen=True)
class BinarySnapshotHeader:
    magic: int
    version: int
    step: int
    time: float
    payload_bytes: int


@dataclass(frozen=True)
class SnapshotFileSet:
    step: int
    binary_path: Optional[Path]
    text_path: Optional[Path]


@dataclass(frozen=True)
class LoadedSnapshot:
    step: int
    path: Path
    format_name: str              # "binary" or "plain_txt"
    time: Optional[float]
    header: Optional[BinarySnapshotHeader]
    data: np.ndarray


# ============================================================
# Filename parsing and discovery
# ============================================================

def _snapshot_regex(prefix: str, ext: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(prefix)}_(\d{{6}}){re.escape(ext)}$")


def _discover_snapshot_steps(data_dir: Path,
                             prefix: str,
                             text_ext: str,
                             binary_ext: str) -> dict[int, SnapshotFileSet]:
    text_re = _snapshot_regex(prefix, text_ext)
    bin_re  = _snapshot_regex(prefix, binary_ext)

    found: dict[int, SnapshotFileSet] = {}

    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue

        m_bin = bin_re.match(path.name)
        if m_bin:
            step = int(m_bin.group(1))
            prev = found.get(step, SnapshotFileSet(step=step, binary_path=None, text_path=None))
            found[step] = SnapshotFileSet(step=step, binary_path=path, text_path=prev.text_path)
            continue

        m_txt = text_re.match(path.name)
        if m_txt:
            step = int(m_txt.group(1))
            prev = found.get(step, SnapshotFileSet(step=step, binary_path=None, text_path=None))
            found[step] = SnapshotFileSet(step=step, binary_path=prev.binary_path, text_path=path)

    return found


def discover_snapshots(ctx: PlotContext) -> list[SnapshotFileSet]:
    found = _discover_snapshot_steps(
        data_dir=ctx.snapshot_dir,
        prefix=ctx.meta.snapshot_prefix,
        text_ext=ctx.meta.snapshot_text_extension,
        binary_ext=ctx.meta.snapshot_binary_extension,
    )
    return [found[k] for k in sorted(found.keys())]


# ============================================================
# Snapshot selection
# ============================================================

def choose_snapshot_file(ctx: PlotContext,
                         requested_step: Optional[int] = None,
                         prefer_binary: bool = True) -> Path:
    snapshots = discover_snapshots(ctx)
    if not snapshots:
        raise FileNotFoundError(f"No snapshot files found in '{ctx.snapshot_dir}'")

    if requested_step is None:
        target = snapshots[-1]
    else:
        matches = [s for s in snapshots if s.step == requested_step]
        if not matches:
            raise FileNotFoundError(
                f"No snapshot found for step {requested_step} in '{ctx.snapshot_dir}'"
            )
        target = matches[0]

    if prefer_binary:
        if target.binary_path is not None:
            return target.binary_path
        if target.text_path is not None:
            return target.text_path
    else:
        if target.text_path is not None:
            return target.text_path
        if target.binary_path is not None:
            return target.binary_path

    raise FileNotFoundError(
        f"Snapshot step {target.step} exists, but no readable file was found."
    )


# ============================================================
# Binary reading
# ============================================================

def read_binary_snapshot_header(path: Path) -> BinarySnapshotHeader:
    with path.open("rb") as f:
        raw = f.read(SNAPSHOT_HEADER_SIZE)

    if len(raw) != SNAPSHOT_HEADER_SIZE:
        raise ValueError(f"Binary snapshot header is too short: {path}")

    magic, version, step, time, payload_bytes = SNAPSHOT_HEADER_STRUCT.unpack(raw)

    if magic != AETHER_MAGIC:
        raise ValueError(
            f"Invalid AETHER magic number in '{path}': got 0x{magic:016x}, expected 0x{AETHER_MAGIC:016x}"
        )

    return BinarySnapshotHeader(
        magic=magic,
        version=version,
        step=step,
        time=time,
        payload_bytes=payload_bytes,
    )


def _binary_shape_from_metadata(ctx: PlotContext) -> tuple[int, int, int, int]:
    ng = ctx.meta.ng if ctx.meta.include_ghosts_default else 0

    nx = ctx.meta.nx + 2 * ng
    ny = ctx.meta.ny + 2 * ng if ctx.meta.dimension >= 2 else 1
    nz = ctx.meta.nz + 2 * ng if ctx.meta.dimension >= 3 else 1

    return (ctx.meta.numvar, nz, ny, nx)


def load_binary_snapshot(path: Path,
                         ctx: PlotContext,
                         trim_ghosts: bool = False) -> LoadedSnapshot:
    hdr = read_binary_snapshot_header(path)

    full_shape = _binary_shape_from_metadata(ctx)
    expected_bytes = int(np.prod(full_shape, dtype=np.int64)) * np.dtype(np.float64).itemsize

    if hdr.payload_bytes != expected_bytes:
        raise ValueError(
            f"Binary payload size mismatch for '{path}'. "
            f"Header says {hdr.payload_bytes} bytes, expected {expected_bytes} from metadata."
        )

    arr = np.memmap(
        path,
        dtype=np.float64,
        mode="r",
        offset=SNAPSHOT_HEADER_SIZE,
        shape=full_shape,
        order="C",
    )

    if trim_ghosts and ctx.meta.include_ghosts_default and ctx.meta.ng > 0:
        ng = ctx.meta.ng
        i_slice = slice(ng, ng + ctx.meta.nx)
        j_slice = slice(ng, ng + ctx.meta.ny) if ctx.meta.dimension >= 2 else slice(0, 1)
        k_slice = slice(ng, ng + ctx.meta.nz) if ctx.meta.dimension >= 3 else slice(0, 1)
        arr = arr[:, k_slice, j_slice, i_slice]

    return LoadedSnapshot(
        step=hdr.step,
        path=path,
        format_name="binary",
        time=hdr.time,
        header=hdr,
        data=arr,
    )


# ============================================================
# Plaintext reading
# ============================================================

def _parse_plaintext_header(lines: list[str]) -> dict[str, object]:
    info: dict[str, object] = {
        "step": None,
        "time": None,
        "dim": None,
        "numvar": None,
        "nx": None,
        "ny": None,
        "nz": None,
        "ng": None,
    }

    patterns: dict[str, re.Pattern[str]] = {
        "step": re.compile(r"\bstep\s*=\s*(-?\d+)"),
        "time": re.compile(r"\bt\s*=\s*([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)"),
        "dim": re.compile(r"\bdim\s*=\s*(\d+)"),
        "numvar": re.compile(r"\bnumvar\s*=\s*(\d+)"),
        "nx": re.compile(r"\bnx\s*=\s*(-?\d+)"),
        "ny": re.compile(r"\bny\s*=\s*(-?\d+)"),
        "nz": re.compile(r"\bnz\s*=\s*(-?\d+)"),
        "ng": re.compile(r"\bng\s*=\s*(-?\d+)"),
    }

    for line in lines:
        if not line.lstrip().startswith("#"):
            break
        s = line.strip()
        for key, pat in patterns.items():
            m = pat.search(s)
            if not m:
                continue
            if key == "time":
                info[key] = float(m.group(1))
            else:
                info[key] = int(m.group(1))

    return info


def _assemble_plaintext_dense(table: np.ndarray,
                              dim: int,
                              numvar: int,
                              trim_ghosts: bool,
                              nx: int,
                              ny: int,
                              nz: int) -> np.ndarray:
    if dim == 1:
        idx = table[:, 0].astype(int)
        vals = table[:, 1:1 + numvar]

        i_min = int(np.min(idx))
        i_max = int(np.max(idx))
        nx_all = i_max - i_min + 1

        out = np.full((numvar, 1, 1, nx_all), np.nan, dtype=float)
        for row_i, ii in enumerate(idx):
            out[:, 0, 0, ii - i_min] = vals[row_i, :]

        if trim_ghosts:
            s = max(0 - i_min, 0)
            out = out[:, :, :, s:s + nx]

        return out

    if dim == 2:
        ij = table[:, :2].astype(int)
        vals = table[:, 2:2 + numvar]

        i_min = int(np.min(ij[:, 0]))
        i_max = int(np.max(ij[:, 0]))
        j_min = int(np.min(ij[:, 1]))
        j_max = int(np.max(ij[:, 1]))

        nx_all = i_max - i_min + 1
        ny_all = j_max - j_min + 1

        out = np.full((numvar, 1, ny_all, nx_all), np.nan, dtype=float)
        for row_i, (ii, jj) in enumerate(ij):
            out[:, 0, jj - j_min, ii - i_min] = vals[row_i, :]

        if trim_ghosts:
            si = max(0 - i_min, 0)
            sj = max(0 - j_min, 0)
            out = out[:, :, sj:sj + ny, si:si + nx]

        return out

    if dim == 3:
        ijk = table[:, :3].astype(int)
        vals = table[:, 3:3 + numvar]

        i_min = int(np.min(ijk[:, 0]))
        i_max = int(np.max(ijk[:, 0]))
        j_min = int(np.min(ijk[:, 1]))
        j_max = int(np.max(ijk[:, 1]))
        k_min = int(np.min(ijk[:, 2]))
        k_max = int(np.max(ijk[:, 2]))

        nx_all = i_max - i_min + 1
        ny_all = j_max - j_min + 1
        nz_all = k_max - k_min + 1

        out = np.full((numvar, nz_all, ny_all, nx_all), np.nan, dtype=float)
        for row_i, (ii, jj, kk) in enumerate(ijk):
            out[:, kk - k_min, jj - j_min, ii - i_min] = vals[row_i, :]

        if trim_ghosts:
            si = max(0 - i_min, 0)
            sj = max(0 - j_min, 0)
            sk = max(0 - k_min, 0)
            out = out[:, sk:sk + nz, sj:sj + ny, si:si + nx]

        return out

    raise ValueError(f"Unsupported dimension {dim}")


def load_plaintext_snapshot(path: Path,
                            ctx: PlotContext,
                            trim_ghosts: bool = False) -> LoadedSnapshot:
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    hdr = _parse_plaintext_header(lines)
    table = np.genfromtxt(path, comments="#")
    if table.ndim == 1:
        table = table[None, :]

    dim = int(hdr["dim"]) if hdr["dim"] is not None else ctx.meta.dimension
    numvar = int(hdr["numvar"]) if hdr["numvar"] is not None else ctx.meta.numvar
    nx = int(hdr["nx"]) if hdr["nx"] is not None else ctx.meta.nx
    ny = int(hdr["ny"]) if hdr["ny"] is not None else ctx.meta.ny
    nz = int(hdr["nz"]) if hdr["nz"] is not None else ctx.meta.nz

    dense = _assemble_plaintext_dense(
        table=table,
        dim=dim,
        numvar=numvar,
        trim_ghosts=trim_ghosts,
        nx=nx,
        ny=ny,
        nz=nz,
    )

    return LoadedSnapshot(
        step=int(hdr["step"]) if hdr["step"] is not None else -1,
        path=path,
        format_name="plain_txt",
        time=float(hdr["time"]) if hdr["time"] is not None else None,
        header=None,
        data=dense,
    )

def choose_snapshot_files(ctx: PlotContext,
                          step_start: Optional[int] = None,
                          step_end: Optional[int] = None,
                          step_stride: int = 1,
                          prefer_binary: bool = True) -> list[Path]:
    snapshots = discover_snapshots(ctx)
    if not snapshots:
        raise FileNotFoundError(f"No snapshot files found in '{ctx.snapshot_dir}'")

    selected: list[SnapshotFileSet] = []

    for s in snapshots:
        if step_start is not None and s.step < step_start:
            continue
        if step_end is not None and s.step > step_end:
            continue
        selected.append(s)

    if not selected:
        raise FileNotFoundError("No snapshots matched the requested animation range.")

    if step_stride <= 0:
        raise ValueError("--step-stride must be positive")

    selected = selected[::step_stride]

    paths: list[Path] = []
    for target in selected:
        if prefer_binary:
            if target.binary_path is not None:
                paths.append(target.binary_path)
                continue
            if target.text_path is not None:
                paths.append(target.text_path)
                continue
        else:
            if target.text_path is not None:
                paths.append(target.text_path)
                continue
            if target.binary_path is not None:
                paths.append(target.binary_path)
                continue

        raise FileNotFoundError(
            f"Snapshot step {target.step} exists, but no readable file was found."
        )

    return paths

def load_snapshot_sequence(ctx: PlotContext,
                           prefer_binary: bool = True,
                           trim_ghosts: Optional[bool] = None) -> list[LoadedSnapshot]:
    if trim_ghosts is None:
        trim_ghosts = ctx.args.trim_ghosts

    if ctx.args.snapshot is not None:
        return [load_snapshot(ctx, prefer_binary=prefer_binary, trim_ghosts=trim_ghosts)]

    paths = choose_snapshot_files(
        ctx,
        step_start=ctx.args.step_start,
        step_end=ctx.args.step_end,
        step_stride=ctx.args.step_stride,
        prefer_binary=prefer_binary,
    )

    loaded: list[LoadedSnapshot] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ctx.meta.snapshot_binary_extension.lower():
            loaded.append(load_binary_snapshot(path, ctx, trim_ghosts=trim_ghosts))
        elif suffix == ctx.meta.snapshot_text_extension.lower():
            loaded.append(load_plaintext_snapshot(path, ctx, trim_ghosts=trim_ghosts))
        else:
            raise ValueError(f"Unsupported snapshot extension '{path.suffix}' for file '{path}'")

    return loaded

# ============================================================
# Unified loading
# ============================================================

def load_snapshot(ctx: PlotContext,
                  requested_step: Optional[int] = None,
                  prefer_binary: bool = True,
                  trim_ghosts: Optional[bool] = None) -> LoadedSnapshot:
    if trim_ghosts is None:
        trim_ghosts = ctx.args.trim_ghosts

    if ctx.args.snapshot is not None:
        path = ctx.args.snapshot
    else:
        path = choose_snapshot_file(
            ctx,
            requested_step=requested_step,
            prefer_binary=prefer_binary,
        )

    suffix = path.suffix.lower()
    if suffix == ctx.meta.snapshot_binary_extension.lower():
        return load_binary_snapshot(path, ctx, trim_ghosts=trim_ghosts)

    if suffix == ctx.meta.snapshot_text_extension.lower():
        return load_plaintext_snapshot(path, ctx, trim_ghosts=trim_ghosts)

    raise ValueError(f"Unsupported snapshot extension '{path.suffix}' for file '{path}'")