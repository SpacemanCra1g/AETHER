#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


# ============================================================
# Dataclasses
# ============================================================

@dataclass(frozen=True)
class PlotArgs:
    animate: bool
    step_start: Optional[int]
    step_end: Optional[int]
    step_stride: int
    fps: int
    cell_axis: bool
    step: Optional[int]
    format_preference: str
    data_dir: Path
    metadata: Optional[Path]
    snapshot: Optional[Path]
    var: int
    trim_ghosts: bool
    output: Optional[Path]
    title: Optional[str]
    contours: Optional[int]
    slice_axis: Optional[str]
    slice_index: Optional[int]
    slice_value: Optional[float]
    plane: Optional[str]
    plane_index: Optional[int]
    plane_value: Optional[float]
    xrange: Optional[tuple[float, float]]
    yrange: Optional[tuple[float, float]]
    zrange: Optional[tuple[float, float]]


@dataclass(frozen=True)
class RunMetadata:
    path: Path
    format_name: str
    version: int

    snapshot_prefix: str
    snapshot_text_extension: str
    snapshot_binary_extension: str

    output_plain_txt: bool
    output_binary: bool

    dimension: int
    numvar: int
    scalar_type: str
    payload_kind: str
    file_order_binary: str
    file_order_text: str
    include_ghosts_default: bool

    nx: int
    ny: int
    nz: int
    ng: int

    x_min: float
    x_max: float
    y_min: Optional[float]
    y_max: Optional[float]
    z_min: Optional[float]
    z_max: Optional[float]

    dx: float
    dy: Optional[float]
    dz: Optional[float]

    gamma: float
    quad: int


@dataclass(frozen=True)
class PlotContext:
    args: PlotArgs
    meta: RunMetadata

    @property
    def snapshot_dir(self) -> Path:
        return self.args.data_dir

    @property
    def prefix(self) -> str:
        return self.meta.snapshot_prefix

    @property
    def dim(self) -> int:
        return self.meta.dimension

    @property
    def numvar(self) -> int:
        return self.meta.numvar

    @property
    def text_enabled(self) -> bool:
        return self.meta.output_plain_txt

    @property
    def binary_enabled(self) -> bool:
        return self.meta.output_binary


# ============================================================
# Metadata parsing helpers
# ============================================================

def _strip_comment(line: str) -> str:
    line = line.strip()
    if not line or line.startswith("#"):
        return ""
    return line


def _parse_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = _strip_comment(raw)
            if not line:
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _get_required(raw: dict[str, str], key: str) -> str:
    if key not in raw:
        raise ValueError(f"Metadata file is missing required key '{key}'")
    return raw[key]


def _get_optional(raw: dict[str, str], key: str) -> Optional[str]:
    return raw.get(key)


def _to_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"true", "1", "yes", "on"}:
        return True
    if v in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"Could not parse boolean value '{value}'")


def _to_int(value: str) -> int:
    return int(value.strip())


def _to_float(value: str) -> float:
    return float(value.strip())


def _opt_int(raw: dict[str, str], key: str, default: int) -> int:
    val = _get_optional(raw, key)
    return default if val is None else _to_int(val)


def _opt_float(raw: dict[str, str], key: str) -> Optional[float]:
    val = _get_optional(raw, key)
    return None if val is None else _to_float(val)


def parse_metadata_file(path: Path) -> RunMetadata:
    raw = _parse_key_value_file(path)

    return RunMetadata(
        path=path,
        format_name=_get_required(raw, "format"),
        version=_to_int(_get_required(raw, "version")),

        snapshot_prefix=_get_required(raw, "snapshot_prefix"),
        snapshot_text_extension=_get_required(raw, "snapshot_text_extension"),
        snapshot_binary_extension=_get_required(raw, "snapshot_binary_extension"),

        output_plain_txt=_to_bool(_get_required(raw, "output_plain_txt")),
        output_binary=_to_bool(_get_required(raw, "output_binary")),

        dimension=_to_int(_get_required(raw, "dimension")),
        numvar=_to_int(_get_required(raw, "numvar")),
        scalar_type=_get_required(raw, "scalar_type"),
        payload_kind=_get_required(raw, "payload_kind"),
        file_order_binary=_get_required(raw, "file_order_binary"),
        file_order_text=_get_required(raw, "file_order_text"),
        include_ghosts_default=_to_bool(_get_required(raw, "include_ghosts_default")),

        nx=_to_int(_get_required(raw, "nx")),
        ny=_opt_int(raw, "ny", 1),
        nz=_opt_int(raw, "nz", 1),
        ng=_to_int(_get_required(raw, "ng")),

        x_min=_to_float(_get_required(raw, "x_min")),
        x_max=_to_float(_get_required(raw, "x_max")),
        y_min=_opt_float(raw, "y_min"),
        y_max=_opt_float(raw, "y_max"),
        z_min=_opt_float(raw, "z_min"),
        z_max=_opt_float(raw, "z_max"),

        dx=_to_float(_get_required(raw, "dx")),
        dy=_opt_float(raw, "dy"),
        dz=_opt_float(raw, "dz"),

        gamma=_to_float(_get_required(raw, "gamma")),
        quad=_to_int(_get_required(raw, "quad")),
    )


# ============================================================
# Metadata discovery
# ============================================================

def find_metadata_file(data_dir: Path, explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"Metadata file not found: {explicit}")
        return explicit.resolve()

    matches = sorted(data_dir.glob("*_metadata.txt"))
    if len(matches) == 1:
        return matches[0].resolve()

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No metadata file found in '{data_dir}'. Expected something like '*_metadata.txt'."
        )

    raise RuntimeError(
        f"Multiple metadata files found in '{data_dir}'. Please specify one with --metadata."
    )


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Plot AETHER snapshots from a directory containing metadata and data files."
    )

    ap.add_argument(
    "--animate",
    action="store_true",
    help="Generate an animation over multiple snapshots instead of plotting a single snapshot.",
    )

    ap.add_argument(
        "--step-start",
        type=int,
        default=None,
        help="First snapshot write number for animation.",
    )

    ap.add_argument(
        "--step-end",
        type=int,
        default=None,
        help="Last snapshot write number for animation.",
    )

    ap.add_argument(
        "--step-stride",
        type=int,
        default=1,
        help="Stride between animation frames.",
    )

    ap.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for animation output.",
    )

    ap.add_argument(
    "--cell-axis",
    action="store_true",
    help="Plot using cell index on the x-axis instead of physical coordinates.",
    )

    ap.add_argument(
        "--step",
        type=int,
        default=None,
        help="Snapshot write number to load. Defaults to the final available snapshot.",
    )

    ap.add_argument(
        "--format",
        choices=["auto", "binary", "plain_txt"],
        default="auto",
        help="Snapshot format preference. Default 'auto' prefers binary when available.",
    )

    ap.add_argument(
        "data_dir",
        help="Directory containing the AETHER metadata file and snapshot files.",
    )
    ap.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Optional explicit metadata file path. Otherwise '*_metadata.txt' is auto-detected.",
    )
    ap.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help="Optional explicit snapshot file path. If omitted, later code can select one automatically.",
    )

    ap.add_argument("--var", type=int, default=0, help="Variable index to plot.")
    ap.add_argument("--trim-ghosts", action="store_true", help="Trim ghost cells before plotting.")
    ap.add_argument("--output", type=str, default=None, help="Optional output image path.")
    ap.add_argument("--title", type=str, default=None, help="Optional plot title.")
    ap.add_argument("--contours", type=int, default=None, help="Overlay contour lines with N levels.")

    ap.add_argument("--slice-axis", choices=["x", "y", "z"], default=None)
    ap.add_argument("--slice-index", type=int, default=None)
    ap.add_argument("--slice-value", type=float, default=None)

    ap.add_argument("--plane", choices=["xy", "xz", "yz"], default=None)
    ap.add_argument("--plane-index", type=int, default=None)
    ap.add_argument("--plane-value", type=float, default=None)

    ap.add_argument("--xrange", type=float, nargs=2, default=None, metavar=("XMIN", "XMAX"))
    ap.add_argument("--yrange", type=float, nargs=2, default=None, metavar=("YMIN", "YMAX"))
    ap.add_argument("--zrange", type=float, nargs=2, default=None, metavar=("ZMIN", "ZMAX"))

    return ap


def parse_args(argv: Optional[Sequence[str]] = None) -> PlotArgs:
    ns = build_parser().parse_args(argv)

    data_dir = Path(ns.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data directory not found: {data_dir}")

    metadata = None if ns.metadata is None else Path(ns.metadata).expanduser().resolve()
    snapshot = None if ns.snapshot is None else Path(ns.snapshot).expanduser().resolve()

    output = None if ns.output is None else Path(ns.output).expanduser().resolve()

    return PlotArgs(
        animate=ns.animate,
        step_start=ns.step_start,
        step_end=ns.step_end,
        step_stride=ns.step_stride,
        fps=ns.fps,
        cell_axis=ns.cell_axis,
        data_dir=data_dir,
        metadata=metadata,
        snapshot=snapshot,
        step=ns.step,
        format_preference=ns.format,
        var=ns.var,
        trim_ghosts=ns.trim_ghosts,
        output=output,
        title=ns.title,
        contours=ns.contours,
        slice_axis=ns.slice_axis,
        slice_index=ns.slice_index,
        slice_value=ns.slice_value,
        plane=ns.plane,
        plane_index=ns.plane_index,
        plane_value=ns.plane_value,
        xrange=None if ns.xrange is None else (float(ns.xrange[0]), float(ns.xrange[1])),
        yrange=None if ns.yrange is None else (float(ns.yrange[0]), float(ns.yrange[1])),
        zrange=None if ns.zrange is None else (float(ns.zrange[0]), float(ns.zrange[1])),
    )


def load_context_from_cli(argv: Optional[Sequence[str]] = None) -> PlotContext:
    args = parse_args(argv)
    metadata_path = find_metadata_file(args.data_dir, args.metadata)
    meta = parse_metadata_file(metadata_path)

    if args.var < 0 or args.var >= meta.numvar:
        raise ValueError(
            f"Requested --var={args.var}, but metadata says numvar={meta.numvar}"
        )

    return PlotContext(args=args, meta=meta)