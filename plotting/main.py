#!/usr/bin/env python3
from __future__ import annotations

import sys

from .animate import animate_snapshots
from .config import load_context_from_cli
from .io import load_snapshot, load_snapshot_sequence
from .plot1d import plot_snapshot_1d
from .plot2d import plot_snapshot_2d
from .plot3d import plot_snapshot_3d


def main(argv: list[str] | None = None) -> int:
    try:
        ctx = load_context_from_cli(argv)
        prefer_binary = (ctx.args.format_preference != "plain_txt")

        if ctx.args.animate:
            snapshots = load_snapshot_sequence(ctx, prefer_binary=prefer_binary)
            outpath = animate_snapshots(
                ctx,
                snapshots,
                save=(ctx.args.output is not None),
                show=True,
            )
        else:
            snapshot = load_snapshot(
                ctx,
                requested_step=ctx.args.step,
                prefer_binary=prefer_binary,
            )

            if ctx.meta.dimension == 1:
                outpath = plot_snapshot_1d(ctx, snapshot, save=(ctx.args.output is not None), show=True)
            elif ctx.meta.dimension == 2:
                outpath = plot_snapshot_2d(ctx, snapshot, save=(ctx.args.output is not None), show=True)
            elif ctx.meta.dimension == 3:
                outpath = plot_snapshot_3d(ctx, snapshot, save=(ctx.args.output is not None), show=True)
            else:
                raise NotImplementedError(f"Unsupported dimension={ctx.meta.dimension}")

        if outpath is not None:
            print(f"[ok] Wrote {outpath}")

        return 0

    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())