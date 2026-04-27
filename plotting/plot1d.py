#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .config import PlotContext
from .fields import prepare_field_1d, get_contact_wave_field
from .io import LoadedSnapshot


def default_output_path(snapshot_path: Path, var: int) -> Path:
    return snapshot_path.with_name(f"{snapshot_path.stem}_v{var}.png")


def build_title(ctx: PlotContext, snapshot: LoadedSnapshot, var: int) -> str:
    if ctx.args.title:
        return ctx.args.title

    if snapshot.time is not None:
        return f"step {snapshot.step}  t={snapshot.time:.6g}  v{var}"

    if snapshot.step >= 0:
        return f"step {snapshot.step}  v{var}"

    return f"v{var}"


def plot_snapshot_1d(ctx: PlotContext,
                     snapshot: LoadedSnapshot,
                     save: bool = False,
                     show: bool = True) -> Optional[Path]:
    field = prepare_field_1d(ctx, snapshot, ctx.args.var)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(field.x, field.values)

    if ctx.args.show_contact_wave:
        contact = get_contact_wave_field(snapshot)
        if contact is not None:
            if contact.ndim != 1:
                raise ValueError(f"Expected 1D contact_wave field, got shape {contact.shape}")

            mask = np.isfinite(contact) & (contact > 0.5)
            if np.any(mask):
                finite_vals = field.values[np.isfinite(field.values)]
                if finite_vals.size > 0:
                    ymin = float(np.min(finite_vals))
                    ymax = float(np.max(finite_vals))
                    yspan = ymax - ymin
                    if yspan <= 0.0:
                        yspan = max(abs(ymax), 1.0)
                else:
                    ymax = 1.0
                    yspan = 1.0

                y_marker = ymax + 0.05 * yspan
                ax.scatter(
                    field.x[mask],
                    np.full(np.count_nonzero(mask), y_marker),
                    color="red",
                    s=18,
                    zorder=5,
                )

                ax.set_ylim(top=ymax + 0.12 * yspan)

    ax.set_xlabel(field.xlabel)
    ax.set_ylabel(field.ylabel)
    ax.set_title(build_title(ctx, snapshot, ctx.args.var))

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