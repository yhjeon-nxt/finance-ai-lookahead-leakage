"""Shared publication-grade matplotlib style for every report figure.

One import (`from leakage.run import figstyle as fs; fs.use()`) gives all figures the same
fonts, grid, spines, palette and export settings, so the report reads as one coherent set
instead of seven different default-matplotlib looks. Helpers below cover the repetitive bits:
semantic colors, de-spining, y-grid, value labels, significance brackets, event markers, save.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- semantic palette (used consistently across every figure) ----------------
# Muted, print-friendly tones; treatment is always the same green, control the same blue, the
# out-of-distribution / time-control the same warm orange, significance the same red.
TREAT = "#2f9e6f"     # treatment / in-distribution / "knows the window"
CTRL = "#3b78c3"      # control-A (different-cutoff model, same dates)
OOD = "#e08434"       # control-B / out-of-distribution time control
SIG = "#c0392b"       # statistically notable (p < .1) accent
NEUTRAL = "#9aa0a8"   # not-significant / "denies the window"
CONFAB = "#e0a32a"    # confabulates the window (knows-on-paper, recall is fabricated)
INK = "#22262b"       # primary text / axis ink
FAINT = "#6b7178"     # secondary text (annotations, subtitles)
GRID = "#dfe2e6"      # gridlines

GROUP_COLORS = {"T-in": TREAT, "C-A": CTRL, "C-B": OOD}


def use() -> None:
    """Install the house style. Idempotent; call once at the top of each figure's plot fn."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
        "axes.labelsize": 11,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#b9bec4",
        "axes.linewidth": 1.0,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#d4d8dc",
        "legend.borderpad": 0.6,
        "axes.grid": False,
        "axes.axisbelow": True,
    })


def despine(ax, *, left: bool = False) -> None:
    """Drop the top/right (and optionally left) spines for an airy, publication look."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)


def ygrid(ax) -> None:
    """Subtle horizontal-only reference grid behind the data."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.9, zorder=0)
    ax.grid(axis="x", visible=False)


def zero_line(ax, *, axis: str = "y") -> None:
    line = ax.axhline if axis == "y" else ax.axvline
    line(0, color=INK, lw=1.0, zorder=2)


def title(ax, main: str, sub: str | None = None) -> None:
    """Bold short title with an optional smaller, lighter subtitle line beneath it.

    Both are placed in point-offsets above the axes (not via set_title), so the subtitle sits
    cleanly under the title at any figure size instead of overprinting it.
    """
    if not sub:
        ax.set_title(main, loc="left", color=INK)
        return
    ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=9.5, color=FAINT, annotation_clip=False)
    ax.annotate(main, xy=(0, 1), xycoords="axes fraction", xytext=(0, 19),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=13, fontweight="bold", color=INK, annotation_clip=False)


def bar_labels(ax, bars, values, *, fmt="{:+.3f}", fontsize=9, dy_frac=0.02,
               color=INK, skip_zero=False) -> None:
    """Print each bar's value just outside its end (above positive, below negative)."""
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    pad = span * dy_frac
    for b, v in zip(bars, values):
        if skip_zero and v == 0:
            continue
        up = v >= 0
        ax.text(b.get_x() + b.get_width() / 2, v + (pad if up else -pad),
                fmt.format(v), ha="center", va="bottom" if up else "top",
                fontsize=fontsize, color=color)


def sig_color(p: float) -> str:
    return SIG if (p is not None and p == p and p < 0.1) else FAINT


def sig_star(p: float) -> str:
    if p is None or p != p:
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "ns"


def bracket(ax, i: int, j: int, y: float, p: float, *, h: float, label: str | None = None) -> None:
    """Significance bracket spanning x=i..j at height y, annotated with p (and stars)."""
    col = sig_color(p)
    ax.plot([i, i, j, j], [y, y + h, y + h, y], lw=1.1, color=col, clip_on=False)
    txt = label if label is not None else f"p = {p:.3f}  {sig_star(p)}"
    ax.text((i + j) / 2, y + h * 1.25, txt, ha="center", va="bottom",
            fontsize=9.5, color=col)


def event_marker(ax, x, label: str, *, y_frac: float = 0.96, va: str = "top",
                 ha: str = "left") -> None:
    """Vertical dashed event line with a clean horizontal label box.

    y_frac pins the box height (0 = axis bottom, 1 = top); set va to match so the box sits
    inside the axis (va='top' for high y_frac, va='bottom' for low).
    """
    ax.axvline(x, color=SIG, ls=(0, (4, 3)), lw=1.1, alpha=0.8, zorder=1)
    lo, hi = ax.get_ylim()
    y = lo + (hi - lo) * y_frac
    dx = 4 if ha == "left" else -4
    ax.annotate(label, xy=(x, y), xytext=(dx, 0), textcoords="offset points",
                ha=ha, va=va, fontsize=8.5, color=SIG, zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=SIG, lw=0.8, alpha=0.92))


def legend(ax, **kw):
    kw.setdefault("frameon", True)
    return ax.legend(**kw)


def save(fig, path) -> None:
    fig.savefig(path)
    plt.close(fig)
