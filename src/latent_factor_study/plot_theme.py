"""Shared publication theme aligned with the thesis figure system."""

from __future__ import annotations

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap


THESIS_PALETTE = {
    "ink": "#111827",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
    "blue": "#0072B2",
    "green": "#009E73",
    "orange": "#D55E00",
    "red": "#D1495B",
}

RETENTION_COLORS = (
    THESIS_PALETTE["blue"],
    THESIS_PALETTE["green"],
    THESIS_PALETTE["orange"],
    THESIS_PALETTE["red"],
)


def apply_thesis_plot_style(*, base_font_size: float = 9.0) -> None:
    """Apply the restrained, colorblind-safe style used across the thesis."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": base_font_size,
            "axes.titlesize": base_font_size + 1.0,
            "axes.titleweight": "bold",
            "axes.labelsize": base_font_size,
            "axes.labelcolor": THESIS_PALETTE["ink"],
            "axes.edgecolor": THESIS_PALETTE["ink"],
            "axes.linewidth": 0.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": THESIS_PALETTE["ink"],
            "xtick.color": THESIS_PALETTE["ink"],
            "ytick.color": THESIS_PALETTE["ink"],
            "xtick.labelsize": max(base_font_size - 1.0, 6.0),
            "ytick.labelsize": max(base_font_size - 1.0, 6.0),
            "legend.fontsize": max(base_font_size - 1.0, 6.0),
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 5.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(axis: Axes, *, grid_axis: str | None = None) -> None:
    """Apply common spine, tick, and optional grid styling to one axis."""
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color(THESIS_PALETTE["ink"])
    axis.tick_params(width=0.8, length=3.0)
    if grid_axis is not None:
        axis.grid(
            axis=grid_axis,
            color=THESIS_PALETTE["light_gray"],
            linewidth=0.7,
            linestyle="-",
        )


def thesis_sequential_cmap(
    *, name: str, color: str, high_is_better: bool
) -> LinearSegmentedColormap:
    """Return a thesis-colored sequential map with the favorable end darkest."""
    pale = "#F8FAFC"
    middle = "#B8D9E8" if color == THESIS_PALETTE["blue"] else "#B7E1D4"
    colors = [pale, middle, color] if high_is_better else [color, middle, pale]
    return LinearSegmentedColormap.from_list(name, colors)
