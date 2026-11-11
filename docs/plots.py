"""Draw the performance charts of the README into docs/img, in a light and a dark version.

The numbers come from `just bench-devices` on an AMD Ryzen 5 3600 (12 threads)
and an NVIDIA GeForce GTX 1060 6GB. mesh2sdf 1.1.0 comes from `just bench-versions`.
After a new benchmark run, update the numbers below and run

    .venv/bin/python docs/plots.py [preview_dir]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).with_name("img")
MACHINE = "AMD Ryzen 5 3600 (12 threads), NVIDIA GeForce GTX 1060 6GB, Python 3.14. Median of 5 rounds of 3 runs."

# median ms at size 128
ORIGINAL = {"Cube": 6059, "Plane": 7220, "Icosphere, 5k triangles": 7409, "Icosphere, 20k triangles": 9012}
# top to bottom in each panel, in the order of the hardware that each device needs
DEVICES = {
    "1 thread": {"Cube": 252.4, "Plane": 930.6, "Icosphere, 5k triangles": 2349.1, "Icosphere, 20k triangles": 3993.6},
    "12 threads": {"Cube": 73.8, "Plane": 238.4, "Icosphere, 5k triangles": 369.9, "Icosphere, 20k triangles": 589.5},
    "OpenCL": {"Cube": 83.3, "Plane": 122.9, "Icosphere, 5k triangles": 167.8, "Icosphere, 20k triangles": 210.8},
    "CUDA": {"Cube": 66.3, "Plane": 96.5, "Icosphere, 5k triangles": 133.0, "Icosphere, 20k triangles": 157.8},
}
# median ms of the plane at sizes 128, 256 and 512
PLANE_BY_SIZE = {
    "1 thread": [930.6, 5883.1, 41355.5],
    "12 threads": [238.4, 1583.7, 10981.7],
    "OpenCL": [122.9, 487.4, 2876.0],
    "CUDA": [96.5, 377.7, 2154.4],
}
SIZES = [128, 256, 512]

# One color for every device, since no device is the answer for every user (dataviz reference palette).
THEMES = {
    "light": {"bar": "#2a78d6", "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781"},
    "dark": {"bar": "#3987e5", "primary": "#f0efec", "secondary": "#c3c2b7", "muted": "#898781"},
}


def header(fig, c, title, subtitle, source):
    fig.text(0.01, 0.97, title, ha="left", va="top", fontsize=13, fontweight="bold", color=c["primary"])
    fig.text(0.01, 0.885, subtitle, ha="left", va="top", fontsize=10, color=c["secondary"])
    fig.text(0.01, 0.02, f"Source: {source}. {MACHINE}", ha="left", va="bottom", fontsize=7.5, color=c["muted"])


def bar_panels(c, panels, label):
    """One panel of horizontal bars per entry of `panels` ({title: {device: value}}), all on one scale."""
    fig, axes = plt.subplots(1, len(panels), figsize=(10, 3.4), sharey=True)
    peak = max(v for values in panels.values() for v in values.values())
    fig.subplots_adjust(left=0.1, right=0.98, top=0.7, bottom=0.12, wspace=0.12)
    for ax, (title, values) in zip(axes, panels.items()):
        for row, value in enumerate(values.values()):
            ax.barh(row, value, height=0.72, color=c["bar"])
            ax.text(value + peak * 0.03, row, label(value), va="center", fontsize=9.5, color=c["primary"])
        ax.set_title(title, loc="left", fontsize=10, color=c["primary"], pad=6)
        ax.set_xlim(0, peak * 1.35)
        ax.set_xticks([])
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(left=False, labelsize=9.5, labelcolor=c["secondary"])
        ax.patch.set_visible(False)
    axes[0].set_yticks(range(len(DEVICES)), list(DEVICES))
    axes[0].invert_yaxis()  # the axes share y, so invert once
    return fig


def speedup_chart(c):
    panels = {w: {d: ORIGINAL[w] / DEVICES[d][w] for d in DEVICES} for w in ORIGINAL}
    fig = bar_panels(c, panels, lambda x: f"{x:.0f}×")
    header(fig, c, "Every device of mesh2sdf2 is 2× to 91× faster than mesh2sdf 1.1.0",
           "Speedup over mesh2sdf 1.1.0 at size=128. All devices compute the same SDF, bit for bit.",
           "just bench-devices and just bench-versions")
    return fig


def size_chart(c):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    fig.subplots_adjust(left=0.06, right=0.84, top=0.72, bottom=0.2)
    ax.hlines(1, SIZES[0], SIZES[-1], color=c["muted"], linewidth=1, linestyle=(0, (4, 3)))
    ax.text(SIZES[-1] * 1.06, 1, "12 threads (1×)", va="center", fontsize=9.5, color=c["muted"])
    # values above the CUDA line and below the OpenCL line, so the labels never meet
    for device, offset, va in (("CUDA", 0.25, "bottom"), ("OpenCL", -0.25, "top")):
        gains = [cpu / gpu for cpu, gpu in zip(PLANE_BY_SIZE["12 threads"], PLANE_BY_SIZE[device])]
        ax.plot(SIZES, gains, color=c["bar"], linewidth=2.5, marker="o", markersize=7)
        for size, gain in zip(SIZES, gains):
            ax.text(size, gain + offset, f"{gain:.1f}×", ha="center", va=va, fontsize=9.5, color=c["primary"])
        ax.text(SIZES[-1] * 1.06, gains[-1], device, va="center", fontsize=10, color=c["primary"])
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES, [str(s) for s in SIZES])
    ax.set_xlim(SIZES[0] / 1.15, SIZES[-1] * 1.15)
    ax.set_ylim(0, 6.2)
    ax.set_yticks([])
    ax.set_xlabel("Grid size (samples per axis)", loc="left", fontsize=9.5, color=c["secondary"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(c["muted"])
    ax.tick_params(axis="x", colors=c["muted"], labelcolor=c["secondary"], labelsize=9.5)
    ax.minorticks_off()
    ax.patch.set_visible(False)
    header(fig, c, "The GPU gain over 12 threads grows with the grid size",
           "Speedup of each GPU path over 12 CPU threads on the plane mesh, "
           "from 2–2.5× at size 128 to 4–5× at size 512.", "just bench-devices")
    return fig


def main():
    preview = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path", "svg.hashsalt": "mesh2sdf2"})
    for name, draw in (("speedup", speedup_chart), ("grid-size", size_chart)):
        for mode, colors in THEMES.items():
            fig = draw(colors)
            fig.savefig(OUT / f"{name}-{mode}.svg", transparent=True, metadata={"Date": None})
            if preview:
                bg = "#ffffff" if mode == "light" else "#0d1117"
                fig.savefig(preview / f"{name}-{mode}.png", dpi=110, facecolor=bg)
            plt.close(fig)


if __name__ == "__main__":
    main()
