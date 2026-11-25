"""Render docs/speed.png: Native ComfyUI KSampler vs MOSS C++ under identical settings.

Numbers: RTX 5080 16 GB, 1024x1024, euler / simple / 50 steps / CFG 3, INT8 ConvRot SFT,
measured from the sampler node start to VAE Decode start (see test_results/speed_chart/results.json).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

NATIVE, CPP, FORECAST = "#8c8c8c", "#257de0", "#f47230"
TEXT, MUTED, GRID = "#191919", "#666666", "#e6e6e6"

# (label, seconds, color, speedup_baseline_seconds or None)
NO_LORA_ROWS = [
    ("Native KSampler (first run)", 88.982, NATIVE, None),
    ("Native KSampler (model cached)", 84.236, NATIVE, None),
    ("MOSS C++ (default min_memory)", 21.914, CPP, 84.236),
    ("MOSS C++ (default + forecast)", 16.024, FORECAST, 84.236),
    ("MOSS C++ (keep_vram + balanced)", 15.711, CPP, 84.236),
    ("MOSS C++ (keep_vram + balanced + forecast)", 11.018, FORECAST, 84.236),
]

LORA_ROWS = [
    ("Native KSampler + LoRA (first load)", 90.349, NATIVE, None),
    ("Native KSampler + LoRA (subsequent cached)", 86.164, NATIVE, None),
    ("MOSS C++ + LoRA (first load, default)", 23.266, CPP, 90.349),
    ("MOSS C++ + LoRA (first load, balanced)", 21.017, CPP, 90.349),
    ("MOSS C++ + LoRA (first load, default + forecast)", 17.358, FORECAST, 90.349),
    ("MOSS C++ + LoRA (cached, keep_vram + balanced)", 17.281, CPP, 86.164),
    ("MOSS C++ + LoRA (cached, keep_vram + balanced + forecast)", 12.118, FORECAST, 86.164),
]


def draw_panel(ax, title, rows, xmax):
    ordered = rows[::-1]
    labels = [r[0] for r in ordered]
    values = [r[1] for r in ordered]
    colors = [r[2] for r in ordered]
    ax.barh(labels, values, color=colors, height=0.62)
    for y, (_, value, _, base) in enumerate(ordered):
        note = "" if base is None else f"  ({base / value:.1f}x)"
        ax.text(value + 1.0, y, f"{value:.1f} s{note}", va="center", fontsize=8.8, color=TEXT)
    ax.set_xlim(0, xmax)
    ax.set_title(title, color=TEXT, fontsize=10.5, loc="left", fontweight="bold", pad=6)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0, labelsize=8.8, colors=TEXT)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)


def main():
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10.8, 6.8),
        dpi=150,
        gridspec_kw={"height_ratios": [len(NO_LORA_ROWS), len(LORA_ROWS)]},
    )
    xmax = 108.0
    draw_panel(ax1, "No LoRA (speedup vs. cached Native KSampler)", NO_LORA_ROWS, xmax)
    draw_panel(ax2, "With LoRA: First Load vs. Subsequent Cached (speedup vs. matching Native KSampler)", LORA_ROWS, xmax)
    ax2.set_xlabel("Sampler node time (seconds, lower is faster; measured from sampler start to VAE Decode start)", color=MUTED, fontsize=9)

    fig.suptitle(
        "Native KSampler vs. MOSS C++ Speed (RTX 5080, 1024x1024, euler / simple / 50 steps / CFG 3)",
        color=TEXT,
        fontsize=11.5,
        fontweight="bold",
        x=0.02,
        ha="left",
        y=0.975,
    )
    handles = [
        Patch(color=NATIVE, label="Native KSampler (baseline)"),
        Patch(color=CPP, label="MOSS C++ (no forecast)"),
        Patch(color=FORECAST, label="MOSS C++ + forecast_denoise"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.015, 0.945), ncol=3, frameon=False, fontsize=8.8)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.895))
    fig.savefig(Path(__file__).with_name("speed.png"), facecolor="white")


if __name__ == "__main__":
    main()
