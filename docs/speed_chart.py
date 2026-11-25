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
    ("官方 KSampler（首次載入）", 88.982, NATIVE, None),
    ("官方 KSampler（已載入模型）", 84.236, NATIVE, None),
    ("MOSS C++（預設 min_memory）", 21.914, CPP, 84.236),
    ("MOSS C++（預設 ＋ 噪聲預測）", 16.024, FORECAST, 84.236),
    ("MOSS C++（keep_vram ＋ balanced）", 15.711, CPP, 84.236),
    ("MOSS C++（keep_vram ＋ balanced ＋ 噪聲預測）", 11.018, FORECAST, 84.236),
]

LORA_ROWS = [
    ("官方 KSampler ＋ LoRA（第一次載入）", 90.349, NATIVE, None),
    ("官方 KSampler ＋ LoRA（後續已載入）", 86.164, NATIVE, None),
    ("MOSS C++ ＋ LoRA（第一次載入，預設）", 23.266, CPP, 90.349),
    ("MOSS C++ ＋ LoRA（第一次載入，balanced）", 21.017, CPP, 90.349),
    ("MOSS C++ ＋ LoRA（第一次載入，預設 ＋ 噪聲預測）", 17.358, FORECAST, 90.349),
    ("MOSS C++ ＋ LoRA（後續已載入，keep_vram ＋ balanced）", 17.281, CPP, 86.164),
    ("MOSS C++ ＋ LoRA（後續已載入，keep_vram ＋ balanced ＋ 噪聲預測）", 12.118, FORECAST, 86.164),
]


def draw_panel(ax, title, rows, xmax):
    ordered = rows[::-1]
    labels = [r[0] for r in ordered]
    values = [r[1] for r in ordered]
    colors = [r[2] for r in ordered]
    ax.barh(labels, values, color=colors, height=0.62)
    for y, (_, value, _, base) in enumerate(ordered):
        note = "" if base is None else f"  ({base / value:.1f}×)"
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
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10.5, 6.8),
        dpi=150,
        gridspec_kw={"height_ratios": [len(NO_LORA_ROWS), len(LORA_ROWS)]},
    )
    xmax = 108.0
    draw_panel(ax1, "無 LoRA（倍率相對官方已載入模型）", NO_LORA_ROWS, xmax)
    draw_panel(ax2, "掛載 LoRA：第一次載入 vs 後續已載入（倍率相對同條件官方 KSampler）", LORA_ROWS, xmax)
    ax2.set_xlabel("採樣節點耗時（秒，越短越快；從採樣節點開始計至 VAE Decode 開始）", color=MUTED, fontsize=9)

    fig.suptitle(
        "官方 KSampler vs MOSS C++ 速度比較（RTX 5080，1024×1024，euler / simple / 50 步 / CFG 3）",
        color=TEXT,
        fontsize=11.5,
        fontweight="bold",
        x=0.02,
        ha="left",
        y=0.975,
    )
    handles = [
        Patch(color=NATIVE, label="官方 KSampler（基準）"),
        Patch(color=CPP, label="MOSS C++（無噪聲預測）"),
        Patch(color=FORECAST, label="MOSS C++ ＋ 噪聲預測（forecast_denoise）"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.015, 0.945), ncol=3, frameon=False, fontsize=8.8)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.895))
    fig.savefig(Path(__file__).with_name("speed.png"), facecolor="white")


if __name__ == "__main__":
    main()
