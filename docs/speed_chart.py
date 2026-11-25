"""Render docs/speed.png: per-image sampling time for the measured MOSS C++ settings.

Numbers: RTX 5080, 1024x1024, CFG 4, measured from the sampler node start to VAE Decode start; see
MATH_ACCELERATION_RESEARCH.md (solver/step sweep) and LOSSLESS_ACCELERATION_RESEARCH.md (keep_engine, 2nd image on).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

BASELINE, CHANGES, LOSSLESS = "#8c8c8c", "#f47230", "#257de0"
TEXT, MUTED, GRID = "#191919", "#666666", "#e6e6e6"
# (label, seconds, color)
ROWS = [
    ("ER-SDE · simple · 50 步（預設）", 22.35, BASELINE),
    ("dpmpp_2m · beta · 20 步", 10.37, CHANGES),
    ("　＋ keep_engine = release_vram", 7.84, LOSSLESS),
    ("　＋ keep_engine = keep_vram", 7.66, LOSSLESS),
    ("　＋ keep_vram ＋ balanced", 6.41, LOSSLESS),
    ("dpmpp_2m · beta · 12 步", 6.91, CHANGES),
    ("dpmpp_2m · beta · 10 步", 6.21, CHANGES),
]


def main():
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    labels = [row[0] for row in ROWS][::-1]
    values = [row[1] for row in ROWS][::-1]
    colors = [row[2] for row in ROWS][::-1]
    ax.barh(labels, values, color=colors, height=0.62)
    base = ROWS[0][1]
    for y, value in enumerate(values):
        note = "" if value == base else f"  ({base / value:.1f}×)"
        ax.text(value + 0.25, y, f"{value:.1f} s{note}", va="center", fontsize=9, color=TEXT)
    ax.set_xlim(0, base * 1.22)
    ax.set_xlabel("每張圖採樣秒數（越短越快）", color=MUTED, fontsize=9)
    ax.set_title("MOSS C++ 採樣速度（RTX 5080，1024²，CFG 4）", color=TEXT, fontsize=11, loc="left")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0, labelsize=9, colors=TEXT)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.legend(handles=[Patch(color=BASELINE, label="預設設定"),
                       Patch(color=CHANGES, label="換求解器／少步數（出圖改變）"),
                       Patch(color=LOSSLESS, label="keep_engine（與 20 步出圖逐位元相同）")],
              loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name("speed.png"), facecolor="white")


if __name__ == "__main__":
    main()
