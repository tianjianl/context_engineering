"""Standalone plot for context rot experiment (IMOBench).

All data is hardcoded — no external files needed. Run in Colab or locally:
    python context_rot_prelim/plot_imobench.py
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Data ──────────────────────────────────────────────────────────────────────
# IMOBench, 200 problems, 5 seeds x 10 turns, N=100 per turn per model.
# Baseline = single-turn (fresh conversation) accuracy.

DATA = {
    "Qwen3-4B": {
        "accs": [0.26, 0.24, 0.21, 0.19, 0.22, 0.24, 0.12, 0.15, 0.16, 0.12],
        "n": 100,
        "baseline": 0.285,
    },
    "Qwen3-30B-A3B": {
        "accs": [0.42, 0.30, 0.35, 0.29, 0.36, 0.30, 0.26, 0.27, 0.21, 0.23],
        "n": 100,
        "baseline": 0.400,
    },
    "DeepSeek-V3.2": {
        "accs": [0.46, 0.32, 0.38, 0.37, 0.42, 0.42, 0.34, 0.41, 0.30, 0.30],
        "n": 100,
        "baseline": 0.385,
    },
    "Gemini-3-Flash": {
        "accs": [0.40, 0.38, 0.42, 0.34, 0.55, 0.41, 0.36, 0.39, 0.32, 0.38],
        "n": 100,
        "baseline": 0.425,
    },
    "MiniMax-M2.5": {
        "accs": [0.34, 0.26, 0.27, 0.15, 0.29, 0.30, 0.28, 0.35, 0.26, 0.27],
        "n": 100,
        "baseline": 0.305,
    },
    "Qwen3.5-397B": {
        "accs": [0.44, 0.37, 0.44, 0.43, 0.50, 0.53, 0.50, 0.52, 0.52, 0.47],
        "n": 100,
        "baseline": 0.330,
    },
}

# Per-problem paired slope (P(correct) change per turn, from regression)
SLOPES = {
    "Qwen3-4B":      {"slope": -0.0181, "se": 0.0058, "sig": True},
    "Qwen3-30B-A3B": {"slope": -0.0151, "se": 0.0061, "sig": True},
    "DeepSeek-V3.2": {"slope": -0.0074, "se": 0.0058, "sig": False},
    "Gemini-3-Flash": {"slope": -0.0007, "se": 0.0065, "sig": False},
    "MiniMax-M2.5":  {"slope": +0.0110, "se": 0.0066, "sig": False},
    "Qwen3.5-397B":  {"slope": +0.0158, "se": 0.0068, "sig": True},
}

# ── Style ─────────────────────────────────────────────────────────────────────

COLORS = {
    "Qwen3-4B":      "#e41a1c",
    "Qwen3-30B-A3B": "#377eb8",
    "DeepSeek-V3.2": "#984ea3",
    "Gemini-3-Flash": "#a65628",
    "MiniMax-M2.5":  "#66c2a5",
    "Qwen3.5-397B":  "#e7298a",
}

MARKERS = {
    "Qwen3-4B": "o", "Qwen3-30B-A3B": "^", "DeepSeek-V3.2": "D",
    "Gemini-3-Flash": "p", "MiniMax-M2.5": "X", "Qwen3.5-397B": "P",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def wilson_ci(p, n, z=1.96):
    """95% Wilson score interval."""
    denom = 1 + z ** 2 / n
    mid = (p + z ** 2 / (2 * n)) / denom
    delta = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denom
    return max(0, mid - delta), min(1, mid + delta)


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot():
    fig, ax = plt.subplots(figsize=(10, 6))
    turns = list(range(1, 11))

    for name, d in DATA.items():
        accs = d["accs"]
        n = d["n"]
        col = COLORS[name]
        mk = MARKERS[name]

        lo = [wilson_ci(p, n)[0] for p in accs]
        hi = [wilson_ci(p, n)[1] for p in accs]

        slope_info = SLOPES[name]
        sig_marker = "*" if slope_info["sig"] else ""
        slope_str = f'{slope_info["slope"]:+.1%}/turn{sig_marker}'
        label = f"{name} ({slope_str})"

        ax.plot(turns, accs, f"-{mk}", color=col, label=label, lw=2.2, ms=8, zorder=3)
        ax.fill_between(turns, lo, hi, color=col, alpha=0.10, zorder=1)
        ax.axhline(d["baseline"], ls=":", color=col, alpha=0.4, lw=1.2)

    ax.set_xlabel("Turn Number", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_title(
        "IMOBench — Accuracy vs Turn Position\n"
        "(dotted lines = single-turn baseline, shaded = 95% CI)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(range(1, 11))
    ax.set_ylim(-0.02, 0.62)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = plot()
    fig.savefig("context_rot_imobench.png", dpi=150, bbox_inches="tight")
    print("Saved → context_rot_imobench.png")
    plt.show()
