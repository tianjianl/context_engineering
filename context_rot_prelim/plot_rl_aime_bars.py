"""Bar plot: AIME accuracy for RL training vs fixed scaffolding baseline."""
import matplotlib.pyplot as plt
import numpy as np

labels = [
    "GRPO baseline\n(no tools)",
    "Tool RL\n(no SFT bonus)",
    "Fixed scaffolding\n(no training)",
]
accs = [61.0, 83.1, 87.6]
colors = ["#377eb8", "#e41a1c", "#4daf4a"]

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(labels, accs, color=colors, width=0.55, edgecolor="white", linewidth=1.5)

for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
            f"{acc:.1f}%", ha="center", va="bottom", fontsize=14, fontweight="bold")

ax.set_ylabel("AIME Accuracy (%)", fontsize=13)
ax.set_ylim(0, 100)
ax.set_title("Qwen3-4B Tool Use: RL Training vs Fixed Scaffolding",
             fontsize=14, fontweight="bold")
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("rl_vs_scaffolding_aime.png", dpi=150, bbox_inches="tight")
plt.show()
