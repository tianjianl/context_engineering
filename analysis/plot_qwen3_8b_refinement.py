import matplotlib.pyplot as plt
import numpy as np

benchmarks = ["HMMT Feb 2025", "HMMT Nov 2025", "IMO-Answerbench"]
baseline = [29.7, 35.9, 30.3]
reasoning_cache = [43.4, 50.3, 47.3]
refine_r12_new = [46.3, 51.6, 44.9]

x = np.arange(len(benchmarks))
width = 0.22

fig, ax = plt.subplots(figsize=(10, 5.5))

bars1 = ax.bar(x - width, baseline, width, label="Baseline (t=16384)", color="#4C72B0", edgecolor="white")
bars2 = ax.bar(x, reasoning_cache, width, label="Reasoning Cache (max rounds=12)", color="#8172B3", edgecolor="white")
bars3 = ax.bar(x + width, refine_r12_new, width, label="Iter. Refinement (max rounds=12, new prompt)", color="#C44E52", edgecolor="white")

ax.set_ylabel("Accuracy (%)", fontsize=13)
ax.set_title("Qwen3-8B: Baseline vs Reasoning Cache vs Iterative Refinement", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(benchmarks, fontsize=12)
ax.legend(fontsize=9.5, loc="upper left")
ax.set_ylim(0, 65)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()

plt.savefig("analysis/qwen3_8b_refinement.png", dpi=200)
plt.savefig("analysis/qwen3_8b_refinement.pdf")
print("Saved to analysis/qwen3_8b_refinement.png and .pdf")
