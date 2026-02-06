import matplotlib.pyplot as plt
import numpy as np

benchmarks = ["HMMT Feb 2025", "HMMT Nov 2025", "IMO-Answerbench"]
baseline = [28.75, 41, 35.56]
refine_r3 = [31.4, 44.3, 38.6]
refine_r12 = [32.5, 45.3, 41.2]

x = np.arange(len(benchmarks))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5.5))

bars1 = ax.bar(x - width, baseline, width, label="Baseline", color="#4C72B0", edgecolor="white")
bars2 = ax.bar(x, refine_r3, width, label="Iterative Refinement (r=3)", color="#DD8452", edgecolor="white")
bars3 = ax.bar(x + width, refine_r12, width, label="Iterative Refinement (r=12)", color="#55A868", edgecolor="white")

ax.set_ylabel("Accuracy (%)", fontsize=13)
ax.set_title("Qwen3-4B-Instruct 2507: Baseline vs Iterative Refinement", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(benchmarks, fontsize=12)
ax.legend(fontsize=10.5, loc="upper left")
ax.set_ylim(0, 55)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()

plt.savefig("analysis/qwen3_4b_refinement.png", dpi=200)
plt.savefig("analysis/qwen3_4b_refinement.pdf")
print("Saved to analysis/qwen3_4b_refinement.png and .pdf")
