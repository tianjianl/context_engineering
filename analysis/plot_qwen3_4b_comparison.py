import matplotlib.pyplot as plt
import numpy as np

benchmarks = ["HMMT Feb 2025", "HMMT Nov 2025", "IMO-Answerbench"]
our_impl = [28.75, 41, 35.56]
official = [31, 39.8, 33.5]

x = np.arange(len(benchmarks))
width = 0.32

fig, ax = plt.subplots(figsize=(8, 5))

bars1 = ax.bar(x - width/2, our_impl, width, label="Our implementation", color="#4C72B0", edgecolor="white")
bars2 = ax.bar(x + width/2, official, width, label="Official numbers", color="#DD8452", edgecolor="white")

ax.set_ylabel("Accuracy (%)", fontsize=13)
ax.set_title("Qwen3-4B-Instruct 2507: Our Implementation vs Official", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(benchmarks, fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(0, 55)

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()

plt.savefig("analysis/qwen3_4b_comparison.png", dpi=200)
plt.savefig("analysis/qwen3_4b_comparison.pdf")
print("Saved to analysis/qwen3_4b_comparison.png and .pdf")
