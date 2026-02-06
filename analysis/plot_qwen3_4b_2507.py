import matplotlib.pyplot as plt
import numpy as np

# --- Data ---

# Overall pass@1 results
benchmarks = ["HMMT", "IMO Bench"]
baseline_pass1 = [28.75, 35.56]
rc_pass1 = [26.46, 21.88]

# RC HMMT by-round breakdown
rounds = list(range(1, 13))
rc_hmmt = {
    "pass@1":  [10.62, 16.01, 20.66, 23.64, 23.45, 25.86, 26.68, 25.13, 27.46, 25.25, 27.33, 26.91],
    "pass@2":  [13.56, 22.20, 27.28, 30.65, 31.60, 32.50, 34.21, 32.27, 34.40, 33.51, 34.99, 33.75],
    "pass@4":  [17.38, 27.99, 34.87, 36.77, 40.24, 37.29, 41.87, 38.24, 40.43, 40.42, 43.22, 39.03],
    "pass@8":  [20.95, 31.76, 41.26, 47.01, 47.27, 40.92, 45.44, 42.52, 47.85, 45.62, 48.11, 44.08],
    "pass@16": [23.33, 37.50, 53.85, 46.67, 50.00, 35.71, 33.33, 33.33, 46.15, 52.94, 50.00, 47.37],
}

# Baseline HMMT reference lines
baseline_hmmt_pass1 = 28.75

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Panel 1: Baseline vs RC bar chart ---
ax1 = axes[0]
x = np.arange(len(benchmarks))
w = 0.3
bars1 = ax1.bar(x - w/2, baseline_pass1, w, label="Baseline (t16384)", color="#4C72B0")
bars2 = ax1.bar(x + w/2, rc_pass1, w, label="RC (t4096, r12)", color="#DD8452")

for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10)
for bar in bars2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10)

ax1.set_ylabel("pass@1 (%)")
ax1.set_title("Qwen3-4B-Instruct-2507: Baseline vs RC")
ax1.set_xticks(x)
ax1.set_xticklabels(benchmarks)
ax1.set_ylim(0, 45)
ax1.legend()
ax1.grid(axis="y", alpha=0.3)

# --- Panel 2: RC HMMT by-round line chart ---
ax2 = axes[1]
markers = ["o", "s", "^", "D", "v"]
for (label, vals), marker in zip(rc_hmmt.items(), markers):
    ax2.plot(rounds, vals, marker=marker, markersize=5, label=label)

ax2.axhline(baseline_hmmt_pass1, color="gray", linestyle="--", alpha=0.7, label=f"Baseline pass@1 ({baseline_hmmt_pass1}%)")
ax2.set_xlabel("Round")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("RC on HMMT by Round (Qwen3-4B-Instruct-2507)")
ax2.set_xticks(rounds)
ax2.set_ylim(0, 60)
ax2.legend(fontsize=8, loc="lower right")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("analysis/qwen3_4b_instruct_2507_results.png", dpi=150, bbox_inches="tight")
plt.savefig("analysis/qwen3_4b_instruct_2507_results.pdf", bbox_inches="tight")
print("Saved to analysis/qwen3_4b_instruct_2507_results.png and .pdf")
