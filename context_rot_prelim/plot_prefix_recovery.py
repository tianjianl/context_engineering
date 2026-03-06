"""Plot prefix recovery rates across models."""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

models = ["Qwen3-4B", "Qwen3-30B-A3B", "Gemini 3 Flash", "MiniMax M2.5", "DeepSeek-V3-2"]
colors = ["#4e79a7", "#59a14f", "#f28e2b", "#b07aa1", "#e15759"]

# Vanilla prefix recovery
incorrect =      [263, 228, 286, 336, 201]
recovered =      [0,   1,   1,   15,  9]
# 2-turn revision (only Gemini so far)
revision_recov = [None, None, 20, None, None]

rates = [r / t * 100 for r, t in zip(recovered, incorrect)]
rev_rates = [r / t * 100 if r is not None else None
             for r, t in zip(revision_recov, incorrect)]

fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(models))
width = 0.35

# Vanilla bars
bars1 = ax.bar([i - width/2 for i in x], rates, width,
               color=colors, alpha=0.85, label="Prefix Recovery")

# Revision bars (only where available)
rev_vals = [r if r is not None else 0 for r in rev_rates]
rev_colors = [c if r is not None else "none" for c, r in zip(colors, rev_rates)]
bars2 = ax.bar([i + width/2 for i in x], rev_vals, width,
               color=rev_colors, alpha=0.45, edgecolor=[c if r is not None else "none" for c, r in zip(colors, rev_rates)],
               linewidth=1.5, label="2-Turn Revision", hatch="//")

for i, (bar, r, t) in enumerate(zip(bars1, recovered, incorrect)):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
            f"{r}/{t}", ha="center", va="bottom", fontsize=9)

for i, (bar, r, t) in enumerate(zip(bars2, revision_recov, incorrect)):
    if r is not None:
        ax.text(bar.get_x() + bar.get_width() / 2, rev_vals[i] + 0.15,
                f"{r}/{t}", ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Recovery Rate (%)", fontsize=13)
ax.set_title("Prefix Recovery vs 2-Turn Revision (IMOBench, temp=0.9, 16K tokens)", fontsize=13)
ax.set_ylim(0, max(max(rates), max(v for v in rev_vals)) * 1.4)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig("prefix_recovery_rates.png", dpi=150)
print("Saved prefix_recovery_rates.png")
