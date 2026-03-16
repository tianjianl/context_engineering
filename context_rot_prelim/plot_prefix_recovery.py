"""Plot prefix recovery and 2-turn revision rates across models."""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

models = ["Qwen3-4B", "Qwen3-30B-A3B", "Gemini 3 Flash", "MiniMax M2.5", "DeepSeek V3.2"]

# Incorrect baselines per model
incorrect =      [263, 228, 227, 336, 201]
# Vanilla prefix recovery (recovered count)
recovered =      [0,   1,   1,   15,  9]
# 2-turn revision (recovered count; None = not run)
revision_recov = [6,   None, 20, 28, 21]

rates = [r / t * 100 for r, t in zip(recovered, incorrect)]
rev_rates = [r / t * 100 if r is not None else None
             for r, t in zip(revision_recov, incorrect)]

fig, ax = plt.subplots(figsize=(9, 5))
x = range(len(models))
width = 0.35

# Recovery bars
bars1 = ax.bar([i - width/2 for i in x], rates, width,
               color="#4e79a7", alpha=0.85, label="Prefix Recovery")

# Revision bars
rev_vals = [r if r is not None else 0 for r in rev_rates]
has_rev = [r is not None for r in rev_rates]
bars2 = ax.bar([i + width/2 for i in x], rev_vals, width,
               color="#e15759", alpha=0.75, label="2-Turn Revision")
# Hide bars where revision wasn't run
for bar, h in zip(bars2, has_rev):
    if not h:
        bar.set_alpha(0)

# Labels on bars
for bar, rate in zip(bars1, rates):
    if rate > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

for bar, rate, h in zip(bars2, rev_rates, has_rev):
    if h and rate is not None:
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.15,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Recovery Rate (%)", fontsize=13)
ax.set_title("Prefix Recovery vs 2-Turn Revision — IMObench", fontsize=13)
ax.set_ylim(0, max(max(rates), max(v for v in rev_vals if v)) * 1.45)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig("prefix_recovery_rates.png", dpi=150)
print("Saved prefix_recovery_rates.png")
