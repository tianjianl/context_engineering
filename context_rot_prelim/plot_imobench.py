"""Standalone context rot plot (IMOBench). No external files needed."""
import numpy as np, matplotlib.pyplot as plt

# (accs_per_turn, baseline, slope_per_turn, sig, color, marker)
M = {
  "Qwen3-4B":      ([.26,.24,.21,.19,.22,.24,.12,.15,.16,.12], .285, -.018, True,  "#e41a1c","o"),
  "Qwen3-30B-A3B": ([.42,.30,.35,.29,.36,.30,.26,.27,.21,.23], .400, -.015, True,  "#377eb8","^"),
  "DeepSeek-V3.2":  ([.46,.32,.38,.37,.42,.42,.34,.41,.30,.30], .385, -.007, False, "#984ea3","D"),
  "Gemini-3-Flash": ([.40,.38,.42,.34,.55,.41,.36,.39,.32,.38], .425, -.001, False, "#a65628","p"),
  "MiniMax-M2.5":   ([.34,.26,.27,.15,.29,.30,.28,.35,.26,.27], .305, +.011, False, "#66c2a5","X"),
  "Qwen3.5-397B":   ([.44,.37,.44,.43,.50,.53,.50,.52,.52,.47], .330, +.016, True,  "#e7298a","P"),
}

def wilson(p, n=100, z=1.96):
    d = 1 + z**2/n
    m = (p + z**2/(2*n)) / d
    w = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / d
    return max(0, m-w), min(1, m+w)

fig, ax = plt.subplots(figsize=(10, 6))
x = range(1, 11)
for name, (accs, bl, slope, sig, col, mk) in M.items():
    lo, hi = zip(*[wilson(p) for p in accs])
    lab = f"{name} ({slope:+.1%}/turn{'*' if sig else ''})"
    ax.plot(x, accs, f"-{mk}", color=col, label=lab, lw=2.2, ms=8, zorder=3)
    ax.fill_between(x, lo, hi, color=col, alpha=.1, zorder=1)
    ax.axhline(bl, ls=":", color=col, alpha=.4, lw=1.2)

ax.set(xlabel="Turn Number", ylabel="Accuracy", xticks=range(1,11), ylim=(-.02,.62))
ax.set_title("IMOBench — Accuracy vs Turn Position\n(dotted = baseline, shaded = 95% CI)",
             fontsize=14, fontweight="bold")
ax.grid(True, alpha=.25)
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig("context_rot_imobench.png", dpi=150, bbox_inches="tight")
plt.show()
