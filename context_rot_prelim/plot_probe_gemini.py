"""Standalone positional probe plot for Gemini-3-Flash on HMMT Nov."""
import numpy as np, matplotlib.pyplot as plt

# Hardcoded: 640 results, 22 problems probed, 30 positions
accs = [.409,.455,.636,.636,.455,.591,.636,.545,.545,.455,
        .571,.714,.667,.381,.524,.571,.667,.476,.429,.381,
        .429,.048,.048,.095,.048,.095,.048,.000,.048,.048]

def wilson(p, n=21, z=1.96):
    d = 1 + z**2/n
    m = (p + z**2/(2*n)) / d
    w = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / d
    return max(0, m-w), min(1, m+w)

x = range(1, 31)
lo, hi = zip(*[wilson(a) for a in accs])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, accs, "-o", color="#a65628", lw=2.2, ms=7, zorder=3,
        label=f"Gemini-3-Flash (overall: {np.mean(accs):.1%})")
ax.fill_between(x, lo, hi, color="#a65628", alpha=0.15, zorder=1)
ax.set(xlabel="Position in Batch (1 = first problem)", ylabel="Accuracy",
       xticks=range(1, 31), ylim=(-0.02, 0.88))
ax.set_title("Gemini-3-Flash — Positional Probe (HMMT Nov 2025)\n"
             "Leave-one-out, 22 problems, 30 positions (shaded = 95% CI)",
             fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.25)
ax.legend(fontsize=10, loc="upper right")
plt.tight_layout()
plt.savefig("context_rot_prelim/probe_gemini_hmmt_nov.png", dpi=150, bbox_inches="tight")
print("Saved probe_gemini_hmmt_nov.png")
