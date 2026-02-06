import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Load data ---
df = pd.read_csv("qwen3_8b_results.csv")

# Focus on the well-sampled refinement runs (n=16, ~5000+ samples)
df_ref = df[(df["method"] == "refinement") & (df["n_samples"] == 16)].copy()
token_budgets = sorted(df_ref["max_tokens"].unique())  # [1024, 2048, 4096]

colors = {1024: "#4C72B0", 2048: "#DD8452", 4096: "#55A868"}
markers = {1024: "o", 2048: "s", 4096: "^"}

# --- Figure 1: pass@1 across rounds, comparing token budgets ---
fig1, ax1 = plt.subplots(figsize=(8, 5))

for tokens in token_budgets:
    subset = df_ref[df_ref["max_tokens"] == tokens].sort_values("round")
    rounds = [int(r[1:]) for r in subset["round"]]
    ax1.plot(rounds, subset["pass@1"].values,
             marker=markers[tokens], markersize=8, linewidth=2.5,
             color=colors[tokens], label=f"t={tokens}")

ax1.set_xlabel("Round", fontsize=12)
ax1.set_ylabel("pass@1 (%)", fontsize=12)
ax1.set_title("Qwen3-8B Refinement on IMO Bench: pass@1 by Token Budget", fontsize=13)
ax1.set_xticks([1, 2, 3, 4])
ax1.set_xticklabels(["R1", "R2", "R3", "R4"])
ax1.set_ylim(0, 30)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("analysis/qwen3_8b_pass1_by_tokens.png", dpi=150, bbox_inches="tight")
plt.savefig("analysis/qwen3_8b_pass1_by_tokens.pdf", bbox_inches="tight")
print("Saved: analysis/qwen3_8b_pass1_by_tokens.png")

# --- Figure 2: Sample attrition across rounds ---
fig2, ax2 = plt.subplots(figsize=(8, 5))

for tokens in token_budgets:
    subset = df_ref[df_ref["max_tokens"] == tokens].sort_values("round")
    rounds = [int(r[1:]) for r in subset["round"]]
    ax2.plot(rounds, subset["total_verified_samples"].values,
             marker=markers[tokens], markersize=8, linewidth=2.5,
             color=colors[tokens], label=f"t={tokens}")

ax2.set_xlabel("Round", fontsize=12)
ax2.set_ylabel("Total Verified Samples", fontsize=12)
ax2.set_title("Qwen3-8B: Sample Attrition Across Rounds", fontsize=13)
ax2.set_xticks([1, 2, 3, 4])
ax2.set_xticklabels(["R1", "R2", "R3", "R4"])
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("analysis/qwen3_8b_sample_attrition.png", dpi=150, bbox_inches="tight")
plt.savefig("analysis/qwen3_8b_sample_attrition.pdf", bbox_inches="tight")
print("Saved: analysis/qwen3_8b_sample_attrition.png")
