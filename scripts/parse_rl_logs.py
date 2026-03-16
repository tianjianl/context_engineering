#!/usr/bin/env python3
"""Parse RL training logs and produce CSV + plot of rewards and AIME eval over time."""

import re
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = "/scratch/dkhasha1/tli104/slurm_logs"

JOBS = {
    "Tool RL + Bonus": f"{LOG_DIR}/tool_rl_no_sft_bonus_1127063.out",
    "Tool RL No Bonus": f"{LOG_DIR}/tool_rl_no_sft_nobonus_1127808.out",
    "Standard GRPO": f"{LOG_DIR}/grpo_polaris_nostd_1128197.out",
}

def parse_rollout_rewards(path):
    """Extract rollout number and raw_reward from data.py lines."""
    pattern = re.compile(r"rollout (\d+):.*?'rollout/raw_reward': ([0-9.]+)")
    results = []
    with open(path) as f:
        for line in f:
            if "data.py" in line and "rollout/raw_reward" in line:
                m = pattern.search(line)
                if m:
                    results.append((int(m.group(1)), float(m.group(2))))
    return results

def parse_sample_accuracy(path):
    """Extract per-sample accuracy from 'First rollout sample' and 'Finish rollout' lines,
    then compute rolling accuracy per rollout batch."""
    pattern = re.compile(r"'acc': (True|False)")
    scores = []
    with open(path) as f:
        for line in f:
            if ("First rollout sample" in line or "Finish rollout" in line) and "'acc':" in line:
                m = pattern.search(line)
                if m:
                    scores.append(1.0 if m.group(1) == "True" else 0.0)
    return scores

def parse_eval(path):
    """Extract eval results."""
    pattern = re.compile(r"eval (\d+):.*?'eval/aime': (-?[0-9.]+)")
    results = []
    with open(path) as f:
        for line in f:
            if "rollout.py" in line and "eval " in line and "eval/aime" in line:
                m = pattern.search(line)
                if m:
                    results.append((int(m.group(1)), float(m.group(2))))
    return results

# Parse all data
all_data = {}
for name, path in JOBS.items():
    rollout_rewards = parse_rollout_rewards(path)
    sample_acc = parse_sample_accuracy(path)
    eval_results = parse_eval(path)
    all_data[name] = {
        "rollout_rewards": rollout_rewards,
        "sample_acc": sample_acc,
        "eval": eval_results,
    }

# Write CSV
csv_path = "/weka/home/tli104/context_engineering/rl_training_results.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "metric", "step", "value"])
    for name, data in all_data.items():
        for rollout, reward in data["rollout_rewards"]:
            w.writerow([name, "raw_reward", rollout, f"{reward:.6f}"])
        # Compute rolling accuracy (window of 5 samples)
        acc = data["sample_acc"]
        if acc:
            window = 5
            for i in range(len(acc)):
                start = max(0, i - window + 1)
                rolling = np.mean(acc[start:i+1])
                w.writerow([name, "sample_acc_rolling", i, f"{rolling:.4f}"])
        for step, score in data["eval"]:
            w.writerow([name, "eval_aime", step, f"{score:.6f}"])

print(f"CSV written to {csv_path}")
print()

# Print summary
for name, data in all_data.items():
    rr = data["rollout_rewards"]
    sa = data["sample_acc"]
    ev = data["eval"]
    print(f"=== {name} ===")
    print(f"  Rollout reward points: {len(rr)}")
    if rr:
        print(f"  First reward: rollout {rr[0][0]} = {rr[0][1]:.4f}")
        print(f"  Last reward:  rollout {rr[-1][0]} = {rr[-1][1]:.4f}")
    print(f"  Sample accuracy points: {len(sa)}")
    if sa:
        print(f"  Overall accuracy: {np.mean(sa):.3f} ({sum(sa):.0f}/{len(sa)})")
    print(f"  Eval points: {len(ev)}")
    for step, score in ev:
        print(f"    eval {step}: AIME = {score:.4f}")
    print()

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = {"Tool RL + Bonus": "#2196F3", "Tool RL No Bonus": "#FF9800", "Standard GRPO": "#4CAF50"}

# Panel 1: Raw reward over rollouts (only bonus has this)
ax = axes[0]
ax.set_title("Training Reward (raw_reward)")
ax.set_xlabel("Rollout")
ax.set_ylabel("Raw Reward")
for name, data in all_data.items():
    rr = data["rollout_rewards"]
    if rr:
        x, y = zip(*rr)
        ax.plot(x, y, "o-", color=colors[name], label=name, markersize=3, alpha=0.7)
        # Trend line
        z = np.polyfit(list(x), list(y), 1)
        p = np.poly1d(z)
        ax.plot(x, p(list(x)), "--", color=colors[name], alpha=0.5)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: Sample accuracy (rolling)
ax = axes[1]
ax.set_title("Sample Accuracy (rolling window=10)")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Accuracy")
for name, data in all_data.items():
    acc = data["sample_acc"]
    if acc and len(acc) > 1:
        window = min(10, len(acc))
        rolling = np.convolve(acc, np.ones(window)/window, mode='valid')
        x = np.arange(window-1, len(acc))
        ax.plot(x, rolling, "-", color=colors[name], label=f"{name} ({np.mean(acc):.1%})", alpha=0.8)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# Panel 3: AIME eval
ax = axes[2]
ax.set_title("AIME 2024 Eval Score")
ax.set_xlabel("Eval Step")
ax.set_ylabel("Score")
bar_width = 0.25
for i, (name, data) in enumerate(all_data.items()):
    ev = data["eval"]
    if ev:
        x = [e[0] + i * bar_width for e in ev]
        y = [e[1] for e in ev]
        ax.bar(x, y, bar_width, color=colors[name], label=name, alpha=0.8)
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_path = "/weka/home/tli104/context_engineering/rl_training_comparison.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to {plot_path}")
