#!/usr/bin/env python3
"""Bin ProofBench-HF performance by generation length and produce a table + plot."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

OUTPUT_DIR = Path("/scratch/dkhasha1/tli104/outputs/qwen3_4b_instruct_2507_proofbench_hf")

METHODS = {
    "Baseline": {
        "raw": OUTPUT_DIR / "baseline_qwen3-4b-instruct-2507_proofbench-hf_t32768_n16_temp0.7_topp0.9.jsonl",
        "graded": OUTPUT_DIR / "baseline_qwen3-4b-instruct-2507_proofbench-hf_t32768_n16_temp0.7_topp0.9_o3_graded.jsonl",
        "format": "baseline",
    },
    "Iterative Refinement (Ours)": {
        "raw": OUTPUT_DIR / "rc_user_qwen3-4b-instruct-2507_proofbench-hf_t32768_rt4096_r12_n16_temp1.0_topp1.0.jsonl",
        "graded": OUTPUT_DIR / "rc_user_qwen3-4b-instruct-2507_proofbench-hf_t32768_rt4096_r12_n16_temp1.0_topp1.0_o3_graded.jsonl",
        "format": "refinement",
    },
    "Reasoning Cache": {
        "raw": OUTPUT_DIR / "rc_fix_qwen3-4b-instruct-2507_proofbench-hf_t16384_r12_n16_temp0.7_topp0.9.jsonl",
        "graded": OUTPUT_DIR / "rc_fix_qwen3-4b-instruct-2507_proofbench-hf_t16384_r12_n16_temp0.7_topp0.9_o3_graded.jsonl",
        "format": "refinement",
    },
}

# Approximate tokens from chars (Qwen tokenizer ~3.5 chars/token for math)
CHARS_PER_TOKEN = 3.5

# Bin edges in tokens (approximate)
BIN_EDGES_TOKENS = [0, 2000, 4000, 8000, 16000, 32000, 64000, 128000, float("inf")]
BIN_LABELS = ["0-2k", "2k-4k", "4k-8k", "8k-16k", "16k-32k", "32k-64k", "64k-128k", "128k+"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def get_gen_lengths_baseline(raw_data):
    """Return dict: (problem_id, sample_within_problem_idx) -> char_length."""
    # Baseline: one line per sample, group by problem_id
    by_problem = defaultdict(list)
    for item in raw_data:
        by_problem[item["problem_id"]].append(item)

    lengths = {}
    for line_idx, (pid, samples) in enumerate(sorted(by_problem.items())):
        for s_idx, sample in enumerate(samples):
            gen = sample.get("generation", "")
            lengths[(line_idx, s_idx)] = len(gen)
    return lengths


def get_gen_lengths_refinement(raw_data):
    """Return dict: (line_idx, sample_idx) -> total char_length across all rounds."""
    lengths = {}
    for line_idx, item in enumerate(raw_data):
        for s_idx, sample in enumerate(item.get("samples", [])):
            total = sum(
                len(r.get("current_round_generation", ""))
                for r in sample.get("rounds", [])
            )
            lengths[(line_idx, s_idx)] = total
    return lengths


def get_scores_from_graded(graded_data):
    """Return dict: (line_idx, sample_idx) -> normalized score (0-1)."""
    scores = {}
    for item in graded_data:
        line_idx = item["line_idx"]
        for gs in item["graded_samples"]:
            s_idx = gs["sample_idx"]
            g = gs.get("grading")
            if g and "score" in g:
                scores[(line_idx, s_idx)] = g["score"] / 7.0
            elif g and "label" in g:
                label_map = {"incorrect": 0, "partial": 0.33, "almost": 0.75, "correct": 1.0}
                scores[(line_idx, s_idx)] = label_map.get(g["label"], 0)
    return scores


def bin_index(char_len):
    tokens = char_len / CHARS_PER_TOKEN
    for i, edge in enumerate(BIN_EDGES_TOKENS[1:]):
        if tokens < edge:
            return i
    return len(BIN_LABELS) - 1


def main():
    all_results = {}

    for method_name, cfg in METHODS.items():
        print(f"Processing {method_name}...")
        raw_data = load_jsonl(cfg["raw"])
        graded_data = load_jsonl(cfg["graded"])

        if cfg["format"] == "baseline":
            lengths = get_gen_lengths_baseline(raw_data)
        else:
            lengths = get_gen_lengths_refinement(raw_data)

        scores = get_scores_from_graded(graded_data)

        # Bin
        bin_scores = defaultdict(list)
        for key, score in scores.items():
            if key in lengths:
                b = bin_index(lengths[key])
                bin_scores[b].append(score)

        all_results[method_name] = bin_scores
        print(f"  {len(scores)} graded samples, {len(lengths)} with lengths")

    # ── Print table ──
    print("\n" + "=" * 90)
    print(f"{'Bin (tokens)':<14}", end="")
    for method in METHODS:
        print(f"{'|':>2} {method:>18} {'n':>6}", end="")
    print()
    print("-" * 90)

    for b_idx, label in enumerate(BIN_LABELS):
        print(f"{label:<14}", end="")
        for method in METHODS:
            scores = all_results[method].get(b_idx, [])
            if scores:
                avg = np.mean(scores) * 100
                print(f"{'|':>2} {avg:>17.2f}% {len(scores):>5}", end="")
            else:
                print(f"{'|':>2} {'—':>18} {'—':>5}", end="")
        print()

    # Total row
    print("-" * 90)
    print(f"{'Total':<14}", end="")
    for method in METHODS:
        all_s = []
        for scores in all_results[method].values():
            all_s.extend(scores)
        avg = np.mean(all_s) * 100 if all_s else 0
        print(f"{'|':>2} {avg:>17.2f}% {len(all_s):>5}", end="")
    print()
    print("=" * 90)

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = {"Baseline": "#2196F3", "Iterative Refinement (Ours)": "#4CAF50", "Reasoning Cache": "#FF9800"}

    # Left: bar chart
    ax = axes[0]
    active_bins = sorted(set(
        b for method_bins in all_results.values() for b in method_bins if method_bins[b]
    ))
    active_labels = [BIN_LABELS[b] for b in active_bins]
    x = np.arange(len(active_bins))
    width = 0.25

    for i, (method, bins) in enumerate(all_results.items()):
        means = []
        for b in active_bins:
            scores = bins.get(b, [])
            means.append(np.mean(scores) * 100 if scores else 0)
        ax.bar(x + (i - 1) * width, means, width, label=method, color=colors[method], alpha=0.85)

    ax.set_xlabel("Generation Length (approx. tokens)")
    ax.set_ylabel("Average Score (%)")
    ax.set_title("ProofBench-HF: Score by Generation Length")
    ax.set_xticks(x)
    ax.set_xticklabels(active_labels, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Right: sample count distribution
    ax2 = axes[1]
    for i, (method, bins) in enumerate(all_results.items()):
        counts = []
        for b in active_bins:
            counts.append(len(bins.get(b, [])))
        ax2.bar(x + (i - 1) * width, counts, width, label=method, color=colors[method], alpha=0.85)

    ax2.set_xlabel("Generation Length (approx. tokens)")
    ax2.set_ylabel("Number of Samples")
    ax2.set_title("ProofBench-HF: Sample Count by Generation Length")
    ax2.set_xticks(x)
    ax2.set_xticklabels(active_labels, rotation=30, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_yscale("log")

    plt.tight_layout()
    out_path = "/weka/home/tli104/context_engineering/proofbench_hf_length_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
