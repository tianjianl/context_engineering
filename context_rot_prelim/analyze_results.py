#!/usr/bin/env python3
"""Analyse context-rot experiment results.

Loads one or more result JSONL files, computes per-turn accuracy, generates
comparison plots, and prints summary tables.

Usage:
    python -m context_rot_prelim.analyze_results \
        /scratch/dkhasha1/tli104/outputs/context_rot/*.jsonl \
        --output_plot context_rot_results.png

    # Export per-turn CSV for further analysis (R / statsmodels)
    python -m context_rot_prelim.analyze_results \
        /scratch/dkhasha1/tli104/outputs/context_rot/*.jsonl \
        --output_csv context_rot_per_turn.csv
"""

import argparse
import csv as csv_mod
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(paths: List[str]) -> List[Dict]:
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _model_short(name: str) -> str:
    return name.split("/")[-1]


def _ci95(arr):
    """95 % Wilson score interval for binary proportions."""
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    p = np.mean(arr)
    z = 1.96
    denom = 1 + z ** 2 / n
    mid = (p + z ** 2 / (2 * n)) / denom
    delta = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denom
    return max(0, mid - delta), min(1, mid + delta)


def build_groups(results):
    """Group results by (model_short, mode)."""
    groups = defaultdict(list)
    for r in results:
        key = (_model_short(r["model"]), r["mode"])
        groups[key].append(r)
    return groups


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(groups, models):
    for model in models:
        print(f"\n{'=' * 65}")
        print(f"  {model}")
        print(f"{'=' * 65}")

        bl = groups.get((model, "baseline"), [])
        if bl:
            c = sum(r["is_correct"] for r in bl)
            lo, hi = _ci95([r["is_correct"] for r in bl])
            print(f"  Baseline: {c}/{len(bl)} = {c / len(bl):.1%}  "
                  f"[{lo:.1%}, {hi:.1%}]")

        sq = groups.get((model, "sequential"), [])
        if not sq:
            continue

        # per-turn
        by_turn = defaultdict(list)
        ctx_by_turn = defaultdict(list)
        for r in sq:
            if not r.get("context_overflow"):
                by_turn[r["turn"]].append(r["is_correct"])
                ctx_by_turn[r["turn"]].append(r["prompt_tokens"])

        turns = sorted(by_turn)
        print(f"\n  {'Turn':>5} {'Acc':>8} {'95% CI':>16} "
              f"{'N':>5} {'Avg ctx tok':>12}")
        print(f"  {'-' * 52}")
        for t in turns:
            arr = by_turn[t]
            acc = np.mean(arr)
            lo, hi = _ci95(arr)
            ctx = np.mean(ctx_by_turn[t])
            print(f"  {t + 1:>5} {acc:>8.1%} [{lo:>6.1%}, {hi:>6.1%}] "
                  f"{len(arr):>5} {ctx:>12,.0f}")

        # early vs late
        if len(turns) >= 4:
            n3 = max(1, len(turns) // 3)
            early = [v for t in turns[:n3] for v in by_turn[t]]
            late = [v for t in turns[-n3:] for v in by_turn[t]]
            e_acc, l_acc = np.mean(early), np.mean(late)
            print(f"\n  Early (turns {[t+1 for t in turns[:n3]]}) "
                  f"avg: {e_acc:.1%}")
            print(f"  Late  (turns {[t+1 for t in turns[-n3:]]}) "
                  f"avg: {l_acc:.1%}")
            print(f"  Delta : {l_acc - e_acc:+.1%}")

        # overflow count
        overflows = sum(1 for r in sq if r.get("context_overflow"))
        if overflows:
            print(f"  Context overflows: {overflows}")


# ---------------------------------------------------------------------------
# CSV export (for R / statsmodels mixed-effects regression)
# ---------------------------------------------------------------------------

def export_csv(results, path):
    """Write one row per evaluation to CSV for external analysis."""
    fields = ["model", "mode", "conversation_id", "turn", "seed",
              "problem_id", "category", "is_correct",
              "prompt_tokens", "response_tokens",
              "thinking_enabled", "context_overflow"]
    with open(path, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fields}
            row["model"] = _model_short(r["model"])
            # Booleans → 0/1 for statistical tools
            for bk in ("is_correct", "thinking_enabled", "context_overflow"):
                if bk in row:
                    row[bk] = int(bool(row[bk]))
            w.writerow(row)
    print(f"\nExported {len(results)} rows → {path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(groups, models, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    cmap = plt.cm.tab10
    colors = {m: cmap(i / max(len(models) - 1, 1))
              for i, m in enumerate(models)}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for model in models:
        sq = groups.get((model, "sequential"), [])
        if not sq:
            continue

        by_turn = defaultdict(list)
        ctx_by_turn = defaultdict(list)
        for r in sq:
            if not r.get("context_overflow"):
                by_turn[r["turn"]].append(r["is_correct"])
                ctx_by_turn[r["turn"]].append(r["prompt_tokens"])

        turns = sorted(by_turn)
        accs = [np.mean(by_turn[t]) for t in turns]
        lo = [_ci95(by_turn[t])[0] for t in turns]
        hi = [_ci95(by_turn[t])[1] for t in turns]
        ctx = [np.mean(ctx_by_turn[t]) for t in turns]
        x = [t + 1 for t in turns]
        col = colors[model]

        # Accuracy vs Turn
        ax = axes[0]
        ax.plot(x, accs, "o-", label=model, color=col, lw=2, ms=5)
        ax.fill_between(x, lo, hi, color=col, alpha=0.12)
        bl = groups.get((model, "baseline"), [])
        if bl:
            bl_acc = np.mean([r["is_correct"] for r in bl])
            ax.axhline(bl_acc, ls="--", color=col, alpha=0.4)

        # Accuracy vs Context tokens
        ax2 = axes[1]
        ax2.plot(ctx, accs, "o-", label=model, color=col, lw=2, ms=5)
        ax2.fill_between(ctx, lo, hi, color=col, alpha=0.12)

    axes[0].set_xlabel("Turn Number", fontsize=12)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_title("Accuracy vs Turn Position", fontsize=14)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Context Tokens (avg)", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("Accuracy vs Context Length", fontsize=14)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Context Rot analysis")
    p.add_argument("files", nargs="+", help="Result JSONL files")
    p.add_argument("--output_plot", default="context_rot_results.png")
    p.add_argument("--output_csv", default=None,
                   help="Export per-row CSV for mixed-effects regression")
    args = p.parse_args()

    results = load_results(args.files)
    print(f"Loaded {len(results)} result rows from {len(args.files)} file(s)")

    groups = build_groups(results)
    models = sorted({_model_short(r["model"]) for r in results})
    print(f"Models: {models}")

    print_summary(groups, models)

    if args.output_csv:
        export_csv(results, args.output_csv)

    plot_results(groups, models, args.output_plot)


if __name__ == "__main__":
    main()
