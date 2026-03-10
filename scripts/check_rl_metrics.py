"""Check RL training metrics and rollout stats from SLURM job logs and dump files.

Usage:
    python scripts/check_rl_metrics.py <job_id>
    python scripts/check_rl_metrics.py <job_id> --rollout 5    # inspect specific rollout
    python scripts/check_rl_metrics.py <job_id> --all-rollouts  # show all rollout summaries
"""

import argparse
import os
import re
import sys
from collections import Counter

SLURM_LOGS = "/scratch/dkhasha1/tli104/slurm_logs"
OUTPUTS = "/scratch/dkhasha1/tli104/outputs"


def find_log(job_id: str) -> tuple:
    for f in os.listdir(SLURM_LOGS):
        if job_id in f and f.endswith(".out"):
            out = os.path.join(SLURM_LOGS, f)
            err = out.replace(".out", ".err")
            return out, err if os.path.exists(err) else None
    return None, None


def find_run_dir(log_path: str) -> str | None:
    with open(log_path) as f:
        for line in f:
            m = re.search(r"(tool_ref_rl_[^\s/\"']+)", line)
            if m:
                candidate = os.path.join(OUTPUTS, m.group(1))
                if os.path.isdir(candidate):
                    return candidate
    return None


def parse_metrics_from_log(log_path: str):
    rollouts = []
    steps = []
    evals = []
    rollout_perfs = []

    with open(log_path) as f:
        for line in f:
            m = re.search(r"rollout\.py:\d+ - eval (\d+): ({.*})", line)
            if m:
                evals.append((int(m.group(1)), eval(m.group(2))))
                continue
            m = re.search(r"rollout\.py:\d+ - perf (\d+): ({.*})", line)
            if m:
                rollout_perfs.append((int(m.group(1)), eval(m.group(2))))
                continue
            m = re.search(r"data\.py:\d+ - rollout (\d+): ({.*})", line)
            if m:
                rollouts.append((int(m.group(1)), eval(m.group(2))))
                continue
            m = re.search(r"model\.py:\d+ - step (\d+): ({.*})", line)
            if m:
                steps.append((int(m.group(1)), eval(m.group(2))))
                continue

    return rollouts, steps, evals, rollout_perfs


def print_summary(rollouts, steps, evals, rollout_perfs):
    if not rollouts and not steps:
        print("No metrics found in log.")
        return

    print(f"\n{'='*80}")
    print(f"Training: {len(rollouts)} rollouts, {len(steps)} steps, {len(evals)} evals")
    print(f"{'='*80}")

    perf_by_id = {pid: m for pid, m in rollout_perfs}

    if rollouts:
        print(f"\n{'Rlout':>6} {'RawRew':>8} {'RespLen':>8} {'Trunc%':>7} {'Rep%':>6} {'std+1':>6} {'std-1':>6}")
        print("-" * 55)
        for rid, m in rollouts:
            raw_rew = m.get("rollout/raw_reward", 0)
            resp_len = m.get("rollout/response_lengths", 0)
            trunc = m.get("rollout/truncated", 0) * 100
            p = perf_by_id.get(rid, {})
            zs_pos = p.get("rollout/zero_std/count_1.0", "")
            zs_neg = p.get("rollout/zero_std/count_-1.0", "")
            rep = p.get("rollout/repetition_frac", 0) * 100
            print(f"{rid:>6} {raw_rew:>8.4f} {resp_len:>8.0f} {trunc:>6.1f}% {rep:>5.1f}% {zs_pos:>6} {zs_neg:>6}")

        if len(rollouts) > 1:
            first = rollouts[0][1]
            last = rollouts[-1][1]
            print(f"\n  Reward: {first.get('rollout/raw_reward',0):.4f} → {last.get('rollout/raw_reward',0):.4f} "
                  f"({last.get('rollout/raw_reward',0)-first.get('rollout/raw_reward',0):+.4f})")
            print(f"  Length: {first.get('rollout/response_lengths',0):.0f} → {last.get('rollout/response_lengths',0):.0f}")

    if evals:
        print(f"\n--- Eval Results ---")
        print(f"{'Step':>6} {'AIME':>8} {'RespLen':>8} {'Trunc%':>7} {'Rep%':>6}")
        print("-" * 40)
        for eid, m in evals:
            score = m.get("eval/aime", 0)
            resp_len = m.get("eval/aime/response_len/mean", 0)
            trunc = m.get("eval/aime/truncated_ratio", 0) * 100
            rep = m.get("eval/aime/repetition_frac", 0) * 100
            print(f"{eid:>6} {score:>8.4f} {resp_len:>8.0f} {trunc:>6.1f}% {rep:>5.1f}%")

    if steps:
        print(f"\n--- Last Train Steps ---")
        print(f"{'Step':>6} {'PGLoss':>10} {'Entropy':>8} {'Clip%':>7} {'KL':>10} {'GradNorm':>10}")
        print("-" * 58)
        for sid, m in steps[-5:]:
            print(f"{sid:>6} {m.get('train/pg_loss',0):>10.6f} {m.get('train/entropy_loss',0):>8.4f} "
                  f"{m.get('train/pg_clipfrac',0)*100:>6.2f}% {m.get('train/ppo_kl',0):>10.6f} "
                  f"{m.get('train/grad_norm',0):>10.4f}")

    # Health check
    print(f"\n--- Health Check ---")
    issues = []

    if rollouts:
        if all(abs(m.get("rollout/rewards", 0)) < 1e-6 for _, m in rollouts[-3:]):
            issues.append("WARN: Normalized rewards stuck at 0 (zero advantage — all samples in group same reward)")
        last_trunc = rollouts[-1][1].get("rollout/truncated", 0) * 100
        if last_trunc > 15:
            issues.append(f"WARN: High truncation ({last_trunc:.1f}%)")
        if len(rollouts) > 3:
            early_len = sum(m.get("rollout/response_lengths", 0) for _, m in rollouts[:3]) / 3
            late_len = sum(m.get("rollout/response_lengths", 0) for _, m in rollouts[-3:]) / 3
            if late_len > early_len * 1.5:
                issues.append(f"WARN: Length growing ({early_len:.0f} → {late_len:.0f}, +{100*(late_len/early_len-1):.0f}%)")
        if len(rollouts) > 5:
            early_rew = sum(m.get("rollout/raw_reward", 0) for _, m in rollouts[:3]) / 3
            late_rew = sum(m.get("rollout/raw_reward", 0) for _, m in rollouts[-3:]) / 3
            if late_rew <= early_rew + 0.01:
                issues.append(f"WARN: Reward stagnant ({early_rew:.4f} → {late_rew:.4f})")

    if steps:
        if steps[-1][1].get("train/grad_norm", 0) > 10:
            issues.append(f"WARN: High grad norm ({steps[-1][1]['train/grad_norm']:.2f})")
        if len(steps) > 10:
            early_ent = sum(m.get("train/entropy_loss", 0) for _, m in steps[:5]) / 5
            late_ent = sum(m.get("train/entropy_loss", 0) for _, m in steps[-5:]) / 5
            if late_ent < early_ent * 0.3:
                issues.append(f"WARN: Entropy collapse ({early_ent:.4f} → {late_ent:.4f})")
        if steps[-1][1].get("train/pg_clipfrac", 0) > 0.3:
            issues.append(f"WARN: High clip fraction ({steps[-1][1]['train/pg_clipfrac']:.1%})")

    if rollout_perfs:
        last_rep = rollout_perfs[-1][1].get("rollout/repetition_frac", 0)
        if last_rep > 0.15:
            issues.append(f"WARN: High repetition ({last_rep:.1%})")

    for issue in issues:
        print(f"  {issue}")
    if not issues:
        print("  All OK")


def _analyze_tool_calls(samples):
    """Analyze tool call positioning and quality."""
    tcs = [s["tool_call_count"] for s in samples]
    accs = [s["reward"]["acc"] for s in samples]

    total_with_tools = sum(1 for t in tcs if t > 0)
    if total_with_tools == 0:
        print(f"  No tool calls in this rollout ({len(samples)} samples)")
        return

    print(f"\n  Tool Usage: {total_with_tools}/{len(samples)} samples ({100*total_with_tools/len(samples):.1f}%)")
    print(f"  Tool count dist: {dict(sorted(Counter(tcs).items()))}")

    # Accuracy split
    acc_with = [accs[i] for i in range(len(samples)) if tcs[i] > 0]
    acc_without = [accs[i] for i in range(len(samples)) if tcs[i] == 0]
    print(f"  Acc WITH tools:    {sum(acc_with)}/{len(acc_with)} ({100*sum(acc_with)/len(acc_with):.1f}%)")
    if acc_without:
        print(f"  Acc WITHOUT tools: {sum(acc_without)}/{len(acc_without)} ({100*sum(acc_without)/len(acc_without):.1f}%)")

    # Tool call positioning
    tc_after_answer = 0
    tc_very_early = 0
    tc_positions = []
    for s in samples:
        if s["tool_call_count"] > 0:
            resp = s["response"]
            tc_pos = resp.find("<tool_call>")
            boxed_pos = resp.find("\\boxed{")
            if tc_pos >= 0:
                frac = tc_pos / max(len(resp), 1)
                tc_positions.append(frac)
                if boxed_pos >= 0 and tc_pos > boxed_pos:
                    tc_after_answer += 1
                if frac < 0.05:
                    tc_very_early += 1

    if tc_positions:
        print(f"\n  Tool Call Positioning (n={len(tc_positions)}):")
        print(f"    Mean position: {sum(tc_positions)/len(tc_positions):.0%} through response")
        print(f"    Very early (<5%): {tc_very_early} ({100*tc_very_early/len(tc_positions):.0f}%)")
        print(f"    After \\boxed answer: {tc_after_answer} ({100*tc_after_answer/len(tc_positions):.0f}%)")
        buckets = [0] * 5
        for p in tc_positions:
            buckets[min(int(p * 5), 4)] += 1
        labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
        print(f"    Distribution: {', '.join(f'{l}:{b}' for l, b in zip(labels, buckets))}")

        # Issues
        if tc_after_answer / len(tc_positions) > 0.5:
            print(f"    ⚠ ISSUE: {tc_after_answer/len(tc_positions):.0%} of tool calls come AFTER a boxed answer — model not learning to call tool proactively")
        if tc_very_early / len(tc_positions) > 0.3:
            print(f"    ⚠ ISSUE: {tc_very_early/len(tc_positions):.0%} of tool calls are very early — model may be calling tool without reasoning first")


def inspect_rollout(run_dir: str, rollout_id: int):
    import torch

    pt_path = os.path.join(run_dir, "dump_details", "rollout_data", f"{rollout_id}.pt")
    if not os.path.exists(pt_path):
        print(f"Rollout file not found: {pt_path}")
        rollout_dir = os.path.join(run_dir, "dump_details", "rollout_data")
        available = sorted(f.replace(".pt", "") for f in os.listdir(rollout_dir) if f.endswith(".pt"))
        print(f"Available: {available}")
        return

    d = torch.load(pt_path, weights_only=False)
    samples = d["samples"]
    accs = [s["reward"]["acc"] for s in samples]
    scores = [s["reward"]["score"] for s in samples]
    tcs = [s["tool_call_count"] for s in samples]
    lens = [s["response_length"] for s in samples]

    print(f"\n{'='*70}")
    print(f"Rollout {rollout_id}: {len(samples)} samples")
    print(f"{'='*70}")
    print(f"  Accuracy: {sum(accs)}/{len(accs)} ({100*sum(accs)/len(accs):.1f}%)")
    print(f"  Score: mean={sum(scores)/len(scores):.3f}")
    print(f"  Length: mean={sum(lens)/len(lens):.0f}, median={sorted(lens)[len(lens)//2]}")

    _analyze_tool_calls(samples)


def all_rollouts_summary(run_dir: str):
    import torch

    rollout_dir = os.path.join(run_dir, "dump_details", "rollout_data")
    if not os.path.isdir(rollout_dir):
        print(f"No rollout data at {rollout_dir}")
        return

    train_files = []
    eval_files = []
    for f in os.listdir(rollout_dir):
        if not f.endswith(".pt"):
            continue
        name = f.replace(".pt", "")
        path = os.path.join(rollout_dir, f)
        if name.startswith("eval_"):
            eval_files.append((int(name.replace("eval_", "")), path))
        else:
            try:
                train_files.append((int(name), path))
            except ValueError:
                continue

    for label, files in [("Train Rollouts", sorted(train_files)), ("Eval Rollouts", sorted(eval_files))]:
        if not files:
            continue
        print(f"\n--- {label} ---")
        print(f"{'ID':>6} {'N':>6} {'Acc%':>7} {'Score':>7} {'Tool%':>7} {'AvgTC':>6} {'AfterAns%':>10} {'AvgLen':>7}")
        print("-" * 65)
        for rid, path in files:
            d = torch.load(path, weights_only=False)
            samples = d["samples"]
            accs = [s["reward"]["acc"] for s in samples]
            scores = [s["reward"]["score"] for s in samples]
            tcs = [s["tool_call_count"] for s in samples]
            lens = [s["response_length"] for s in samples]

            # Tool call after answer %
            after_ans = 0
            with_tool = 0
            for s in samples:
                if s["tool_call_count"] > 0:
                    with_tool += 1
                    resp = s["response"]
                    tc_pos = resp.find("<tool_call>")
                    boxed_pos = resp.find("\\boxed{")
                    if tc_pos >= 0 and boxed_pos >= 0 and tc_pos > boxed_pos:
                        after_ans += 1
            after_pct = 100 * after_ans / with_tool if with_tool else 0

            print(f"{rid:>6} {len(samples):>6} {100*sum(accs)/len(accs):>6.1f}% "
                  f"{sum(scores)/len(scores):>7.3f} "
                  f"{100*sum(1 for t in tcs if t>0)/len(tcs):>6.1f}% "
                  f"{sum(tcs)/len(tcs):>6.2f} "
                  f"{after_pct:>9.0f}% "
                  f"{sum(lens)/len(lens):>7.0f}")


def job_name_from_log(log_path: str) -> str:
    """Extract job name from log filename."""
    base = os.path.basename(log_path)
    # Format: <name>_<jobid>.out
    parts = base.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0]
    return base


def compare_jobs(job_ids: list[str]):
    """Compare eval performance across multiple jobs side by side."""
    job_data = []
    for jid in job_ids:
        log_out, _ = find_log(jid)
        if not log_out:
            print(f"  Skipping {jid}: no log found")
            continue
        rollouts, steps, evals, rollout_perfs = parse_metrics_from_log(log_out)
        name = job_name_from_log(log_out)
        job_data.append({
            "id": jid,
            "name": name,
            "rollouts": rollouts,
            "evals": evals,
            "rollout_perfs": rollout_perfs,
        })

    if not job_data:
        print("No jobs with data found.")
        return

    # --- Eval comparison ---
    any_evals = any(d["evals"] for d in job_data)
    if any_evals:
        # Collect all eval steps across jobs
        all_eval_steps = sorted(set(eid for d in job_data for eid, _ in d["evals"]))

        # Header
        name_width = max(max(len(d["name"]) for d in job_data), 10)
        print(f"\n{'='*80}")
        print("AIME Eval Comparison")
        print(f"{'='*80}")
        print(f"\n{'Job':<{name_width}}  {'ID':>8}", end="")
        for step in all_eval_steps:
            print(f"  {'step '+str(step):>10}", end="")
        print(f"  {'delta':>8}")
        print("-" * (name_width + 12 + 12 * len(all_eval_steps) + 10))

        for d in job_data:
            eval_by_step = {eid: m for eid, m in d["evals"]}
            print(f"{d['name']:<{name_width}}  {d['id']:>8}", end="")
            scores = []
            for step in all_eval_steps:
                if step in eval_by_step:
                    score = eval_by_step[step].get("eval/aime", 0)
                    scores.append(score)
                    print(f"  {score:>9.1%}", end="")
                else:
                    print(f"  {'--':>10}", end="")
            if len(scores) >= 2:
                print(f"  {scores[-1]-scores[0]:>+7.1%}", end="")
            else:
                print(f"  {'--':>8}", end="")
            print()

    # --- Rollout reward comparison ---
    print(f"\n{'='*80}")
    print("Rollout Reward Comparison")
    print(f"{'='*80}")
    name_width = max(max(len(d["name"]) for d in job_data), 10)
    print(f"\n{'Job':<{name_width}}  {'ID':>8}  {'Rlouts':>6}  {'Start':>8}  {'End':>8}  {'Delta':>8}  {'Len0':>6}  {'LenN':>6}")
    print("-" * (name_width + 70))
    for d in job_data:
        n = len(d["rollouts"])
        if n == 0:
            print(f"{d['name']:<{name_width}}  {d['id']:>8}  {'0':>6}  {'--':>8}  {'--':>8}  {'--':>8}  {'--':>6}  {'--':>6}")
            continue
        first_r = d["rollouts"][0][1].get("rollout/raw_reward", 0)
        last_r = d["rollouts"][-1][1].get("rollout/raw_reward", 0)
        first_l = d["rollouts"][0][1].get("rollout/response_lengths", 0)
        last_l = d["rollouts"][-1][1].get("rollout/response_lengths", 0)
        print(f"{d['name']:<{name_width}}  {d['id']:>8}  {n:>6}  {first_r:>8.4f}  {last_r:>8.4f}  {last_r-first_r:>+8.4f}  {first_l:>6.0f}  {last_l:>6.0f}")


def main():
    parser = argparse.ArgumentParser(description="Check RL training metrics from SLURM job")
    parser.add_argument("job_id", nargs="+", help="SLURM job ID(s). Multiple IDs triggers comparison mode.")
    parser.add_argument("--rollout", type=int, help="Inspect a specific rollout .pt file")
    parser.add_argument("--all-rollouts", action="store_true", help="Summary across all rollout .pt files")
    parser.add_argument("--run-dir", help="Override run directory")
    args = parser.parse_args()

    if len(args.job_id) > 1:
        compare_jobs(args.job_id)
        return

    job_id = args.job_id[0]
    log_out, _ = find_log(job_id)
    if not log_out:
        print(f"No log found for job {job_id} in {SLURM_LOGS}")
        sys.exit(1)
    print(f"Log: {log_out}")

    run_dir = args.run_dir
    if not run_dir:
        run_dir = find_run_dir(log_out)
    if run_dir:
        print(f"Run: {run_dir}")

    rollouts, steps, evals, rollout_perfs = parse_metrics_from_log(log_out)
    print_summary(rollouts, steps, evals, rollout_perfs)

    if run_dir and args.rollout is not None:
        inspect_rollout(run_dir, args.rollout)
    elif run_dir and args.all_rollouts:
        all_rollouts_summary(run_dir)
    elif not run_dir and (args.rollout is not None or args.all_rollouts):
        print("\nCould not find run directory. Use --run-dir to specify.")


if __name__ == "__main__":
    main()
