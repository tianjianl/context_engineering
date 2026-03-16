Check RL training internals for all currently running RL jobs. Produce a concise table summary of progress and AIME test metrics.

Do not ask for permission — run all commands directly.

Steps:
1. Run `squeue -u tli104 -p a100,h100,h200 -o "%.10i %.30j %.10P %.8T %.12M" | grep -i "tool_rl\|grpo"` to find running RL jobs.
2. For each running RL job, use `scripts/check_rl_metrics.py` to analyze it:
   - Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate /scratch/dkhasha1/tli104/vllm && python3 /weka/home/tli104/context_engineering/scripts/check_rl_metrics.py <job_id>`
   - This parses log metrics (rollout rewards, train steps, eval results) and runs health checks.
3. For the latest rollout of each job, inspect the .pt dump to check tool call behavior:
   - Run: `python3 /weka/home/tli104/context_engineering/scripts/check_rl_metrics.py <job_id> --rollout <latest_rollout_id>`
   - This shows accuracy, tool call distribution, and whether tool calls happen before/after the boxed answer.
4. If there are multiple running jobs, run a head-to-head comparison:
   - Run: `python3 /weka/home/tli104/context_engineering/scripts/check_rl_metrics.py <job_id_1> <job_id_2> ...` (pass all job IDs)
   - This produces a side-by-side table of AIME eval scores and rollout rewards across jobs.
5. Present a **table summary** in markdown with the following columns for each job:

### Progress Table

| Job ID | Name | Rollouts | Reward (start) | Reward (end) | Delta | Resp Len (start) | Resp Len (end) | Trunc% | Rep% |
|--------|------|----------|----------------|--------------|-------|-------------------|----------------|--------|------|

### AIME Test Metrics Table

| Job ID | Name | Eval Step | AIME Score | Resp Len | Trunc% | Rep% |
|--------|------|-----------|------------|----------|--------|------|

For each job, show ALL eval steps so the AIME score trajectory is visible.

### Health Issues

List any warnings from the health check (length growth, reward stagnation, entropy collapse, high clip fraction, grad norm spikes, high repetition, zero advantage).

Key metrics to watch:
- `rollout/raw_reward`: should trend upward
- `rollout/response_lengths`: length growing = reward hacking
- `rollout/truncated`: >15% means too many samples hitting max length
- `train/entropy_loss`: collapsing entropy = mode collapse
- `train/pg_clipfrac`: >30% = updates too aggressive
- `train/grad_norm`: spikes = instability
- `rollout/repetition_frac`: >15% = degenerate outputs
- `zero_std(+1)` / `zero_std(-1)`: ratio of groups where all correct vs all wrong
- Tool calls after \\boxed answer: model not using tool proactively = tool not being learned properly
