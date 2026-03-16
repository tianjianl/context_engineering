"""Custom rollout log function that logs reward components separately.

Logs: correctness_reward, tool_bonus, format_penalty, used_tool, num_tool_calls
as separate metrics alongside the default logging.

Tool refinement panel metrics (logged under tool_ref/):
  - raw_accuracy: correctness reward only (no tool bonus), mapped to [0,1]
  - summed_reward: total reward (correctness + tool bonus + format penalty)
  - tool_call_reward: tool bonus component only
  - avg_tool_calls: average number of tool calls per sample
"""

import numpy as np


# Keys from reward dict to log as mean values
REWARD_COMPONENT_KEYS = [
    "correctness_reward",
    "tool_bonus",
    "format_penalty",
    "used_tool",
    "num_tool_calls",
]


def _extract_reward_values(samples, key):
    """Extract float values for a reward dict key from samples."""
    values = []
    for s in samples:
        if isinstance(s.reward, dict) and key in s.reward:
            values.append(float(s.reward[key]))
    return values


def _compute_tool_ref_metrics(samples):
    """Compute the tool refinement panel metrics from samples."""
    metrics = {}

    correctness = _extract_reward_values(samples, "correctness_reward")
    tool_bonus = _extract_reward_values(samples, "tool_bonus")
    format_penalty = _extract_reward_values(samples, "format_penalty")
    num_tool_calls = _extract_reward_values(samples, "num_tool_calls")

    if correctness:
        # raw_accuracy: map correctness from {-1, +1} to {0, 1}
        metrics["tool_ref/raw_accuracy"] = np.mean([(v + 1) / 2 for v in correctness]).item()

    if correctness and tool_bonus and format_penalty:
        # summed_reward: total reward (correctness + tool bonus + format penalty)
        n = min(len(correctness), len(tool_bonus), len(format_penalty))
        summed = [correctness[i] + tool_bonus[i] + format_penalty[i] for i in range(n)]
        metrics["tool_ref/summed_reward"] = np.mean(summed).item()

    if tool_bonus:
        # tool_call_reward: just the tool bonus component
        metrics["tool_ref/tool_call_reward"] = np.mean(tool_bonus).item()

    if num_tool_calls:
        # avg_tool_calls: average number of tool calls per sample
        metrics["tool_ref/avg_tool_calls"] = np.mean(num_tool_calls).item()

    return metrics


def log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    """Custom rollout log function. Returns False to also run default logging."""
    if rollout_extra_metrics is None:
        rollout_extra_metrics = {}

    # Extract and aggregate reward components from sample reward dicts
    for key in REWARD_COMPONENT_KEYS:
        values = _extract_reward_values(samples, key)
        if values:
            rollout_extra_metrics[f"reward/{key}"] = np.mean(values).item()

    # Tool call count distribution (fraction of samples with 0, 1, 2, 3+ calls)
    tc_values = [int(v) for v in _extract_reward_values(samples, "num_tool_calls")]
    if tc_values:
        n = len(tc_values)
        rollout_extra_metrics["reward/tc_0"] = sum(1 for v in tc_values if v == 0) / n
        rollout_extra_metrics["reward/tc_1"] = sum(1 for v in tc_values if v == 1) / n
        rollout_extra_metrics["reward/tc_2"] = sum(1 for v in tc_values if v == 2) / n
        rollout_extra_metrics["reward/tc_3plus"] = sum(1 for v in tc_values if v >= 3) / n

    # Tool refinement panel metrics
    rollout_extra_metrics.update(_compute_tool_ref_metrics(samples))

    # Return False so default logging still runs
    return False


def log_eval_rollout(rollout_id, args, data, extra_metrics):
    """Custom eval rollout log function. Returns False to also run default logging."""
    if extra_metrics is None:
        extra_metrics = {}

    for dataset_key in data.keys():
        samples = data[dataset_key].get("samples")
        if samples is None:
            continue
        for key in REWARD_COMPONENT_KEYS:
            values = _extract_reward_values(samples, key)
            if values:
                extra_metrics[f"eval/{dataset_key}/{key}"] = np.mean(values).item()

        # Tool refinement panel metrics for eval
        for k, v in _compute_tool_ref_metrics(samples).items():
            extra_metrics[f"eval/{dataset_key}/{k.split('/')[-1]}"] = v

    # Return False so default logging still runs
    return False
