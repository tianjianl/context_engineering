"""
Tool Refinement RL training on H100s with Qwen3-4B-Instruct-2507.

Trains the model via GRPO to learn when/how to use the llm_refine tool
for iterative context compression during math problem solving.

Based on run_dapo_h100.py with custom generate function for multi-turn
tool-calling rollouts.

Single node:
    python run_tool_refinement_rl.py

Multi-node (via SLURM, run on each node):
    python run_tool_refinement_rl.py
    SLURM_JOB_NUM_NODES is read automatically.
"""

import os
import sys
from dataclasses import dataclass
from typing import Literal

import typer

# Add slime to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "slime"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Megatron-LM"))

import slime.utils.external_utils.command_utils as U


SCRATCH = "/scratch/dkhasha1/tli104"
MODEL_NAME = "Qwen3-4B-Instruct-2507"
MEGATRON_MODEL_TYPE = "qwen3-4B-Instruct-2507"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    num_gpus_per_node: int = 4  # H100 nodes have 4 GPUs each
    dynamic_sampling: bool = False
    enable_eval: bool = True
    disable_grpo_std_norm: bool = False
    disable_rewards_norm: bool = False
    model_path: str = None  # Custom HF model path (e.g. SFT checkpoint). Defaults to base model.
    dataset_path: str = f"{SCRATCH}/datasets/dapo-math-17k/dapo-math-17k.jsonl"
    wandb_project: str = None
    lr: str = "1e-6"
    global_batch_size: int = 512


def _get_model_paths(args: ScriptArgs):
    """Return (hf_path, torch_dist_path) based on model_path or default."""
    if args.model_path:
        hf_path = args.model_path.rstrip("/")
        torch_dist_path = f"{hf_path}_torch_dist"
    else:
        hf_path = f"{SCRATCH}/models/{MODEL_NAME}"
        torch_dist_path = f"{SCRATCH}/models/{MODEL_NAME}_torch_dist"
    return hf_path, torch_dist_path


def prepare(args: ScriptArgs):
    hf_path, dst = _get_model_paths(args)
    if os.path.exists(dst):
        print(f"Skipping conversion — {dst} already exists")
        return

    slime_dir = os.path.join(os.path.dirname(__file__), "slime")
    megatron_dir = os.path.join(os.path.dirname(__file__), "Megatron-LM")

    multinode_args = ""
    if args.num_nodes > 1:
        job_hostnames = os.environ["SLURM_JOB_HOSTNAMES"].strip().split("\n")
        master_addr = job_hostnames[0]
        node_rank = int(os.environ["SLURM_NODEID"])
        multinode_args = (
            f"--master-addr {master_addr} --master-port 23456 "
            f"--nnodes={args.num_nodes} --node-rank {node_rank} "
        )

    U.exec_command(
        f"source {slime_dir}/scripts/models/{MEGATRON_MODEL_TYPE}.sh && "
        f"PYTHONPATH={megatron_dir}:{slime_dir} "
        f"torchrun --nproc-per-node {args.num_gpus_per_node} "
        f"{multinode_args}"
        f"{slime_dir}/tools/convert_hf_to_torch_dist.py "
        f"${{MODEL_ARGS[@]}} "
        f"--hf-checkpoint {hf_path} "
        f"--save {dst}"
    )


def execute(args: ScriptArgs):
    load_save_path = f"{SCRATCH}/outputs/tool_ref_rl_{MODEL_NAME}_{args.run_id}_bs{args.global_batch_size}_lr{args.lr}/checkpoints"
    dump_path = f"{SCRATCH}/outputs/tool_ref_rl_{MODEL_NAME}_{args.run_id}_bs{args.global_batch_size}_lr{args.lr}/dump_details"

    hf_path, torch_dist_path = _get_model_paths(args)
    ckpt_args = (
        f"--hf-checkpoint {hf_path} "
        f"--ref-load {torch_dist_path} "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        f"--save-interval {2 if args.mode == 'debug_minimal' else 20} "
        f"--save-retain-interval {2 if args.mode == 'debug_minimal' else 20} "
    )

    # Custom generate function for multi-turn tool refinement.
    # NOTE: --apply-chat-template is NOT used — the custom generate function
    # handles chat template formatting internally (with tools=[LLM_REFINE_TOOL]).
    tool_ref_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tool_refinement_rl"
    )
    custom_args = (
        f"--custom-generate-function-path generate.generate "
        f"--custom-rm-path reward.reward_func "
        f"--custom-rollout-log-function-path log_rewards.log_rollout "
        f"--custom-eval-rollout-log-function-path log_rewards.log_eval_rollout "
    )

    debug = args.mode == "debug_minimal"
    rollout_args = (
        f"--prompt-data {args.dataset_path} "
        "--input-key prompt "
        "--label-key label "
        # No --apply-chat-template: custom generate handles formatting
        "--rollout-shuffle "
        "--rm-type dapo "
        "--reward-key score "
        f"--num-rollout {10 if debug else 3000} "
        f"--rollout-batch-size {4 if debug else 256} "
        f"--n-samples-per-prompt {4 if debug else 8} "
        # Longer max response for multi-turn (5 rounds × ~4K tokens + summaries)
        f"--rollout-max-response-len {200 if debug else 16384} "
        "--rollout-temperature 1.0 "
        f"--global-batch-size {16 if debug else args.global_batch_size} "
        "--balance-data "
    )

    if args.dynamic_sampling and args.mode != "debug_minimal":
        rollout_args += (
            "--over-sampling-batch-size 64 "
            "--dynamic-sampling-filter-path "
            "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        )

    eval_args = ""
    if args.enable_eval and args.mode != "debug_minimal":
        eval_args = (
            "--eval-interval 20 "
            f"--eval-prompt-data aime {SCRATCH}/datasets/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 16 "
            "--eval-max-response-len 16384 "
            "--eval-top-p 0.7 "
        )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        + ("--disable-grpo-std-normalization " if args.disable_grpo_std_norm else "")
        + ("--disable-rewards-normalization " if args.disable_rewards_norm else "")
    )

    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    # Megatron parallelism for 4B model:
    #   H200 (4 GPUs): TP=1, CP=2 → 2 GPUs per actor replica (2 replicas)
    #   A100 (8 GPUs): TP=2, CP=4 → 8 GPUs per actor replica
    tp = 1 if args.num_gpus_per_node == 4 else 2
    cp = args.num_gpus_per_node // tp
    train_backend_args = (
        f"--tensor-model-parallel-size {tp} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        f"--context-parallel-size {cp} "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--train-memory-margin-bytes 3221225472 "
    )

    # Lower sglang-mem-fraction-static for longer KV caches
    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-chunked-prefill-size 4096 "
        "--sglang-mem-fraction-static 0.6 "
    )

    # Higher max-tokens-per-gpu for longer multi-turn sequences
    perf_args = "--use-dynamic-batch-size --max-tokens-per-gpu 16384 "

    misc_args = (
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        "--use-fault-tolerance "
        f"--dump-details {dump_path} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{custom_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id, wandb_project=args.wandb_project)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{train_backend_args} "
        f"{misc_args} "
    )

    slime_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slime")
    megatron_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Megatron-LM")
    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        train_script=os.path.join(slime_dir, "train.py"),
        extra_env_vars={
            "PYTHONPATH": f"{megatron_dir}:{tool_ref_dir}",
            **({"WANDB_MODE": os.environ["WANDB_MODE"]} if os.environ.get("WANDB_MODE") else {}),
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
