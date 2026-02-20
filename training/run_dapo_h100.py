"""
DAPO RL training on H100s with Qwen3-4B-Instruct-2507.

Single node:
    python run_dapo_h100.py

Multi-node (via SLURM, run on each node):
    python run_dapo_h100.py
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


def prepare(args: ScriptArgs):
    dst = f"{SCRATCH}/models/{MODEL_NAME}_torch_dist"
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
        f"PYTHONPATH={megatron_dir} "
        f"torchrun --nproc-per-node {args.num_gpus_per_node} "
        f"{multinode_args}"
        f"{slime_dir}/tools/convert_hf_to_torch_dist.py "
        f"${{MODEL_ARGS[@]}} "
        f"--hf-checkpoint {SCRATCH}/models/{MODEL_NAME} "
        f"--save {dst}"
    )


def execute(args: ScriptArgs):
    load_save_path = f"{SCRATCH}/outputs/dapo_{MODEL_NAME}_{args.run_id}/checkpoints"
    dump_path = f"{SCRATCH}/outputs/dapo_{MODEL_NAME}_{args.run_id}/dump_details"

    ckpt_args = (
        f"--hf-checkpoint {SCRATCH}/models/{MODEL_NAME} "
        f"--ref-load {SCRATCH}/models/{MODEL_NAME}_torch_dist "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        f"--save-interval {2 if args.mode == 'debug_minimal' else 20} "
        f"--save-retain-interval {2 if args.mode == 'debug_minimal' else 20} "
    )

    rollout_args = (
        f"--prompt-data {SCRATCH}/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type dapo "
        "--reward-key score "
        f"--num-rollout {10 if args.mode == 'debug_minimal' else 3000} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 8192} "
        "--rollout-temperature 0.8 "
        "--global-batch-size 256 "
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
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    # Megatron parallelism for 4B model on 4 H100s (per node):
    #   TP=2, CP=2 → 2×2=4 GPUs per actor replica
    tp = 2
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

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-chunked-prefill-size 4096 "
        "--sglang-mem-fraction-static 0.7 "
    )

    perf_args = "--use-dynamic-batch-size --max-tokens-per-gpu 9216 "

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
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
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
        extra_env_vars={"PYTHONPATH": megatron_dir},
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
