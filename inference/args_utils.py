"""Shared argparse argument builders for inference scripts."""


def add_common_args(parser):
    """Add arguments shared by all inference scripts.

    Adds: --dataset, --input_file, --cache_dir, --model, --num_tokens,
          --output_file, --num_samples, --temperature, --top_p, --top_k
    """
    parser.add_argument(
        "--dataset", type=str, choices=["hmmt", "imobench", "imobench_v2", "constory"], required=True,
        help="Dataset to use: 'hmmt', 'imobench', 'imobench_v2', or 'constory'"
    )
    parser.add_argument(
        "--input_file", type=str, default=None,
        help="Input file (required for hmmt/constory, optional for imobench)"
    )
    parser.add_argument(
        "--cache_dir", type=str,
        default="/scratch/dkhasha1/tli104/imobench",
        help="Directory to cache downloaded datasets"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
        help="Model name or path to use for generation"
    )
    parser.add_argument(
        "--num_tokens", type=int, required=True,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--output_file", type=str, default="output.jsonl",
        help="Output JSONL file for results"
    )
    parser.add_argument(
        "--num_samples", type=int, default=1,
        help="Number of samples to generate per question (default: 1)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9,
        help="Top-p sampling parameter (default: 0.9)"
    )
    parser.add_argument(
        "--top_k", type=int, default=-1,
        help="Top-k sampling parameter (default: -1, disabled)"
    )


def add_dp_args(parser):
    """Add data-parallelism arguments.

    Adds: --num_gpus, --max_model_len, --gpu_memory_utilization,
          --tensor_parallel_size
    """
    parser.add_argument(
        "--num_gpus", type=int, default=None,
        help="Number of GPUs to use (default: auto-detect)"
    )
    parser.add_argument(
        "--max_model_len", type=int, default=None,
        help="Maximum model context length (default: auto-calculated)"
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.95,
        help="GPU memory utilization fraction (default: 0.95)"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1,
        help="Number of GPUs for tensor parallelism per worker (default: 1)"
    )


def add_refinement_args(parser):
    """Add refinement-related arguments shared by context_refinement and agentic_refinement.

    Adds: --max_refinement_tokens, --preserve_answer, --strip_answer,
          --disable_thinking_for_refinement, --strip_thinking_from_refinement,
          --keep_thinking_in_refinement, --strip_thinking_from_generation,
          --keep_thinking_in_generation
    """
    parser.add_argument(
        "--max_refinement_tokens", type=int, default=None,
        help="Max tokens for refinement (default: 16384)"
    )
    parser.add_argument(
        "--preserve_answer", action="store_true", default=True,
        help="Preserve final answer in refined context"
    )
    parser.add_argument(
        "--strip_answer", action="store_true",
        help="Strip final answer from refined context (sets preserve_answer=False)"
    )
    parser.add_argument(
        "--disable_thinking_for_refinement", action="store_true",
        help="Append /no_think to refinement prompts"
    )
    parser.add_argument(
        "--strip_thinking_from_refinement", action="store_true", default=True,
        help="Strip <think> tags from refinement output (default: True)"
    )
    parser.add_argument(
        "--keep_thinking_in_refinement", action="store_true",
        help="Keep <think> tags in refinement output (overrides strip_thinking_from_refinement)"
    )
    parser.add_argument(
        "--strip_thinking_from_generation", action="store_true", default=True,
        help="Strip <think> tags from generation output (default: True)"
    )
    parser.add_argument(
        "--keep_thinking_in_generation", action="store_true",
        help="Keep <think> tags in generation output (overrides strip_thinking_from_generation)"
    )


def post_process_args(args):
    """Apply shared post-parse logic.

    - strip_answer -> sets preserve_answer = False
    - keep_thinking_in_* -> sets strip_thinking_from_* = False
    - Default max_refinement_tokens = 16384
    - Default max_model_len calculation
    """
    if getattr(args, 'strip_answer', False):
        args.preserve_answer = False

    if getattr(args, 'keep_thinking_in_refinement', False):
        args.strip_thinking_from_refinement = False
    if getattr(args, 'keep_thinking_in_generation', False):
        args.strip_thinking_from_generation = False

    if getattr(args, 'max_refinement_tokens', None) is None:
        args.max_refinement_tokens = 16384

    if getattr(args, 'max_model_len', None) is None:
        max_refinement = getattr(args, 'max_refinement_tokens', 16384)
        accumulate_raw = getattr(args, 'accumulate_raw', False)
        rounds = getattr(args, 'rounds', None)
        if accumulate_raw and rounds is not None:
            args.max_model_len = min(args.num_tokens * (rounds + 2), 131072)
        else:
            args.max_model_len = min(args.num_tokens + max_refinement + 8192, 131072)


def validate_args(args, parser):
    """Validate parsed arguments.

    - hmmt dataset requires --input_file
    """
    if args.dataset == "hmmt" and args.input_file is None:
        parser.error("--input_file is required when using --dataset hmmt")
    if args.dataset == "constory" and args.input_file is None:
        parser.error("--input_file is required when using --dataset constory")
