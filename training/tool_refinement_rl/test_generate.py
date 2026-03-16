"""
Verification test for the tool refinement RL generate function.

Launches a standalone sglang server, runs a few samples through the
generate function, and validates token tracking, loss masks, and logprobs.

Usage (on a GPU node):
    cd /weka/home/tli104/context_engineering/training
    python tool_refinement_rl/test_generate.py

Requires: sglang installed, GPU available, model downloaded.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from argparse import Namespace

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "slime"))
sys.path.insert(0, os.path.dirname(__file__))

SCRATCH = "/scratch/dkhasha1/tli104"
MODEL = f"{SCRATCH}/models/Qwen3-4B-Instruct-2507"
PORT = 30100  # Avoid conflict with existing servers


def start_sglang_server():
    """Start a standalone sglang server for testing."""
    print(f"Starting sglang server on port {PORT} with model {MODEL}...")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", MODEL,
            "--port", str(PORT),
            "--mem-fraction-static", "0.7",
            "--trust-remote-code",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for server to be ready
    import urllib.request
    for attempt in range(120):
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/health")
            urllib.request.urlopen(req, timeout=2)
            print(f"sglang server ready after {attempt + 1}s")
            return proc
        except Exception:
            time.sleep(1)
    print("ERROR: sglang server failed to start within 120s")
    proc.kill()
    sys.exit(1)


def make_mock_args():
    """Create a mock args namespace matching what slime's GenerateState expects."""
    return Namespace(
        hf_checkpoint=MODEL,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=PORT,
        sglang_server_concurrency=64,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        rollout_temperature=0.8,
        rollout_top_p=0.9,
        rollout_top_k=-1,
        rollout_max_response_len=8192,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        rollout_max_context_len=None,
        context_parallel_size=1,
        max_tokens_per_gpu=32768,
        partial_rollout=False,
        use_rollout_routing_replay=False,
        use_slime_router=False,
        sglang_speculative_algorithm=None,
        sglang_enable_deterministic_inference=False,
        ci_test=False,
        use_http2=False,
        use_distributed_post=False,
    )


async def test_generate():
    """Test the generate function with a simple math problem."""
    from slime.utils.types import Sample
    from slime.utils.http_utils import init_http_client
    from generate import generate

    args = make_mock_args()

    # Initialize the HTTP client (required before post() works)
    init_http_client(args)

    # Hard AIME problem + explicit instruction to use the tool.
    # We also cap per-turn tokens to 1024 so the model can't finish in one shot.
    sample = Sample(
        prompt=(
            "Let $S$ be the set of all positive rational numbers $r$ such that "
            "the decimal representation of $r$ has the property that the digits "
            "of $r$ form an eventually periodic sequence. Find the number of "
            "elements in $S$ that can be written as $\\frac{a}{b}$ where $a$ "
            "and $b$ are positive integers with $a + b \\leq 1000$.\n\n"
            "IMPORTANT: You MUST call the llm_refine tool at least once to "
            "compress your work before giving a final answer. First explore "
            "approaches, then call llm_refine to summarize your progress, "
            "then continue to the solution."
        ),
        label="",  # We don't care about correctness, just tool call mechanics
        index=0,
        group_index=0,
    )

    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.9,
        "max_new_tokens": 1024,  # Cap per-turn to force multi-turn
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }

    print("\n" + "=" * 60)
    print("TEST: Running generate function")
    print("=" * 60)

    result = await generate(args, sample, sampling_params)

    # Validate results
    errors = []

    # 1. Check token lengths
    prompt_len = len(result.tokens) - result.response_length
    if prompt_len <= 0:
        errors.append(f"Prompt length should be > 0, got {prompt_len}")

    if result.response_length != len(result.loss_mask):
        errors.append(
            f"response_length ({result.response_length}) != "
            f"len(loss_mask) ({len(result.loss_mask)})"
        )

    if result.rollout_log_probs is not None:
        if result.response_length != len(result.rollout_log_probs):
            errors.append(
                f"response_length ({result.response_length}) != "
                f"len(rollout_log_probs) ({len(result.rollout_log_probs)})"
            )

    # 2. Check loss mask values
    unique_mask_values = set(result.loss_mask)
    if not unique_mask_values.issubset({0, 1}):
        errors.append(f"loss_mask should only contain 0/1, got {unique_mask_values}")

    num_trained = sum(result.loss_mask)
    num_masked = len(result.loss_mask) - num_trained
    print(f"\nToken stats:")
    print(f"  Prompt tokens: {prompt_len}")
    print(f"  Response tokens: {result.response_length}")
    print(f"  Trained tokens (loss_mask=1): {num_trained}")
    print(f"  Masked tokens (loss_mask=0): {num_masked}")
    print(f"  Tool call count: {getattr(result, 'tool_call_count', 'N/A')}")
    print(f"  Status: {result.status}")

    # 3. Check response content
    print(f"\nResponse preview (first 1000 chars):")
    print(result.response[:1000])
    if len(result.response) > 1000:
        print(f"\n... ({len(result.response)} total chars) ...")
        print(f"\nResponse tail (last 500 chars):")
        print(result.response[-500:])

    if "\\boxed{" in result.response or "\\boxed " in result.response:
        print("\n[OK] Response contains \\boxed{}")
    else:
        print("\n[WARN] Response does not contain \\boxed{} — may need more tokens")

    # Check for tool_call / tool_response markers in response
    if "<tool_call>" in result.response:
        print(f"[OK] Response contains <tool_call> tags")
    if "<tool_response>" in result.response:
        print(f"[OK] Response contains <tool_response> tags (injected summaries)")

    # 4. Check tool call tracking
    tool_calls = getattr(result, "tool_call_count", 0)
    if tool_calls > 0:
        print(f"\n[OK] Model called llm_refine {tool_calls} time(s)")
        if num_masked == 0:
            errors.append(
                "Tool was called but no tokens have loss_mask=0 (summary tokens should be masked)"
            )
        # Verify logprobs: masked tokens should have 0.0 logprobs
        if result.rollout_log_probs:
            for i, (mask, logp) in enumerate(
                zip(result.loss_mask, result.rollout_log_probs)
            ):
                if mask == 0 and logp != 0.0:
                    errors.append(
                        f"Token {i}: loss_mask=0 but logprob={logp} (expected 0.0)"
                    )
                    break
    else:
        print("\n[INFO] Model did not call llm_refine — this is valid but not ideal for testing")

    # Report
    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("PASSED: All checks passed!")
    print("=" * 60)

    return len(errors) == 0


def main():
    server_proc = None
    try:
        server_proc = start_sglang_server()
        success = asyncio.run(test_generate())
        sys.exit(0 if success else 1)
    finally:
        if server_proc:
            print("\nShutting down sglang server...")
            server_proc.send_signal(signal.SIGTERM)
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
