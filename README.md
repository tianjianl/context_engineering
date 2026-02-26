# Architect of Thought
Learning to summarize & refine intermediate generations.

## Inference Scripts

All inference scripts live in `inference/` and share common utilities (`args_utils.py`, `data_utils.py`, `dp_utils.py`). They use vLLM for generation and support multi-GPU data parallelism.

### Methods

#### Baseline (`baseline_vllm.py`)
Standard single-pass generation with no refinement. Generates `n` samples per problem in one shot.

```bash
python -m inference.baseline_vllm \
  --dataset imobench --model Qwen/Qwen3-8B \
  --num_tokens 32768 --num_samples 16 \
  --output_file output_baseline.jsonl
```

Supports both tensor parallelism (`--tensor_parallel_size`) and data parallelism (`--data_parallel --num_dp_workers`).

#### Context Refinement (`context_refinement_dp.py`)
Multi-round generate-then-refine loop. Each round: generate a solution, compress it into a refined context, then use that context to guide the next generation.

```bash
python -m inference.context_refinement_dp \
  --dataset imobench --model Qwen/Qwen3-8B \
  --num_tokens 16384 --rounds 4 --num_samples 16 \
  --output_file output_refinement.jsonl
```

**Refinement modes:**

| Flag | Method | Description |
|------|--------|-------------|
| *(default)* | Accumulate | Compress generation, append refined context as assistant prefix for next round |
| `--accumulate` | Accumulate | Same as default but accumulates context across rounds |
| `--rc` | Reasoning Cache (RC) | Summarize previous attempt; summary replaces (not appends to) context each round |
| `--rc_verify` | RC + Verify | Like RC but the summary critically evaluates the solution and suggests improvements |
| `--rc_user` | RC User | Like RC but embeds the summary in the user prompt instead of as an assistant prefix |

Additional options: `--accumulate_raw` (accumulate raw generation instead of refined), `--strip_answer` (strip final answers from refined context to force re-derivation).

#### Tool Refinement (`tool_refinement.py`)
The model autonomously decides when to call `llm_refine` using native tool-calling format. When `<tool_call>` is emitted, generation stops, the partial work is summarized, and the summary is returned as a `<tool_response>`. Designed for models with tool-calling support (e.g., Qwen3-4B-Instruct-2507).

```bash
python -m inference.tool_refinement \
  --dataset imobench --model Qwen/Qwen3-4B-Instruct-2507 \
  --num_tokens 16384 --max_rounds 12 --num_samples 16 \
  --output_file output_tool_refinement.jsonl
```

Options: `--compact_context` (only keep prompt + latest summary, discard generation history), `--refiner_base_url` (use external API for summarization), `--system_prompt` / `--system_prompt_file`.

#### Agentic Refinement (`agentic_refinement.py`)
Similar to tool refinement but uses a Jinja2 template for prompt construction (instead of the tokenizer's built-in `tools=` parameter) and supports thinking mode handling (`--strip_thinking_from_generation`, `--strip_thinking_from_refinement`). Uses a system prompt that instructs the model to call `llm_refine` after every reasoning step.

```bash
python -m inference.agentic_refinement \
  --dataset imobench --model Qwen/Qwen3-8B \
  --num_tokens 16384 --max_rounds 12 --num_samples 1 \
  --output_file output_agentic.jsonl
```

### Verification & Grading

| Script | Purpose |
|--------|---------|
| `verify_solutions.py` | Verify extracted `\boxed{}` answers against ground truth |
| `verify_by_round.py` | Per-round accuracy breakdown for refinement outputs |
| `verify_cumulative.py` | Cumulative accuracy across rounds |
| `grade_proofs_gemini.py` | Grade proof-style solutions via Gemini API |
| `grade_proofs_o3.py` | Grade proof-style solutions via OpenAI o3 |

### Shared Utilities

| Module | Contents |
|--------|----------|
| `args_utils.py` | Shared argparse builders: `add_common_args`, `add_dp_args`, `add_refinement_args`, `post_process_args`, `validate_args` |
| `data_utils.py` | Dataset loading, text processing, `apply_chat_template`, `apply_chat_template_with_prefix`, refinement prompts |
| `dp_utils.py` | Multi-GPU orchestration: `detect_gpus`, `shard_data`, `run_data_parallel`, `SampleState`, `save_intermediate_results` |
| `verify_utils.py` | Math verification utilities |
| `test_utils.py` | Unit tests for all utility modules |

### Datasets

| Name | Flag | Problems |
|------|------|----------|
| IMOBench | `--dataset imobench` | ~343 |
| HMMT | `--dataset hmmt --input_file <path>` | ~88 |
| ProofBench | `--dataset hmmt --input_file <path>` | 60 |
| ProofBench HF | `--dataset hmmt --input_file <path>` | 435 |
