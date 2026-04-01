# Prefix Recovery Experiment Plan

## Context
We want to test whether LMs can recover after producing an incorrect solution. The experiment:
1. Run baseline on IMOBench (~343 problems) for all 5 models
2. For each model, collect its OWN incorrect solutions
3. Feed those incorrect solutions back as assistant prefixes
4. Let the model continue generating from the prefix
5. Check if the model recovers (produces a correct answer in the continuation)

**Models**: Gemini 3 Flash, MiniMax M2.5, DeepSeek V3.2 (via OpenRouter API), Qwen3-4B-Instruct-2507, Qwen3-30B-A3B-Instruct-2507 (local vLLM)

## Key Design Decisions

**Prefix injection**:
- **vLLM (local models)**: True prefix continuation using `apply_chat_template_with_prefix()` from `data_utils.py`. The model literally continues generating from the end of its incorrect solution.
- **API models (OpenRouter)**: Messages `[user: problem, assistant: incorrect_solution]`. The model generates a new assistant message having seen its own incorrect work. This tests self-correction rather than pure continuation.

**Prefix content**: Full generation including `\boxed{wrong_answer}`. For vLLM models, include `<think>` tags if present (natural self-context). For API models, use the raw API response.

**Verification**: Use `verify_utils.verify_batch()` to check if the continuation/response contains a correct `\boxed{}` answer.

## Files to Create

### 1. `inference/prefix_recovery_vllm.py`
**Purpose**: Run prefix recovery for local vLLM models (Qwen3-4B, Qwen3-30B-A3B)

**Logic**:
1. Load baseline JSONL via `data_utils.load_jsonl()`
2. Verify each generation with `verify_utils.verify_batch()`, filter for incorrect ones
3. For each incorrect solution:
   - Use `apply_chat_template_with_prefix(tokenizer, problem_prompt, incorrect_generation)`
   - Run `llm.generate()` to continue from the prefix
4. Verify continuations and save results

**Args**: `--model`, `--baseline_file`, `--output_file`, `--num_tokens` (default 16384), `--temperature` (default 0.7), `--max_model_len`, `--tensor_parallel_size`

**Output format**:
```jsonl
{
  "problem_id": "...",
  "prompt": "...",
  "answer": "...",
  "incorrect_prefix": "...",
  "continuation": "...",
  "recovered": true/false,
  "model": "..."
}
```

### 2. `inference/prefix_recovery_api.py`
**Purpose**: Run prefix recovery for API models via OpenRouter

**Logic**:
1. Load baseline JSONL, verify, filter for incorrect solutions
2. For each incorrect solution:
   - Send `[{"role": "user", "content": prompt}, {"role": "assistant", "content": incorrect_generation}]`
   - Model generates a new response seeing its own incorrect work
3. Verify responses and save results

**Args**: `--model` (OpenRouter model ID), `--baseline_file`, `--output_file`, `--max_tokens` (default 16384), `--temperature` (default 0.7), `--concurrency` (default 10)

**Uses**: `AsyncOpenAI` with OpenRouter base URL (pattern from `context_rot_prelim/openrouter_inference.py`)

**Same output format** as the vLLM script.

### 3. `inference/analyze_prefix_recovery.py`
**Purpose**: Compare recovery rates across all models

**Logic**:
1. Load recovery output files for all 5 models
2. For each model: compute recovery rate = (# recovered) / (# incorrect baseline)
3. Print comparison table + per-category breakdown if available
4. Also report what % of total problems each model got wrong in baseline (context for recovery rate)

## SLURM Jobs

### Phase 1: Baselines (can all run in parallel)

All outputs go to `/scratch/dkhasha1/tli104/outputs/prefix_recovery/`

| Job | Script | Partition | Env | Model |
|-----|--------|-----------|-----|-------|
| baseline_qwen3_4b | `baseline_vllm.py` | h200 (4 GPU) | vllm | `Qwen/Qwen3-4B-Instruct-2507` |
| baseline_qwen3_30b_a3b | `baseline_vllm.py` | h200 (4 GPU) | vllm_0_8_4 | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| baseline_api | `openrouter_inference.py` (baseline mode) | cpu | vllm | 3 API models sequentially |

**Baseline params**: `--dataset imobench --num_tokens 16384 --temperature 0.7 --num_samples 1`

**API models** (OpenRouter IDs):
- `google/gemini-3-flash-preview`
- `minimax/minimax-m2.5`
- `deepseek/deepseek-v3.2`

### Phase 2: Recovery (after baselines complete)

| Job | Script | Partition | Env | Baseline input |
|-----|--------|-----------|-----|----------------|
| recovery_qwen3_4b | `prefix_recovery_vllm.py` | h200 (4 GPU) | vllm | baseline_qwen3_4b output |
| recovery_qwen3_30b_a3b | `prefix_recovery_vllm.py` | h200 (4 GPU) | vllm_0_8_4 | baseline_qwen3_30b_a3b output |
| recovery_api | `prefix_recovery_api.py` | cpu | vllm | 3 API baseline outputs |

**Recovery params**: `--num_tokens 16384 --temperature 0.7`

### SLURM Scripts to Create (in `/scratch/dkhasha1/tli104/slurm_scripts/`)

1. `prefix_recovery_baseline_qwen3_4b.sh`
2. `prefix_recovery_baseline_qwen3_30b.sh`
3. `prefix_recovery_baseline_api.sh` (runs 3 models)
4. `prefix_recovery_qwen3_4b.sh`
5. `prefix_recovery_qwen3_30b.sh`
6. `prefix_recovery_api.sh` (runs 3 models)

## Verification

1. Run Phase 1 baselines, check outputs are non-empty JSONL
2. Run `verify_solutions.py` on each baseline to confirm accuracy numbers
3. Run Phase 2 recovery, check outputs have `recovered` field
4. Run `analyze_prefix_recovery.py` to compare recovery rates across all 5 models
5. Expected: even strong models rarely recover from their own incorrect prefixes
