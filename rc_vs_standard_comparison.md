# Standard Iterative Refinement (`context_refinement_dp.py --rc`) vs RC Reimplementation (`rc/inference/generate_complete.py`)

## Critical Differences

### 1. What Gets Summarized (THE BIGGEST DIFFERENCE)

**RC Reimpl**: Feeds the **THINKING** (chain-of-thought) to the summarizer.
- `update_reasoning()` takes content BEFORE `</think>`, strips `<think>` tag
- So `curr_reasoning` = the raw chain of thought (the thinking process)
- This is passed as `{reasoning}` to the summarization prompt

**Standard RC**: Feeds the **ANSWER** (post-thinking visible response) to the summarizer.
- `strip_thinking()` takes content AFTER `</think>`
- So `current_gen` = the model's visible answer/solution
- This is passed as `{reasoning}` to the RC refinement prompt

**Impact**: The RC reimpl summarizes the full reasoning process (potentially 16K tokens of chain-of-thought), giving the summarizer much more information about the problem-solving approach, dead ends, key insights, etc. The standard RC only summarizes the short visible answer, which may be just a final boxed answer with minimal explanation.

### 2. How the Summary Is Fed Back to the Generator

**RC Reimpl**: Summary goes into a **structured user-message prompt**.
```
<|im_start|>user
You are given a maths problem...
### PROBLEM
[problem]
### SUMMARY OF PREVIOUS ATTEMPT
[summary]
### INSTRUCTIONS
If a summary of a previous attempt is provided, your task is to improve upon this attempt...
Reason step-by-step and return your final answer in \boxed{}.
<|im_end|>
<|im_start|>assistant
<think>
```
The model receives explicit instructions about what to do with the summary.

**Standard RC**: Summary is injected as an **assistant prefix** (continuation).
```
<|im_start|>user
[original problem prompt]<|im_end|>
<|im_start|>assistant
[summary]
```
The model simply continues generating after the summary, with NO instruction about how to use it. The summary appears as if the assistant already wrote it.

**Impact**: The RC reimpl gives the model clear context about what the summary represents and explicit instructions to improve upon it. The standard RC forces the model to continue from the summary as if it's mid-response, which may confuse the model about what role the summary plays.

### 3. Thinking Mode (`enable_thinking`)

**RC Reimpl**: Explicitly enables thinking via chat template.
- `tokenizer.apply_chat_template(..., enable_thinking=True)` for Qwen models
- If `<think>` not already in output, appends it manually
- Model ALWAYS produces `<think>...</think>` structured output

**Standard RC**: Does NOT enable thinking.
- Uses plain `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`
- No `enable_thinking` parameter passed
- Model may or may not produce thinking tags depending on its default behavior

**Impact**: With explicit thinking enabled, the RC reimpl guarantees structured thinking output and can reliably extract the chain-of-thought. The standard RC relies on the model's default behavior, which for Qwen3-Instruct models may or may not include `<think>` tags.

### 4. Summarization Token Budget

**RC Reimpl**: `max_summarization_tokens = 2048` (default, used in practice)

**Standard RC**: `max_refinement_tokens = 16384` (default)

**Impact**: The RC reimpl produces short, focused summaries (2K tokens max). The standard RC allows up to 16K tokens for the summary, which defeats the purpose of compression — the summary could be as long as the original generation.

### 5. Temperature & Sampling

| Parameter | RC Reimpl (defaults) | RC Reimpl (actual run) | Standard RC (defaults) |
|-----------|---------------------|----------------------|----------------------|
| temperature | 0.6 | 1.0 | 0.7 |
| top_p | 0.95 | 1.0 | 0.9 |
| top_k | N/A | N/A | -1 (disabled) |
| seed | 42 | 42 | None |

**Impact**: The actual RC reimpl runs use temp=1.0 / top_p=1.0 (maximum diversity), while standard RC uses temp=0.7 / top_p=0.9 (moderate diversity). Higher temperature allows more exploration across rounds.

### 6. Number of Samples

**RC Reimpl**: `n=4` (default, used in practice)

**Standard RC**: `n=16` (typical, set via `--num_samples`)

**Impact**: RC reimpl uses fewer samples but runs more steps. The higher sample count in standard RC means more diverse initial attempts but the same summary-based refinement applied to each.

---

## Detailed Side-by-Side Comparison

| Aspect | RC Reimpl (`generate_complete.py`) | Standard RC (`context_refinement_dp.py --rc`) |
|--------|-----------------------------------|-----------------------------------------------|
| **Summarization prompt** | `prompts/summarization_prompt.txt` | `create_rc_refinement_prompt()` — **IDENTICAL TEXT** |
| **Generation prompt (step 1)** | `reasoning_prompt.txt` with `{problem}` and `{curr_summary}=""` | Original dataset prompt (e.g. "Solve the following math problem...") |
| **Generation prompt (step 2+)** | `reasoning_prompt.txt` filled with `{problem}` and `{curr_summary}=summary`, wrapped in chat template | `apply_chat_template_with_prefix(original_prompt, summary)` |
| **Input to summarizer** | Thinking content (before `</think>`) | Post-thinking content (after `</think>`) |
| **Summary token budget** | 2048 | 16384 |
| **Generation token budget** | `max_thinking_tokens` (16384 in practice) | `num_tokens` (varies: 4096, 8192, 16384) |
| **Thinking enabled** | Yes (`enable_thinking=True`) | No (default chat template) |
| **`<think>` tag appended** | Yes, explicitly added if missing | No |
| **Summary replacement** | `prefix = refined_ctx` (replacement) | `prefix = refined_ctx` (replacement) — **SAME** |
| **vLLM dtype** | `bfloat16` (explicit) | Default (auto) |
| **GPU memory util** | 0.70 default (0.90 in practice) | 0.95 default |
| **max_model_len** | Not explicitly set | `num_tokens + max_refinement_tokens + 4096` |
| **Stop tokens** | None | None |
| **Early stopping** | None (all samples run all steps) | None (all samples run all rounds) |
| **DP sharding** | Round-robin by problem | Round-robin by problem — **SAME** |
| **Output format** | JSON (list of dicts, `reasoning_store`/`summarization_store` arrays) | JSONL (one line per problem, `samples[].rounds[]` nested structure) |
| **Intermediate saves** | No | Yes (per-GPU JSONL, overwritten each round) |
| **Saves raw thinking** | Yes (`reasoning_store` has raw output) | Yes (`current_round_generation_raw`) |
| **Refinement fallback** | No fallback logic | If refinement strip is empty, falls back to using the stripped generation |
| **System prompt** | None | None |
| **N per step 1** | `n` samples per unique problem (batched) | `num_samples` per prompt (batched) — **SAME pattern** |
| **N per step 2+** | 1 per sample | 1 per (question, sample) pair — **SAME** |

---

## The Generation Prompt Difference in Detail

### RC Reimpl — `reasoning_prompt.txt`

```
You are given a maths problem. You may also be given a summary of a previous attempt
to solve it. This previous attempt may or may not be correct.

### PROBLEM
{problem}

### SUMMARY OF PREVIOUS ATTEMPT
{curr_summary}

### INSTRUCTIONS
If no summary of a previous attempt is provided, solve the problem from scratch.

If a summary of a previous attempt is provided, your task is to improve upon this
attempt. You should rely on this summary to guide your thinking.
Some strategies you could use include:
- Verifying the previous solution.
- Proving the result in a different way.
- Finding alternative problem-solving strategies.
- Continuing from where the previous solution left off, assuming that the previous
  solution is incomplete.

Reason step-by-step and return your final answer in \boxed{}.
```

This prompt:
- Explicitly frames the summary as a "previous attempt"
- Gives concrete improvement strategies
- Instructs the model to reason step-by-step
- Asks for `\boxed{}` answer

### Standard RC — No explicit generation prompt

The model just sees:
```
User: [original dataset prompt, e.g. "Solve the following math problem step by step..."]
Assistant: [summary from last round]
```
And continues generating from there. No explicit instructions about the summary.

---

## The Thinking Strip Difference in Detail

### RC Reimpl `update_reasoning()`:
```python
# Model output: "<think>long chain of thought</think>The answer is \boxed{42}."
response = "<think>long chain of thought</think>The answer is \\boxed{42}."

# Step 1: Take content BEFORE </think>
curr_chunk = response.split("</think>")[0]  # "<think>long chain of thought"

# Step 2: Remove <think> tag
curr_chunk = curr_chunk.replace("<think>", "")  # "long chain of thought"

# Result: curr_reasoning = "long chain of thought" (THE THINKING)
```

### Standard RC `strip_thinking()`:
```python
# Model output: "<think>long chain of thought</think>The answer is \boxed{42}."
text = "<think>long chain of thought</think>The answer is \\boxed{42}."

# Find </think> position
think_end = text.find('</think>')  # found

# Take content AFTER </think>
return text[think_end + 8:].strip()  # "The answer is \boxed{42}." (THE ANSWER)
```

**What gets summarized:**
- RC Reimpl summarizes: `"long chain of thought"` (potentially 16K tokens of reasoning)
- Standard RC summarizes: `"The answer is \boxed{42}."` (potentially just 1 line)

---

## Flow Diagram

### RC Reimpl Flow (Step N, N > 0):
```
[reasoning_prompt.txt filled with problem + prev_summary]
    → apply_chat_template(enable_thinking=True)
    → append <think> if missing
    → vLLM generate (max_thinking_tokens)
    → OUTPUT: <think>THINKING</think>ANSWER
    → Extract THINKING (before </think>)
    → [summarization_prompt.txt filled with problem + prev_summary + THINKING]
    → apply_chat_template
    → vLLM generate (max_summarization_tokens=2048)
    → Extract summary (after </think>)
    → NEW_SUMMARY replaces prev_summary
```

### Standard RC Flow (Round N, N > 0):
```
[apply_chat_template_with_prefix(original_prompt, prev_summary)]
    → vLLM generate (num_tokens)
    → OUTPUT: possibly <think>THINKING</think>ANSWER  or  just ANSWER
    → strip_thinking → ANSWER (after </think>)
    → [create_rc_refinement_prompt(problem, prev_summary, ANSWER)]
    → apply_chat_template
    → vLLM generate (max_refinement_tokens=16384)
    → strip_thinking → NEW_SUMMARY
    → NEW_SUMMARY replaces prev_summary
```

---

## The Closed-Turn Problem in Standard RC

The standard RC uses `apply_chat_template_with_prefix(prompt, summary)` which produces:
```
<|im_start|>user
[problem]<|im_end|>
<|im_start|>assistant
[summary]<|im_end|>       ← TURN CLOSED
```
The assistant turn is terminated with `<|im_end|>`. The model generates after a **closed** turn, which is outside normal chat structure. The model may produce confused output or start a new turn marker.

The RC reimpl produces:
```
<|im_start|>user
[problem + summary + instructions]<|im_end|>
<|im_start|>assistant
<think>                    ← FRESH OPEN TURN
```
Clean chat structure with an open assistant turn and thinking explicitly started.

## The `</think>` Problem in Standard RC

Three scenarios when `strip_thinking()` processes generation output:

| Scenario | `<think>`? | `</think>`? | Result | What gets summarized |
|----------|-----------|------------|--------|---------------------|
| Full thinking | Yes | Yes | Text after `</think>` | Short answer only |
| Ran out of tokens | Yes | **No** | **`""`** (empty) | Nothing — empty `{reasoning}` |
| No thinking | No | No | Full text unchanged | Full response (best case) |

Scenario 2 is the worst case: the model uses the entire token budget for thinking, never closes `</think>`, and the standard RC has **nothing** to summarize. The RC reimpl would have the full 16K of chain-of-thought in this case.

## Empirical Findings from Qwen3-4B-Instruct-2507 Trajectories

### Finding 1: No Thinking Mode in Either Implementation

Both implementations produce output **without** `<think>` tags. The `enable_thinking=True` parameter has no effect on Qwen3-4B-Instruct-2507's chat template — the template output is identical with or without it. The RC reimpl appends `<think>` to the prompt, but the model does not generate `</think>` in response — it just generates normally.

**Consequence**: The "what gets summarized" difference (thinking vs answer) does NOT apply. Both implementations summarize the full model response, which includes the reasoning since there's no `<think>` block to strip.

### Finding 2: Massive Empty Generation Rate in Standard RC (THE KILLER BUG)

The `<|im_end|>` closed-turn issue causes the model to generate empty strings in 15-54% of rounds > 0:

| Standard RC File | R1 Empty | R2-R12 Empty |
|-----------------|----------|--------------|
| IMOBench t4096 | 0.0% | **53.5%** |
| HMMT t4096 | 0.0% | 13.8% |
| HMMT t8192 | 0.0% | 19.1% |
| HMMT Combined t16k | 0.0% | 15.6% |

IMOBench t4096 per-round empty rates:
```
R1:  0.0%  R2: 46.5%  R3: 47.5%  R4: 50.2%  R5: 52.6%  R6: 53.5%
R7: 54.1%  R8: 56.1%  R9: 56.0%  R10: 57.1%  R11: 57.7%  R12: 57.7%
```

**RC Reimpl**: 0.0% empty across ALL 1600 entries x 4 steps.

### Finding 3: Wasted Refinement on Empty Generations

When generation is empty, the refinement step still runs (summarizing the problem + existing summary with no new content). This produces a rephrased summary but adds no new information:

| Gen Empty? | Mean Refinement Length | Median |
|-----------|----------------------|--------|
| Yes | 1,343 chars | 1,313 chars |
| No | 1,667 chars | 1,566 chars |

### Finding 4: Prompt Structure Difference

**Standard RC (rounds > 0)** — the prompt ends with a CLOSED assistant turn:
```
<|im_start|>user
[problem]<|im_end|>
<|im_start|>assistant
[summary]<|im_end|>     ← CLOSED, model often generates EOS immediately
```

**RC Reimpl (steps > 0)** — fresh open turn with instructions:
```
<|im_start|>user
[problem + summary + improvement instructions]<|im_end|>
<|im_start|>assistant
[model generates normally, 15K+ chars per step]
```

### Finding 5: Effective Compute Utilization

| Metric | Standard RC (IMO t4096) | RC Reimpl (IMO t16384) |
|--------|------------------------|----------------------|
| Samples | 16 per problem | 4 per problem |
| Rounds/Steps | 12 | 4 |
| Generation calls | 12 × 16 = 192 per problem | 4 × 4 = 16 per problem |
| Empty gens | ~54% of R2-R12 | 0% |
| Effective gens | ~96 per problem | 16 per problem |
| Reasoning per gen | varies (0 when empty) | ~15K chars mean |
| Summary per step | ~1.5K chars | ~1.7K chars |

## Summary of Why RC Reimpl Outperforms Standard RC

1. **~50% of rounds are wasted (THE MAIN ISSUE)**: The `<|im_end|>` closed-turn prompt causes the model to generate empty output in up to 54% of rounds > 0. The RC reimpl's fresh-prompt approach has 0% empty generations.

2. **Explicit generation instructions**: The RC reimpl tells the model exactly what to do with the summary — verify, try alternative strategies, continue from where it left off. Standard RC just injects the summary as a continuation prefix with no instructions, and the model often just generates EOS.

3. **~~Summarizes reasoning, not just answers~~** (NOT confirmed): In practice, neither implementation uses thinking mode — both summarize the full model response. This difference is theoretical only for Qwen3-4B-Instruct-2507.

4. **~~Guaranteed thinking mode~~** (NOT confirmed): `enable_thinking=True` has no effect on the Qwen3-4B-Instruct-2507 chat template. Neither implementation produces `<think>` tagged output.

5. **Constrained summaries**: 2K token summary budget forces concise, information-dense summaries. Standard RC's 16K budget can produce bloated summaries that aren't truly compressed.

6. **Higher temperature**: temp=1.0 (RC reimpl) vs temp=0.7 (standard RC) enables more diverse exploration across rounds.

7. **All compute is productive**: RC reimpl generates ~15K chars of reasoning per step with 0% waste. Standard RC wastes half its generation calls on empty output.

---

## What's Actually Identical

1. **Summarization prompt text** — word-for-word identical
2. **Summary replacement strategy** — both replace (not accumulate) the prefix each round
3. **Data parallelism pattern** — round-robin sharding, spawn multiprocessing, CUDA_VISIBLE_DEVICES isolation
4. **No stop tokens** — both let the model run to max_tokens or EOS
5. **No early stopping** — both run all samples through all rounds
6. **No filtering between rounds** — no best-of-n selection
