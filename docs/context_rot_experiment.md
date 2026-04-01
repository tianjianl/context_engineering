# Context Rot Experiment Plan

## Preliminary Implementation: `context_rot_prelim/`

**Models** (2×2 design: size × thinking):
- Qwen3-4B-Instruct-2507 (non-thinking)
- Qwen3-4B-Thinking-2507 (thinking)
- Qwen3-30B-A3B-Instruct-2507 (non-thinking)
- Qwen3-30B-A3B-Thinking-2507 (thinking)

**Datasets**: HMMT Feb+Nov 2025 (60 problems), IMOBench (200 problems)

**Design**: 5 random seeds × 10 turns/conversation. Each seed produces different
orderings so the same problem appears at different positions across seeds.
Mixed-effects regression separates position effects from problem difficulty.

**Scripts**: `run_experiment.py` (batched + data-parallel), `analyze_results.py`,
`launch_all.sh` (8 SLURM jobs total).

---

## Motivation

**Context rot** is the hypothesized phenomenon where language model performance degrades as conversation context accumulates across turns. Even when each new question is independent, the growing history of prior questions and answers may cause:
- Attention dilution over irrelevant prior context
- Semantic interference from prior problems/solutions
- Positional bias (models attend less to content in the "middle" of long contexts)
- Increased likelihood of hallucination or reasoning shortcuts

This experiment rigorously measures whether context rot exists, quantifies its magnitude, and identifies which factors drive it.

---

## Hypotheses

| ID | Hypothesis | Prediction |
|----|-----------|------------|
| H1 | **Primary**: Multi-turn context degrades accuracy | Accuracy at turn 20 < accuracy at turn 1 |
| H2 | Smaller models are more susceptible | Accuracy drop is larger for 4B/8B than 32B/70B+ |
| H3 | Effect is driven by token count, not turn count | Padding context with filler produces similar degradation |
| H4 | Reasoning models (o4-mini) are more resistant | Smaller accuracy drop for chain-of-thought models |
| H5 | Same-topic sequences cause more interference | Algebra→Algebra degrades more than Algebra→Geometry |
| H6 | Prior incorrect answers increase rot | Conversations with injected wrong answers degrade faster |

---

## Dataset

### Source: MATH (Hendrycks et al.)

The MATH benchmark is ideal because:
- **Verifiable answers** — exact numerical/symbolic answers, auto-gradable via `math-verify`
- **Difficulty levels** (1–5) — we can control for difficulty
- **Subject categories** (7 topics) — enables H5 (topic interference)
- **Large pool** (~12.5K problems) — ample for counterbalancing

### Problem Selection

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Difficulty** | Level 3–4 only | Avoid ceiling (L1–2) and floor (L5) effects |
| **Total problems** | 400 | 200 for main experiment + 200 for replication/filler conditions |
| **Selection** | Stratified random across 7 categories | Balanced topic representation |
| **Answer type** | Numerical/short symbolic only | Exclude proofs and long-form answers for clean verification |

### Data Preparation

```
datasets/context_rot/
├── problems_main.jsonl         # 200 problems for main experiment
├── problems_filler.jsonl       # 200 for filler/replication
├── conversations/              # Pre-generated conversation orderings
│   ├── random_order_1.json     # 10 conversations × 20 turns
│   ├── random_order_2.json     # different randomization
│   └── ...
└── same_topic_order.json       # grouped by category (for H5)
```

---

## Models

### Tier 1: Primary Models (run all conditions)

| Model | Type | Access | Context Window | Notes |
|-------|------|--------|---------------|-------|
| **Qwen3-4B-Instruct** | Small local | vLLM | 32K | Smallest, expect largest rot |
| **Qwen3-8B** | Medium local | vLLM | 128K | Medium baseline |
| **Qwen3-32B** | Large local | vLLM | 128K | Larger local model |
| **GPT-4o** | Large API | OpenAI API | 128K | Strong proprietary baseline |
| **Claude Sonnet 4** | Large API | Anthropic API | 200K | Large context window |
| **Gemini 2.5 Flash** | Large API | Google API | 1M | Largest context window |

### Tier 2: Extended Models (main condition only, if budget allows)

| Model | Type | Access | Context Window | Notes |
|-------|------|--------|---------------|-------|
| **o4-mini** | Reasoning API | OpenAI API | 200K | Tests H4 (reasoning robustness) |
| **GPT-4.1** | Large API | OpenAI API | 1M | Latest GPT, large context |
| **Qwen3-30B-A3B** | MoE local | vLLM (vllm_0_8_4 env) | 128K | MoE architecture comparison |
| **Gemini 2.5 Pro** | Large API | Google API | 1M | Strongest Gemini |

---

## Experimental Conditions

### Condition 1: Baseline (Independent Single-Turn)

Each problem is asked in a fresh, isolated conversation. This is the control — the "true" accuracy without any context accumulation.

```
[Conversation 1]  User: Solve: {Q1}   →  Assistant: {A1}
[Conversation 2]  User: Solve: {Q2}   →  Assistant: {A2}
...
[Conversation 200] User: Solve: {Q200} →  Assistant: {A200}
```

**Purpose**: Establishes ceiling performance for each model.

### Condition 2: Sequential Multi-Turn (Main)

Problems are asked sequentially in the same conversation. The full history is retained.

```
[Single conversation]
User: Solve the following math problem. Show your reasoning step by step.
      {Q1}
Assistant: {A1}
User: Solve the following math problem. Show your reasoning step by step.
      {Q2}
Assistant: {A2}
...
User: Solve the following math problem. Show your reasoning step by step.
      {Q20}
Assistant: {A20}
```

**Design**:
- 10 conversations × 20 turns = 200 problem instances
- Each problem appears exactly once
- Problem order is randomized within each conversation
- 3 independent random orderings (seeds) → 600 instances total for statistical power

**Purpose**: Measures the core context rot effect.

### Condition 3: Filler-Padded Multi-Turn

Same structure as Condition 2, but between each Q&A pair, inject a block of irrelevant filler text (e.g., random Wikipedia paragraphs) to inflate context length without adding more math content.

```
User: {Q1}
Assistant: {A1}
User: [2000 tokens of filler text]
Assistant: I understand, thank you for sharing.
User: {Q2}
Assistant: {A2}
User: [2000 tokens of filler text]
...
```

**Purpose**: Disentangles context length from semantic interference (H3). If filler causes similar degradation, the effect is length-driven. If not, it's interference-driven.

### Condition 4: Same-Topic Sequences

Group problems by MATH category (Algebra, Geometry, Number Theory, etc.) and present 20 same-topic problems in sequence. Compare against mixed-topic sequences from Condition 2.

**Purpose**: Tests whether topically related prior context causes more interference (H5).

### Condition 5: Error Injection

Same as Condition 2, but for turns 5, 10, and 15, replace the model's actual answer with a plausible but incorrect answer. Measure whether subsequent turns are affected.

**Purpose**: Tests whether incorrect prior reasoning "poisons" subsequent generations (H6).

### Condition 6: Context Refresh (Mitigation)

Every 5 turns, insert a "context break" message:

```
User: [System note: The previous problems are complete. Please focus only on the
      next problem. Disregard all previous mathematical content.]
      {Q_next}
```

Or alternatively, start a new conversation but include a brief summary: "You have solved 5 math problems so far. Here is a new one:"

**Purpose**: Tests whether explicit context management mitigates rot. Suggests practical interventions.

---

## Implementation

### Conversation Format

Each turn uses this user prompt template:
```
Solve the following math problem. Show your reasoning step by step, then provide your final answer in \boxed{}.

Problem: {problem_text}
```

- No system prompt (or a minimal one: "You are a helpful math assistant.") — consistent across all models
- Temperature: **0.0** for all runs (deterministic, isolates the context effect)
- Max tokens per response: **4096** (sufficient for MATH L3-4 reasoning)

### Script Architecture

```
inference/context_rot.py           # Main experiment runner
inference/context_rot_prepare.py   # Dataset preparation & conversation generation
inference/context_rot_analyze.py   # Analysis & visualization
```

**`context_rot.py`** supports:
- `--condition {baseline, sequential, filler, same_topic, error_inject, refresh}`
- `--model <model_name>`
- `--num_conversations 10`
- `--turns_per_conversation 20`
- `--seed <int>` (for randomization)
- `--backend {vllm, openai, anthropic, google}` (or auto-detect from model name)

For **local models (vLLM)**: Use the OpenAI-compatible server API so the multi-turn conversation is handled properly (the server manages the KV cache and full context).

For **API models**: Use native multi-turn chat APIs (pass full message history each call).

### Output Format

```jsonl
{
  "condition": "sequential",
  "conversation_id": 3,
  "turn": 14,
  "problem_id": "algebra_1234",
  "category": "Algebra",
  "difficulty": 3,
  "prompt": "...",
  "ground_truth": "42",
  "generation": "...",
  "parsed_answer": "42",
  "is_correct": true,
  "context_tokens": 28450,
  "response_tokens": 876,
  "model": "Qwen3-8B",
  "seed": 42
}
```

---

## Execution Plan

### Phase 1: Data Preparation (Day 1)

1. Download MATH dataset, filter to L3–4, numerical answers only
2. Stratified sample 400 problems (200 main + 200 reserve)
3. Generate conversation orderings (3 seeds × 10 conversations × 20 turns)
4. Generate same-topic orderings
5. Prepare filler text blocks

### Phase 2: Local Model Runs (Days 1–3)

| Job | Model | Condition | Partition | GPUs | Est. Time |
|-----|-------|-----------|-----------|------|-----------|
| 1 | Qwen3-4B | Baseline | h200 | 1 | 2h |
| 2 | Qwen3-4B | Sequential (3 seeds) | h200 | 1 | 6h |
| 3 | Qwen3-4B | Filler | h200 | 1 | 4h |
| 4 | Qwen3-4B | Same-topic | h200 | 1 | 3h |
| 5 | Qwen3-8B | Baseline | h200 | 1 | 3h |
| 6 | Qwen3-8B | Sequential (3 seeds) | h200 | 1 | 8h |
| 7 | Qwen3-8B | Filler | h200 | 1 | 5h |
| 8 | Qwen3-8B | Same-topic | h200 | 1 | 4h |
| 9 | Qwen3-32B | Baseline | h200 | 4 | 4h |
| 10 | Qwen3-32B | Sequential (3 seeds) | h200 | 4 | 10h |
| 11 | Qwen3-32B | Filler | h200 | 4 | 6h |
| 12 | Qwen3-32B | Same-topic | h200 | 4 | 5h |

Run models via vLLM OpenAI-compatible server to handle multi-turn properly.

### Phase 3: API Model Runs (Days 2–4)

Run sequentially to manage rate limits and costs.

| Model | Condition | Est. API Cost |
|-------|-----------|--------------|
| GPT-4o | All conditions | ~$15 |
| Claude Sonnet 4 | All conditions | ~$15 |
| Gemini 2.5 Flash | All conditions | ~$5 |
| o4-mini | Baseline + Sequential only | ~$20 |

**Estimated total API cost**: ~$55

### Phase 4: Analysis (Days 4–5)

See Analysis section below.

---

## Analysis Plan

### Primary Analysis: Accuracy vs. Turn Position

For each model × condition:
- Plot accuracy (y-axis) vs. turn number (x-axis, 1–20)
- Fit a linear regression: `accuracy ~ turn_number`
- Report the **slope** (context rot rate) and **p-value**
- Use a logistic mixed-effects model: `correct ~ turn + (1|conversation) + (1|problem)` to account for random effects

### Secondary Analyses

1. **Accuracy vs. Context Tokens**: Plot accuracy against cumulative input token count (more direct measure than turn number). Fit `accuracy ~ log(context_tokens)`.

2. **Model Size Scaling**: Plot context rot rate (slope) vs. model parameter count. Test H2.

3. **Condition Comparison**: For each model, compare accuracy curves across conditions:
   - Sequential vs. Baseline → measures total rot
   - Sequential vs. Filler → disentangles length vs. interference
   - Sequential vs. Same-topic → measures topic interference (H5)
   - Sequential vs. Error-inject → measures error propagation (H6)
   - Sequential vs. Refresh → measures mitigation effectiveness

4. **Per-Category Breakdown**: Does rot affect some math categories more than others?

5. **Response Quality**: Beyond accuracy, measure:
   - Response length (tokens) vs. turn — do models get lazier?
   - Reasoning completeness — do later responses skip steps?

### Statistical Tests

- **Paired t-test / Wilcoxon**: Compare accuracy at turns 1–5 vs. turns 16–20
- **Linear mixed-effects model**: `correct ~ turn + model_size + condition + (1|problem_id) + (1|conversation_id)`
- **Bonferroni correction** for multiple comparisons across models
- **Effect size**: Report Cohen's d for the baseline vs. multi-turn comparison

### Visualizations

1. **Main figure**: Line plot with 95% CI bands — accuracy vs. turn for all Tier 1 models (one panel per model, or overlaid)
2. **Heatmap**: Models × turn position, colored by accuracy
3. **Bar chart**: Context rot magnitude (accuracy drop from T1–5 to T16–20) per model
4. **Scatter**: Rot magnitude vs. model size (log scale)
5. **Condition comparison**: Grouped bar chart comparing conditions within each model

---

## Controls & Validity

| Threat | Mitigation |
|--------|-----------|
| Problem difficulty varies by position | Randomized ordering + 3 seeds + mixed-effects model with problem random effect |
| Model randomness | Temperature 0.0 for deterministic outputs |
| API model versioning | Record exact model version/snapshot IDs |
| Position confound with difficulty | Verify difficulty is uniform across positions post-hoc |
| Prompt format differences across models | Use identical user prompts; only chat template differs |
| Context window overflow | Monitor token counts; 20 turns × 4K ≈ 80K tokens, well within all models' windows |
| vLLM vs. API differences | Both use the chat/completions endpoint format |

---

## Expected Outcomes & Impact

### If H1 is confirmed (context rot exists):
- Quantify the degradation rate per model family
- Identify whether it's a length or interference effect
- Propose mitigations (context refresh, summarization)
- Implications for agentic workflows that accumulate long conversation histories

### If H1 is rejected (no context rot):
- Also an important finding — would mean multi-turn accumulation is safe
- Still valuable to confirm across model families
- May reveal that models handle context well but with other costs (latency, cost)

### Potential Paper Framing
"Context Rot: How Multi-Turn Conversations Degrade Language Model Reasoning"
- Systematic benchmark across 6+ models
- Disentangles length vs. interference vs. error propagation
- Proposes and evaluates mitigations
- Directly relevant to the agentic AI paradigm where models run for many turns

---

## Extension Ideas (Future Work)

1. **Longer sequences**: 50 or 100 turns to test extreme context accumulation
2. **Non-math domains**: Code generation, factual QA, logical reasoning
3. **Context window boundary**: What happens as models approach their context limit?
4. **KV cache quantization**: Does cache compression accelerate rot?
5. **RAG comparison**: Does retrieval-augmented context show similar degradation?
6. **Fine-tuning mitigation**: Can models be trained to resist context rot?
