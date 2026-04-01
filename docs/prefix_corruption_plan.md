# Prefix Corruption Experiment Plan

## Research Question

**Does corrupting arithmetic in a model's own correct reasoning prefix cause it to produce wrong answers?**

We take prefixes from solutions the model already got right, introduce arithmetic errors (perturbed values, swapped numbers), then let the model continue generating. The clean prefix baseline should yield high accuracy (the prefix was already on a correct trajectory), so corruption-induced failures produce a large, measurable gap. This tests whether reasoning traces function as active computation that downstream steps depend on, or passive context the model can route around.

## Background

### Existing infrastructure

All baselines and clean prefix ablations already exist in `/scratch/dkhasha1/tli104/outputs/prefix_recovery/`:

**Baselines** (IMOBench, 343 problems, temp=0.9, 1 sample):
- `baseline_qwen3_4b_imobench_t16384_n1_temp0.9.jsonl`
- `baseline_qwen3_30b_a3b_imobench_t16384_n1_temp0.9.jsonl`

**Clean prefix ablation from incorrect solutions** (16 samples per problem, temp=0.9):
- Qwen3-4B-Instruct-2507: p0, p100, p400, p800, p1000
- Qwen3-30B-A3B-Instruct-2507: p0, p50, p100, p200, p400, p800, p1000, p2000, p3000, p4000

### Implemented scripts

| Script | Purpose |
|--------|---------|
| `inference/arithmetic_corruption.py` | Corruption utilities (find numbers, perturb, swap) |
| `inference/prefix_corruption_experiment.py` | Experiment runner (vLLM, multi-sample). **Currently filters for incorrect prefixes — needs `--use_correct` flag** |
| `inference/prefix_length_ablation.py` | Clean prefix ablation (provides `filter_incorrect`, `truncate_to_tokens`) |

### Required script changes

`prefix_corruption_experiment.py` currently calls `filter_incorrect()` to select prefixes from wrong solutions. We need to:

1. Add a `filter_correct()` function (inverse of `filter_incorrect`): return items whose baseline generation is verified correct.
2. Add a `--use_correct` flag (default True) to select correct vs. incorrect prefixes.
3. The rest of the pipeline (truncate, corrupt, generate, verify) stays the same.

We also need a **clean correct-prefix control** run for each (model, prefix_length) — take correct prefixes, truncate without corruption, generate continuations. This tells us the baseline accuracy when continuing from a clean correct prefix.

## Experimental Design

### Why correct prefixes?

Starting from **correct** prefixes gives a stronger experimental signal:
- **Clean correct prefix**: the model was already on the right track → high continuation accuracy (ceiling)
- **Corrupted correct prefix**: same trajectory but with arithmetic errors injected → accuracy drops
- The **gap** between clean and corrupted directly measures how much the model depends on arithmetic correctness in its reasoning chain

Contrast with incorrect prefixes: the model was already failing, so continuation accuracy is low in both the clean and corrupted conditions, making the corruption effect harder to detect.

### Independent Variables

| Variable | Levels | Rationale |
|----------|--------|-----------|
| **Model** | Qwen3-4B-Instruct-2507, Qwen3-30B-A3B-Instruct-2507 | Small vs. medium model, tests if larger models are more robust |
| **Prefix length** | 200, 400, 800 tokens | Enough content for corruptions. 200 = early reasoning, 400 = mid-solution, 800 = deep into solution |
| **Corruption strategy** | `result_perturbation`, `number_swap` | Perturbation changes values (injects wrong arithmetic); swap preserves the number set but misplaces them (injects wrong associations) |
| **Num corruptions** | 1, 3, 5 | Dose-response: single error vs. pervasive corruption |

### Controls

| Control | Description |
|---------|-------------|
| **No prefix (fresh generation)** | Existing `prefix_ablation_*_p0.jsonl` — re-rolling from scratch |
| **Clean correct prefix** | New runs needed: truncate correct prefixes to p{200,400,800}, generate continuations without corruption |

The clean correct-prefix control is the critical baseline — it establishes the ceiling accuracy that corruption degrades from.

### Fixed Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Dataset | IMOBench (343 problems) | Matches baselines |
| Source prefixes | Correct baseline solutions only | Maximizes clean-prefix accuracy for a large corruption gap |
| Samples per problem | 16 | pass@k estimation |
| Temperature | 0.9 | Matches prior ablation runs |
| Top-p | 0.9 | Matches prior ablation runs |
| Max tokens | 16384 | Matches prior ablation runs |
| Corruption seed | 42 (+ per-problem offset) | Reproducible; same corruptions across prefix lengths |

### Run Matrix

**Corruption runs**: 2 models × 3 prefix lengths × 2 strategies × 3 corruption counts = **36 runs**.
**Clean correct-prefix controls**: 2 models × 3 prefix lengths = **6 runs**.
**Total**: **42 runs**.

#### Qwen3-4B-Instruct-2507

**Clean correct-prefix controls (3 runs)**:

| Prefix | Output file |
|--------|-------------|
| 200 | `correct_clean_4b_p200.jsonl` |
| 400 | `correct_clean_4b_p400.jsonl` |
| 800 | `correct_clean_4b_p800.jsonl` |

**Corruption runs (18 runs)**:

| Prefix | Strategy | Corruptions | Output file |
|--------|----------|-------------|-------------|
| 200 | result_perturbation | 1 | `correct_corrupt_4b_p200_rp_c1.jsonl` |
| 200 | result_perturbation | 3 | `correct_corrupt_4b_p200_rp_c3.jsonl` |
| 200 | result_perturbation | 5 | `correct_corrupt_4b_p200_rp_c5.jsonl` |
| 200 | number_swap | 1 | `correct_corrupt_4b_p200_ns_c1.jsonl` |
| 200 | number_swap | 3 | `correct_corrupt_4b_p200_ns_c3.jsonl` |
| 200 | number_swap | 5 | `correct_corrupt_4b_p200_ns_c5.jsonl` |
| 400 | result_perturbation | 1 | `correct_corrupt_4b_p400_rp_c1.jsonl` |
| 400 | result_perturbation | 3 | `correct_corrupt_4b_p400_rp_c3.jsonl` |
| 400 | result_perturbation | 5 | `correct_corrupt_4b_p400_rp_c5.jsonl` |
| 400 | number_swap | 1 | `correct_corrupt_4b_p400_ns_c1.jsonl` |
| 400 | number_swap | 3 | `correct_corrupt_4b_p400_ns_c3.jsonl` |
| 400 | number_swap | 5 | `correct_corrupt_4b_p400_ns_c5.jsonl` |
| 800 | result_perturbation | 1 | `correct_corrupt_4b_p800_rp_c1.jsonl` |
| 800 | result_perturbation | 3 | `correct_corrupt_4b_p800_rp_c3.jsonl` |
| 800 | result_perturbation | 5 | `correct_corrupt_4b_p800_rp_c5.jsonl` |
| 800 | number_swap | 1 | `correct_corrupt_4b_p800_ns_c1.jsonl` |
| 800 | number_swap | 3 | `correct_corrupt_4b_p800_ns_c3.jsonl` |
| 800 | number_swap | 5 | `correct_corrupt_4b_p800_ns_c5.jsonl` |

#### Qwen3-30B-A3B-Instruct-2507

Same matrix, with `30b` in filenames (3 clean controls + 18 corruption runs).

## SLURM Execution

### Job configuration

| Model | Partition | GPUs | Env | Exclude |
|-------|-----------|------|-----|---------|
| Qwen3-4B-Instruct-2507 | nvl | 1 | `vllm` | n10 |
| Qwen3-30B-A3B-Instruct-2507 | h200 | 4 | `vllm_0_8_4` | h205 |

### Batching Strategy

Each SLURM job runs a **single** (model, prefix_length, strategy, num_corruptions) combination. This gives fine-grained checkpointing and makes it easy to re-run failures.

**Naming conventions**:
- Clean controls: `ccln_{model}_p{prefix}` — e.g., `ccln_4b_p400`
- Corruption: `ccorr_{model}_p{prefix}_{strategy}_{corruptions}` — e.g., `ccorr_4b_p400_rp_c3`

### Job Template — Clean Correct-Prefix Control

```bash
#!/bin/bash
#SBATCH --job-name=ccln_4b_p400
#SBATCH --partition=nvl
#SBATCH --gres=gpu:1
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=40
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/dkhasha1/tli104/slurm_logs/ccln_4b_p400_%j.out
#SBATCH --error=/scratch/dkhasha1/tli104/slurm_logs/ccln_4b_p400_%j.err
#SBATCH --exclude=n10

export SCRATCH_DIR=/scratch/dkhasha1/tli104
export HF_HOME=${SCRATCH_DIR}/hf_model_cache
export HF_DATASETS_CACHE=${SCRATCH_DIR}/hf_datasets_cache
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${SCRATCH_DIR}/vllm

cd /weka/home/tli104/context_engineering

python -m inference.prefix_corruption_experiment \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --baseline_file ${SCRATCH_DIR}/outputs/prefix_recovery/baseline_qwen3_4b_imobench_t16384_n1_temp0.9.jsonl \
    --output_file ${SCRATCH_DIR}/outputs/prefix_recovery/correct_clean_4b_p400.jsonl \
    --prefix_length 400 \
    --use_correct \
    --no_corruption \
    --num_samples 16 \
    --num_tokens 16384 \
    --temperature 0.9 \
    --tensor_parallel_size 1
```

### Job Template — Corruption Run

```bash
#!/bin/bash
#SBATCH --job-name=ccorr_4b_p400_rp_c3
#SBATCH --partition=nvl
#SBATCH --gres=gpu:1
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=40
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/dkhasha1/tli104/slurm_logs/ccorr_4b_p400_rp_c3_%j.out
#SBATCH --error=/scratch/dkhasha1/tli104/slurm_logs/ccorr_4b_p400_rp_c3_%j.err
#SBATCH --exclude=n10

export SCRATCH_DIR=/scratch/dkhasha1/tli104
export HF_HOME=${SCRATCH_DIR}/hf_model_cache
export HF_DATASETS_CACHE=${SCRATCH_DIR}/hf_datasets_cache
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${SCRATCH_DIR}/vllm

cd /weka/home/tli104/context_engineering

python -m inference.prefix_corruption_experiment \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --baseline_file ${SCRATCH_DIR}/outputs/prefix_recovery/baseline_qwen3_4b_imobench_t16384_n1_temp0.9.jsonl \
    --output_file ${SCRATCH_DIR}/outputs/prefix_recovery/correct_corrupt_4b_p400_rp_c3.jsonl \
    --prefix_length 400 \
    --use_correct \
    --corruption_strategy result_perturbation \
    --num_corruptions 3 \
    --corruption_seed 42 \
    --num_samples 16 \
    --num_tokens 16384 \
    --temperature 0.9 \
    --tensor_parallel_size 1
```

### Execution Order

1. **Phase 0** — Implement `--use_correct` and `--no_corruption` flags in `prefix_corruption_experiment.py`
2. **Phase 1** — Run clean correct-prefix controls (6 jobs: 2 models × 3 prefix lengths)
3. **Phase 2** — Run all 36 corruption jobs (can submit in parallel per partition)
4. **Phase 3** — Analysis

Phases 1 and 2 can run concurrently (controls don't need to finish before corruption runs start, since they use the same baselines independently). Phase 2 jobs for the two models run on different partitions (NVL vs H200) so they don't compete.

## Analysis Plan

### Primary Metric

**Accuracy** = fraction of samples where the continuation produces a correct `\boxed{}` answer. Computed at two granularities:
- **Per-sample**: `correct / total_samples` (fine-grained)
- **Per-problem (pass@k)**: fraction of problems where at least 1 of 16 samples is correct

Note the framing change from "recovery rate" (incorrect prefix → correct answer) to "accuracy" (correct prefix → still correct after corruption). The clean correct-prefix control should have high accuracy; corruption should degrade it.

### Comparisons

#### 1. Corruption vs. Clean Correct Prefix (main result)

For each (model, prefix_length), compare:
- Clean correct-prefix accuracy (from new control runs)
- Corrupted correct-prefix accuracy (from corruption runs)

The gap = accuracy lost due to arithmetic corruption.

**Test**: Two-proportion z-test per cell; mixed-effects logistic regression across all cells:
```
correct ~ corrupted * num_corruptions * prefix_length + (1|problem_id)
```

**Expected figure**: Grouped bar chart — accuracy on y-axis, prefix length on x-axis, bars grouped by condition (clean, rp×1, rp×3, rp×5, ns×1, ns×3, ns×5). Clean bar should be high; corruption bars should step down.

#### 2. Dose-Response

Plot accuracy vs. num_corruptions for each (model, prefix_length, strategy). A steeper decline = higher sensitivity to arithmetic errors.

**Expected figure**: Line plot — accuracy vs. num_corruptions (0 = clean control, 1, 3, 5), one line per prefix length, faceted by model.

#### 3. Strategy Comparison

Compare `result_perturbation` vs. `number_swap` at matched corruption counts. Result perturbation changes values (injects wrong arithmetic); number swap misplaces values (injects wrong associations). If result perturbation hurts more, models are sensitive to arithmetic correctness specifically.

#### 4. Corruption Coverage Diagnostic

Report per run:
- Fraction of problems where corruption was actually applied (some short prefixes may have no corruptible numbers)
- Number of corruptible numbers found vs. requested corruptions

This is critical — if corruption can't be applied at p200 for many problems, those results should be interpreted with care (or the prefix length should be increased).

#### 5. Model Size Effect

Compare 4B vs. 30B degradation from corruption. Hypothesis: larger models are more robust (can "route around" corrupted arithmetic).

### Visualization Checklist

1. **Main figure**: Recovery rate bars — clean vs. corrupted, grouped by prefix length, faceted by model
2. **Dose-response**: Line plot, recovery vs. num_corruptions
3. **Strategy comparison**: Paired bars (rp vs. ns) at each prefix length
4. **Corruption coverage**: Histogram of `num_corruptions_applied` per run
5. **Per-problem heatmap**: Problems (rows) × conditions (columns), colored by recovery

## Possible Outcomes & Interpretation

| Outcome | Interpretation |
|---------|---------------|
| Corruption strongly reduces accuracy | Models actively use prefix arithmetic as scaffolding; corrupted intermediate results derail continuation even when the original reasoning was correct |
| Corruption has no effect | Models re-derive from the problem statement; prefix content is largely ignored during continuation |
| Effect at p800 but not p200 | Short prefixes don't contain enough committed reasoning; corruption matters only when the prefix is deep into a solution path where later steps depend on earlier arithmetic |
| Result perturbation hurts more than number swap | Models are sensitive to arithmetic correctness specifically, not just number placement |
| 4B more affected than 30B | Larger models have more redundant reasoning capacity to "route around" errors |
| Effect only at c>=3, not c=1 | Models can tolerate isolated errors but break down under pervasive corruption |
| Accuracy drops to near-zero at p800 + c5 | The model is fully dependent on prefix correctness by that depth — it cannot re-derive independently |

## Extensions (if results are interesting)

1. **Targeted corruption**: Corrupt only the final intermediate result (most recent `=`) vs. early results — tests whether recency of error matters
2. **Semantic corruption**: Replace correct reasoning steps with plausible-but-wrong text (not just numbers)
3. **API models**: Extend to Gemini/DeepSeek via OpenRouter (multi-turn messages instead of prefix continuation)
4. **Incorrect prefix baseline**: Compare against the existing incorrect-prefix corruption data to see if corruption effect differs when the reasoning was already wrong
