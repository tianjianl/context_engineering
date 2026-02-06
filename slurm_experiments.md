## 2026-02-06 Agentic Refinement Tests

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 934574 | test_agentic_h200 | h200 | 4 | TEST: Qwen3-4B-Instruct-2507 agentic refinement on H200, 1 problem, 1 sample, t4096, refine2048, max_rounds=3 |
| 934568 | test_agentic | nvl | 1 | TEST: Qwen3-4B-Instruct-2507 agentic refinement, 1 problem, 1 sample, t4096, refine2048, max_rounds=3 |

## 2026-02-06 Qwen3-4B-Instruct-2507 HMMT Combined t16k Refine4k Experiments

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 934566 | refine_qwen3_4b_hmmt_t16k_r4k | nvl | 2 | Qwen3-4B-Instruct-2507 refinement on hmmt-combined, t16384, refine4096, r12, n16, temp1.0, topp1.0, dp=2, --accumulate, with intermediate saving |

## 2026-02-06 Qwen3-30B-A3B Baseline Experiments (NVL)

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 934542 | baseline_qwen3_30b_hmmt | nvl | 2 | FAILED - vllm_0_8_3 env not found |
| 934543 | baseline_qwen3_30b_imobench | nvl | 2 | FAILED - vllm_0_8_3 env not found |
| 934548 | baseline_qwen3_30b_hmmt | nvl | 2 | FAILED - OOM, no max_model_len set |
| 934549 | baseline_qwen3_30b_imobench | nvl | 2 | FAILED - OOM, no max_model_len set |
| 934550 | baseline_qwen3_30b_hmmt | nvl | 2 | FAILED - CUDA device visibility error with vllm_0_8_4 on NVL |
| 934551 | baseline_qwen3_30b_imobench | nvl | 2 | FAILED - CUDA device visibility error with vllm_0_8_4 on NVL |
| 934553 | baseline_qwen3_30b_hmmt | h200 | 1 | Qwen3-30B-A3B-Instruct-2507 baseline on hmmt, t32768, n16, temp0.7, topp0.9, tp=1, max_model_len=40960, env=vllm_0_8_4 |
| 934554 | baseline_qwen3_30b_imobench | h200 | 1 | Qwen3-30B-A3B-Instruct-2507 baseline on imobench, t32768, n16, temp0.7, topp0.9, tp=1, max_model_len=40960, env=vllm_0_8_4 |

## 2026-02-06 Qwen3-4B-Instruct-2507 IMOBench t16k Refinement Experiments

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 934187 | refine_qwen3_4b_imobench_t16k | nvl | 2 | Qwen3-4B-Instruct-2507 refinement on imobench, t16384, refine2048, r12, n16, temp1.0, topp1.0, dp=2, --accumulate |
| 934188 | rc_qwen3_4b_imobench_t16k | nvl | 2 | Qwen3-4B-Instruct-2507 RC on imobench, t16384, rc2048, r12, n16, temp1.0, topp1.0, dp=2 |

## 2026-02-06 Qwen3-4B-Instruct-2507 HMMT Combined (Feb+Nov) t16k Refinement Experiments

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 933491 | refine_qwen3_4b_hmmt_t16k | h200 | 2 | Qwen3-4B-Instruct-2507 refinement on hmmt-combined (feb+nov), t16384, refine2048, r12, n16, temp1.0, topp1.0, dp=2, --accumulate |
| 933492 | rc_qwen3_4b_hmmt_t16k | h200 | 2 | Qwen3-4B-Instruct-2507 RC on hmmt-combined (feb+nov), t16384, rc2048, r12, n16, temp1.0, topp1.0, dp=2 |

## 2026-02-06 Qwen3-4B-Instruct-2507 HMMT Nov 2025 Experiments

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 932810 | baseline_qwen3_4b_hmmt_nov2025 | nvl | 2 | Qwen3-4B-Instruct-2507 baseline on hmmt_nov_2025, t16384, n16, temp0.7, topp0.9, dp=2 |

## 2026-02-05 Qwen3-4B-Instruct-2507 IMOBench Experiments

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 921044 | baseline_qwen3_4b_imobench | nvl | 2 | Qwen3-4B-Instruct-2507 baseline on imobench, t16384, n16, temp0.7, topp0.9, dp=2 |
| 921045 | rc_qwen3_4b_imobench | h200 | 4 | Qwen3-4B-Instruct-2507 RC on imobench, t4096, r12, n16, temp0.7, topp0.9, dp=4 |
| 921046 | refine_qwen3_4b_imobench | h200 | 4 | Qwen3-4B-Instruct-2507 refinement on imobench, t4096, r12, n16, temp0.7, topp0.9, dp=4, --accumulate |

## 2026-02-05 Qwen3-4B-Instruct-2507 HMMT Experiments

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 921037 | baseline_qwen3_4b_hmmt | nvl | 2 | Qwen3-4B-Instruct-2507 baseline on hmmt, t16384, n16, temp0.7, topp0.9, dp=2 |
| 921038 | rc_qwen3_4b_hmmt | h200 | 4 | Qwen3-4B-Instruct-2507 RC on hmmt, t4096, r12, n16, temp0.7, topp0.9, dp=4 |
| 921039 | refine_qwen3_4b_hmmt | h200 | 4 | Qwen3-4B-Instruct-2507 refinement on hmmt, t4096, r12, n16, temp0.7, topp0.9, dp=4, --accumulate (CANCELLED - stuck) |
| 929535 | refine_qwen3_4b_hmmt | h200 | 4 | Qwen3-4B-Instruct-2507 refinement on hmmt, t4096, r12, n16, temp0.7, topp0.9, dp=4, --accumulate (rerun of 921039) |
| 929536 | rc_qwen3_4b_hmmt_t8192 | h200 | 4 | Qwen3-4B-Instruct-2507 RC on hmmt, t8192, r12, n16, temp0.7, topp0.9, dp=4 |
| 929537 | refine_qwen3_4b_hmmt_t8192 | h200 | 4 | Qwen3-4B-Instruct-2507 refinement on hmmt, t8192, r12, n16, temp0.7, topp0.9, dp=4, --accumulate (CANCELLED - moved to NVL) |
| 932808 | refine_qwen3_4b_hmmt_t8192 | nvl | 2 | Qwen3-4B-Instruct-2507 refinement on hmmt, t8192, r12, n16, temp0.7, topp0.9, dp=2, --accumulate (resubmit of 929537) |

## 2026-02-04 Qwen3-30B-A3B Baseline Experiments (H100)

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 918474 | baseline_qwen3_30b_a3b_h100 | h100 | 4 | Qwen3-30B-A3B baseline on imobench, t32768, n16, tp=1, dp=4 |

## 2026-02-03 Qwen3-30B-A3B Baseline Experiments

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 850474 | baseline_qwen3_30b_a3b_imobench | nvl | 2 | Qwen3-30B-A3B-Instruct-2507 baseline on imobench, t32768, n16 |
| 850475 | baseline_qwen3_30b_a3b_imobench_t65536 | nvl | 2 | Qwen3-30B-A3B-Instruct-2507 baseline on imobench, t65536, n16 |

## 2026-02-02 Qwen3-30B-A3B Baseline Experiments

| Job ID | Description | Partition |
|--------|-------------|-----------|
| 848107 | hmmt_baseline_t16384 (Qwen3-30B-A3B-Instruct-2507, 2 GPUs) | h100 |

## 2026-01-28 Qwen3-8B DP Experiments

| Job ID | Description | Partition |
|--------|-------------|-----------|
| 820347 | imobench_t16384_r1_dp4 (max_model_len=32768, gpu_mem=0.90) | h200 |
| 820348 | imobench_t16384_r2_dp4 (max_model_len=32768, gpu_mem=0.90) | h200 |
| 820349 | imobench_t16384_r3_dp4 (max_model_len=32768, gpu_mem=0.90) | h200 |
