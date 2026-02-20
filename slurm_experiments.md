## 2026-02-19 Verify — Qwen3-4B ProofBench (60) All Methods (CPU)

verify_solutions.py + verify_by_round.py on all 4 Qwen3-4B ProofBench (60) files. Note: ProofBench-HF skipped here (empty answer fields) — o3 grading (1036107) serves as verification for HF.

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1036140 | verify_4b_pb | cpu | Verify Qwen3-4B ProofBench (60): baseline (t32768 n16), rc (t16384 r12 n16), rc_user (t32768 rt4096 r12 n16), compact_tool_refinement (t16384 r12 n16). verify_solutions + verify_by_round (except baseline). |

## 2026-02-19 O3 Grading — Qwen3-4B All ProofBench + ProofBench-HF (CPU)

Grade all 9 Qwen3-4B-Instruct-2507 ProofBench files with o3 (reasoning_effort=medium, --max-samples -1, 4 workers, --resume). Two schemes auto-detected: rubric 0-7 for ProofBench-HF (marking_scheme); categorical incorrect/partial/almost/correct for ProofBench-60 (grading_guidelines). Cancelled 1036107 and resubmitted as 1036151.

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1036151 | grade_o3_4b_pbhf | cpu | O3 grade: ProofBench-HF (5 files: baseline, rc, rc_fix, rc_user, ctr) + ProofBench-60 (4 files: baseline, rc, rc_user, ctr). 9 files total, sequential. |

## 2026-02-19 RC User 30B ProofBench (NVL) — t16k, rt2k/4k/8k

RC User on ProofBench (60 problems) for Qwen3-30B-A3B-Instruct-2507. Three rt variants on NVL partition (TP=2). t=16384, r=12, n=16, temp=1.0, topp=1.0, env=vllm_0_8_4.

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 1035999 | rc_user_30b_pb_rt2k | nvl | 2 (TP=2) | RC user, Qwen3-30B-A3B, ProofBench (60), t16384, rt2048, r12, n16, temp1.0, topp1.0, env=vllm_0_8_4 |
| 1036000 | rc_user_30b_pb_rt4k | nvl | 2 (TP=2) | RC user, Qwen3-30B-A3B, ProofBench (60), t16384, rt4096, r12, n16, temp1.0, topp1.0, env=vllm_0_8_4 |
| 1036001 | rc_user_30b_pb_rt8k | nvl | 2 (TP=2) | RC user, Qwen3-30B-A3B, ProofBench (60), t16384, rt8192, r12, n16, temp1.0, topp1.0, env=vllm_0_8_4 |

## 2026-02-19 Tool SFT Merged Training — 3 Epochs Full Dataset (H200)

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 1035984 | tool_sft_merged | h200 | 4 | SFT: train on tool_sft_train (11140 ex = correct_sft+synthesized_sft+reconstructed_sft merged+deduped), val=tool_sft_val (1237 ex). YAML: qwen3_tool_sft_merged.yaml, lr=5e-6, 3 epochs, batch=64 (per_device=1, grad_accum=16, 4 GPUs), save_steps=1000, DeepSpeed Z2, output=tool_sft_merged_checkpoints. HF dataset: dogtooth/llm_tool_self_context_management_sft. |

## 2026-02-19 SFT v2 Training — Fix Format Mismatch (H200)

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 1021944 | sft_v2_train | h200 | 4 | SFT v2: train on tool_sft_train (8578 ex = correct_sft+synthesized_sft, patched observation turns with CONTINUATION_INSTRUCTIONS). Fix: compact_context now keeps last generation in assistant turn (matches training format). YAML: qwen3_tool_sft_v2.yaml, lr=5e-6, 1 epoch, DeepSpeed Z2, output=tool_sft_checkpoints. |

## 2026-02-18 SFT Ep1 Compact Tool Refinement Eval (H200)

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 1021528 | sft_ep1_ctr_imo | h200 | 4 (DP=4) | Compact tool refinement, SFT Qwen3-4B ep1 (9432 train, 1047 val, lr=5e-6), IMOBench, t8192, r12, n16, temp0.7, topp0.9, --compact_context, env=vllm |
| 1021529 | sft_ep1_ctr_hmmt | h200 | 4 (DP=4) | Compact tool refinement, SFT Qwen3-4B ep1 (9432 train, 1047 val, lr=5e-6), HMMT Combined, t8192, r12, n16, temp0.7, topp0.9, --compact_context, env=vllm |

## 2026-02-18 Verify SFT Ep1 Compact Tool Refinement (CPU)

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1021583 | verify_sft_ep1_imo | cpu | Verify + per-round breakdown: compact tool refinement SFT Qwen3-4B ep1 on IMOBench, t8192 r12 n16. Depends on 1021528. |
| 1021584 | verify_sft_ep1_hmmt | cpu | Verify + per-round breakdown: compact tool refinement SFT Qwen3-4B ep1 on HMMT Combined, t8192 r12 n16. Depends on 1021529. |

## 2026-02-17 ProofBench HF Regrading (CPU)

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1018112 | regrade_pbhf | cpu | Regrade 4B ProofBench HF with Gemini (--resume to fill nulls for baseline/RC/compact TR, fresh grade for RC User + RC Fix + ProofBench 60 RC User). --max-samples -1, 8 workers. |

## 2026-02-17 Verify SFT Compact Tool Refinement (CPU)

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1018116 | verify_sft_ctr | cpu | Verify + per-round breakdown: compact tool refinement SFT Qwen3-4B (Gemini synthesized) on IMOBench, t8192 r12 n16 |

## 2026-02-17 SFT Compact Tool Refinement Eval (H200)

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 1018025 | sft_ctr_imo | h200 | 4 (DP=4) | Compact tool refinement, SFT Qwen3-4B (Gemini synthesized), IMOBench, t8192, r12, n16, temp0.7, topp0.9, --compact_context, env=vllm |

## 2026-02-17 Comprehensive Verification (CPU)

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1018014 | verify_all_tables | cpu | Verify all baseline, RC User, RC Reimpl results for Qwen3-4B and Qwen3-30B-A3B on IMOBench and HMMT Feb. Caches results as _results.json next to each input file. Output: baseline_rc_user_reimpl_tables.md |

## 2026-02-17 Tool SFT Training (H200)

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 1018002 | tool_sft_train | h200 | 4 | Full SFT Qwen3-4B-Instruct-2507 on 9432 train / 1047 val examples (5971 correct + 4511 synthesized with injected tool calls). LlamaFactory, DeepSpeed ZeRO-2, lr=5e-6, 3 epochs, cutoff=32768, batch=32 effective. Config: qwen3_tool_sft.yaml |

## 2026-02-17 Reconstruct Trajectories for SFT (H200)

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 1017989 | reconstruct_traj | h200 | 4 (DP=4) | Reconstruct 2849 truncated trajectories from Gemini annotations, Qwen3-4B-Instruct-2507, t8192, r5, temp0.7, topp0.9, env=vllm |

## 2026-02-17 Baseline HMMT Nov 2025 (NVL)

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 1017946 | bl_4b_hmmt_nov | nvl | 2 (DP=2) | Baseline, Qwen3-4B-Instruct-2507, HMMT Nov 2025, t16384, n16, temp0.7, topp0.9, env=vllm |
| 1017947 | bl_30b_hmmt_nov | nvl | 2 (TP=2) | Baseline, Qwen3-30B-A3B-Instruct-2507, HMMT Nov 2025, t16384, n16, temp0.7, topp0.9, max_model_len=32768, env=vllm_0_8_4 |

## 2026-02-17 Verify RC User vs RC Reimpl vs Baseline Comparison

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1017827 | verify_4b_imobench_cmp | cpu | Qwen3-4B IMOBench: baseline + rc_user (rt2048, rt4096) + rc_reimpl (per-step 1-12) |
| 1017828 | verify_4b_hmmt_cmp | cpu | Qwen3-4B HMMT: baseline + rc_user (rt2048, rt4096, rt8192) |
| 1017829 | verify_30b_hmmt_cmp | cpu | Qwen3-30B-A3B HMMT: baseline (t16384) + rc_user (rt2048, rt4096, rt8192) |

## 2026-02-16 RC User ProofBench (H200)

RC User on ProofBench and ProofBench HF for both 4B and 30B-A3B models. Larger token limits (t32768, rt4096).

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 1001836 | rc_user_4b_pbhf | h200 | 4 (DP=4) | RC user, Qwen3-4B, ProofBench HF (435), t32768, rt4096, r12, n16, temp1.0, topp1.0, env=vllm |
| 1001837 | rc_user_4b_pb | h200 | 4 (DP=4) | RC user, Qwen3-4B, ProofBench (60), t32768, rt4096, r12, n16, temp1.0, topp1.0, env=vllm |
| 1001838 | rc_user_30b_pbhf | h200 | 4 (TP=4) | RC user, Qwen3-30B-A3B, ProofBench HF (435), t32768, rt4096, r12, n16, temp1.0, topp1.0, max_model_len=131072, env=vllm_0_8_4 |
| 1001839 | rc_user_30b_pb | h200 | 4 (TP=4) | RC user, Qwen3-30B-A3B, ProofBench (60), t32768, rt4096, r12, n16, temp1.0, topp1.0, max_model_len=131072, env=vllm_0_8_4 |

## 2026-02-16 RC Reimpl 4B HMMT (H200)

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 1002038 | rc_reimpl_4b_hmmt_r12 | h200 | 4 (DP=4) | RC reimpl, Qwen3-4B, HMMT combined (60), n16, steps12, t16384, sum2048, temp1.0, topp1.0, env=vllm_0_8_4 |

## 2026-02-16 Verify RC User 4B

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1001819 | verify_rc_user_4b_hmmt_rt2048 | cpu | Verify rc_user Qwen3-4B HMMT-combined t16384 rt2048 r12 n16 |
| 1001967 | verify_rc_user_4b_imo_partial | cpu | Verify rc_user Qwen3-4B IMO t16384 rt2048 (rounds 1-7 from intermediates) |

## 2026-02-15 RC User Qwen3-235B-A22B (H200)

First run with the largest Qwen3 MoE model. TP=4 across 4 H200s, single worker (no DP). vllm_0_8_4 env.

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 1000747 | rc_user_235b_imo | h200 | 4 (TP=4) | RC user, Qwen3-235B-A22B, IMOBench, t32768, rt4096, r10, n4, temp1.0, topp1.0, max_model_len=131072, env=vllm_0_8_4 |

## 2026-02-15 RC User (NVL) + RC Reimpl n=16 r=12 (H200)

RC User: same pipeline as RC-fix but summary placed in user prompt (like RC reimpl) instead of assistant prefix. Uses RC reimpl hparams: temp=1.0, top_p=1.0, rt=2048.
RC Reimpl: original RC reimpl code with n=16 (up from n=4) and 12 steps.

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 997536 | rc_reimpl_30b_imo_r12 | h200 | 4 | RC reimpl, Qwen3-30B-A3B, IMOBench, n=16, 12 steps, t16k, temp1.0, topp1.0 |
| 997537 | rc_reimpl_4b_imo_r12 | h200 | 4 | RC reimpl, Qwen3-4B, IMOBench, n=16, 12 steps, t16k, temp1.0, topp1.0 |
| 997538 | rc_user_4b_imo | nvl | 2 | RC user, Qwen3-4B, IMOBench, t16384, rt2048, r12, n16, temp1.0, topp1.0 |
| 997539 | rc_user_4b_hmmt | nvl | 2 | RC user, Qwen3-4B, HMMT, t16384, rt2048, r12, n16, temp1.0, topp1.0 |
| 997546 | rc_user_4b_hmmt_rt4096 | nvl | 2 | RC user, Qwen3-4B, HMMT, t16384, rt4096, r12, n16, temp1.0, topp1.0 |
| 997547 | rc_user_4b_hmmt_rt8192 | nvl | 2 | RC user, Qwen3-4B, HMMT, t16384, rt8192, r12, n16, temp1.0, topp1.0 |
| 997548 | rc_user_4b_imo_rt4096 | nvl | 2 | RC user, Qwen3-4B, IMOBench, t16384, rt4096, r12, n16, temp1.0, topp1.0 |
| 997549 | rc_user_4b_imo_rt8192 | nvl | 2 | RC user, Qwen3-4B, IMOBench, t16384, rt8192, r12, n16, temp1.0, topp1.0 |

## 2026-02-15 Resubmit RC User 30B (NVL) — vllm_0_8_4 env fix

Original 30B NVL jobs crashed with `topk_softmax` MoE error due to wrong vllm env. Resubmitted with `vllm_0_8_4`.

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 1000652 | rc_user_30b_imo | nvl | 2 | RC user, Qwen3-30B-A3B, IMOBench, t16384, rt2048, r12, n16, temp1.0, topp1.0, env=vllm_0_8_4 |
| 1000653 | rc_user_30b_hmmt | nvl | 2 | RC user, Qwen3-30B-A3B, HMMT, t16384, rt2048, r12, n16, temp1.0, topp1.0, env=vllm_0_8_4 |
| 1000654 | rc_user_30b_imo_rt4096 | nvl | 2 | RC user, Qwen3-30B-A3B, IMOBench, t16384, rt4096, r12, n16, temp1.0, topp1.0, env=vllm_0_8_4 |
| 1000655 | rc_user_30b_imo_rt8192 | nvl | 2 | RC user, Qwen3-30B-A3B, IMOBench, t16384, rt8192, r12, n16, temp1.0, topp1.0, env=vllm_0_8_4 |
| 1000656 | rc_user_30b_hmmt_rt4096 | nvl | 2 | RC user, Qwen3-30B-A3B, HMMT, t16384, rt4096, r12, n16, temp1.0, topp1.0, env=vllm_0_8_4 |
| 1000657 | rc_user_30b_hmmt_rt8192 | nvl | 2 | RC user, Qwen3-30B-A3B, HMMT, t16384, rt8192, r12, n16, temp1.0, topp1.0, env=vllm_0_8_4 |

## 2026-02-14 RC Fix rt4k/rt8k (H200) — Qwen3-30B-A3B All Benchmarks

Same as NVL rt2048 jobs but with higher refinement token budgets (4096 and 8192) on H200 with 4 GPUs, vllm_0_8_4 env.

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 982560 | rc_fix_30b_imo_rt4k | h200 | 4 | RC fix, Qwen3-30B-A3B-Instruct-2507, IMOBench, t16384, rt4096, r12, n16, temp0.7, topp0.9, dp=4. |
| 982561 | rc_fix_30b_imo_rt8k | h200 | 4 | RC fix, Qwen3-30B-A3B-Instruct-2507, IMOBench, t16384, rt8192, r12, n16, temp0.7, topp0.9, dp=4. |
| 982562 | rc_fix_30b_hmmt_rt4k | h200 | 4 | RC fix, Qwen3-30B-A3B-Instruct-2507, HMMT Combined, t16384, rt4096, r12, n16, temp0.7, topp0.9, dp=4. |
| 982563 | rc_fix_30b_hmmt_rt8k | h200 | 4 | RC fix, Qwen3-30B-A3B-Instruct-2507, HMMT Combined, t16384, rt8192, r12, n16, temp0.7, topp0.9, dp=4. |
| 982564 | rc_fix_30b_pbhf_rt4k | h200 | 4 | RC fix, Qwen3-30B-A3B-Instruct-2507, ProofBench HF, t16384, rt4096, r12, n16, temp0.7, topp0.9, dp=4. |
| 982565 | rc_fix_30b_pbhf_rt8k | h200 | 4 | RC fix, Qwen3-30B-A3B-Instruct-2507, ProofBench HF, t16384, rt8192, r12, n16, temp0.7, topp0.9, dp=4. |

## 2026-02-14 Annotate Incorrect Tool Refinement Trajectories — Gemini-3-Flash

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 982542 | annotate_tr_t8192_r12 | cpu | 0 | Gemini annotation of incorrect trajectories, tool_refinement t8192 r12 n16 (pass@1=32.22%), max 4 samples/problem, ~$4 |
| 982543 | annotate_tr_v2_t8192_r12 | cpu | 0 | Gemini annotation of incorrect trajectories, tool_refinement_v2 t8192 r12 n16 (pass@1=28.84%), max 4 samples/problem, ~$4.50 |
| 982544 | annotate_tr_t16384_r6 | cpu | 0 | Gemini annotation of incorrect trajectories, tool_refinement t16384 r6 n16 (pass@1=32.23%), max 4 samples/problem, ~$5 |

## 2026-02-14 Verify Tool Refinement — Qwen3-4B IMOBench (3 files)

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 982535 | verify_tool_ref_t8192_r12 | cpu | 0 | Verify tool_refinement Qwen3-4B IMOBench t8192 r12 n16 |
| 982536 | verify_tool_ref_v2_t8192_r12 | cpu | 0 | Verify tool_refinement_v2 Qwen3-4B IMOBench t8192 r12 n16 |
| 982537 | verify_tool_ref_t16384_r6 | cpu | 0 | Verify tool_refinement Qwen3-4B IMOBench t16384 r6 n16 |

## 2026-02-14 RC Fix (NVL, rt2048) — Qwen3-30B-A3B All Benchmarks

Cancelled h200 jobs 982528/982529/982530. Resubmitted on NVL with `--max_refinement_tokens 2048`, `vllm_0_8_4` env. Output files include `rt2048` to reflect refinement token budget.

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 982531 | rc_fix_30b_imo | nvl | 2 | RC fix, Qwen3-30B-A3B-Instruct-2507, IMObench (~343), t16384, rt2048, r12, n16, temp0.7, topp0.9, dp=2. |
| 982532 | rc_fix_30b_hmmt | nvl | 2 | RC fix, Qwen3-30B-A3B-Instruct-2507, HMMT Combined (~88), t16384, rt2048, r12, n16, temp0.7, topp0.9, dp=2. |
| 982533 | rc_fix_30b_pbhf | nvl | 2 | RC fix, Qwen3-30B-A3B-Instruct-2507, ProofBench HF (435), t16384, rt2048, r12, n16, temp0.7, topp0.9, dp=2. |

## 2026-02-13 RC Fix — Qwen3-30B-A3B IMObench & HMMT

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 982417 | rc_fix_30b_imo | h200 | 4 | RC with closed-turn fix, Qwen3-30B-A3B-Instruct-2507, IMObench (~343 problems), t16384, r12, n16, temp0.7, topp0.9, dp=4. |
| 982418 | rc_fix_30b_hmmt | h200 | 4 | RC with closed-turn fix, Qwen3-30B-A3B-Instruct-2507, HMMT Combined (~88 problems), t16384, r12, n16, temp0.7, topp0.9, dp=4. |

## 2026-02-13 Verify RC Fix — Qwen3-4B IMObench

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 982414 | verify_rc_fix_4b_imo_t8k | cpu | 0 | Verify rc_fix Qwen3-4B-Instruct-2507 IMObench, t8192, r12, n16. Per-round + cumulative. |
| 982415 | verify_rc_fix_4b_imo_t16k | cpu | 0 | Verify rc_fix Qwen3-4B-Instruct-2507 IMObench, t16384, r12, n16. Per-round + cumulative. |

## 2026-02-13 RC Fix — Qwen3-4B ProofBench HF

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 972907 | rc_fix_4b_pbhf | h200 | 4 | Standard RC with closed-turn fix, Qwen3-4B-Instruct-2507, ProofBench HF (435 problems), t16384, r12, n16, temp0.7, topp0.9, dp=4. |
| 972975 | rc_fix_30b_pbhf | h200 | 4 | Standard RC with closed-turn fix, Qwen3-30B-A3B-Instruct-2507, ProofBench HF (435 problems), t16384, r12, n16, temp0.7, topp0.9, dp=4. |

## 2026-02-13 Verify RC Fix — Qwen3-4B HMMT Combined

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 970933 | verify_rc_fix_4b_hmmt_t8k | cpu | 0 | Verify rc_fix Qwen3-4B-Instruct-2507 HMMT Combined (60 problems), t8192, r12, n16. Per-round + cumulative. |
| 970934 | verify_rc_fix_4b_hmmt_t16k | cpu | 0 | Verify rc_fix Qwen3-4B-Instruct-2507 HMMT Combined (60 problems), t16384, r12, n16. Per-round + cumulative. |

## 2026-02-13 Standard RC Fix (remove `<|im_end|>` closed turn) — Qwen3-4B IMOBench

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 969465 | rc_fix_4b_imo_t8k | nvl | 2 | Standard RC with closed-turn fix (open assistant prefix instead of `<|im_end|>`), Qwen3-4B-Instruct-2507, IMOBench, t8192, r12, n16, temp0.7, topp0.9, dp=2. Compare against broken RC (21.89% at t4096) and RC reimpl (40.81%). |
| 969466 | rc_fix_4b_imo_t16k | nvl | 2 | Same fix, t16384, r12, n16. Compare against broken RC t16k+rc2k (32.88%) and RC reimpl (40.81%). |
| 969476 | rc_fix_4b_hmmt_t8k | nvl | 2 | Standard RC with closed-turn fix, Qwen3-4B-Instruct-2507, HMMT Combined (60 problems), t8192, r12, n16, temp0.7, topp0.9, dp=2. Compare against broken RC t8192 (33.75% on HMMT Feb only). |
| 969477 | rc_fix_4b_hmmt_t16k | nvl | 2 | Same fix, HMMT Combined, t16384, r12, n16. Compare against broken RC t16k+rc2k (40.10%). |

## 2026-02-13 RC Reimpl — Qwen3-4B HMMT Feb 2025

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 967809 | rc_reimpl_q4b_hmmt | nvl | 2 | RC decoding (rc/inference/generate_complete.py) on HMMT Feb 2025 (30 problems), Qwen3-4B-Instruct-2507, n=4, max_steps=12, t16384, summ2048, temp1.0, topp1.0, use_think_tags, dp=2, tp=1, gpu_mem=0.90. Output: rc_reimpl_qwen3_4b_hmmt_feb_dp2_n4_steps12_t16384.json |

## 2026-02-12 Verify Tool Refinement t16k r6 — Qwen3-4B IMOBench

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 951973 | verify_tr_4b_t16k_r6 | cpu | 0 | Verify tool refinement Qwen3-4B-Instruct-2507 IMOBench (t16384, r6, n16). Overall, per-round, cumulative. |

## 2026-02-12 Qwen3-4B Tool Refinement t16k r6 IMOBench

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 951925 | tr_4b_imo_t16k_r6 | h200 | 4 | Qwen3-4B-Instruct-2507 standard tool refinement on IMOBench, t16384, r6, n16, temp0.7, topp0.9, dp=4. Compare against baseline (35.61%) and compact variants. |

## 2026-02-12 Verify CTR v3 vs v1 vs Baseline — Qwen3-4B IMOBench

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 951918 | verify_ctr_v3_4b | cpu | 0 | Verify compact tool refinement v3 (t8192, r12, n16) + v1 (t8192) + baseline (t16384) for Qwen3-4B-Instruct-2507 IMOBench. Overall, per-round, cumulative. |

## 2026-02-12 Verify RC Reimpl Qwen3-30B-A3B IMOBench

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 951842 | verify_rc_30b | cpu | 0 | Verify RC reimpl Qwen3-30B-A3B-Instruct-2507 IMOBench (n4, steps4, t16384). Converts JSON→JSONL, runs verify_solutions.py on final step + per-step breakdown. |

## 2026-02-12 Compact Tool Refinement v3 (RC-style summarization) — Qwen3-4B IMOBench

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 951839 | ctr_v3_4b_imo | h200 | 4 | Qwen3-4B-Instruct-2507 compact tool refinement **v3** (RC-style summarization prompt, fresh-start context, continuation instructions) on IMOBench, t8192, r12, n16, temp0.7, topp0.9, dp=4, --compact_context. Compare against v1 compact (32.94%) and RC reimpl (40.81%). |
| 951840 | ctr_v3_30b_imo | h200 | 4 | Qwen3-30B-A3B-Instruct-2507 compact tool refinement **v3** (RC-style summarization) on IMOBench, t16384, r12, n16, temp0.7, topp0.9, dp=4, --compact_context, env=vllm_0_8_4. |

## 2026-02-12 RC vs Iterative Refinement Comparison Verification

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 951836 | verify_rc_cmp | cpu | 0 | Compare all RC/refinement methods for Qwen3-4B IMOBench: RC t4096, RC t16k-rc2k, RC reimpl (n4 steps4), tool refinement v1, compact tool refinement. Per-round + cumulative. |

## 2026-02-12 Verify RC Reimpl Qwen3-4B IMOBench

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 951813 | verify_rc_4b | cpu | 0 | Verify RC reimpl Qwen3-4B-Instruct-2507 IMOBench (dp4, n4, steps4, t16384). Converts JSON→JSONL, runs verify_solutions.py on final step + per-step breakdown. |

## 2026-02-12 RC Reimplementation (from rc/ codebase) — Qwen3-30B-A3B IMOBench

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 951792 | rc_reimpl_q30b_imobench | h200 | 4 | RC decoding (rc/inference/generate_complete.py) on IMOBench (400 problems), Qwen3-30B-A3B-Instruct-2507, n=4, max_steps=4, t16384, summ2048, temp1.0, topp1.0, use_think_tags, tp=4, gpu_mem=0.85, env=vllm_0_8_4. Output: /scratch/dkhasha1/tli104/outputs/rc_reimpl/rc_reimpl_qwen3_30b_a3b_imobench_n4_steps4_t16384.json |
| 951808 | rc_reimpl_q4b_imo_dp | h200 | 4 | RC decoding **with DP** (dp=4, tp=1), Qwen3-4B-Instruct-2507 on IMOBench (400 problems), n=4, max_steps=4, t16384, summ2048, temp1.0, topp1.0, use_think_tags, gpu_mem=0.90. Output: /scratch/dkhasha1/tli104/outputs/rc_reimpl/rc_reimpl_qwen3_4b_imobench_dp4_n4_steps4_t16384.json |

## 2026-02-12 Qwen3-30B-A3B Baseline Verification

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 951463 | verify_30b_baselines | cpu | 0 | Verify all 3 Qwen3-30B-A3B baselines: IMOBench t32768, HMMT t32768, HMMT t16384 (n=16 each) |

## 2026-02-11 V1 vs V2 Prompt Comparison

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 950756 | verify_v1v2_4b | cpu | 0 | Verify v1 vs v2 prompt comparison for Qwen3-4B IMOBench (t8192, r12, n16). Both merged from intermediates. |
| 950766 | tr_30b_imo_v1 | h200 | 4 | Qwen3-30B-A3B tool refinement with **v1 prompt** (strict workflow) via --system_prompt_file, IMOBench, t16384, r12, n16, temp0.7, topp0.9, dp=4. Compare against v2 (job 944409). |

## 2026-02-11 Qwen3-30B-A3B IMOBench Verification

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 950735 | verify_30b_imo | cpu | 0 | Verify Qwen3-30B-A3B IMOBench baseline (t32768) + tool refinement v1 (t16384, merged intermediates up to r3). Runs verify_solutions, verify_by_round, verify_cumulative. |

## 2026-02-11 Qwen3-30B-A3B Tool Refinement t32768 Experiments

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 950660 | ctr_30b_imo | h200 | 4 | Qwen3-30B-A3B compact tool refinement on IMOBench, t16384, r12, n16, temp0.7, topp0.9, dp=4, --compact_context |
| 950672 | tr_30b_hmmt_t32k | h200 | 4 | Qwen3-30B-A3B tool refinement on HMMT, t32768, r12, n16, temp0.7, topp0.9, dp=4 |
| 950673 | ctr_30b_hmmt_t32k | h200 | 4 | Qwen3-30B-A3B compact tool refinement on HMMT, t32768, r12, n16, temp0.7, topp0.9, dp=4, --compact_context |
| 950674 | tr_30b_imo_t32k | h200 | 4 | Qwen3-30B-A3B tool refinement on IMOBench, t32768, r12, n16, temp0.7, topp0.9, dp=4 |
| 950675 | ctr_30b_imo_t32k | h200 | 4 | Qwen3-30B-A3B compact tool refinement on IMOBench, t32768, r12, n16, temp0.7, topp0.9, dp=4, --compact_context |

## 2026-02-10 Verification of Recent Outputs (HMMT + IMOBench)

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 947598 | verify_recent | cpu | 0 | Verify all 6 recent merged outputs (Qwen3-30B-A3B + Qwen3-4B hmmt/imobench, tool/compact_tool refinement). Runs verify_solutions.py + verify_by_round.py. Results saved to verification_results_*.txt |

## 2026-02-09 Tool Refinement v2 — Mid-Reasoning Prompts

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 944427 | tr_4b_imo_v2 | nvl | 2 | Qwen3-4B-Instruct-2507 tool refinement v2 (updated prompts encouraging mid-reasoning tool calls) on IMOBench, t8192, r12, n16, temp0.7, topp0.9, dp=2 |
| 944431 | tr_4b_hmmt_v2 | nvl | 2 | Qwen3-4B-Instruct-2507 tool refinement v2 (updated prompts encouraging mid-reasoning tool calls) on HMMT, t8192, r12, n16, temp0.7, topp0.9, dp=2 |

## 2026-02-09 Qwen3-4B-Instruct-2507 Compact Tool Refinement IMOBench t16k (NVL)

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 944420 | ctr_4b_imo_t16k | nvl | 2 | Qwen3-4B-Instruct-2507 compact tool refinement on IMOBench, t16384, r12, n16, temp0.7, topp0.9, dp=2, --compact_context |

## 2026-02-09 Qwen3-30B-A3B Tool Refinement Experiments

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 944407 | tr_30b_hmmt | h200 | 4 | Qwen3-30B-A3B-Instruct-2507 tool refinement on HMMT, t16384, r12, n16, temp0.7, topp0.9, dp=4, env=vllm_0_8_4 |
| 944409 | tr_30b_imo | h200 | 4 | Qwen3-30B-A3B-Instruct-2507 tool refinement on IMOBench, t16384, r12, n16, temp0.7, topp0.9, dp=4, env=vllm_0_8_4 |
| 944410 | ctr_30b_hmmt | h200 | 4 | Qwen3-30B-A3B-Instruct-2507 compact tool refinement on HMMT, t16384, r12, n16, temp0.7, topp0.9, dp=4, --compact_context, env=vllm_0_8_4 |
| 944411 | ctr_30b_imo | h200 | 4 | Qwen3-30B-A3B-Instruct-2507 compact tool refinement on IMOBench, t16384, r12, n16, temp0.7, topp0.9, dp=4, --compact_context, env=vllm_0_8_4 |

## 2026-02-09 HMMT t16k Tool Refinement Experiments

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 944403 | tr_hmmt_t16k | h200 | 4 | Qwen3-4B-Instruct-2507 tool refinement on HMMT, t16384, r12, n16, temp0.7, topp0.9, dp=4 |
| 944405 | ctr_hmmt_t16k | h200 | 4 | Qwen3-4B-Instruct-2507 compact tool refinement on HMMT, t16384, r12, n16, temp0.7, topp0.9, dp=4, --compact_context |

## 2026-02-09 Resubmit compact_tr_pbhf (rerun of cancelled 944072)

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 944143 | compact_tr_pbhf | nvl | 2 | Qwen3-4B-Instruct-2507 compact tool refinement on HF ProofBench (435 problems), t16384, r12, n16, temp0.7, topp0.9, dp=2, --compact_context |

## 2026-02-09 Verify Compact Tool Refinement Results

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 944141 | verify_ctr_imo | cpu | 0 | Verify compact tool refinement on IMOBench (Qwen3-4B, t8192, r12, n16) |
| 944142 | verify_ctr_hmmt | cpu | 0 | Verify compact tool refinement on HMMT (Qwen3-4B, t8192, r12, n16) |

## 2026-02-09 RC Inference on ProofBench

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 944086 | rc_4b_proof | h200 | 4 | Qwen3-4B-Instruct-2507 RC on proofbench (60 problems), t16384, r12, n16, temp0.7, topp0.9, dp=4 |
| 944087 | rc_4b_pbhf | h200 | 4 | Qwen3-4B-Instruct-2507 RC on HF ProofBench (435 problems), t16384, r12, n16, temp0.7, topp0.9, dp=4 |

## 2026-02-09 HF ProofBench Experiments (wenjiema02/ProofBench, 435 problems)

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 944071 | baseline_4b_pbhf | nvl | 2 | Qwen3-4B-Instruct-2507 baseline on HF ProofBench, t32768, n16, temp0.7, topp0.9, dp=2 |
| 944072 | compact_tr_pbhf | nvl | 2 | Qwen3-4B-Instruct-2507 compact tool refinement on HF ProofBench, t16384, r12, n16, temp0.7, topp0.9, dp=2, --compact_context |

## 2026-02-09 ProofBench Experiments (Qwen3-4B-Instruct-2507)

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 944060 | baseline_4b_proof | h200 | 4 | Qwen3-4B-Instruct-2507 baseline on proofbench, t32768, n16, temp0.7, topp0.9, dp=4 |
| 944061 | compact_tr_proof | h200 | 4 | Qwen3-4B-Instruct-2507 compact tool refinement on proofbench, t16384, r12, n16, temp0.7, topp0.9, dp=4, --compact_context |

## 2026-02-09 Compact Tool Refinement Experiments (no generation in context)

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 944057 | compact_tr_hmmt | h200 | 4 | Qwen3-4B-Instruct-2507 compact tool refinement on hmmt, t8192, r12, n16, temp0.7, topp0.9, dp=4, --compact_context (only prompt+summary in context) |
| 944058 | compact_tr_imo | h200 | 4 | Qwen3-4B-Instruct-2507 compact tool refinement on imobench, t8192, r12, n16, temp0.7, topp0.9, dp=4, --compact_context (only prompt+summary in context) |

## 2026-02-08 Tool-Calling Refinement Experiments

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 942727 | tool_refine_qwen3_4b_imobench_t8k | h200 | 4 | Qwen3-4B-Instruct-2507 tool-calling refinement on imobench, t8192, r12, n16, temp0.7, topp0.9, dp=4, with feedback+redirect refiner prompt |
| 942746 | tool_refine_qwen3_4b_hmmt_t8k | h200 | 4 | Qwen3-4B-Instruct-2507 tool-calling refinement on hmmt, t8192, r12, n16, temp0.7, topp0.9, dp=4, with feedback+redirect refiner prompt |

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
| 929535 | refine_qwen3_4b_hmmt | h200 | 4 | Qwen3-4B-Instruct-2507 refinement on hmmt, t4096, r12, n16, temp0.7, topp0.9, dp=4, --accumulate |
| 929536 | rc_qwen3_4b_hmmt_t8192 | h200 | 4 | Qwen3-4B-Instruct-2507 RC on hmmt, t8192, r12, n16, temp0.7, topp0.9, dp=4 |
| 932808 | refine_qwen3_4b_hmmt_t8192 | nvl | 2 | Qwen3-4B-Instruct-2507 refinement on hmmt, t8192, r12, n16, temp0.7, topp0.9, dp=2, --accumulate |

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
