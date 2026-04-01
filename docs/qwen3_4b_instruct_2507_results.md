# Qwen3-4B-Instruct-2507 — All Results

Generated: 2026-02-13

## Summary Table (Final pass@1)

| Dataset | Method | Tokens | Rounds | N | pass@1 |
|---------|--------|--------|--------|---|--------|
| HMMT Feb (30) | Baseline | 16384 | — | 16 | 28.75% |
| HMMT Feb (30) | RC | 4096 | 12 | 16 | 26.46% |
| HMMT Feb (30) | RC | 8192 | 12 | 16 | 33.75% |
| HMMT Feb (30) | Tool Refinement v1 | 8192 | 12 | 16 | 29.58% |
| HMMT Feb (30) | Compact TR v1 | 8192 | 12 | 16 | 29.38% |
| HMMT Feb (30) | Compact TR v1 | 16384 | 12 | 16 | — (a) |
| HMMT Nov (30) | Baseline | 16384 | — | 16 | 40.21% |
| HMMT Combined (60) | RC (t16k+rc2k) | 16384 | 12 | 16 | 40.10% |
| IMOBench (400) | Baseline | 16384 | — | 16 | 35.61% |
| IMOBench (400) | RC | 4096 | 12 | 16 | 21.89% |
| IMOBench (400) | RC (t16k+rc2k) | 16384 | 12 | 16 | 32.88% |
| IMOBench (400) | RC Reimpl | 16384 | 4 | 4 | 40.81% |
| IMOBench (400) | Tool Refinement v1 | 8192 | 12 | 16 | 32.22% |
| IMOBench (400) | Tool Refinement v2 | 8192 | 12 | 16 | 28.83% |
| IMOBench (400) | Tool Refinement v1 | 16384 | 6 | 16 | 32.22% |
| IMOBench (400) | Compact TR v1 | 8192 | 12 | 16 | 32.94% |
| IMOBench (400) | Compact TR v1 | 16384 | 12 | 16 | — (b) |
| IMOBench (400) | Compact TR v3 | 8192 | 12 | 16 | 19.25% |
| ProofBench (60) | Baseline | 32768 | — | 16 | (c) |
| ProofBench (60) | RC | 16384 | 12 | 16 | (c) |
| ProofBench (60) | Compact TR v1 | 16384 | 12 | 16 | (c) |
| ProofBench HF (435) | Baseline | 32768 | — | 16 | (c) |
| ProofBench HF (435) | RC | 16384 | 12 | 16 | (c) |
| ProofBench HF (435) | Compact TR v1 | 16384 | 12 | 16 | (c) |

Notes:
- (a) Overall verify_solutions.py failed; per-round R1=29.38%, R2=28.94%
- (b) Overall verify_solutions.py failed; per-round R1=34.62%, R2=35.30%
- (c) ProofBench requires Gemini grading (grade_proofs_gemini.py) — not yet verified
- Sampling: temp=0.7, top_p=0.9 unless noted
- RC (t16k+rc2k) and RC Reimpl use temp=1.0, top_p=1.0
- Tool Refinement v2 uses guidance-style prompts; v1 uses strict workflow prompts
- Compact TR v3 uses RC-style summarization (failed approach — accuracy collapsed)

---

## Per-Round pass@1

### RC — HMMT Feb, t4096, r12, n16

| Round | pass@1 | Samples |
|-------|--------|---------|
| R1 | 10.62 | 480 |
| R2 | 16.01 | 430 |
| R3 | 20.87 | 420 |
| R4 | 23.64 | 419 |
| R5 | 23.45 | 413 |
| R6 | 25.86 | 419 |
| R7 | 26.68 | 406 |
| R8 | 25.13 | 413 |
| R9 | 27.46 | 401 |
| R10 | 25.25 | 407 |
| R11 | 27.33 | 408 |
| R12 | 26.91 | 416 |

### RC — HMMT Feb, t8192, r12, n16

| Round | pass@1 | Samples |
|-------|--------|---------|
| R1 | 27.71 | 480 |
| R2 | 27.93 | 353 |
| R3 | 28.59 | 389 |
| R4 | 31.11 | 393 |
| R5 | 30.97 | 390 |
| R6 | 32.67 | 391 |
| R7 | 32.13 | 399 |
| R8 | 32.64 | 389 |
| R9 | 33.69 | 394 |
| R10 | 33.12 | 386 |
| R11 | 33.02 | 389 |
| R12 | 33.91 | 397 |

### RC (t16k+rc2k) — HMMT Combined, r12, n16

| Round | pass@1 | Samples |
|-------|--------|---------|
| R1 | 34.48 | 960 |
| R2 | 34.70 | 690 |
| R3 | 38.27 | 821 |
| R4 | 38.67 | 834 |
| R5 | 39.45 | 839 |
| R6 | 38.35 | 831 |
| R7 | 41.21 | 828 |
| R8 | 41.48 | 823 |
| R9 | 42.60 | 790 |
| R10 | 42.27 | 826 |
| R11 | 41.62 | 814 |
| R12 | 41.70 | 819 |

### RC — IMOBench, t4096, r12, n16

| Round | pass@1 | Samples |
|-------|--------|---------|
| R1 | 11.17 | 6400 |
| R2 | 14.86 | 3422 |
| R3 | 16.93 | 3360 |
| R4 | 17.46 | 3185 |
| R5 | 18.77 | 3035 |
| R6 | 17.70 | 2979 |
| R7 | 19.23 | 2938 |
| R8 | 19.37 | 2809 |
| R9 | 19.08 | 2816 |
| R10 | 19.40 | 2747 |
| R11 | 17.98 | 2710 |
| R12 | 19.62 | 2708 |

### RC (t16k+rc2k) — IMOBench, r12, n16

| Round | pass@1 | Samples |
|-------|--------|---------|
| R1 | 35.16 | 6400 |
| R2 | 35.44 | 2306 |
| R3 | 35.63 | 3204 |
| R4 | 35.82 | 3099 |
| R5 | 37.34 | 3010 |
| R6 | 36.56 | 3011 |
| R7 | 37.40 | 3020 |
| R8 | 37.60 | 2944 |
| R9 | 38.44 | 2890 |
| R10 | 38.15 | 3002 |
| R11 | 38.02 | 2833 |
| R12 | 38.38 | 2850 |

### RC Reimpl — IMOBench, n4, t16384 (per-step)

| Step | pass@1 |
|------|--------|
| 1 | 34.88 |
| 2 | 38.50 |
| 3 | 40.19 |
| 4 | 40.81 |

### Tool Refinement v1 — IMOBench, t8192, r12, n16 (cumulative)

| Round | pass@1 |
|-------|--------|
| R1 | 31.95 |
| R2 | 32.22 |
| R3 | 32.22 |

Note: Only 2-3 effective rounds; tool refinement converges early at t8192.

### Tool Refinement v2 — IMOBench, t8192, r12, n16 (cumulative)

| Round | pass@1 |
|-------|--------|
| R1 | 25.78 |
| R2 | 28.83 |

Note: Only 2 effective rounds; v2 guidance prompts underperform v1 strict prompts.

### Tool Refinement v1 — IMOBench, t16384, r6, n16 (cumulative)

| Round | pass@1 |
|-------|--------|
| R1 | 22.50 |
| R2 | 23.89 |
| R3 | 27.86 |
| R4 | 31.92 |
| R5 | 32.20 |
| R6 | 32.22 |

Note: More rounds are effective at t16384 (6 vs 2 at t8192), but same final accuracy (32.22%).

### Tool Refinement v1 — IMOBench, t16384, r6, n16 (per-round)

| Round | pass@1 | Samples |
|-------|--------|---------|
| R1 | 34.35 | 3970 |
| R2 | 34.80 | 3382 |
| R3 | 26.75 | 1004 |
| R4 | 21.39 | 1311 |
| R5 | 19.62 | 148 |
| R6 | 12.50 | 16 |

### Compact TR v1 — IMOBench, t8192, r12, n16 (cumulative)

| Round | pass@1 |
|-------|--------|
| R1 | 32.56 |
| R2 | 32.59 |
| R3 | 32.83 |
| R4 | 32.91 |
| R5 | 32.91 |
| R6 | 32.94 |
| R7–R12 | 32.94 |

Note: Converges by R6; nearly all improvement in R1-R2.

### Compact TR v1 — HMMT Feb, t8192, r12, n16 (per-round)

| Round | pass@1 | Samples |
|-------|--------|---------|
| R1 | 28.54 | 480 |
| R2 | 31.22 | 293 |
| R3–R7 | 0.00 | ≤11 |

### Compact TR v1 — HMMT Feb, t16384, r12, n16 (per-round)

| Round | pass@1 | Samples |
|-------|--------|---------|
| R1 | 29.38 | 480 |
| R2 | 28.94 | 366 |
| R3–R9 | ≤12.50 | ≤22 |

### Compact TR v3 — IMOBench, t8192, r12, n16 (per-round)

| Round | pass@1 | Samples |
|-------|--------|---------|
| R1 | 32.11 | 3949 |
| R2 | 39.55 | 187 |
| R3 | 51.37 | 105 |
| R4 | 41.67 | 78 |
| R5 | 40.15 | 79 |
| R6 | 42.80 | 50 |
| R7 | 28.79 | 53 |
| R8 | 39.39 | 33 |
| R9 | 42.59 | 33 |
| R10 | 46.43 | 31 |
| R11 | 41.67 | 37 |
| R12 | 47.92 | 27 |

Overall: 19.25% — RC-style summarization caused massive sample attrition after R1 (6400→3949→187).

### Tool Refinement v1 — HMMT Feb, t8192, r12, n16 (per-round)

| Round | pass@1 | Samples |
|-------|--------|---------|
| R1 | 28.96 | 480 |
| R2 | 30.93 | 312 |
| R3 | 0.00 | 1 |

Note: Only 2 effective rounds on HMMT.

### Compact TR v1 — IMOBench, t16384, r12, n16 (per-round)

| Round | pass@1 | Samples |
|-------|--------|---------|
| R1 | 34.62 | 6400 |
| R2 | 35.30 | 4305 |
| R3 | 17.12 | 205 |
| R4–R12 | ≤17.02 | ≤60 |

### Tool Refinement v1 — IMOBench, t8192, r12, n16 (per-round)

| Round | pass@1 | Samples |
|-------|--------|---------|
| R1 | 31.95 | 6400 |
| R2 | 36.05 | 3492 |
| R3 | 0.00 | 5 |

### Tool Refinement v2 — IMOBench, t8192, r12, n16 (per-round)

| Round | pass@1 | Samples |
|-------|--------|---------|
| R1 | 25.81 | 6400 |
| R2 | 29.16 | 6121 |

Note: v2 retains more samples into R2 (6121 vs 3492 for v1) but lower accuracy.
