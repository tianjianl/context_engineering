Get an overview of all current and recent SLURM jobs to orient yourself at the start of a session.

Steps:

1. Run `myjobs` to see all currently running and pending jobs.

2. Run `sacct -u $USER --format=JobID,JobName%40,Partition,State%12,Elapsed,Start,End,ExitCode --starttime=$(date -d '48 hours ago' +%Y-%m-%dT%H:%M:%S) 2>/dev/null || sacct -u $USER --format=JobID,JobName%40,Partition,State%12,Elapsed,Start,End,ExitCode -S now-2days` to see recently finished jobs (last 48 hours). Filter out `.batch` and `.extern` sub-steps — only show the main job entries.

3. For each RUNNING job:
   a. Find its log files in `/scratch/dkhasha1/tli104/slurm_logs/` using glob for `*_<jobid>.out` and `*_<jobid>.err`.
   b. Read the LAST 100 lines of the .out and .err files to check progress.
   c. Look for progress patterns: `[GPU X] Round Y/Z`, tqdm bars (`Processed prompts: XX%`), `[GPU X] Completed processing`, error/traceback lines.
   d. Determine if the job is actively progressing or appears stuck (no new output, errors, hanging).

4. For each RECENTLY FINISHED job (COMPLETED, FAILED, CANCELLED, TIMEOUT from sacct):
   a. If FAILED/CANCELLED/TIMEOUT: find the log files in `/scratch/dkhasha1/tli104/slurm_logs/` and read the LAST 50 lines of the .err file to extract the error message. Keep it to one short line.
   b. If COMPLETED: note it completed successfully.

5. For ALL jobs (running + recent), determine from the job name and logs:
   - **Model**: e.g., Qwen3-4B, Qwen3-8B, Qwen3-30B-A3B
   - **Benchmark**: e.g., IMOBench, HMMT, ProofBench, ProofBench-HF
   - **Method**: e.g., baseline, refinement, RC, tool refinement, compact tool refinement
   - Cross-reference with `slurm_experiments.md` in the project root for descriptions if the job name alone is ambiguous.

6. Present a single summary table with ALL jobs (running first, then pending, then recently finished):

```
| Job ID | Name | Status | Model | Benchmark | Method | Progress / Error |
|--------|------|--------|-------|-----------|--------|------------------|
| 123456 | rc_4b_proof | RUNNING | Qwen3-4B | ProofBench | RC | Round 5/12, batch 60% |
| 123457 | baseline_8b | PENDING | Qwen3-8B | IMOBench | baseline | Waiting (Resources) |
| 123450 | refine_4b | COMPLETED | Qwen3-4B | HMMT | refinement | Done |
| 123449 | tool_ref_4b | FAILED | Qwen3-4B | IMOBench | tool refine | OOM: CUDA out of memory |
```

7. After the table, give a brief 2-3 sentence summary: how many jobs running, pending, completed, failed. Highlight anything that needs attention (stuck or failed jobs).

IMPORTANT:
- Do NOT attempt to fix any failed jobs. Just report the error.
- Do NOT re-submit or modify any jobs.
- Keep error descriptions short (one line max in the table).
- If there are no jobs at all, say so clearly.
