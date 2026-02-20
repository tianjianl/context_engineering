Get an overview of all current and recently finished SLURM jobs, then present a concise job status table.

Steps:

1. Run `squeue -u $USER -o "%.10i %.40j %.10P %.6t %.10M %L" --sort=-i` for running/pending jobs.
2. Find recently finished experiments using BOTH of these approaches (sacct often misses jobs outside 24h):
   a. Run `sacct -u $USER --format=JobID%-12,JobName%40,Partition%10,State%12,Elapsed%12,End%22,ExitCode -S now-7days --noheader` for jobs that finished in the last 7 days. Filter out `.batch`/`.extern`/`.0` sub-steps, RUNNING jobs (already covered by squeue), and interactive sessions.
   b. Run `find /scratch/dkhasha1/tli104/outputs/ -name '*.jsonl' -mtime -3 -printf '%T@ %Tc %p\n' | sort -rn | head -30` to find recently written output files (both final and intermediate).
   c. Run `find /scratch/dkhasha1/tli104/slurm_logs/ -name '*.out' -mtime -3 -printf '%T@ %Tc %p\n' | sort -rn | head -20` to find recently modified SLURM logs.
   Use the output files and SLURM logs to identify experiments that completed recently, even if sacct doesn't show them.
3. For each RUNNING job: use `grep -E '\[GPU [0-9]\] (Round|Completed)' /scratch/dkhasha1/tli104/slurm_logs/*_<jobid>.out | tail -15` to get the latest round progress. Also check `.err` for CUDA errors or other failures that might indicate a hung job.
4. For FAILED/TIMEOUT/CANCELLED jobs: read last 50 lines of the `.err` log for a one-line error summary. For TIMEOUT jobs, check how many GPUs completed via `grep 'Completed' *_<jobid>.out` — partial completion means intermediate files may be usable.
5. Cross-reference `slurm_experiments.md` and job names to determine Model, Benchmark, and Method for each job.
6. Output a **single markdown table** containing **both** running/pending jobs (from step 1) **and** recently finished jobs (from step 2). Use these exact columns:

```
| Job ID | Name | Partition | State | Elapsed | Model | Benchmark | Method | Notes |
```

- **Notes** column: for RUNNING show progress (e.g. "Round 8/12, GPU1 stuck"); for COMPLETED leave blank or note elapsed; for FAILED/TIMEOUT/CANCELLED show short error reason.
- One line per job. No verbose log dumps. No results/accuracy numbers.
- **Model** column: use full model names as they appear in `slurm_experiments.md` (e.g. "Qwen3-4B-Instruct-2507", not "Qwen3-4B").

7. After the table, give a 1-2 sentence summary (e.g., "3 running, 2 completed, 1 failed OOM"). Do NOT fix or re-submit any jobs.
