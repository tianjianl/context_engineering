Check all currently running SLURM jobs, read their logs, summarize progress, and estimate remaining time.

Steps:

1. Run `squeue -p h200,h100,nvl -u $USER -o "%.10i %.30j %.10P %.4D %.4C %.10m %.10M %.6t %S %e %L" --sort=-i` to get all running/pending jobs with start time, end time, and time left.

2. For each RUNNING job (state = R):
   a. Find its log files in `/scratch/dkhasha1/tli104/slurm_logs/` using the job ID (glob for `*_<jobid>.out` and `*_<jobid>.err`).
   b. Read the LAST 200 lines of each log file (both .out and .err) to find progress indicators.
   c. Look for these progress patterns:
      - `[GPU X] Round Y/Z` lines — indicates which round each GPU is on out of total rounds
      - `Processed prompts: XX%|... | N/M [elapsed<remaining, speed]` — tqdm-style progress bars showing batch progress within a round
      - `[GPU X] Completed processing` — indicates a GPU finished all its work
      - `Saving results to` — indicates job is nearly done
      - `Loaded N problems` — total problem count
      - `Rounds: N` — total number of rounds
      - Error/traceback lines — indicates a failure
   d. Determine:
      - What model is being run
      - Current round / total rounds for each GPU
      - Within-round progress (percent of prompts processed)
      - Overall progress percentage: approximate as `((completed_rounds + current_round_fraction) / total_rounds) * 100%`
      - Time elapsed (from squeue or log timestamps)
      - Estimated time remaining: use the SLURM time-left column (`%L`) as an upper bound, but also extrapolate from progress rate if enough data is available (e.g., if 40% done in 1 hour, estimate ~1.5 hours remaining)

3. For each PENDING job (state = PD):
   - Report it as pending and show the reason (e.g., Resources, Priority).

4. Present a clear summary table for all jobs:

```
Job ID | Name | Status | Model | Progress | Elapsed | Est. Remaining
-------|------|--------|-------|----------|---------|----------------
123456 | ... | Running | Qwen3-8B | Round 3/5 (60%), batch 45% | 2:30:00 | ~1:40:00
123457 | ... | Pending | - | Waiting (Resources) | - | -
```

5. If there are NO running or pending jobs, say so clearly.
