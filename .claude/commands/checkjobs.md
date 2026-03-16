Check SLURM job status and summarize progress. Do not ask for permission — run all commands directly.

Steps:
1. Run `squeue -p h100,h200,nvl,a100,cpu -u tli104 -o "%.10i %.30j %.10P %.8T %.12M %.12l %.6D %R"` to get all running/pending jobs.
2. Read `slurm_experiments.md` to get context on recent experiments (focus on the top entries — most recent).
3. For each running job, use `checklogs <job_id>` to find its log files, then read the tail of the `.out` log (last ~50 lines) to check progress. Do this in parallel for all running jobs.
4. For recently finished jobs (in slurm_experiments.md but not in squeue), check if their output logs exist and briefly check completion status from the tail of logs.
5. Summarize:
   - Currently running jobs: name, partition, runtime, and brief progress from logs
   - Recently completed jobs: whether they succeeded or failed
   - Any pending jobs
   - Keep it concise — a few lines per job max
