Check the SLURM logs for job ID: $ARGUMENTS

1. Read both the stdout (.out) and stderr (.err) log files from `/scratch/dkhasha1/tli104/slurm_logs/` for this job ID. Use glob to find the files matching `*_$ARGUMENTS.out` and `*_$ARGUMENTS.err`.
2. Summarize the job status: is it still running, completed successfully, or failed?
3. If running, report progress (e.g., how many prompts processed).
4. If failed, identify the error and suggest a fix.
5. Report key stats: model loaded, GPU memory, KV cache size, throughput if available.
