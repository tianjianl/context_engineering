Check all my currently running and recently completed SLURM jobs.

1. Run `squeue -u $USER -o "%.10i %.30j %.10P %.4D %.4C %.10m %.10M %.6t %R" --sort=-i` to see all running/pending jobs.
2. Run `sacct -u $USER --format=JobID,JobName%30,Partition,State,Elapsed,Start,End --starttime=$(date -d '24 hours ago' +%Y-%m-%dT%H:%M:%S) 2>/dev/null || sacct -u $USER --format=JobID,JobName%30,Partition,State,Elapsed,Start,End -S now-1day` to see recent job history.
3. Summarize: how many jobs running, pending, recently completed/failed.
4. For any failed jobs, briefly note the failure if visible.
