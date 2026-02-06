Submit SLURM scripts: $ARGUMENTS

1. If a directory path is given, submit all .sh scripts in that directory using `for script in <dir>/*.sh; do sbatch "$script"; done`.
2. If a single script path is given, submit it with `sbatch <script>`.
3. If just a name/keyword is given, look for matching scripts under `/scratch/dkhasha1/tli104/slurm_scripts/`.
4. Report all submitted job IDs.
5. Log each submission in `slurm_experiments.md` following the tracking format:
   | Job ID | Job Name | Partition | GPUs | Description |
