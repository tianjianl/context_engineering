Create and submit a new SLURM experiment: $ARGUMENTS

Parse the arguments to determine: model, dataset (hmmt/imobench), partition (h200/nvl), method (baseline/refinement/rc), and any extra params.

Follow these rules from CLAUDE.md:
- H200: `--gres=gpu:4`, `--qos=h200_8`, `--time=10:00:00`
- NVL: `--gres=gpu:2`, no QOS, `--time=72:00:00`
- Always set `--mem=256GB`, `--cpus-per-task=40`

Steps:
1. Create the SLURM script under `/scratch/dkhasha1/tli104/slurm_scripts/` in an appropriately named subdirectory.
2. Use the correct conda environment (default: `/scratch/dkhasha1/tli104/vllm`; for Qwen3-30B use `/scratch/dkhasha1/tli104/vllm_0_8_4`).
3. Set output/error log paths to `/scratch/dkhasha1/tli104/slurm_logs/`.
4. Show the script to the user for review before submitting.
5. After approval, submit with `sbatch` and log in `slurm_experiments.md`.
