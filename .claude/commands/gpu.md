We are currently in a GPU environment (interactive/allocated node). Do NOT use sbatch or create SLURM scripts for running jobs.

Rules for this session:
1. Run Python commands directly (e.g., `python inference/context_refinement_dp.py ...`) instead of wrapping them in SLURM scripts.
2. Do NOT create sbatch scripts or submit jobs via sbatch.
3. Still use the standard environment setup (conda activate, HF_HOME, etc.) if needed.
4. GPU resources are already available — no need to request them.
