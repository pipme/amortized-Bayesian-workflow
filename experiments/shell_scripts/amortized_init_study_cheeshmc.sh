#!/bin/bash
#SBATCH --job-name=cheeshmc_init
#SBATCH --cpus-per-gpu=2
#SBATCH --partition=gpu
#SBATCH --mem-per-gpu=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH -M Anonymous
#SBATCH -G 1
#SBATCH --output=./slurm/cheeshmc_init_%A_%a.out   # Note: added %a for array task ID
#SBATCH --error=./slurm/cheeshmc_init_%A_%a.err

source ~/.bashrc
# forge
conda activate abw_review
cd ~/project/sbi_mcmc/experiments

timestamp=$(date +"%Y-%m-%d_%H:%M:%S")
echo "[$timestamp]"
# Define array of task names
task_names=("CustomDDM(dt-0.0001)" "psychometric_curve_overdispersion" "BernoulliGLM" "GEV")

# Select task name based on SLURM_ARRAY_TASK_ID
task_name=${task_names[$SLURM_ARRAY_TASK_ID]}

output_dir=results/$task_name/notebook_output/
mkdir -p "$output_dir"
notebook=chess_hmc_initialization_comparison
mcmc_method="ChEES-HMC"
echo "Running task: $task_name, notebook: $notebook"
python -mpapermill $notebook.ipynb $output_dir/${mcmc_method}_$notebook.ipynb -p mcmc_method $mcmc_method -p task_name $task_name
