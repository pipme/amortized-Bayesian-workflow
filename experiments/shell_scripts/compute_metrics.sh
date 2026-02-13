#!/bin/bash
#SBATCH --job-name=metrics
#SBATCH --cpus-per-task=1
#SBATCH --partition=short
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH -M Anonymous
#SBATCH --output=./slurm/metrics_%A_%a.out   # Note: added %a for array task ID
#SBATCH --error=./slurm/metrics_%A_%a.err

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
notebook="check_results"
echo "Running task: $task_name, notebook: $notebook"
python -mpapermill $notebook.ipynb $output_dir/$notebook.ipynb -p task_name $task_name
