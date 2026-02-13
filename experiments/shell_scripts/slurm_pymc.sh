#!/bin/bash
#SBATCH --job-name=pymc     # create a short name for your job
#SBATCH --cpus-per-task=4       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --partition=short
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=03:00:00          # total run time limit (HH:MM:SS)
#SBATCH --array=51-250
#SBATCH -M Anonymous
#SBATCH --output=./slurm/test_%a.out   # Output file; %A for job ID, %a for array task ID
#SBATCH --error=./slurm/test_%a.err

start=$((20 * SLURM_ARRAY_TASK_ID))
end=$((20 * (SLURM_ARRAY_TASK_ID + 1)))
source ~/.bashrc
cd ~/project/sbi_mcmc/experiments
# source ../.sbi_mcmc_bf/bin/activate
# forge
conda activate abw_review
export TQDM_MININTERVAL=60
echo "start running.."

for task_name in GEV BernoulliGLM psychometric_curve CustomDDM; do
    echo "Running task: $task_name"
    python run_pymc.py --task_name "$task_name" -r $start $end
done
