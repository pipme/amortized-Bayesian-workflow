#!/bin/bash
#SBATCH --job-name=notebook
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=gpu
#SBATCH --nodelist=dgx1-01
#SBATCH --mem-per-gpu=16G
#SBATCH --time=08:00:00
#SBATCH -M Anonymous
#SBATCH -G 1
#SBATCH --output=./slurm/notebook_%A.out   # Output file; %A for job ID, %a for array task ID
#SBATCH --error=./slurm/notebook_%A.err

source ~/.bashrc
# forge
conda activate abw_review
cd ~/project/sbi_mcmc/experiments

declare -A task_datasets

# Define datasets for each task
task_datasets["CustomDDM(dt-0.0001)"]="test_dataset_chunk_1 test_dataset_chunk_3 test_dataset_chunk_4"
task_datasets["GEV"]="test_dataset_chunk_1"
task_datasets["psychometric_curve_overdispersion"]="test_dataset_chunk_1 test_dataset_chunk_2"
task_datasets["BernoulliGLM"]="test_dataset_chunk_1 test_dataset_chunk_2"

# for task_name in  "CustomDDM(dt-0.0001)" psychometric_curve_overdispersion BernoulliGLM GEV; do
# for task_name in psychometric_curve_overdispersion BernoulliGLM; do
for task_name in "CustomDDM(dt-0.0001)"; do
    for test_dataset_name in ${task_datasets["$task_name"]}; do
        python config/modify_config.py hydra.run.dir=./config task_name="'$task_name'" overwrite_stats=true test_dataset_name="$test_dataset_name"

        output_dir=results/$task_name/notebook_output/$test_dataset_name
        mkdir -p "$output_dir"
        for notebook in 04_amortized_inference 03_inference_phase_ood 05_inference_phase_psis 06_chees_hmc; do
            # for notebook in draft; do
            echo "Running task: $task_name, dataset: $test_dataset_name, notebook: $notebook"
            python -mpapermill $notebook.ipynb $output_dir/$notebook.ipynb
        done
    done
done
