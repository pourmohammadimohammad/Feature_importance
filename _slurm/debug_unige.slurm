#!/bin/bash -l

#SBATCH --mail-type ALL
#SBATCH --array=0-9
#SBATCH --job-name run_experiment_one_debug

#SBATCH --output=./out/exp_%a.out
#SBATCH --error=./out/exp_%a.err

#SBATCH --partition=debug-cpu

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=0
#SBATCH --cpus-per-task 4
#SBATCH --time 00:15:00

module load Anaconda3

source activate leave_one_out

#python3 ./run_experiment_one.py ${SLURM_ARRAY_TASK_ID}


python3 ./run_experiment_one.py ${SLURM_ARRAY_TASK_ID}