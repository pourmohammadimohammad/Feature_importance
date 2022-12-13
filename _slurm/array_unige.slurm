#!/bin/bash -l

#SBATCH --mail-type ALL
#SBATCH --array=0-47
#SBATCH --job-name server_code
#SBATCH --output=./out/exp_%a.out
#SBATCH --error=./out/exp_%a.err

#SBATCH --partition=shared-cpu

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=0
#SBATCH --cpus-per-task 4
#SBATCH --time 12:00:00

module load Anaconda3

source activate leave_one_out

python3 ./server_code.py ${SLURM_ARRAY_TASK_ID}


