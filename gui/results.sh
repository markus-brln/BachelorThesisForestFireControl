#!/bin/bash

#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=30G
# send to GPU if needed #SBATCH --partition=gpu
#SBATCH --job-name=data_translator_job

# Load the module for R
module load Python/3.7.4-GCCcore-8.3.0
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4
module load numba/0.47.0-fosscuda-2019b-Python-3.7.4


module load matplotlib/3.1.1-foss-2019b-Python-3.7.4

echo Overview of modules that are loaded
module list

# Run R using the code from mandelbrot.R
echo starting main
python main.py 1 4      # architecture, experiment


# execute with:
# sbatch some_job.sh 

# info about running jobs
# squeue -u $USER

# jobinfo 10674730 (job ID)

# watch last 15 lines
# watch tail -n 15 slurmfile
