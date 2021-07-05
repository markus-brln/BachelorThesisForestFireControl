#!/bin/bash

# Longer time because training all networks and translating all data
#SBATCH --time=00:25:00 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=20G
#SBATCH --job-name=data_translator_job

# Load the module for R
module load Python/3.7.4-GCCcore-8.3.0
module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4
#module load  Python/3.6.4-foss-2018a

module load matplotlib/3.1.1-foss-2019b-Python-3.7.4

echo Overview of modules that are loaded
module list

echo starting data_translator
python3 data_translator_segments.py $1


# execute with:
# sbatch some_job.sh 

# info about running jobs
# squeue -u $USER

# jobinfo 10674730 (job ID)

# watch last 15 lines
# watch tail -n 15 slurmfile
