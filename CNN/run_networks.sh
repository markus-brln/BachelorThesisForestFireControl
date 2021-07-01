#!/bin/bash

# Longer time because training all networks and translating all data
#SBATCH --time=03:15:00 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=20G
#SBATCH --partition=gpu
#SBATCH --job-name=network_training

# Load the module for R
module load Python/3.7.4-GCCcore-8.3.0
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4
#module load  Python/3.6.4-foss-2018a

module load matplotlib/3.1.1-foss-2019b-Python-3.7.4

echo Overview of modules that are loaded
module list

echo "training networks"
for i in {0..4};
do
  for j in {0..9};
  do
    python3 CNN_angle.py $i $j
  done
done
