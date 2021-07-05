#!/bin/bash              
#SBATCH --time=01:30:00                                                                                                                      
#SBATCH --nodes=1                                                                                                                            
#SBATCH --ntasks-per-node=1                                                                                                                  
#SBATCH --mem=25G                                                                                                                            
#SBATCH --partition=gpu                                                                                                                      
#SBATCH --job-name=CNN_job                                                                                                                   
#SBATCH --gres=name[[:type]:count]                                                                                                           
#SBATCH --gres=gpu:1                                                                                                                         
#SBATCH --cpus-per-task=2                                                                                                     

module load Python/3.7.4-GCCcore-8.3.0
module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4
module load matplotlib/3.1.1-foss-2019b-Python-3.7.4


echo Overview of modules that are loaded
echo starting CNN

# the number behind again stands for the variant!
python CNN_segments.py $1


# execute with:                                                                                                                              
# sbatch some_job.sh                                                                                                                         

# info about running jobs                                                                                                                   
# squeue -u $USER                                                                                                                           
# jobinfo 10674730 (job ID)                                                                                                                   

