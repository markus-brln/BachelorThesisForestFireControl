#!/bin/bash              
#SBATCH --time=00:10:00                                                                                                                      
#SBATCH --nodes=1                                                                                                                            
#SBATCH --ntasks-per-node=1                                                                                                                  
#SBATCH --mem=40G                                                                                                                            
#SBATCH --partition=gpu                                                                                                                      
#SBATCH --job-name=CNN_job                                                                                                                   
#SBATCH --gres=name[[:type]:count]                                                                                                           
#SBATCH --gres=gpu:1                                                                                                                         
#SBATCH --cpus-per-task=2                                                                                                     

module load Python/3.7.4-GCCcore-8.3.0
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4
module load matplotlib/3.1.1-foss-2019b-Python-3.7.4

echo Overview of modules that are loaded
echo starting CNN
python train_models.py


# execute with:                                                                                                                              
# sbatch some_job.sh                                                                                                                         

# info about running jobs                                                                                                                   
# squeue -u $USER                                                                                                                           
# jobinfo 10674730 (job ID)                                                                                                                   

