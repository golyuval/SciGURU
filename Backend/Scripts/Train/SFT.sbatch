#!/bin/bash
#SBATCH --partition main                         
#SBATCH --time 0-12:00:00                      
#SBATCH --job-name SFT            
#SBATCH --output=Log/Cluster/%J-sft.out              
#SBATCH --mail-user=golyuval@post.bgu.ac.il     
#SBATCH --mail-type=ALL                       
#SBATCH --gpus=rtx_4090:1                            

echo -e "job ID             ${RED}$SLURM_JOBID${RESET}"
echo -e "job Nodes      ${YELLOW}$SLURM_JOB_NODELIST${RESET}\n"

module load anaconda
source activate my_env      
python Code/Train/SFT/train_model.py
