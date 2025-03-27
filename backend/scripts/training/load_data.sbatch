#!/bin/bash
#SBATCH --partition main                         
#SBATCH --time 0-12:00:00                      
#SBATCH --job-name SFT            
#SBATCH --output=output/%J-load_data.out              
#SBATCH --mail-user=yuvalx123x@gmail.com        
#SBATCH --mail-type=ALL                       
#SBATCH --gpus=rtx_3090:1                            

echo -e "job ID             ${RED}$SLURM_JOBID${RESET}"
echo -e "job Nodes      ${YELLOW}$SLURM_JOB_NODELIST${RESET}\n"

module load anaconda
source activate my_env      
python finetuning/d.py
