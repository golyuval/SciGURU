#!/bin/bash
module load anaconda
source activate myenv
python infer.py 

source deactivate myenv
module unload anaconda