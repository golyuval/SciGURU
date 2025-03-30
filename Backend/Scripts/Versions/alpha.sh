#!/bin/bash
module load anaconda
source activate myenv
python code/Infer/alpha.py 

source deactivate myenv
module unload anacondas