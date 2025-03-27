---
datasets:
- llamafactory/PubMedQA
metrics:
- bleu
- rouge
- meteor
- bertscore
base_model:
- meta-llama/Llama-3.1-8B-Instruct
---

### connect to cluster master node

(me)        - ssh username@slurm.bgu.ac.il
(me)        - password
(me)        - sinteractive --time 0-04:00:00 --gpu rtx_3090:1

### connect to computation node (GPU)

(me)        - ssh username@slurm.bgu.ac.il
(me)        - password
(me)        - module load anaconda
(me)        - source activate myenv
(me)        - python backend/code/infer/infer.py

(me)        - TALK TO THE MODEL
