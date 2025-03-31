
# Developers Guide ( SLURM )

</br>


## Clone SciGURU repository

### 1 - Enter root directory

- via MobaXtrem - open terminal
- via SSH session </br>

    ```bash
    ssh <username>:slurm.bgu.ac.il
    <password>
    ```

### 2 - Clone repository
```bash
git clone https://github.com/golyuval/SciGURU
username : golyuval
password : <repository access token>
```

</br>

--- 



## Reset Environment



### 1 - Enter SciGURU directory
```bash
cd SciGURU/
```

### 2 - Install packages
```bash
./Backend/Scripts/reset_env.sh
```

### 3 - Activate my_env (if not activated)
```bash
conda activate my_env
```

### 4 - Save critical tokens
```bash
export HF_READ_TOKEN=<hugging_face_read_token>
export HF_WRITE_TOKEN=<hugging_face_write_token>
```

--- 



## Run Scripts


### 1 - Enter backend directory
```bash
cd SciGURU/Backend/
```

### 2 - Run Scripts

**Load Data** - task for loading data from hugging face into **Code/Train/SFT/datasets**
```bash
sbatch Scripts/Train/load_data.sbatch
```

**SFT** - task for performing SFT training (rtx_4090) ---> save model to **Versions** 
```bash
sbatch Scripts/Train/SFT.sbatch
```

**SFT_** - same task as SFT (rtx_3090) 
```bash
sbatch Scripts/Train/SFT_.sbatch
```
</br>

