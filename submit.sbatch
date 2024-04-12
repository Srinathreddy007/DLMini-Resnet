#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --time=12:00:00

#SBATCH --job-name=dlmp
#SBATCH --mem=4GB

#SBATCH --gres=gpu:1

#SBATCH --mail-type=all
#SBATCH --mail-user=ap8638@nyu.edu
#SBATCH --output=slurm_%j.out

module load anaconda3/2020.07
cd /scratch/ap8638/DL-MiniProject/

source activate gpu
pip install -r requirements.txt

python main.py --batch_size_train=256 --num_blocks='[2, 3, 2, 2]' --channel_size='[64, 128, 192, 256]' --zero_init_residual=True --epochs=90 --learning_rate=0.001 --weight_decay=1e-02 --model_name=18_zero_BB_256_AdamW
