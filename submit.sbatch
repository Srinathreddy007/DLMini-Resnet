#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --time=12:00:00

#SBATCH --job-name=dlmp
#SBATCH --mem=4GB

#SBATCH --gres=gpu:v100:1

#SBATCH --mail-type=all
#SBATCH --mail-user=ap8638@nyu.edu
#SBATCH --output=slurm_%j.out

module load anaconda3/2020.07
cd /scratch/ap8638/DL-MiniProject/

source activate gpu
pip install -r requirements.txt

python main.py --batch_size_train=64 --num_blocks='[3, 4, 6, 3]' --channel_size='[64, 96, 128, 188]' --he_init=True --epochs=250 --learning_rate=0.01 --weight_decay=5e-04 --model_name=18_he_BN_188
