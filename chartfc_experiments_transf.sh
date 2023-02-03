#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/%j.out
#SBATCH --mem=160GB
python "/scratch/users/k20116188/prefil/main.py" --data_root="/scratch/users/k20116188/prefil/data" --img_encoder="fc" --txt_encoder="bert" --fusion="transf"
python "/scratch/users/k20116188/prefil/main.py" --data_root="/scratch/users/k20116188/prefil/data" --img_encoder="alexnet" --txt_encoder="bert" --fusion="transf"
python "/scratch/users/k20116188/prefil/main.py" --data_root="/scratch/users/k20116188/prefil/data" --img_encoder="resnet" --txt_encoder="bert" --fusion="transf"
python "/scratch/users/k20116188/prefil/main.py" --data_root="/scratch/users/k20116188/prefil/data" --img_encoder="densenet" --txt_encoder="bert" --fusion="transf"
python "/scratch/users/k20116188/prefil/main.py" --data_root="/scratch/users/k20116188/prefil/data" --img_encoder="vit" --txt_encoder="bert" --fusion="transf"

python "/scratch/users/k20116188/prefil/main.py" --data_root="/scratch/users/k20116188/prefil/data" --img_encoder="fc" --txt_encoder="word_embedding" --fusion="transf"
python "/scratch/users/k20116188/prefil/main.py" --data_root="/scratch/users/k20116188/prefil/data" --img_encoder="alexnet" --txt_encoder="word_embedding" --fusion="transf"
python "/scratch/users/k20116188/prefil/main.py" --data_root="/scratch/users/k20116188/prefil/data" --img_encoder="resnet" --txt_encoder="word_embedding" --fusion="transf"
python "/scratch/users/k20116188/prefil/main.py" --data_root="/scratch/users/k20116188/prefil/data" --img_encoder="densenet" --txt_encoder="word_embedding" --fusion="transf"
python "/scratch/users/k20116188/prefil/main.py" --data_root="/scratch/users/k20116188/prefil/data" --img_encoder="vit" --txt_encoder="word_embedding" --fusion="transf"

python "/scratch/users/k20116188/prefil/main.py" --data_root="/scratch/users/k20116188/prefil/data" --img_encoder="fc" --txt_encoder="lstm" --fusion="transf"
python "/scratch/users/k20116188/prefil/main.py" --data_root="/scratch/users/k20116188/prefil/data" --img_encoder="alexnet" --txt_encoder="lstm" --fusion="transf"
python "/scratch/users/k20116188/prefil/main.py" --data_root="/scratch/users/k20116188/prefil/data" --img_encoder="resnet" --txt_encoder="lstm" --fusion="transf"
python "/scratch/users/k20116188/prefil/main.py" --data_root="/scratch/users/k20116188/prefil/data" --img_encoder="densenet" --txt_encoder="lstm" --fusion="transf"
python "/scratch/users/k20116188/prefil/main.py" --data_root="/scratch/users/k20116188/prefil/data" --img_encoder="vit" --txt_encoder="lstm" --fusion="transf"