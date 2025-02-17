#!/bin/bash

## 01 scripts explanation
# Calculating communicability of each ROIs

## 02 environment setting
# conda activate {your environment}
# cd {your directory which contains 'communicability.py'}

# Perlmutter:
# cd /pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main
# salloc -A m4750_g -C gpu -q interactive -t 4:00:00 -N 1 --gpus 1
# source mbbn/bin/activate
# ./scripts/main_experiments/02_pretraining/compute_communicability.sh

python communicability.py --dataset_name UKB --ROI_num 304