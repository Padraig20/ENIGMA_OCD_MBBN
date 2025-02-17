#!/bin/bash

## 01 scripts explanation
# 1-1. you can change 'communicability_option'
## if communicability_option==remove_high_comm_node, node with high communicability will be removed and model will learn to fill high-communicable nodes.
## elif communicability_option==remove_low_comm_node, node with low communicability will be removed and model will learn to fill low-communicable nodes.

# 1-2. you can change 'num_hub_ROIs' - number of nodes that you remove.
## num_hub_ROIs === 380 for Schaefer 400 atlas
## num_hub_ROIs == 170 for HCPMMP1 symmetric atlas

## 02 environment setting
# conda activate {your environment}
# cd {your directory which contains 'main.py'}

# Perlmutter:
# cd /pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main
# salloc -A m4750_g -C gpu -q interactive -t 4:00:00 -N 1 --gpus 1
# source mbbn/bin/activate
# ./scripts/main_experiments/02_pretraining/pretrain-UKB_random.sh

python main.py --dataset_name UKB  --ukb_path /pscratch/sd/p/pakmasha/UKB_304_ROIs --wandb_mode offline \
--step 3 --batch_size_phase3 32 --lr_init_phase3 3e-5 \
--workers_phase3 16 --target reconstruction \
--fmri_type divided_timeseries --transformer_hidden_layers 8 \
--seq_part head --fmri_dividing_type four_channels \
--spatiotemporal --spat_diff_loss_type minus_log  --spatial_loss_factor 4.0 \
--exp_name pretraining_ukb_700_random_100roi_seed1_check --seed 1  --sequence_length_phase3 464 \
--intermediate_vec 304 --nEpochs_phase3 1000 --num_heads 4 --filtering_type Boxcar \
--use_mask_loss --masking_method spatiotemporal --spatial_masking_type random_ROIs --num_random_ROIs 100 \
--temporal_masking_type time_window --temporal_masking_window_size 20 --window_interval_rate 2  \
2> /pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main/failed_experiments/enigma_ocd_pretrain_error.log

