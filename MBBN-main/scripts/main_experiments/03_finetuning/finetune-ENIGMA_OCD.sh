#!/bin/bash
## 01 scripts explanation
# ABCD : name of dataset
# sex : name of task
# MBBN: name of model (step : 2)
# seed 1 : seed is set as 1. this decides dataset splits

## 02 environment setting
# conda activate {your environment}
# cd /pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main
# salloc -A m4750_g -C gpu -q interactive -t 4:00:00 -N 1 --gpus 1
# source mbbn/bin/activate
# ./scripts/main_experiments/03_finetuning/finetune-ENIGMA_OCD.sh

## 03 experiment
# ENIGMA_OCD
python main.py --dataset_name ENIGMA_OCD --base_path /pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main --enigma_path /pscratch/sd/p/pakmasha/MBBN_data \
--step 2 --batch_size_phase2 32 --lr_init_phase2 3e-5 --lr_policy_phase2 step \
--workers_phase2 32 --fine_tune_task binary_classification --target OCD \
--fmri_type divided_timeseries --transformer_hidden_layers 8 \
--seq_part head --fmri_dividing_type four_channels \
--spatiotemporal --spat_diff_loss_type minus_log --spatial_loss_factor 1.0 \
--exp_name finetune_enigma_hub100_epoch1408_lower_lr_seed1 --seed 1 --sequence_length_phase2 700 \
--intermediate_vec 316 --nEpochs_phase2 200 --num_heads 4 \
--finetune --pretrained_model_weights_path /pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main/experiments/ENIGMA_OCD_divfreqBERT_reconstruction_reconstruction_pretraining_hub_700_100roi_highprec_seed1/ENIGMA_OCD_divfreqBERT_reconstruction_reconstruction_pretraining_hub_700_100roi_highprec_seed1_epoch_1408_BEST_val_loss.pth \
2> /pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main/failed_experiments/enigma_ocd_error_finetuning.log

#--prepare_visualization