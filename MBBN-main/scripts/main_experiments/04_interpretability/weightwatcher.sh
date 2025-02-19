#!/bin/bash
## 01 scripts explanation
# after pretraining, you can run main.py with --weightwatcher option

## 02 environment setting
# conda activate {your environment}
# cd {your directory which contains 'main.py'}

# Perlmutter:
# cd /pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main
# salloc -A m4750_g -C gpu -q interactive -t 4:00:00 -N 1 --gpus 1
# source mbbn/bin/activate
# ./scripts/main_experiments/04_interpretability/weightwatcher.sh

## 03 experiment
python main.py --step 2 --fmri_type divided_timeseries --wandb_mode offline \
--transformer_hidden_layers 8 --num_heads 4 --exp_name enigma_random_300roi_epoch_1209 \
--spatiotemporal --spat_diff_loss_type minus_log --intermediate_vec 316 --fmri_dividing_type four_channels \
--pretrained_model_weights_path /pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main/experiments/ENIGMA_OCD_divfreqBERT_reconstruction_reconstruction_pretraining_random_700_300roi_seed1/ENIGMA_OCD_divfreqBERT_reconstruction_reconstruction_pretraining_random_700_300roi_seed1_epoch_1209_BEST_val_loss.pth \
--wandb_mode offline --weightwatcher --fine_tune_task binary_classification \
--weightwatcher_save_dir /pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main/weightwatcher \
