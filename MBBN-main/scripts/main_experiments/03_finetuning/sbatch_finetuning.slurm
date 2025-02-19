#!/bin/bash
#SBATCH -A m4750
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 04:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=pakmasha99@gmail.com
#SBATCH --output=/pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main/slurm/output_log/slurm_%j.out
#SBATCH --error=/pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main/slurm/error_log/slurm_%j.err

cd /pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main

module load conda
module load cudnn/9.1.0
module load nccl/2.21.5
conda init bash
source mbbn/bin/activate

export SLURM_CPU_pAKmashA99!263968
BIND="cores"

python main.py --dataset_name ENIGMA_OCD --base_path /pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main --enigma_path /pscratch/sd/p/pakmasha/MBBN_data \
--step 2 --batch_size_phase2 32 --lr_init_phase2 3e-5 --lr_policy_phase2 step \
--workers_phase2 32 --fine_tune_task binary_classification --target OCD \
--fmri_type divided_timeseries --transformer_hidden_layers 8 \
--seq_part head --fmri_dividing_type four_channels \
--spatiotemporal --spat_diff_loss_type minus_log --spatial_loss_factor 1.0 \
--exp_name finetune_enigma_random_290_epoch1956_seed1 --seed 1 --sequence_length_phase2 700 \
--intermediate_vec 316 --nEpochs_phase2 200 --num_heads 4 \
--finetune --pretrained_model_weights_path /pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main/experiments/ENIGMA_OCD_divfreqBERT_reconstruction_reconstruction_pretraining_random_700_290roi_highprec_seed1/ENIGMA_OCD_divfreqBERT_reconstruction_reconstruction_pretraining_random_700_290roi_highprec_seed1_epoch_1956_BEST_val_loss.pth \
2> /pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main/failed_experiments/enigma_ocd_error_finetuning.log