#!/bin/bash
#SBATCH -A m4750
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gpus-per-task=4
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

export SLURM_CPU_BIND="cores"

python main.py --dataset_name ENIGMA_OCD \
--step 3 --batch_size_phase3 32 --lr_init_phase3 0.0003  --lr_policy_phase3 step \
--workers_phase3 32 --target reconstruction \
--fmri_type divided_timeseries --transformer_hidden_layers 8 \
--seq_part head --fmri_dividing_type four_channels \
--spatiotemporal --spat_diff_loss_type minus_log  --spatial_loss_factor 4.0 \
--exp_name hub_700_300roi_maskrate0.5_seed1 --seed 1  --sequence_length_phase3 700 \
--intermediate_vec 316 --nEpochs_phase3 3000 --num_heads 4 --filtering_type Boxcar \
--use_mask_loss --masking_method spatiotemporal --spatial_masking_type hub_ROIs --num_hub_ROIs 300 --communicability_option remove_high_comm_node \
--temporal_masking_type time_window --temporal_masking_window_size 20 --window_interval_rate 2  \
--weight_decay_phase3 0.0039 --lr_gamma_phase3 0.93 --lr_step_phase3 3500 --lr_warmup_phase3 400 --masking_rate 0.5 \
2> /pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main/failed_experiments/enigma_ocd_pretrain_error.log