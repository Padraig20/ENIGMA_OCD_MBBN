#!/bin/bash
#SBATCH -A m4750
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
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
--step 3 --batch_size_phase3 32 --lr_init_phase3 3e-5  --lr_policy_phase3 step \
--workers_phase3 32 --target reconstruction \
--fmri_type divided_timeseries --transformer_hidden_layers 8 \
--seq_part head --fmri_dividing_type four_channels \
--spatiotemporal --spat_diff_loss_type minus_log  --spatial_loss_factor 4.0 \
--exp_name pretraining_random_700_300roi_seed1_4node_checkrun --seed 1  --sequence_length_phase3 700 \
--intermediate_vec 316 --nEpochs_phase3 3000 --num_heads 4 --filtering_type Boxcar \
--use_mask_loss --masking_method spatiotemporal --spatial_masking_type random_ROIs --num_random_ROIs 300 \
--temporal_masking_type time_window --temporal_masking_window_size 20 --window_interval_rate 2  \
2> /pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main/failed_experiments/enigma_ocd_pretrain_error.log

