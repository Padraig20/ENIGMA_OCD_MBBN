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

export SLURM_CPU_BIND="cores"

python3 main.py --dataset_name ENIGMA_OCD --base_path /pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main --enigma_path /pscratch/sd/p/pakmasha/MBBN_data \
--step 2 --batch_size_phase2 8 --lr_init_phase2 3e-5 --lr_policy_phase2 step \
--workers_phase2 8 --fine_tune_task binary_classification --target OCD \
--fmri_type divided_timeseries --transformer_hidden_layers 8 \
--seq_part head --fmri_dividing_type four_channels \
--spatiotemporal --spat_diff_loss_type minus_log --spatial_loss_factor 4.0 \
--exp_name vmd_four_ch_700_final_from_scratch_seed3 --seed 3 --sequence_length_phase2 700 \
--intermediate_vec 316 --nEpochs_phase2 100 --num_heads 4 \
2> /pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main/failed_experiments/enigma_ocd_error_from_scratch.log