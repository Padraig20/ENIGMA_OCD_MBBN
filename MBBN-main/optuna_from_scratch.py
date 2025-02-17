import optuna
import subprocess
import os
import wandb
import torch
import sys

# Restrict PyTorch from using all GPU memory at once
torch.cuda.set_per_process_memory_fraction(0.9, 0)  # Limit to 90% usage

# # Initialize WandB
# WANDB_PROJECT = "enigma-ocd_mbbn"
# WANDB_ENTITY = "pakmasha-seoul-national-university"
# WANDB_API_KEY = "285aadcd46b8af7731dca6cf50c7051164415461"

# os.environ["WANDB_API_KEY"] = WANDB_API_KEY
# wandb.login(key=WANDB_API_KEY)
# wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, reinit=True, name="optuna_tuning")

# Define output log file path
log_file_path = "/pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main/slurm/output_log/optuna_tuning.out"

def log_to_file(message):
    """Helper function to log messages to file and print to console."""
    with open(log_file_path, "a") as log_file:
        log_file.write(message + "\n")
    print(message)  # Also print to console

def objective(trial):
    """Objective function for Optuna to optimize hyperparameters."""

    # Log the trial number
    log_to_file(f"Starting Trial: {trial.number}")

    # Define the hyperparameter search space
    # batch_size = trial.suggest_categorical("batch_size_phase2", [8, 16, 32, 64])
    try:
        batch_size = trial.suggest_categorical("batch_size_phase2", [8, 16])
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            log_to_file("Warning: Reducing batch size due to memory constraints.")
            batch_size = 8  # Use the smallest batch size

    # lr_init = trial.suggest_float("lr_init_phase2", 1e-5, 3e-4, log=True)
    lr_policy = trial.suggest_categorical("lr_policy_phase2", ["step", "SGDR"])  # step이 더 잘 나옴
    # spat_diff_loss_type = trial.suggest_categorical("spat_diff_loss_type", 
    #                                                 ["minus_log", "reciprocal_log", "exp_minus", "log_loss", "exp_whole"])
    # spatial_loss_factor = trial.suggest_float("spatial_loss_factor", 1.0, 5.0, log=True)
    # optimizer = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])  # Adam이 더 잘 나옴
    # lr_warmup = trial.suggest_int("lr_warmup_phase2", 100, 2000, step=100)
    # lr_decay = trial.suggest_float("lr_gamma_phase2", 0.90, 0.99, log=True)
    # weight_decay = trial.suggest_float("weight_decay_phase2", 1e-4, 1e-2, log=True)
    # lr_step = trial.suggest_int("lr_step_phase2", 1000, 5000, step=500)

    # Construct the command to run your training script
    command = (
        f"python main.py --dataset_name ENIGMA_OCD --wandb_mode offline"
        f"--step 3 --batch_size_phase3 32 --lr_init_phase3 3e-5 --lr_policy_phase3 {lr_policy} "
        f"--workers_phase3 32 --target reconstruction "
        f"--fmri_type divided_timeseries --transformer_hidden_layers 8 "
        f"--seq_part head --fmri_dividing_type four_channels "
        f"--spatiotemporal --spat_diff_loss_type minus_log --spatial_loss_factor 4.0 "
        f"--exp_name pretraining_random_700_lr_policy_seed1 --seed 1 --sequence_length_phase3 700 "
        f"--intermediate_vec 316 --nEpochs_phase3 5 --num_heads 4 --filtering_type Boxcar "
        f"--use_mask_loss --masking_method spatiotemporal --spatial_masking_type random_ROIs --num_random_ROIs 290 "
        f"--temporal_masking_type time_window --temporal_masking_window_size 20 --window_interval_rate 2"
    )

    # Run the command
    process = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Print command output for debugging
    log_to_file("STDOUT: " + process.stdout)
    log_to_file("STDERR: " + process.stderr)

    # Check if the training script encountered an error
    if process.returncode != 0:
        log_to_file(f"Error: Training script failed with exit code {process.returncode}")
        return float("nan")
    
    # Extract AUROC from the logs
    auroc = None
    for line in process.stdout.split("\n"):
        if "val_AUROC" in line:
            try:
                auroc = float(line.split(":")[-1].strip())

                # Check for NaN AUROC
                if auroc is None or auroc != auroc:
                    raise ValueError("AUROC returned NaN or None")
                break
            except ValueError:
                log_to_file("Warning: AUROC extraction failed or returned NaN.")
                return float("nan")

    # Ensure AUROC was found
    if auroc is None:
        log_to_file("Error: AUROC value not found in training logs.")
        return float("nan")

    # Extract the AUROC result from the training logs
    auroc = None
    for line in process.stdout.split("\n"):
        if "val_AUROC" in line:
            try:
                auroc = float(line.split(":")[-1].strip())
                break
            except ValueError:
                return float("nan")

    # Handle cases where AUROC is still None
    if auroc is None or auroc != auroc:  # Additional safeguard
        log_to_file("Warning: Trial returned NaN AUROC. Skipping trial.")
        return float("nan")

    # # Log hyperparameters and results to WandB
    # wandb.log({
    #     "trial": trial.number,
    #     "batch_size_phase2": batch_size,
    #     "lr_init_phase2": lr_init,
    #     "lr_policy_phase2": lr_policy,
    #     "spat_diff_loss_type": spat_diff_loss_type,
    #     "spatial_loss_factor": spatial_loss_factor,
    #     "optimizer": optimizer,
    #     "lr_warmup_phase2": lr_warmup,
    #     "lr_gamma_phase2": lr_decay,
    #     "weight_decay_phase2": weight_decay,
    #     "lr_step_phase2": lr_step,
    #     "val_AUROC": auroc
    # })


    # Free up GPU memory before returning
    torch.cuda.empty_cache()
    log_to_file("Freed GPU memory after Trial: " + str(trial.number))

    return auroc if auroc else float("nan")

# Optuna Study
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study.optimize(objective, n_trials=100)  # Adjust n_trials as needed

    # Get the best trial and its parameters
    best_trial_message = (
        f"\nBest is trial {study.best_trial.number} with value: {study.best_value}\n"
        f"Best Hyperparameters: {study.best_trial.params}"
    )

    # Log to file and force writing to output
    log_to_file(best_trial_message)
    sys.stdout.flush()

    # wandb.finish()