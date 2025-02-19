import os
import optuna
from main import get_arguments
from trainer import Trainer

def objective(trial):
    # Get default arguments. Use os.getcwd() as the base_path.
    args = get_arguments(base_path=os.getcwd())

    """ 
    Fixed parameters
    """
    args.dataset_name = "ENIGMA_OCD"
    args.wandb_mode = "offline"
    args.step = 3
    args.batch_size_phase3 = 32
    args.target = "reconstruction"
    args.fmri_type = "divided_timeseries"
    args.transformer_hidden_layers = 8
    args.seq_part = "head"
    args.fmri_dividing_type = "four_channels"
    args.spatiotemporal = True
    args.workers_phase3 = 32
    args.seed = 1
    args.sequence_length = 700
    args.intermediate_vec = 316
    args.num_heads = 4
    args.use_mask_loss = True
    args.masking_method = "spatiotemporal"
    args.distributed = False
    args.temporal_masking_type = "time_window"

    """
    Learning rate parameters
    """
    args.lr_init_phase3 = 0.0003
    args.lr_policy_phase3 = "step"
    args.weight_decay_phase3 = 0.0039
    args.lr_gamma_phase3 = 0.93
    args.lr_step_phase3 = 3500
    args.lr_warmup_phase3 = 400
    # args.lr_init_phase3 = trial.suggest_float("lr_init_phase2", 1e-5, 3e-4)
    # args.lr_policy_phase3 = trial.suggest_categorical("lr_policy_phase3", ["step", "SGDR"])
    # args.weight_decay_phase3 = trial.suggest_float("weight_decay_phase2", 1e-4, 1e-2)
    # args.lr_gamma_phase3 = trial.suggest_float("lr_gamma_phase2", 0.90, 0.99)
    # args.lr_step_phase3 = trial.suggest_int("lr_step_phase2", 1000, 5000, step=500)
    # args.lr_warmup_phase3 = trial.suggest_int("lr_warmup_phase2", 100, 2000, step=100)

    """
    Spatial difference loss
    """
    args.spat_diff_loss_type = "minus_log"
    # args.spatial_loss_factor = 4.0
    args.spatial_loss_factor = trial.suggest_float("spatial_loss_factor", 1.0, 5.0)

    """
    Spatial masking
    """
    args.spatial_masking_type = "hub_ROIs"
    args.hub_ROIs = 300
    args.communicability_option == "remove_high_comm_node"

    """
    Temporal masking
    """
    args.temporal_masking_window_size = 20
    args.window_interval_rate = 2
    args.masking_rate = 0.1

    """
    Optimizer
    """
    args.optim_phase3 = "AdamW"

    # For tuning speed set a small number of epochs.
    args.nEpochs_phase3 = 3
  
    # When preparing visualization (or depending on your settings), sets is chosen accordingly.
    # For tuning, we simply use all.
    sets = ['train', 'val', 'test']
  
    # Create the Trainer with the updated parameters.
    args.workers = args.workers_phase3
    args.batch_size = args.batch_size_phase3
    args.nEpochs = args.nEpochs_phase3 
    args.task = "MBBN_pretraining"
    args.experiment_folder = "/pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main/optuna"
    args.loaded_model_weights_path = False
    args.gpu = None
    args.optim = args.optim_phase3
    args.weight_decay = args.weight_decay_phase3
    args.lr_policy = args.lr_policy_phase3
    args.lr_step = args.lr_step_phase3
    args.lr_init = args.lr_init_phase3
    args.lr_gamma = 0.5 if args.lr_policy == 'SGDR' else args.lr_gamma_phase3
    args.lr_warmup = int(args.total_iterations * 0.05) if args.lr_warmup_phase3 is None else args.lr_warmup_phase3
    args.experiment_title = args.experiment_folder

    trainer = Trainer(sets=sets, **vars(args))
    
    # Run training. The training() method in your Trainer returns a tuple:
    # (best_AUROC, best_loss, best_MAE)
    best_AUROC, best_loss, best_MAE = trainer.training()
  
    # Since in reconstruction mode the validation/test metrics are not computed,
    # we rely on the loss history recorded by the writer.
    try:
        # Check if the writer has recorded the total loss history for the training set.
        if hasattr(trainer.writer, "total_train_loss_history") and trainer.writer.total_train_loss_history:
            print(trainer.writer.total_train_loss_history)
            min_loss = min(trainer.writer.total_train_loss_history)
            print(f"Using minimum loss from history: {min_loss}")
        else:
            print("No total_train_loss_history found; using best_loss from training()")
            min_loss = best_loss
    except Exception as e:
        print("Error retrieving training loss history:", e)
        min_loss = best_loss

    return min_loss

def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials = 20)  # Adjust the number of trials as needed

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save the best trial and all trial results to an output file.
    output_file = "/pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main/optuna/optuna_trial_results.txt"
    with open(output_file, "w") as f:
        f.write("Best trial:\n")
        f.write(f"  Value: {trial.value}\n")
        f.write("  Params:\n")
        for key, value in trial.params.items():
            f.write(f"    {key}: {value}\n")
        f.write("\nAll trial results:\n")
        for t in study.trials:
            f.write(f"Trial {t.number}: Value = {t.value}, Params = {t.params}\n")

if __name__ == "__main__":
    main()