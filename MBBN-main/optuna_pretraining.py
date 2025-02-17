import os
import optuna
from main import get_arguments
from trainer import Trainer

def objective(trial):
    # Get default arguments. Use os.getcwd() as the base_path.
    args = get_arguments(base_path=os.getcwd())

    # Override selected hyperparameters for tuning.
    # For phase3 training (e.g., pretraining) tune the learning rate and policy.
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
    args.masking_rate = 0.1

    args.lr_init_phase3 = 3e-5
    # args.lr_policy_phase3 = "step"
    args.lr_policy_phase3 = trial.suggest_categorical("lr_policy_phase3", ["step", "SGDR"])
    

    args.spat_diff_loss_type = "minus_log"
    args.spatial_loss_factor = 4.0

    
    args.spatial_masking_type = "random_ROIs"
    args.num_random_ROIs = 290
    args.temporal_masking_type = "time_window"
    args.temporal_masking_window_size = 20
    args.window_interval_rate = 2
    args.optim_phase3 = "AdamW"
    args.weight_decay_phase3 = 1e-2
    args.lr_gamma_phase3 = 0.97
    args.lr_step_phase3 = 3000
    args.lr_warmup_phase3 = 500
    
    # args.lr_init_phase3 = trial.suggest_float("lr_init_phase3", 3e-5, 3e-4, log=True)
    
    # For tuning speed set a small number of epochs.
    args.nEpochs_phase3 = 2
  
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
  
    # Choose the objective metric based on the task.
    # For regression, we minimize loss; for classification, assume maximization of AUROC,
    # so we return negative AUROC to make it a minimization problem.
    if args.fine_tune_task == 'binary_classification':
        return -best_AUROC
    else:
        return best_loss

def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=3)  # Adjust the number of trials as needed

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