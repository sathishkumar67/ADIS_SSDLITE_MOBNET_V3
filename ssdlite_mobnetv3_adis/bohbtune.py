# constants
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
WARMUP_EPOCHS = 10
NUM_EPOCHS = 100
PATIENCE = 10
END_FACTOR = 1.0

# define the objective function
def objective(trial):
    # define callback to report intermidiate results
    def on_train_epoch_end(score, epoch):
        trial.report(score, step=epoch)  
        if trial.should_prune():
            raise optuna.TrialPruned()
        
    # suggest hyperparameters for the model
    INITIAL_LR = trial.suggest_float("INITIAL_LR", 1e-4, 1e-1, log=True)
    LR_FACTOR = trial.suggest_float("LR_FACTOR", 1e-4, 1e-1, log=True)
    START_FACTOR = trial.suggest_float("START_FACTOR", 1e-4, 1e-1, log=True)
    WEIGHT_DECAY = trial.suggest_float("WEIGHT_DECAY", 1e-4, 1e-1, log=True)
    MOMENTUM = trial.suggest_float("MOMENTUM", 0.7, 0.99)
    
    # create the model
    model = SSDLITE_MOBILENET_V3_Large(num_classes_with_bg=NUM_CLASSES_WITH_BG)
    # move the model to device
    model.to(DEVICE)
    
    # create the optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=INITIAL_LR,
        betas=(MOMENTUM, 0.999),
        weight_decay=WEIGHT_DECAY,
        eps=1e-8,
        fused=True
    )
    
    # tune the model
    best_val_loss = bohb_tunner(
        args={
            "device": DEVICE,
            "warmup_epochs": WARMUP_EPOCHS,
            "num_epochs": NUM_EPOCHS,
            "patience": PATIENCE,
            "initial_lr": INITIAL_LR,
            "lr_factor": LR_FACTOR,
            "start_factor": START_FACTOR,
            "end_factor": END_FACTOR
        },
        model=model,
        optimizer=optimizer,
        dataloaders={"train":train_loader, "val":val_loader},
        callback=on_train_epoch_end
    )
    # return the best validation loss
    return best_val_loss

# define the number of trials
NUM_TRIALS = 5

# load the study
study = optuna.create_study(direction='minimize', 
                            sampler=optuna.samplers.TPESampler(), 
                            pruner=optuna.pruners.HyperbandPruner(),
                            study_name="ssd_mobnetv3_adis_epu_bohbtune",
                            load_if_exists=True)

# Optimize with a callback to stop after NUM_TRIALS complete trials
study.optimize(objective, n_trials=NUM_TRIALS)

# save the study
joblib.dump(study, f"{LOCAL_DIR}/ssd_mobnetv3_adis_bohbtune_study1.pkl")