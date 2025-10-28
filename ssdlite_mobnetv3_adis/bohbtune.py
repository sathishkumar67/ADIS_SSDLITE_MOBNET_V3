"""Hyperparameter tuning helpers using BOHB/Optuna.

This module provides an example objective function for tuning the SSDLite
MobileNetV3 model using Optuna and the BOHB tuner. The code is provided as a
convenient script-like helper: edit constants (data paths, dataloaders, etc.)
before running in your environment.
"""

from __future__ import annotations
import torch
from .model import SSDLITE_MOBILENET_V3_Large
from .trainer import bohb_tunner
from torch import optim
import optuna
import joblib
from tqdm import tqdm

# Constants (update these before running in your environment)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
WARMUP_EPOCHS = 10
NUM_EPOCHS = 100
PATIENCE = 10
END_FACTOR = 1.0
LOCAL_DIR = ""  # replace with your local directory
NUM_CLASSES_WITH_BG = 0  # replace with your number of classes including background

# Placeholder dataloaders — set these before running
train_loader = None  # replace with your train dataloader
val_loader = None  # replace with your validation dataloader


def objective(trial):
    """Optuna objective for tuning the SSDLite model.

    This function builds a model and optimizer from trial-suggested
    hyperparameters, runs the `bohb_tunner` training loop and reports
    intermediate results via the provided callback. It returns the best
    validation loss observed for the trial.

    Args:
        trial (optuna.trial.Trial): Optuna trial object used to suggest hyperparameters.

    Returns:
        float: Best validation loss observed by the tuner for this trial.
    """
    # define callback to report intermidiate results
    def on_train_epoch_end(score, epoch):
        trial.report(score, step=epoch)  
        if trial.should_prune():
            raise optuna.TrialPruned()
        
    # suggest hyperparameters for the model
    INITIAL_LR = trial.suggest_float("INITIAL_LR", 1e-5, 1e-1, log=True)
    START_FACTOR = trial.suggest_float("START_FACTOR", 1e-5, 1e-1, log=True)
    WEIGHT_DECAY = trial.suggest_float("WEIGHT_DECAY", 1e-5, 1e-1, log=True)
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
            "start_factor": START_FACTOR,
            "end_factor": END_FACTOR
        },
        model=model,
        optimizer=optimizer,
        dataloaders={"train":train_loader, "val":val_loader},
        callback=on_train_epoch_end
    )
    # Return the best validation loss observed by the tuner
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