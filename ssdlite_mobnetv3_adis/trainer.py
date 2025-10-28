"""Training utilities and helpers.

This module provides training loops used by the project including a standard
`train` loop and a `bohb_tunner` wrapper that reports progress to hyperparameter
tuning callbacks. The implementations use PyTorch and provide common
functionality: warmup schedulers, early stopping, checkpointing, and logging.
"""

from __future__ import annotations
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR



def train(
    args: dict,
    model: nn.Module,
    optimizer: optim.Optimizer,
    dataloaders: dict[str, torch.utils.data.DataLoader]
) -> None:
    """
    Train an object detection model with linear warmup, early stopping on val loss, and loss tracking.

    Args:
        args (dict): Dictionary containing training parameters:
            - device (torch.device): Device to train on (e.g., 'cuda' or 'cpu').
            - warmup_epochs (int): Number of epochs for linear warmup.
            - num_epochs (int): Total number of epochs for training.
            - patience (int): Early stopping patience in epochs (val loss based).
            - start_factor (float): Start factor for linear warmup.
            - end_factor (float): End factor for linear warmup.
            - output_dir (str): Directory to save the best model checkpoint.
        model (nn.Module): The detection model.
        optimizer (optim.Optimizer): Optimizer instance.
        dataloaders (dict): Dict with 'train' and 'val' DataLoader.
    """
    # Create output directory if it doesn't exist
    os.makedirs(args["output_dir"], exist_ok=True)
    # Unpack dataloaders
    train_loader, val_loader = dataloaders['train'], dataloaders['val']

    # Prepare lists for tracking loss
    training_loss = []
    validation_loss = []

    # Set up linear warmup scheduler
    scheduler = LinearLR(
        optimizer,
        start_factor=args["start_factor"],
        end_factor=args["end_factor"],
        total_iters=args["warmup_epochs"]
    )

    # Initialize best validation loss and patience counter
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(1, args["num_epochs"] + 1):
        model.train()
        total_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args['num_epochs']}", unit="batch")
        for images, targets in train_bar:
            images = images.to(args["device"])
            targets = [{k: v.to(args["device"]) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        avg_train_loss = total_loss / len(train_loader)
        training_loss.append(avg_train_loss)

        # Step scheduler
        scheduler.step()

        # Validation
        total_val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validating", unit="batch"):
                images = images.to(args["device"])
                targets = [{k: v.to(args["device"]) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                total_val_loss += sum(loss for loss in loss_dict.values()).item()
        avg_val_loss = total_val_loss / len(val_loader)
        validation_loss.append(avg_val_loss)

        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        # Early stopping on val loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }
            torch.save(checkpoint, os.path.join(args["output_dir"], 'best_checkpoint.pth'))
            print(f"Saved new best model (Val Loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args["patience"]:
                print(f"Early stopping at epoch {epoch} (no improvement for {args['patience']} epochs)")
                break

    # After training, save loss arrays
    training_loss = np.array(training_loss)
    validation_loss = np.array(validation_loss)
    np.save(os.path.join(args["output_dir"], 'training_loss.npy'), training_loss)
    np.save(os.path.join(args["output_dir"], 'validation_loss.npy'), validation_loss)
    print("Saved training and validation loss arrays")
            
                        
def bohb_tunner(
    args: dict,
    model: nn.Module,
    optimizer: optim.Optimizer,
    dataloaders: dict[str, torch.utils.data.DataLoader],
    callback
) -> float:
    """
    Train an object detection model with linear warmup, early stopping on val loss, and BOHB callback.

    Args:
        args (dict): Dictionary containing training parameters:
            - device (torch.device): Device to train on (e.g., 'cuda' or 'cpu').
            - warmup_epochs (int): Number of epochs for linear warmup.
            - num_epochs (int): Total number of epochs for training.
            - patience (int): Early stopping patience in epochs (val loss based).
            - start_factor (float): Start factor for linear warmup.
            - end_factor (float): End factor for linear warmup.
        model (nn.Module): The detection model.
        optimizer (optim.Optimizer): Optimizer instance.
        dataloaders (dict): Dict with 'train' and 'val' DataLoader.
        callback (Callable): Callback function for BOHB, receiving (val_loss, epoch).

    Returns:
        float: Best validation loss achieved.
    """
    # Unpack dataloaders
    train_loader, val_loader = dataloaders['train'], dataloaders['val']

    # Set up linear warmup scheduler
    scheduler = LinearLR(
        optimizer,
        start_factor=args["start_factor"],
        end_factor=args["end_factor"],
        total_iters=args["warmup_epochs"]
    )

    # Initialize best validation loss and patience counter
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args["num_epochs"] + 1):
        # Training phase
        model.train()
        total_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{args['num_epochs']}", unit="batch"):
            images = images.to(args["device"])
            targets = [{k: v.to(args["device"]) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Step scheduler
        scheduler.step()

        # Validation phase
        total_val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validating", unit="batch"):
                images = images.to(args["device"])
                targets = [{k: v.to(args["device"]) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                total_val_loss += sum(loss for loss in loss_dict.values()).item()
        avg_val_loss = total_val_loss / len(val_loader)

        # Report to BOHB
        callback(avg_val_loss, epoch)
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args["patience"]:
                print(f"Early stopping at epoch {epoch} (no improvement for {args['patience']} epochs)")
                break

    return best_val_loss