from __future__ import annotations
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR



def train(
    args: dict,
    model: nn.Module,
    optimizer: optim.Optimizer,
    dataloaders: dict[str, torch.utils.data.DataLoader]
) -> None:
    """
    Train an object detection model with linear warmup, cosine decay, EMA, and early stopping on val loss.

    Args:
        args (dict): Dictionary containing training parameters:
            - device (torch.device): Device to train on (e.g., 'cuda' or 'cpu').
            - warmup_epochs (int): Number of epochs for linear warmup.
            - num_epochs (int): Total number of epochs for training.
            - patience (int): Early stopping patience in epochs (val loss based).
            - initial_lr (float): Initial learning rate.
            - lr_factor (float): Factor to reduce learning rate after warmup.
            - ema_decay (float): Decay factor for EMA (0 < ema_decay < 1).
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

    # Set up LR schedulers: linear warmup then cosine annealing
    scheduler_warmup = LinearLR(optimizer, start_factor=args["start_factor"], end_factor=args["end_factor"], total_iters=args["warmup_epochs"])
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=(args["num_epochs"] - args["warmup_epochs"]), eta_min=args["initial_lr"] * args["lr_factor"])
    # SequentialLR to combine warmup and cosine annealing
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[args["warmup_epochs"]])

    # Initialize EMA weights
    ema_model = {name: param.detach().cpu().clone() for name, param in model.state_dict().items()}

    # Initialize best validation loss and patience counter
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(1, args["num_epochs"] + 1):
        # Training
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

            with torch.no_grad():
                for name, param in model.state_dict().items():
                    ema_model[name] = args["ema_decay"] * ema_model[name] + (1 - args["ema_decay"]) * param.detach().cpu()

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
                'ema_state_dict': ema_model,
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
) -> None:
    """
    Train an object detection model with linear warmup, cosine decay, EMA, and early stopping on val loss.

    Args:
        args (dict): Dictionary containing training parameters:
            - device (torch.device): Device to train on (e.g., 'cuda' or 'cpu').
            - warmup_epochs (int): Number of epochs for linear warmup.
            - num_epochs (int): Total number of epochs for training.
            - patience (int): Early stopping patience in epochs (val loss based).
            - initial_lr (float): Initial learning rate.
            - lr_factor (float): Factor to reduce learning rate.
            - start_factor (float): Start factor for linear warmup.
            - end_factor (float): End factor for linear warmup.
        model (nn.Module): The detection model.
        optimizer (optim.Optimizer): Optimizer instance.
        dataloaders (dict): Dict with 'train' and 'val' DataLoader.
        callback (Callable): Callback function for BOHB.
    """
    # Unpack dataloaders
    train_loader, val_loader = dataloaders['train'], dataloaders['val']

    # Set up LR schedulers: linear warmup then cosine annealing
    scheduler_warmup = LinearLR(optimizer, start_factor=args["start_factor"], end_factor=args["end_factor"], total_iters=args["warmup_epochs"])
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=(args["num_epochs"] - args["warmup_epochs"]), eta_min=args["initial_lr"] * args["lr_factor"])
    # SequentialLR to combine warmup and cosine annealing
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[args["warmup_epochs"]])

    # Initialize best validation loss and patience counter
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(1, args["num_epochs"] + 1):
        # Training
        model.train()
        # Initialize total loss for the epoch
        total_loss = 0.0
        # Create a tqdm progress bar for training
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args['num_epochs']}", unit="batch")
        for images, targets in train_bar:
            # Move images to device
            images = images.to(args["device"])
            # Move targets to device
            targets = [{k: v.to(args["device"]) for k, v in t.items()} for t in targets]

            # Forward pass and compute loss
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        # Calculate average loss for the epoch
        avg_train_loss = total_loss / len(train_loader)

        # Step scheduler
        scheduler.step()

        # Validation
        total_val_loss = 0.0
        # no gradient calculation for validation
        with torch.no_grad():
            # Create a tqdm progress bar for validation
            for images, targets in tqdm(val_loader, desc="Validating", unit="batch"):
                # Move images to device
                images = images.to(args["device"])
                # Move targets to device
                targets = [{k: v.to(args["device"]) for k, v in t.items()} for t in targets]
                # Forward pass and compute loss
                loss_dict = model(images, targets)
                # Accumulate validation loss
                total_val_loss += sum(loss for loss in loss_dict.values()).item()
                
        # Calculate average validation loss
        avg_val_loss = total_val_loss / len(val_loader)
        
        # report the average validation loss to the BOHB callback
        callback(avg_val_loss, epoch)

        # Print training and validation loss
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        # Early stopping on val loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args["patience"]:
                print(f"Early stopping at epoch {epoch} (no improvement for {args['patience']} epochs)")
                break
    
    # return the best validation loss
    return best_val_loss