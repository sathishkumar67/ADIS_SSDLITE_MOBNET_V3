"""Plotting helpers: precision-recall curves and confusion matrices.

This module provides convenient plotting functions used in the notebook and
scripts to visualize model performance (precision-recall per class and
confusion matrices). The plotting functions accept PyTorch models and
DataLoaders and use matplotlib / seaborn for visualization.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
from torchvision.ops import box_iou

# Plot precision-recall curves for all classes function
def plot_precision_recall_curves(
    CLASSES: List[str],
    num_classes: int,
    IOU_THRESHOLD: float,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    dataset_name: str = "Validation",
    figsize: Tuple[int, int] = (8, 8),
    show_mean: bool = True
    ) -> None:
    """
    Plots precision-recall curves for all classes and the average curve, with AUCs.

    Args:
        CLASSES (List[str]): List of class names.
        num_classes (int): Number of classes.
        IOU_THRESHOLD (float): IoU threshold for true positive.
        model (torch.nn.Module): Trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation.
        device (torch.device): Device to run model on.
        figsize (Tuple[int, int], optional): Figure size for plot. Defaults to (8, 8).
        show_mean (bool, optional): Whether to plot the average curve. Defaults to True.

    Returns:
        None. Displays the precision-recall plot.
    """
    # Initialize containers for scores and true labels per class
    class_scores: List[List[float]] = [[] for _ in range(num_classes)]
    class_true: List[List[int]] = [[] for _ in range(num_classes)]

    # Set model to evaluation mode for inference
    model.eval()
    # Progress bar for inference
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Inference", leave=True):
            # Move images to device if not already
            images = images.to(device) if isinstance(images, torch.Tensor) else [img.to(device) for img in images]
            outputs = model(images)
            for out, tgt in zip(outputs, targets):
                gt_boxes  = tgt["boxes"].to(device)
                gt_labels = tgt["labels"].to(device)
                pred_boxes  = out["boxes"]
                pred_scores = out["scores"]
                pred_labels = out["labels"]
                if pred_boxes.numel() == 0:
                    continue
                # Compute IoU matrix between predicted and ground truth boxes
                ious = box_iou(pred_boxes.to(device), gt_boxes)
                for idx in range(pred_boxes.size(0)):
                    score = float(pred_scores[idx].item())
                    label = int(pred_labels[idx].item())
                    mask = (gt_labels == label)
                    if mask.sum() > 0:
                        iou_vals = ious[idx, mask]
                        max_iou = float(iou_vals.max().item())
                        is_tp = 1 if max_iou >= IOU_THRESHOLD else 0
                    else:
                        is_tp = 0
                    # Only consider valid class labels
                    if 0 < label <= num_classes:
                        class_scores[label-1].append(score)
                        class_true[label-1].append(is_tp)

    # Plotting
    plt.figure(figsize=figsize)
    mean_precisions: List[np.ndarray] = []
    mean_recalls: List[np.ndarray] = []
    mean_aucs: List[float] = []

    for i, cls in enumerate(CLASSES):
        # Skip classes with insufficient data for PR curve
        if len(class_true[i]) == 0 or len(set(class_true[i])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(class_true[i], class_scores[i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{cls} (AUC={pr_auc:.3f})')
        # Interpolate for averaging
        mean_precisions.append(np.interp(np.linspace(0,1,100), recall[::-1], precision[::-1]))
        mean_recalls.append(np.linspace(0,1,100))
        mean_aucs.append(pr_auc)

    # Plot average PR curve if requested and data exists
    if show_mean and mean_precisions:
        avg_precision = np.mean(mean_precisions, axis=0)
        avg_recall = mean_recalls[0]
        avg_auc = auc(avg_recall, avg_precision)
        plt.plot(avg_recall, avg_precision, 'k--', linewidth=2, label=f'Average (AUC={avg_auc:.3f})')

    # Plot formatting
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision–Recall Curves for All Classes (IoU ≥ {IOU_THRESHOLD:.2f}) ({dataset_name})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




# Plotting confusion matrix function
def plot_confusion_matrix(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str],
    num_classes: int,
    figsize: Optional[tuple] = (10, 8),
    title: Optional[str] = None
    ) -> None:
    """
    Plots the confusion matrix for a given model and dataloader.

    Args:
        model (torch.nn.Module): Trained model for evaluation.
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation.
        device (torch.device): Device to run model on.
        class_names (List[str]): List of class names for axis labels.
        num_classes (int): Number of classes (excluding background).
        figsize (tuple, optional): Size of the plot. Defaults to (10, 8).
        title (str, optional): Title for the plot. Defaults to None.

    Returns:
        None. Displays the confusion matrix plot.
    """
    print("[Confusion Matrix] Initializing collection of predictions and ground truths...")
    all_gt: List[int] = []
    all_pred: List[int] = []
    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Inference", leave=True):
            # Move images to device
            images = images.to(device) if isinstance(images, torch.Tensor) else [img.to(device) for img in images]
            outputs = model(images)
            for out, tgt in zip(outputs, targets):
                gt_labels = tgt["labels"].cpu().numpy()
                pred_labels = out["labels"].cpu().numpy()
                # If no predictions, count all GT as missed (-1 for missed predictions)
                if len(pred_labels) == 0:
                    all_gt.extend(gt_labels)
                    all_pred.extend([-1]*len(gt_labels))
                else:
                    # Match predictions to GT by order (for summary, not strict evaluation)
                    min_len = min(len(gt_labels), len(pred_labels))
                    all_gt.extend(gt_labels[:min_len])
                    all_pred.extend(pred_labels[:min_len])
                    
    print("[Confusion Matrix] Computing confusion matrix...")
    cm = confusion_matrix(all_gt, all_pred, labels=list(range(1, num_classes+1)))
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plot_title = title if title is not None else 'Confusion Matrix'
    plt.title(plot_title)
    plt.tight_layout()
    plt.show()
    print("[Confusion Matrix] Plot displayed. Function complete.")