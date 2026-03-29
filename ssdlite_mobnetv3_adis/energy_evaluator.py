"""
Advanced evaluation script with energy consumption, FPS, latency, and FPR metrics.
"""

import os
import time
import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from torchvision.ops import box_iou
from .utils import GPUMonitor
from .dataset import collate_fn, SSDLITEOBJDET_DATASET, CachedSSDLITEOBJDET_DATASET

def evaluate_with_energy(
    model, 
    dataloader, 
    device, 
    class_names, 
    conf_thresh=0.2, 
    iou_thresh=0.5
):
    """
    Evaluates the model and tracks energy, FPS, latency, and FPR.
    """
    model.eval()
    model.to(device)
    
    counters = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
    total_images = 0
    total_inference_time = 0.0
    # True Negatives: Images where a class is NOT present and NOT detected
    true_negatives_per_class = defaultdict(int) 
    
    device_idx = device.index if device.type == 'cuda' and device.index is not None else 0
    
    print(f"Starting advanced evaluation on {device}...")
    
    with GPUMonitor(device_index=device_idx) as monitor:
        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc="Evaluating"):
                batch_size = images.size(0)
                total_images += batch_size
                
                # Move images to device
                images = images.to(device)
                
                # Inference
                outputs = model(images)
                
                for out, tgt in zip(outputs, targets):
                    pred_boxes = out["boxes"].cpu()
                    pred_scores = out["scores"].cpu()
                    pred_labels = out["labels"].cpu()
                    true_boxes = tgt["boxes"].cpu()
                    true_labels = tgt["labels"].cpu()
                    
                    # Filter detections by confidence
                    keep = pred_scores > conf_thresh
                    pred_boxes = pred_boxes[keep]
                    pred_labels = pred_labels[keep]
                    pred_scores = pred_scores[keep]
                    
                    # Classes present in ground truth for this image
                    gt_classes_in_image = set(true_labels.tolist())
                    # Classes detected by model for this image
                    pred_classes_in_image = set(pred_labels.tolist())

                    # Update True Negatives per class
                    for cls_idx, _ in enumerate(class_names, 1):
                        if cls_idx not in gt_classes_in_image:
                            if cls_idx not in pred_classes_in_image:
                                true_negatives_per_class[cls_idx] += 1

                    # Track support
                    for lbl in true_labels.tolist():
                        counters[int(lbl)]["support"] += 1
                        
                    # Handle empty GT or Predictions for matching
                    if true_boxes.numel() == 0:
                        # If no GT, every detection is an FP
                        for lbl in pred_labels.tolist():
                            counters[int(lbl)]["fp"] += 1
                        continue
                        
                    if pred_boxes.numel() == 0:
                        # If no detections, every GT is an FN
                        for lbl in true_labels.tolist():
                            counters[int(lbl)]["fn"] += 1
                        continue
                        
                    # Matching (TP and remaining FP/FN)
                    iou_matrix = box_iou(pred_boxes, true_boxes)
                    matches = torch.nonzero(iou_matrix > iou_thresh, as_tuple=False)
                    matched_pred, matched_true = set(), set()
                    
                    for pi, ti in matches.tolist():
                        if pi in matched_pred or ti in matched_true:
                            continue
                        p_lbl = int(pred_labels[pi].item())
                        t_lbl = int(true_labels[ti].item())
                        
                        if p_lbl == t_lbl:
                            counters[p_lbl]["tp"] += 1
                            matched_pred.add(pi)
                            matched_true.add(ti)
                    
                    # Unmatched detections are FP
                    for pi in range(len(pred_boxes)):
                        if pi not in matched_pred:
                            lbl = int(pred_labels[pi].item())
                            counters[lbl]["fp"] += 1
                            
                    # Unmatched GT are FN
                    for ti in range(len(true_boxes)):
                        if ti not in matched_true:
                            lbl = int(true_labels[ti].item())
                            counters[lbl]["fn"] += 1
    
    energy_metrics = monitor.get_metrics()
    
    results = {}
    total_tp = 0
    total_fp = 0
    total_tn = 0
    
    for cls_idx, name in enumerate(class_names, 1): 
        cnt = counters[cls_idx]
        tp, fp, fn, support = cnt["tp"], cnt["fp"], cnt["fn"], cnt["support"]
        tn = true_negatives_per_class[cls_idx]
        
        total_tp += tp
        total_fp += fp
        total_tn += tn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # FPR = FP / (FP + TN)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        results[name] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Support": support,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "FPR": fpr
        }
        
    df = pd.DataFrame(results).T
    
    # Overall metrics
    overall_fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0.0
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Average GPU Power: {energy_metrics['avg_power_w']:.2f} W")
    print(f"Total GPU Energy: {energy_metrics['total_energy_j']:.2f} J")
    print(f"Overall False Positive Rate: {overall_fpr:.4f}")
    print("="*50)
    
    return df, energy_metrics

def evaluate_video_energy(model, video_path, device, class_names, conf_thresh=0.2):
    """
    Evaluates model on a video and tracks energy and FPS.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
        
    model.eval()
    model.to(device)
    
    total_frames = 0
    total_inference_time = 0.0
    
    device_idx = device.index if device.type == 'cuda' and device.index is not None else 0
    
    print(f"Starting video evaluation: {video_path}")
    
    with GPUMonitor(device_index=device_idx) as monitor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            total_frames += 1
            
            # Preprocess
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (320, 320))
            img_tensor = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            # Inference
            start_inf = time.time()
            with torch.no_grad():
                _ = model(img_tensor)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            inf_time = time.time() - start_inf
            total_inference_time += inf_time
            
            if total_frames % 100 == 0:
                print(f"Processed {total_frames} frames...")
                
    cap.release()
    
    energy_metrics = monitor.get_metrics()
    fps = total_frames / total_inference_time if total_inference_time > 0 else 0
    avg_latency_ms = (total_inference_time / total_frames) * 1000
    
    print("\n" + "="*50)
    print("VIDEO EVALUATION SUMMARY")
    print("="*50)
    print(f"Average GPU Power: {energy_metrics['avg_power_w']:.2f} W")
    print(f"Total GPU Energy: {energy_metrics['total_energy_j']:.2f} J")
    print(f"FPS: {fps:.2f}")
    print(f"Latency per frame: {avg_latency_ms:.2f} ms")
    print("="*50)
    
    return energy_metrics
