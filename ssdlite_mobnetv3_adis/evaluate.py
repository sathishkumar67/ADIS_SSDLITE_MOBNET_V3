# from typing import Tuple
# import pandas as pd
# from torchvision.ops import box_iou
# from collections import defaultdict
# from torchmetrics.detection import MeanAveragePrecision


# def evaluate_model(model, dataloader, device, iou_threshold=0.5):
#     metric = MeanAveragePrecision(
#         box_format='xyxy',
#         iou_type='bbox',
#         iou_thresholds=[iou_threshold],
#         class_metrics=True,
#         extended_summary=True
#     )
    
#     model.eval()
#     with torch.no_grad():
#         for images, targets in dataloader:
#             # Move images to the device
#             outputs = model(images.to(device))
            
#             # Convert outputs to TorchMetrics format
#             preds = []
#             for i, output in enumerate(outputs):
#                 preds.append({
#                     'boxes': output['boxes'].cpu(),
#                     'scores': output['scores'].cpu(),
#                     'labels': output['labels'].cpu()
#                 })
            
#             # Convert targets to TorchMetrics format
#             targs = []
#             for target in targets:
#                 targs.append({
#                     'boxes': target['boxes'].cpu(),
#                     'labels': target['labels'].cpu()
#                 })
            
#             metric.update(preds, targs)
    
#     # Compute metrics
#     results = metric.compute()
#     return results


# def calculate_per_class_with_iou(model, dataloader, device, classes,
#                                 conf_thresh=0.2, iou_thresh=0.5):
#     counters  = defaultdict(lambda: {"tp":0,"fp":0,"fn":0,"support":0})
#     iou_sums   = defaultdict(float)
#     iou_counts = defaultdict(int)

#     model.eval()
#     with torch.no_grad():
#         for images, targets in dataloader:
#             outputs = model([img.to(device) for img in images])
#             for output, target in zip(outputs, targets):
#                 # Prepare tensors
#                 pred_boxes  = output["boxes"].cpu()
#                 pred_scores = output["scores"].cpu()
#                 pred_labels = output["labels"].cpu()
#                 true_boxes  = target["boxes"]
#                 true_labels = target["labels"]

#                 # Filter by confidence
#                 keep = pred_scores > conf_thresh
#                 pred_boxes  = pred_boxes[keep]
#                 pred_labels = pred_labels[keep]

#                 # Count support
#                 for lbl in true_labels.tolist():
#                     counters[lbl]["support"] += 1

#                 # No predictions → all GT are FN
#                 if pred_boxes.numel() == 0:
#                     for lbl in true_labels.tolist():
#                         counters[lbl]["fn"] += 1
#                     continue

#                 # Compute IoU matrix and find matches
#                 iou_matrix = box_iou(pred_boxes, true_boxes)
#                 matches    = torch.nonzero(iou_matrix > iou_thresh, as_tuple=False)

#                 matched_pred, matched_true = set(), set()
#                 for pi, ti in matches.tolist():
#                     matched_pred.add(pi); matched_true.add(ti)
#                     p_lbl = int(pred_labels[pi].item())
#                     t_lbl = int(true_labels[ti].item())

#                     if p_lbl == t_lbl:
#                         counters[p_lbl]["tp"] += 1
#                         iou_sums[p_lbl]   += iou_matrix[pi, ti].item()
#                         iou_counts[p_lbl] += 1
#                     else:
#                         counters[p_lbl]["fp"] += 1
#                         counters[t_lbl]["fn"] += 1

#                 # Unmatched → FP or FN
#                 for pi in range(len(pred_boxes)):
#                     if pi not in matched_pred:
#                         cls = int(pred_labels[pi].item())
#                         counters[cls]["fp"] += 1
#                 for ti in range(len(true_boxes)):
#                     if ti not in matched_true:
#                         cls = int(true_labels[ti].item())
#                         counters[cls]["fn"] += 1

#     # Build results
#     results = {}
#     for cls, cnt in counters.items():
#         tp, fp, fn, sup = cnt["tp"], cnt["fp"], cnt["fn"], cnt["support"]
#         prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#         rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#         f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
#         avg_iou = iou_sums[cls]/iou_counts[cls] if iou_counts[cls]>0 else 0.0
#         accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
#         # Store results
#         results[cls] = {
#             "count": sup,
#             "precision": prec,
#             "recall":    rec,
#             "f1_score":  f1,
#             "accuracy":  accuracy,
#             "avg_iou":   avg_iou
#         }
        
#     # Convert to DataFrame for better readability
#     df_metrics = pd.DataFrame(results).T
#     df_metrics.index = [classes[idx-1] for idx in df_metrics.index]
#     df_metrics = df_metrics.sort_index()
#     map_score =  evaluate_model(model, dataloader, device)["map_per_class"].cpu().tolist()
#     df_metrics["mAP"] = map_score
#     return df_metrics

# import time
# start_time = time.time()
# df_metrics = calculate_per_class_with_iou(model, test_loader, device, classes=CLASSES)
# df_metrics.loc["Average"] = df_metrics.mean()
# print(f"Per-class metrics for test set:\n{df_metrics}")
# end_time = time.time()
# print(f"Time taken for set evaluation: {end_time - start_time:.2f} seconds")