"""
Significance testing for model comparison.
Provides Wilcoxon Signed-Rank Test and Paired t-Test to compare two model versions 
(e.g., untuned vs. BOHB-tuned) across per-class metrics.
"""

import os
import torch
import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate
from .evaluate import compute_average_metrics

def cohen_d(x, y):
    """Calculates Cohen's d for effect size."""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)
    pooled_std = np.sqrt(((nx-1)*std_x**2 + (ny-1)*std_y**2) / dof)
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std

def run_significance_tests(df_a, df_b, model_a_name, model_b_name, alpha=0.05):
    """
    Performs Wilcoxon and Paired t-test between two metric DataFrames.
    df_a: Baseline model metrics (untuned)
    df_b: Comparison model metrics (tuned)
    """
    # Filter rows to include only common classes (excluding 'Average')
    common_classes = [c for c in df_a.index if c in df_b.index and c != 'Average']
    df_a = df_a.loc[common_classes]
    df_b = df_b.loc[common_classes]
    
    metrics = ['precision', 'recall', 'mAP@50', 'mAP@[50:95]', 'f1_score']
    # Mapping for pretty printing
    display_metrics = {
        'precision': 'Precision',
        'recall': 'Recall',
        'mAP@50': 'mAP50',
        'mAP@[50:95]': 'mAP50-95',
        'f1_score': 'F1-Score'
    }
    
    wilcoxon_results = []
    ttest_results = []
    
    for metric in metrics:
        if metric not in df_a.columns or metric not in df_b.columns:
            continue
            
        data_a = df_a[metric].values
        data_b = df_b[metric].values
        
        # Mean Difference
        mean_diff = np.mean(data_b) - np.mean(data_a)
        
        # Check if data are identical to avoid warnings
        if np.array_equal(data_a, data_b):
            w_stat, w_p, w_sig, w_interp = 0.0, 1.0, "No", "Data are identical"
            t_stat, t_p, t_sig, t_cohen, t_interp = 0.0, 1.0, "No", 0.0, "Data are identical"
        else:
            # Wilcoxon Signed-Rank Test
            try:
                w_stat, w_p = stats.wilcoxon(data_a, data_b)
                w_sig = "Yes" if w_p < alpha else "No"
                w_interp = f"{model_b_name} is significantly better (p={w_p:.4f})" if w_sig == "Yes" and mean_diff > 0 else \
                           f"{model_a_name} is significantly better (p={w_p:.4f})" if w_sig == "Yes" and mean_diff < 0 else \
                           f"No significant difference (p={w_p:.4f})"
            except (ValueError, RuntimeWarning):
                w_stat, w_p, w_sig, w_interp = 0.0, 1.0, "No", "N/A (insufficient variance)"

            # Paired t-Test
            try:
                t_stat, t_p = stats.ttest_rel(data_a, data_b)
                t_sig = "Yes" if t_p < alpha else "No"
                t_cohen = cohen_d(data_b, data_a)
                t_interp = f"{model_b_name} is significantly better (p={t_p:.4f})" if t_sig == "Yes" and mean_diff > 0 else \
                           f"{model_a_name} is significantly better (p={t_p:.4f})" if t_sig == "Yes" and mean_diff < 0 else \
                           f"No significant difference (p={t_p:.4f})"
            except (ValueError, RuntimeWarning):
                t_stat, t_p, t_sig, t_cohen, t_interp = 0.0, 1.0, "No", 0.0, "N/A (insufficient variance)"

        wilcoxon_results.append({
            "Metric": display_metrics[metric],
            "Statistic": w_stat,
            "p-value": w_p,
            "Significant": w_sig,
            "Mean Diff (B-A)": mean_diff,
            "Interpretation": w_interp
        })
        
        ttest_results.append({
            "Metric": display_metrics[metric],
            "t-Statistic": t_stat,
            "p-value": t_p,
            "Significant": t_sig,
            "Mean Diff (B-A)": mean_diff,
            "Cohen's d": t_cohen,
            "Interpretation": t_interp
        })
    
    return pd.DataFrame(wilcoxon_results), pd.DataFrame(ttest_results)

def print_comparison_report(df_a, df_b, model_a_name, model_b_name, w_results, t_results, alpha=0.05):
    """Prints a styled report similar to the YOLO significance test output."""
    print("=" * 90)
    print("  STATISTICAL SIGNIFICANCE TESTING — SSDLite-MobileNetV3 Model Comparison")
    print("=" * 90)
    print(f"  Models compared  : {model_a_name}, {model_b_name}")
    print(f"  Number of classes: {len(df_a.index) - 1 if 'Average' in df_a.index else len(df_a.index)}")
    print(f"  Base alpha       : {alpha}")
    print("=" * 90)
    
    print("\n  Per-Model Mean Metrics:")
    print("-" * 90)
    summary_cols = ['precision', 'recall', 'mAP@50', 'mAP@[50:95]', 'f1_score']
    # Mapping for output display
    print_map = {'precision': 'Precision', 'recall': 'Recall', 'mAP@50': 'mAP50', 'mAP@[50:95]': 'mAP50-95', 'f1_score': 'F1'}
    
    mean_a = df_a.loc['Average', summary_cols] if 'Average' in df_a.index else df_a[summary_cols].mean()
    mean_b = df_b.loc['Average', summary_cols] if 'Average' in df_b.index else df_b[summary_cols].mean()
    
    means_df = pd.DataFrame([mean_a, mean_b], index=[model_a_name, model_b_name])
    means_df.columns = [print_map[c] for c in means_df.columns]
    print(means_df.round(4).to_string())
    
    print("-" * 90)
    print(f"  PAIRWISE COMPARISON: {model_a_name} vs {model_b_name}")
    print(f"  (alpha = {alpha:.6f})")
    print("-" * 90)
    
    print("\n  >> Wilcoxon Signed-Rank Test:")
    print(tabulate(w_results, headers='keys', tablefmt='psql', showindex=False))
    
    print("\n  >> Paired t-Test:")
    print(tabulate(t_results, headers='keys', tablefmt='psql', showindex=False))
    
    sig_w = w_results['Significant'].value_counts().get('Yes', 0)
    sig_t = t_results['Significant'].value_counts().get('Yes', 0)
    
    print("\n" + "=" * 90)
    print("  SUMMARY")
    print("=" * 90)
    print(f"  [+] Wilcoxon: {sig_w}/{len(w_results)} metrics showed significant differences")
    print(f"  [+] t-test:   {sig_t}/{len(t_results)} metrics showed significant differences")
    print("=" * 90)

def compare_checkpoints(ckpt_a_path, ckpt_b_path, dataloader, device, class_names, model_gen):
    """Loads checkpoints, evaluates them, and runs tests."""
    model_a_name = os.path.basename(ckpt_a_path).replace(".pth", "").replace(".pt", "")
    model_b_name = os.path.basename(ckpt_b_path).replace(".pth", "").replace(".pt", "")
    
    print(f"Evaluating {model_a_name}...")
    model_a = model_gen()
    ckpt_a = torch.load(ckpt_a_path, map_location=device)
    model_a.load_state_dict(ckpt_a['model_state_dict'])
    model_a.to(device)
    df_a = compute_average_metrics(model_a, dataloader, device, class_names)
    
    print(f"\nEvaluating {model_b_name}...")
    model_b = model_gen()
    ckpt_b = torch.load(ckpt_b_path, map_location=device)
    model_b.load_state_dict(ckpt_b['model_state_dict'])
    model_b.to(device)
    df_b = compute_average_metrics(model_b, dataloader, device, class_names)
    
    w_results, t_results = run_significance_tests(df_a, df_b, model_a_name, model_b_name)
    print_comparison_report(df_a, df_b, model_a_name, model_b_name, w_results, t_results)
    
    return df_a, df_b, w_results, t_results
