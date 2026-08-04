import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def generate_comparison_charts(report_dir: Path, comparison_csv_path: Path, plot_results: list[dict[str, object]]):
    """Generates bar charts, ROC, PR, and Radar charts."""
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(comparison_csv_path)
    df_test = df[df['split'] == 'test'].copy()
    df_test = df_test.sort_values(by='macro_f1', ascending=False)
    
    # 1. Macro F1 Bar Chart
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='macro_f1', y='method', data=df_test, hue='method', legend=False, palette='viridis')
    plt.title('Macro F1 Score Comparison (Test Set)', fontsize=16)
    plt.xlabel('Macro F1 Score', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.xlim(0, 1.0)
    for i, v in enumerate(df_test['macro_f1']):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(report_dir / 'f1_comparison_bar.png', dpi=300)
    plt.close()

    # 2. Detailed Metrics Bar Chart
    df_melted = pd.melt(df_test, id_vars=['method'], 
                        value_vars=['macro_precision', 'macro_recall', 'macro_f1'],
                        var_name='Metric', value_name='Score')
    df_melted['Metric'] = df_melted['Metric'].map({
        'macro_precision': 'Precision',
        'macro_recall': 'Recall',
        'macro_f1': 'F1 Score'
    })
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Score', y='method', hue='Metric', data=df_melted, palette='muted')
    plt.title('Detailed Metric Comparison (Test Set)', fontsize=16)
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.xlim(0, 1.0)
    plt.legend(title='Metrics', loc='lower right')
    plt.tight_layout()
    plt.savefig(report_dir / 'detailed_metrics_bar.png', dpi=300)
    plt.close()

    # 3. ROC and PR Curves (For classes 1 and 2, assuming 0 is normal, 1 is fire, 2 is smoke)
    # We will plot Macro ROC and PR if we have probabilities
    plt.figure(figsize=(10, 8))
    for res in plot_results:
        y_true = res['y_true']
        y_proba = res['y_proba']
        name = res['name']
        
        if y_proba is not None and y_proba.shape[1] == 3: # Needs 3 classes
            # Simple Macro-average ROC
            fpr, tpr, roc_auc = {}, {}, {}
            for i in range(3):
                fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Approximate macro by interpolating
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(3):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= 3
            roc_auc_macro = auc(all_fpr, mean_tpr)
            
            plt.plot(all_fpr, mean_tpr, label=f"{name} (Macro AUC = {roc_auc_macro:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Macro ROC Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(report_dir / 'roc_curves.png', dpi=300)
    plt.close()

    # 4. Radar Chart
    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for index, row in df_test.iterrows():
        name = row['method']
        values = [row['accuracy'], row['macro_precision'], row['macro_recall'], row['macro_f1']]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=name)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0, 1)
    plt.title('Model Radar Chart', size=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(report_dir / 'radar_chart.png', dpi=300)
    plt.close()
