import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")

# Load data
df = pd.read_csv("reports/classical/method_comparison.csv")
df_test = df[df['split'] == 'test'].drop_duplicates(subset=['method']).copy()

# Sort by macro F1 score
df_test = df_test.sort_values(by='macro_f1', ascending=False)

# Shorten method names for better display
name_map = {
    'HOG+LBP+colour RBF-SVM': 'SVM',
    'LBP+GLCM+Contours Random Forest': 'Random Forest',
    'Extra Trees': 'Extra Trees',
    'XGBoost': 'XGBoost',
    'LightGBM': 'LightGBM',
    'colour+morphology': 'Baseline (Colour)'
}
df_test['method_short'] = df_test['method'].map(lambda x: name_map.get(x, x))

# 1. Create a figure for Macro F1
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='macro_f1', y='method_short', data=df_test, hue='method_short', legend=False, palette='viridis')
plt.title('Macro F1 Score Comparison (Test Set)', fontsize=16)
plt.xlabel('Macro F1 Score', fontsize=12)
plt.ylabel('Model', fontsize=12)
plt.xlim(0, 1.0)

# Add value labels to bars
for i, v in enumerate(df_test['macro_f1']):
    ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('reports/classical/f1_comparison_bar.png', dpi=300)
plt.close()

# 2. Create a grouped bar chart for Precision, Recall, and F1
df_melted = pd.melt(df_test, id_vars=['method_short'], 
                    value_vars=['macro_precision', 'macro_recall', 'macro_f1'],
                    var_name='Metric', value_name='Score')

# Clean metric names
df_melted['Metric'] = df_melted['Metric'].map({
    'macro_precision': 'Precision',
    'macro_recall': 'Recall',
    'macro_f1': 'F1 Score'
})

plt.figure(figsize=(12, 7))
sns.barplot(x='Score', y='method_short', hue='Metric', data=df_melted, palette='muted')
plt.title('Detailed Metric Comparison (Test Set)', fontsize=16)
plt.xlabel('Score', fontsize=12)
plt.ylabel('Model', fontsize=12)
plt.xlim(0, 1.0)
plt.legend(title='Metrics', loc='lower right')
plt.tight_layout()
plt.savefig('reports/classical/detailed_metrics_bar.png', dpi=300)
plt.close()

# 3. Radar Chart
categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(polar=True))
colors = sns.color_palette("tab10", len(df_test))
for (index, row), color in zip(df_test.iterrows(), colors):
    name = row['method_short']
    values = [row['accuracy'], row['macro_precision'], row['macro_recall'], row['macro_f1']]
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=name, color=color)
    ax.fill(angles, values, alpha=0.1, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=12)
ax.set_ylim(0, 1)
plt.title('Model Comparison Radar Chart (Test Set)', size=16, y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), frameon=True)
plt.tight_layout()
plt.savefig('reports/classical/radar_chart.png', dpi=300, bbox_inches='tight')
plt.close()

print("Charts generated successfully!")
