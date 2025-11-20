"""
Generate publication-quality visualization for research paper.
Creates a comprehensive results visualization showing overall accuracy
and class-wise performance metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for academic papers
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Data from classification report
classes = ['Brown Bright', 'Grey Bright', 'Healthy', 'Helioptis', 'Leaf Scorch', 'Red Rust']
precision = [0.98, 0.98, 1.00, 1.00, 0.99, 0.99]
recall = [0.99, 0.99, 1.00, 0.99, 0.99, 0.99]
f1_score = [0.99, 0.98, 1.00, 0.99, 0.99, 0.99]
support = [309, 280, 293, 294, 315, 309]

overall_accuracy = 99.17

# Create figure with subplots
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, height_ratios=[1, 1.5])

# Main title with overall accuracy
ax_title = fig.add_subplot(gs[0, :])
ax_title.axis('off')
ax_title.text(0.5, 0.5, f'Overall Accuracy: {overall_accuracy}%', 
              ha='center', va='center', fontsize=32, fontweight='bold',
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

# Bar chart for metrics comparison
ax1 = fig.add_subplot(gs[1, 0])
x = np.arange(len(classes))
width = 0.25

bars1 = ax1.bar(x - width, precision, width, label='Precision', color='#2ecc71', alpha=0.8)
bars2 = ax1.bar(x, recall, width, label='Recall', color='#3498db', alpha=0.8)
bars3 = ax1.bar(x + width, f1_score, width, label='F1-Score', color='#e74c3c', alpha=0.8)

ax1.set_xlabel('Disease Classes', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Class-wise Performance Metrics', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
ax1.set_ylim([0.95, 1.02])
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_axisbelow(True)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

# Horizontal bar chart for F1-scores (alternative visualization)
ax2 = fig.add_subplot(gs[1, 1])
colors = ['#e74c3c' if score < 1.0 else '#27ae60' for score in f1_score]
bars = ax2.barh(classes, f1_score, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

ax2.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
ax2.set_title('F1-Score by Class', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlim([0.95, 1.01])
ax2.grid(True, alpha=0.3, axis='x')
ax2.set_axisbelow(True)

# Add value labels
for i, (bar, score) in enumerate(zip(bars, f1_score)):
    width = bar.get_width()
    ax2.text(width + 0.002, bar.get_y() + bar.get_height()/2.,
            f'{score:.2f}',
            ha='left', va='center', fontsize=10, fontweight='bold')
    # Add support count
    ax2.text(0.955, bar.get_y() + bar.get_height()/2.,
            f'n={support[i]}',
            ha='left', va='center', fontsize=9, style='italic', alpha=0.7)

# Add text box with summary statistics
textstr = f'Macro Avg: 0.99\nWeighted Avg: 0.99\nTotal Samples: 1,800'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, fontweight='bold')

# Overall figure title
fig.suptitle('Tea Leaf Disease Classification Results', 
             fontsize=18, fontweight='bold', y=0.98)

# Save with high DPI for publication
plt.savefig('research_results.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✅ Research visualization saved to research_results.png")

plt.close()

