import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LinearRegression

# Standard plot styling for presentation slides
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- PART A: Feature Weight Bar Chart ---
features = ['Z-Drift\n(Texture Shift)', 'XY-Var\n(Spatial Spread)', 'Tortuosity\n(Path Complexity)']
weights = [0.6136, 0.2709, -0.7186] # From your regression output
colors = ['#2b5c8f' if w > 0 else '#d95f02' for w in weights]

bars = ax1.bar(features, weights, color=colors, width=0.5, edgecolor='black', linewidth=1)
ax1.axhline(0, color='black', linewidth=1)
ax1.set_ylabel("Standardized Regression Weight", fontsize=12)
ax1.set_title("Unified 3D Diagnostic Feature Importance", fontsize=13, fontweight='bold')
ax1.set_ylim(-1.0, 1.0)

for bar, w in zip(bars, weights):
    yval = bar.get_height()
    offset = 0.05 if yval >= 0 else -0.08
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + offset, f"{w:+.4f}", 
             ha='center', va='bottom' if yval >= 0 else 'top', fontsize=11, fontweight='bold')
rank_z = rankdata(z_drift)
rank_xy = rankdata(xy_var)
rank_tort = rankdata(tortuosity)
rank_y = rankdata(true_porosity_var)

X_ranks = np.column_stack((rank_z, rank_xy, rank_tort))
reg = LinearRegression().fit(X_ranks, rank_y)
predicted_ranks = reg.predict(X_ranks)

ax2.scatter(rank_y, predicted_ranks, color='#2b5c8f', edgecolors='k', s=70, alpha=0.8, zorder=3)

m, b = np.polyfit(rank_y, predicted_ranks, 1)
ax2.plot(rank_y, m * rank_y + b, color='#e7298a', linestyle='--', linewidth=2, label=f'Spearman ρ = 0.5704')

ax2.set_xlabel("Actual 3D Porosity Variance Rank", fontsize=12)
ax2.set_ylabel("Predicted Diagnostic Rank", fontsize=12)
ax2.set_title("Model Rank Alignment (Ground Truth vs. Diagnostic)", fontsize=13, fontweight='bold')
ax2.legend(loc='upper left', fontsize=11, frameon=True)

plt.tight_layout()
plt.savefig("results/Unified_Feature_Importance_and_Scatter.png", dpi=300)
plt.show()