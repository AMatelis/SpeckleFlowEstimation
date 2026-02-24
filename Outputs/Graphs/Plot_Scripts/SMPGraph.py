import matplotlib.pyplot as plt
import numpy as np

# Model names
models = [
    "3D CNN Regression", 
    "3D CNN Class-Based", 
    "2D CNN Regression", 
    "2D CNN Class-Based", 
    "2D Linear Regression", 
    "3D CNN + Dropout/BatchNorm"
]

# Example representative accuracies for low, mid, high flows (as fractions 0-1)
# Replace these with your actual results from predictions.csv or metrics
low_flow_acc = [0.734, 1.0, 0.65, 1.0, 0.7, 0.97]
mid_flow_acc = [0.92, 1.0, 0.93, 1.0, 0.95, 0.94]
high_flow_acc = [0.988, 1.0, 0.99, 1.0, 0.99, 0.95]

# X positions for plotting
x = np.arange(len(models))

# Set up the figure
plt.figure(figsize=(12, 6))
plt.plot(x, low_flow_acc, marker='o', linestyle='-', color='#1f77b4', label='Low Flow (0–50 µL/min)')
plt.plot(x, mid_flow_acc, marker='s', linestyle='--', color='#ff7f0e', label='Mid Flow (50–150 µL/min)')
plt.plot(x, high_flow_acc, marker='^', linestyle='-.', color='#2ca02c', label='High Flow (100–400 µL/min)')

# Labels and title
plt.xticks(x, models, rotation=25, ha='right')
plt.ylabel("Accuracy / Prediction Agreement")
plt.ylim(0.6, 1.05)
plt.title("Comparison of Blood Flow Prediction Models Across Flow Ranges", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='lower right')
plt.tight_layout()

# Save figure as high-resolution PNG
plt.savefig("bloodflow_model_comparison.png", dpi=300)

# Show figure
plt.show()