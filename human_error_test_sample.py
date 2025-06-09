import numpy as np
import matplotlib.pyplot as plt
import utils.mnist_reader as mnist_reader
from joblib import load
import os

# Load the test data
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

# Load the scaler and model
scaler = load('scaler.joblib')  # Update path if needed
final_mlp = load('final_mlp.joblib')  # Update path if needed

# Scale the test data
X_te_scaled = scaler.transform(X_test)

# Predict
y_pred = final_mlp.predict(X_te_scaled)

# Target values to inspect
target_values = [2, 4, 6]

# Find misclassified indices where true label is 2, 4, or 6 and prediction is wrong
misclassified_idx = [i for i in range(len(y_test)) if (y_test[i] in target_values and y_pred[i] != y_test[i])]

# Sample 15 or fewer if not enough
sampled_idx = misclassified_idx[:15]

# Create directory
os.makedirs('human_error_test', exist_ok=True)

# Save images and info
for i, idx in enumerate(sampled_idx):
    img = X_test[idx].reshape(28, 28)
    plt.imsave(f'human_error_test/img_{i}_true_{y_test[idx]}_pred_{y_pred[idx]}.png', img, cmap='gray')

# Save info as a CSV for reference
import csv
with open('human_error_test/info.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Index', 'True Label', 'Predicted Label'])
    for i, idx in enumerate(sampled_idx):
        writer.writerow([idx, y_test[idx], y_pred[idx]])

print(f"Saved {len(sampled_idx)} misclassified images with true labels 2, 4, or 6 to 'human_error_test/'.") 