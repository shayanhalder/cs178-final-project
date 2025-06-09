import numpy as np
import matplotlib.pyplot as plt
import utils.mnist_reader as mnist_reader
from joblib import load

# Load the test data
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

# Load the scaler and model (assuming you have them saved, adjust if needed)
from sklearn.preprocessing import StandardScaler
scaler = load('scaler.joblib')  # Update path if needed
final_mlp = load('final_mlp.joblib')  # Update path if needed

# Scale the test data
X_te_scaled = scaler.transform(X_test)

# Predict
y_pred = final_mlp.predict(X_te_scaled)

# Find misclassified indices
misclassified_idx = np.where(y_pred != y_test)[0]

# Save misclassified images and their info
np.savez('misclassified_images.npz',
         images=X_test[misclassified_idx],
         true_labels=y_test[misclassified_idx],
         pred_labels=y_pred[misclassified_idx],
         indices=misclassified_idx)

print(f"Saved {len(misclassified_idx)} misclassified images to 'misclassified_images.npz'.")

# Optionally, save a few sample images for quick inspection
import os
os.makedirs('misclassified_samples', exist_ok=True)
for i, idx in enumerate(misclassified_idx[:20]):
    img = X_test[idx].reshape(28, 28)
    plt.imsave(f'misclassified_samples/img_{i}_true_{y_test[idx]}_pred_{y_pred[idx]}.png', img, cmap='gray') 